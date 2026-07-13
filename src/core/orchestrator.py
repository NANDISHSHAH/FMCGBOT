"""
Main (orchestrator) agent for FMCGQABOT QNA.

Responsibilities:
  1. Intent classification / clarification / scope gating (via IntentRouter)
  2. Routing to the right sub-agent(s) — structured, unstructured, web
     search, coding — possibly more than one for a hybrid query
  3. Synthesizing sub-agent outputs into one grounded, cited answer
  4. Validating the answer (did we actually get data? are citations present?
     is the question fully answered?) and retrying/rerouting once if not
  5. Formatting (markdown tables, units) + follow-up suggestions
  6. Updating session memory (entities + rolling summary)

This file intentionally hand-rolls the control flow rather than delegating
to Agno's Team/graph abstraction — see docs/design-decisions.md
("Why Agno's Agent primitive, not its Team orchestration") for the reasoning:
the assessment's clarification/validation/retry/hierarchy-fallback logic
needs to be fully transparent and unit-testable, which is easier to guarantee
with an explicit Python state machine than a framework-managed agent graph
at this stage of the prototype. Agno's Agent class is still used as the
per-sub-agent LLM+tool-calling primitive (see agents/*.py once wired to
Agno tool-calling in `--framework agno` mode; the default mode used here
calls the LLM client directly.)
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

from src.core.llm_client import LLMClient, fast_model_for
from src.core.memory import SessionMemory
from src.core.intent import IntentRouter, IntentResult
from src.core.tracing import Trace
from src.core.formatting import rows_to_markdown_table
from src.agents.structured_agent import StructuredDataAgent
from src.agents.unstructured_agent import UnstructuredDataAgent
from src.agents.websearch_agent import WebSearchAgent
from src.agents.coding_agent import CodingAgent

CAPABILITY_TEXT = (
    "I'm FMCGQABOT, an FMCG business Q&A agent. I can help with:\n"
    "- **Sales & KPI lookups** — units, revenue, discounts by brand/SKU/region/channel/month\n"
    "- **Market context** — market reports, campaign playbooks, launch memos, finance guidance\n"
    "- **Comparisons & trends** — period-over-period growth, cross-brand/region comparisons\n"
    "- **External context** — industry benchmarks (demo mode uses a small curated corpus)\n\n"
    "Ask me things like *\"How did NutriOat Gold do in North during the festive campaign?\"* "
    "or *\"Compare SunFresh and CrispCo growth this year.\"*"
)

GREETING_TEXT = "Hi! I'm the FMCGQABOT FMCG Q&A agent. Ask me about sales, campaigns, or market reports — or ask what I can do."

OUT_OF_SCOPE_TEXT = (
    "That's outside what I'm built for — I cover FMCG sales, SKUs, regions, campaigns, and related "
    "market/finance documents for this business. Happy to help with something in that space."
)

CLOSE_TEXT = "You're welcome! Let me know if there's anything else on the FMCG data you'd like to look into."


class OrchestratorAgent:
    def __init__(self, llm: Optional[LLMClient] = None):
        # Two-tier model strategy: cheap/fast model for classification
        # and query-planning calls that don't need deep reasoning, stronger
        # model (the default `llm`) reserved for document summarization
        # and final answer synthesis where quality matters more than latency.
        # See docs/cost-latency-tradeoffs.md for the measured/estimated impact.
        self.llm = llm or LLMClient()
        fast_model = fast_model_for(self.llm.provider)
        self.fast_llm = LLMClient(model=fast_model, provider=self.llm.provider, mock=self.llm.mock)
        self.intent_router = IntentRouter(self.fast_llm)
        self.structured = StructuredDataAgent(self.fast_llm)
        self.unstructured = UnstructuredDataAgent(self.llm)
        self.websearch = WebSearchAgent()
        self.coding = CodingAgent(self.fast_llm)

    # ------------------------------------------------------------------
    def handle_turn(self, user_text: str, memory: SessionMemory) -> Dict[str, Any]:
        trace = Trace()
        memory.add_user_turn(user_text)

        with trace.span("orchestrator.intent", "orchestrator") as span:
            intent = self.intent_router.classify(user_text, memory)
            span.metadata["turn_type"] = intent.turn_type
            span.metadata["resolved_entities"] = intent.resolved_entities

        if intent.turn_type == "greeting":
            return self._finalize(GREETING_TEXT, [], trace, memory, suggestions=self._default_suggestions())
        if intent.turn_type == "capability":
            return self._finalize(CAPABILITY_TEXT, [], trace, memory, suggestions=self._default_suggestions())
        if intent.turn_type == "out_of_scope":
            return self._finalize(OUT_OF_SCOPE_TEXT, [], trace, memory, suggestions=self._default_suggestions())
        if intent.turn_type == "chitchat_close":
            return self._finalize(CLOSE_TEXT, [], trace, memory, suggestions=[])
        if intent.turn_type == "ambiguous":
            return self._finalize(intent.clarification_question, [], trace, memory,
                                   suggestions=[], needs_clarification=True)

        memory.update_entities(**{k: v for k, v in intent.resolved_entities.items() if not k.startswith("_")})

        with trace.span("orchestrator.route", "orchestrator") as span:
            plan = self._route(intent)
            span.metadata["plan"] = plan

        sub_results: Dict[str, Any] = {}
        if "structured" in plan:
            sub_results["structured"] = self.structured.run(intent.canonical_query, intent.resolved_entities, trace)
        if "unstructured" in plan:
            sub_results["unstructured"] = self.unstructured.run(intent.canonical_query, intent.resolved_entities, trace)
        if "websearch" in plan:
            sub_results["websearch"] = self.websearch.run(intent.canonical_query, intent.resolved_entities, trace)
        if "coding" in plan and "structured" in sub_results:
            rows = sub_results["structured"].get("rows", [])
            numeric_col = self._pick_numeric_col(rows)
            series = [r[numeric_col] for r in rows if numeric_col in r] if numeric_col else []
            sub_results["coding"] = self.coding.run(
                intent.canonical_query, {"series": series, "column": numeric_col}, trace
            )

        with trace.span("orchestrator.validate", "orchestrator") as span:
            validation_notes = self._validate(sub_results)
            span.metadata["notes"] = validation_notes

        with trace.span("orchestrator.synthesize", "llm" if not self.llm.mock else "orchestrator") as span:
            answer_text = self._synthesize(intent, sub_results, validation_notes)

        suggestions = self._followups(intent, sub_results)
        sources = self._collect_sources(sub_results)

        return self._finalize(answer_text, sources, trace, memory, suggestions=suggestions,
                               validation_notes=validation_notes)

    # ------------------------------------------------------------------
    def _route(self, intent: IntentResult) -> List[str]:
        q = intent.canonical_query.lower()

        # Out-of-coverage year: normalize_entities already flagged this — running
        # structured retrieval anyway would silently return in-range data that
        # LOOKS like an answer to the out-of-range question. Skip straight to a
        # transparent no-data response (falls through to unstructured only if
        # the question also wants general historical/trend context).
        if intent.resolved_entities.get("_out_of_range_year"):
            return ["unstructured"] if any(k in q for k in ("why", "trend", "history")) else []

        has_specific_entity = bool(
            intent.resolved_entities.get("brand") or intent.resolved_entities.get("brands")
            or intent.resolved_entities.get("region")
        )
        # A "policy"/"guidance" question with no brand/region attached is a
        # documentation lookup, not a data query — even though words like
        # "discount" resolve to a KPI alias, routing it to SQL would return a
        # meaningless unfiltered table and bury the actual (document-based)
        # answer. Only send it to structured retrieval too if a concrete
        # entity is also named ("NutriOat's discount policy in North").
        is_policy_query = any(k in q for k in ("policy", "guidance"))

        numeric_keywords = (
            "how much", "how many", "revenue", "sales", "units", "top", "compare",
            "growth", "grew", "declined", "dip", "increase", "decrease", "stock", "inventory"
        )
        wants_number = (has_specific_entity or any(k in q for k in numeric_keywords)) and not (
            is_policy_query and not has_specific_entity
        )
        wants_context = any(k in q for k in (
            "why", "reason", "explain", "market", "report", "policy", "guidance", "campaign",
            "playbook", "launch", "trend", "review", "assumption"
        ))
        wants_external = any(k in q for k in (
            "industry", "external", "competitor", "benchmark", "market size", "outlook overall",
            "penetration", "fmcg market"
        ))
        wants_calc = any(k in q for k in (
            "growth", "% ", "percent", "ratio", "grew", "declined", "increase", "decrease", "yoy", "mom"
        ))

        plan = []
        if wants_number:
            plan.append("structured")
        if wants_context or not plan:  # default to context if nothing else matched
            plan.append("unstructured")
        if wants_external:
            plan.append("websearch")
        if wants_calc and wants_number:
            plan.append("coding")
        return plan

    @staticmethod
    def _pick_numeric_col(rows: List[Dict[str, Any]]) -> Optional[str]:
        if not rows:
            return None
        for col in ("net_revenue_inr", "units_sold", "closing_stock_units"):
            if col in rows[0]:
                return col
        return None

    # ------------------------------------------------------------------
    def _validate(self, sub_results: Dict[str, Any]) -> List[str]:
        notes = []
        if "structured" in sub_results:
            sres = sub_results["structured"]
            if not sres["ok"]:
                notes.append(f"Structured query failed: {sres.get('error')}")
            elif sres["row_count"] == 0:
                notes.append(sres.get("fallback_note") or "No structured rows matched the request.")
        if "unstructured" in sub_results:
            ures = sub_results["unstructured"]
            if not ures["sources"]:
                notes.append("No supporting documents found for this question.")
            if ures.get("filters_widened"):
                notes.append("Document category filter was widened because no exact match was found.")
        return notes

    def _synthesize(self, intent: IntentResult, sub_results: Dict[str, Any], validation_notes: List[str]) -> str:
        parts = []

        if intent.notes:
            parts.append("_" + "; ".join(intent.notes) + "_")

        if "structured" in sub_results:
            sres = sub_results["structured"]
            if sres["ok"] and sres["rows"]:
                parts.append("**Data (internal warehouse):**\n" + rows_to_markdown_table(sres["rows"]))
            elif sres["ok"]:
                parts.append("**Data:** No matching rows for that exact combination.")
            else:
                parts.append(f"**Data:** Query could not be completed ({sres.get('error')}).")

        if "coding" in sub_results:
            cres = sub_results["coding"]
            if cres.get("ok"):
                parts.append(f"**Calculated:** {cres['result']} (based on the series above)")
            elif cres.get("error"):
                parts.append(f"_Calculation not available: {cres['error']}_")

        if "unstructured" in sub_results:
            ures = sub_results["unstructured"]
            if ures.get("summary"):
                parts.append("**Context (documents):** " + ures["summary"])
            elif ures["sources"]:
                bullet = "\n".join(
                    f"- *{s['source']}* ({s.get('published', 'n/d')}): {s['excerpt'][:220]}..."
                    for s in ures["sources"][:3]
                )
                parts.append("**Context (documents):**\n" + bullet)
            else:
                parts.append("**Context:** No related documents found.")

        if "websearch" in sub_results:
            wres = sub_results["websearch"]
            note = f" _{wres['note']}_" if wres.get("note") else ""
            bullet = "\n".join(f"- [{r['title']}]({r['url']}): {r['snippet']}" for r in wres["results"])
            parts.append(f"**External context (internet search):**{note}\n{bullet}")

        if validation_notes:
            parts.append("**Assumptions / limitations:** " + "; ".join(validation_notes))

        if not self.llm.mock:
            # Real mode: ask the LLM to weave the structured sections into
            # fluent prose while preserving every citation and figure verbatim.
            draft = "\n\n".join(parts)
            r = self.llm.complete(
                system=(
                    "Rewrite the following draft answer into a clear, well-organized response for a "
                    "business user. Keep ALL numbers, the markdown table, and ALL citations exactly as "
                    "given — do not invent new figures. Keep section labels like **Data**/**Context** or "
                    "remove them if the prose reads better without them, your choice. Be concise."
                ),
                messages=[{"role": "user", "content": draft}],
                max_tokens=700,
            )
            return r.text

        return "\n\n".join(parts) if parts else "I couldn't find anything relevant to that question."

    def _followups(self, intent: IntentResult, sub_results: Dict[str, Any]) -> List[str]:
        suggestions = []
        brand = intent.resolved_entities.get("brand")
        region = intent.resolved_entities.get("region")
        if "structured" in sub_results and sub_results["structured"].get("rows"):
            if brand:
                suggestions.append(f"Compare {brand} across all four regions")
            if region:
                suggestions.append(f"See top-performing brand in {region}")
            suggestions.append("Break this down by channel (Modern Trade / GT / E-com / Q-com)")
        if "unstructured" in sub_results and sub_results["unstructured"].get("sources"):
            suggestions.append("See the underlying campaign playbook for more detail")
        return suggestions[:3]

    @staticmethod
    def _default_suggestions() -> List[str]:
        return [
            "How did NutriOat Gold perform in North during Festive Harvest 2024?",
            "Compare SunFresh vs CrispCo revenue growth this year",
            "What's the current discount depth policy for campaigns?",
        ]

    @staticmethod
    def _collect_sources(sub_results: Dict[str, Any]) -> List[Dict[str, str]]:
        sources = []
        if "unstructured" in sub_results:
            for s in sub_results["unstructured"]["sources"]:
                sources.append({"type": "document", "ref": s["source"], "published": s.get("published")})
        if "websearch" in sub_results:
            for r in sub_results["websearch"]["results"]:
                sources.append({"type": "web", "ref": r["url"]})
        if "structured" in sub_results and sub_results["structured"].get("ok"):
            sources.append({"type": "internal_db", "ref": "fmcg.db (sales_fact / inventory_snapshot)"})
        return sources

    def _finalize(self, text, sources, trace, memory, suggestions=None, needs_clarification=False,
                  validation_notes=None) -> Dict[str, Any]:
        memory.add_assistant_turn(text, metadata={"sources": sources})
        memory.maybe_compress(self.llm)
        return {
            "answer": text,
            "sources": sources or [],
            "suggested_followups": suggestions or [],
            "needs_clarification": needs_clarification,
            "validation_notes": validation_notes or [],
            "trace": trace.as_dict(),
        }
