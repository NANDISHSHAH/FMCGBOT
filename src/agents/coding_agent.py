"""
Coding sub-agent — used when the answer needs a derived calculation that
structured retrieval alone doesn't give (YoY/MoM growth, ratios, weighted
averages) over numbers the orchestrator already pulled from the structured
agent. Real mode: LLM writes the snippet given the numbers in context. Mock
mode: falls back to the pct_growth() helper for the common growth-rate case,
otherwise reports that the calculation type isn't covered by the mock
planner (documented limitation, same reasoning as structured_agent).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from src.core.llm_client import LLMClient
from src.tools.code_tool import run_code, pct_growth


class CodingAgent:
    name = "coding_agent"

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def run(self, instruction: str, numeric_context: Dict[str, Any], trace) -> Dict[str, Any]:
        with trace.span("coding_agent.plan_snippet", "llm") as span:
            snippet = self._plan_snippet(instruction, numeric_context)
            span.metadata["snippet"] = snippet

        with trace.span("coding_agent.execute", "tool") as span:
            result = run_code(snippet) if snippet else {"ok": False, "error": "No snippet planned."}
            span.metadata["ok"] = result["ok"]

        return {"agent": self.name, "snippet": snippet, **result}

    def _plan_snippet(self, instruction: str, numeric_context: Dict[str, Any]) -> Optional[str]:
        if not self.llm.mock:
            r = self.llm.complete(
                system="Write a short Python snippet (builtins + math/statistics only, no imports) "
                       "that computes the requested value and assigns it to a variable named `result`. "
                       "Output ONLY the code.",
                messages=[{"role": "user", "content": f"Task: {instruction}\nAvailable numbers: {numeric_context}"}],
                max_tokens=200,
            )
            return r.text.strip().strip("`").replace("python\n", "")

        # mock fallback: handle the common "growth" pattern deterministically
        series = numeric_context.get("series")
        if series and len(series) >= 2:
            old, new = series[0], series[-1]
            r = pct_growth(old, new)
            if r["ok"]:
                return f"result = {r['result']}"
        return None
