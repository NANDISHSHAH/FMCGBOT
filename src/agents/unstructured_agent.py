"""
Unstructured Data Retrieval sub-agent — wraps doc_retrieval_tool and (in
real mode) uses the LLM to write a short grounded summary of the retrieved
excerpts with inline source citations; in mock mode returns the raw ranked
excerpts with citations directly (still fully cited, just not re-summarized).
"""
from __future__ import annotations
from typing import Any, Dict, Optional

from src.core.llm_client import LLMClient
from src.tools.doc_retrieval_tool import retrieve, list_metadata


class UnstructuredDataAgent:
    name = "unstructured_data_agent"

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def run(self, query: str, entities: Dict[str, Any], trace) -> Dict[str, Any]:
        category = self._infer_category(entities)
        with trace.span("unstructured_agent.retrieve", "tool") as span:
            hits = retrieve(query, top_k=4, category=category)
            span.metadata["result_count"] = len(hits["results"])

        summary_text = None
        if not self.llm.mock and hits["results"]:
            with trace.span("unstructured_agent.summarize", "llm") as span:
                context = "\n\n".join(
                    f"[{h['source']}] ({h['published']}): {h['excerpt']}" for h in hits["results"]
                )
                r = self.llm.complete(
                    system="Summarize these document excerpts to answer the user's question. "
                           "Cite each fact with its [source] filename inline. Be concise (3-5 sentences).",
                    messages=[{"role": "user", "content": f"Question: {query}\n\nExcerpts:\n{context}"}],
                    max_tokens=300,
                )
                summary_text = r.text
                span.metadata.update(r.as_dict())

        return {
            "agent": self.name,
            "filters_widened": hits["filters_widened_due_to_no_match"],
            "sources": hits["results"],
            "summary": summary_text,
        }

    def metadata(self) -> Dict[str, Any]:
        return list_metadata()

    @staticmethod
    def _infer_category(entities: Dict[str, Any]) -> Optional[str]:
        brand_to_category = {
            "NutriOat": "Breakfast Cereals", "SunFresh": "Juices & Beverages",
            "CrispCo": "Savory Snacks", "PureWave": "Home Care", "HomeGlow": "Personal Care",
        }
        return brand_to_category.get(entities.get("brand"))
