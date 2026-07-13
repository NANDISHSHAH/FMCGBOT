"""Internet Search sub-agent — clearly labeled as external, never mixed
silently with internal data (each citation carries its own provenance so the
synthesizer/response can distinguish internal vs external claims)."""
from __future__ import annotations
from typing import Any, Dict

from src.tools.websearch_tool import WebSearchTool


class WebSearchAgent:
    name = "websearch_agent"

    def __init__(self, tool: WebSearchTool = None):
        self.tool = tool or WebSearchTool()

    def run(self, query: str, entities: Dict[str, Any], trace) -> Dict[str, Any]:
        with trace.span("websearch_agent.search", "tool") as span:
            result = self.tool.search(query, num_results=3)
            span.metadata["result_count"] = len(result["results"])
            span.metadata["is_mock"] = self.tool.is_mock
        return {"agent": self.name, **result}
