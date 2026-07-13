"""
Internet Search sub-agent tool.

The sandbox this prototype was built in has no egress to general web/search
APIs (network allow-list covers package registries + GitHub only), and the
assessment doesn't supply a search API key. So this ships a MockSearchProvider
with a small curated set of "external market" snippets about the same FMCG
categories used in the synthetic data (so it can plausibly be cross-referenced
against internal data in a demo answer), behind the exact interface a real
provider would implement.

Swapping in a real provider (Tavily, SerpAPI, Bing) is a one-class change:
implement `search(query, num_results)` returning the same shape and pass it
into WebSearchTool(provider=...). No calling code changes.
"""
from __future__ import annotations
import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Protocol


class SearchProvider(Protocol):
    def search(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        ...


_MOCK_CORPUS = [
    {
        "title": "India Breakfast Cereals Market Outlook 2025",
        "url": "https://example-industry-reports.test/india-cereals-2025",
        "snippet": "Industry analysts estimate the Indian breakfast cereals market grew "
                   "12-15% YoY in 2024-25, with premiumization (oat-based, whole-grain "
                   "variants) cited as the primary growth driver, particularly during "
                   "the festive quarter (Oct-Nov).",
    },
    {
        "title": "Summer 2025 Beverage Demand Trends — India",
        "url": "https://example-industry-reports.test/india-beverages-summer-2025",
        "snippet": "Juice and beverage volumes in India typically rise 30-40% during the "
                   "Mar-Jun summer window, with South and West regions showing the "
                   "highest elasticity to in-store chilled-display promotions.",
    },
    {
        "title": "FMCG Quick Commerce Penetration Report",
        "url": "https://example-industry-reports.test/quick-commerce-fmcg",
        "snippet": "Quick commerce contributes an estimated 8-12% of FMCG volume in urban "
                   "India as of 2025, up from under 3% in 2022, but remains "
                   "underpenetrated for small-format snacks and beverage SKUs.",
    },
    {
        "title": "Savory Snacks Category Competitive Landscape",
        "url": "https://example-industry-reports.test/savory-snacks-competitive",
        "snippet": "Combo/bundle promotions in savory snacks are a common trial-generation "
                   "tactic in India, typically run at 8-12% discount depth versus everyday "
                   "pricing, per category trade press.",
    },
]


class MockSearchProvider:
    def search(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        q = query.lower()
        scored = []
        for item in _MOCK_CORPUS:
            overlap = sum(1 for w in q.split() if w in (item["title"] + item["snippet"]).lower())
            scored.append((overlap, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:num_results]]


class LiveSearchProvider:
    """Live web search via external providers selected by environment variables.

    Supported providers:
      - serpapi  (requires SERPAPI_API_KEY)
      - tavily   (requires TAVILY_API_KEY)

    Configuration:
      WEBSEARCH_PROVIDER=serpapi|tavily   (default: serpapi)
      WEBSEARCH_TIMEOUT_SECONDS=10         (default: 10)
    """

    def __init__(self):
        self.provider = (os.environ.get("WEBSEARCH_PROVIDER") or "serpapi").strip().lower()
        self.timeout = float(os.environ.get("WEBSEARCH_TIMEOUT_SECONDS", "10"))

    def search(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        if self.provider == "tavily":
            return self._tavily_search(query, num_results)
        return self._serpapi_search(query, num_results)

    def _serpapi_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            raise RuntimeError("SERPAPI_API_KEY is not set")

        params = urllib.parse.urlencode(
            {
                "q": query,
                "engine": "google",
                "api_key": api_key,
                "num": max(1, min(num_results, 10)),
            }
        )
        url = f"https://serpapi.com/search.json?{params}"
        req = urllib.request.Request(url=url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        results = []
        for item in payload.get("organic_results", [])[:num_results]:
            results.append(
                {
                    "title": item.get("title", "Untitled"),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
        return results

    def _tavily_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set")

        body = json.dumps(
            {
                "query": query,
                "search_depth": "basic",
                "max_results": max(1, min(num_results, 10)),
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url="https://api.tavily.com/search",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        results = []
        for item in payload.get("results", [])[:num_results]:
            results.append(
                {
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                }
            )
        return results


class WebSearchTool:
    def __init__(self, provider: SearchProvider = None):
        self.provider = provider or self._auto_provider()
        self.is_mock = isinstance(self.provider, MockSearchProvider)

    @staticmethod
    def _auto_provider() -> SearchProvider:
        provider = (os.environ.get("WEBSEARCH_PROVIDER") or "").strip().lower()
        if provider in {"serpapi", "tavily"}:
            try:
                return LiveSearchProvider()
            except Exception:
                return MockSearchProvider()
        if os.environ.get("SERPAPI_API_KEY") or os.environ.get("TAVILY_API_KEY"):
            try:
                return LiveSearchProvider()
            except Exception:
                return MockSearchProvider()
        return MockSearchProvider()

    def search(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        note = None
        try:
            results = self.provider.search(query, num_results)
        except Exception as e:
            # Graceful fallback to curated corpus when live providers fail.
            fallback = MockSearchProvider()
            results = fallback.search(query, num_results)
            self.provider = fallback
            self.is_mock = True
            note = f"LIVE provider unavailable ({e}); served from curated mock corpus."

        return {
            "query": query,
            "provider": "mock_curated_corpus" if self.is_mock else "live",
            "note": note or (
                "MOCK: no live internet search configured in this environment; "
                "results are drawn from a small curated corpus for demo purposes."
                if self.is_mock else None
            ),
            "results": results,
        }
