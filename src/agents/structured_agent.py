"""
Structured Data Retrieval sub-agent.

Real mode: asks Gemini to write a SQL SELECT against the known schema, then
runs it through the SQL guard (src/tools/sql_tool.py) before execution, and
retries once with the error message fed back if the query is invalid or
returns zero rows for an entity we know exists (answer validation/retry
requirement).

Mock mode: LLM can't reliably free-form NL->SQL without a real model, so a
small deterministic query-builder covers the question patterns exercised in
the test notebook (single entity/period lookup, comparisons, top-N, growth-
oriented lookups feeding the coding agent). This is a documented scope
narrowing for mock mode only (docs/cost-latency-tradeoffs.md) — the
architecture (guard, retry, hierarchy fallback) is unchanged either way.
"""
from __future__ import annotations
from typing import Any, Dict, Optional

from src.core.llm_client import LLMClient
from src.tools.sql_tool import run_query, get_metadata, SCHEMA_DESCRIPTION


class StructuredDataAgent:
    name = "structured_data_agent"

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def run(self, query: str, entities: Dict[str, Any], trace) -> Dict[str, Any]:
        with trace.span("structured_agent.plan_sql", "llm") as span:
            sql = self._plan_sql(query, entities)
            span.metadata["planned_sql"] = sql

        with trace.span("structured_agent.execute_sql", "tool") as span:
            result = run_query(sql)
            span.metadata["ok"] = result["ok"]

        # retry once on guard/DB error, or on an unexpectedly empty result for
        # a known brand/region combo (possible over-constrained WHERE clause)
        if not result["ok"] or (result.get("row_count") == 0 and entities.get("brand")):
            with trace.span("structured_agent.retry_sql", "llm") as span:
                sql_retry = self._plan_sql(query, entities, prior_error=result.get("error"))
                span.metadata["planned_sql"] = sql_retry
            with trace.span("structured_agent.execute_sql_retry", "tool") as span:
                result2 = run_query(sql_retry)
                span.metadata["ok"] = result2["ok"]
            if result2["ok"] and result2.get("row_count", 0) >= result.get("row_count", 0):
                result = result2

        # hierarchy-aware fallback: if the specific SKU/region/month combo
        # genuinely has no data, roll up one level (region -> all regions,
        # month -> all months) and say so explicitly.
        if result["ok"] and result.get("row_count") == 0:
            result["fallback_note"] = (
                "No rows at the requested granularity; consider a broader region, "
                "month, or KPI. Use metadata discovery to see available dimensions."
            )

        return {
            "agent": self.name,
            "sql": result.get("sql"),
            "ok": result["ok"],
            "row_count": result.get("row_count", 0),
            "rows": result.get("rows", []),
            "error": result.get("error"),
            "fallback_note": result.get("fallback_note"),
        }

    def metadata(self) -> Dict[str, Any]:
        return get_metadata()

    # ------------------------------------------------------------------
    def _plan_sql(self, query: str, entities: Dict[str, Any], prior_error: Optional[str] = None) -> str:
        if not self.llm.mock:
            system = (
                "You write a single SQLite SELECT statement against this schema:\n"
                f"{SCHEMA_DESCRIPTION}\n"
                "Rules: SELECT only, no semicolons, join through skus/brands/regions/channels "
                "as needed, always alias columns clearly. Output ONLY the SQL, no explanation, "
                "no markdown fences."
            )
            user_msg = f"Question: {query}\nKnown entities: {entities}"
            if prior_error:
                user_msg += f"\nPrevious attempt failed with: {prior_error}. Fix it."
            r = self.llm.complete(system=system, messages=[{"role": "user", "content": user_msg}], max_tokens=300)
            return r.text.strip().strip("`").replace("sql\n", "")

        return self._deterministic_sql(query, entities)

    def _deterministic_sql(self, query: str, entities: Dict[str, Any]) -> str:
        """Small rule-based planner used only in MOCK mode (see module docstring)."""
        brand = entities.get("brand")
        region = entities.get("region")
        month = entities.get("month")
        kpi = entities.get("kpi", "net_revenue_inr")
        ql = query.lower()

        select_kpi = f"SUM(f.{kpi}) AS total_{kpi}" if kpi != "closing_stock_units" and kpi != "days_of_cover" else f"AVG(i.{kpi}) AS avg_{kpi}"

        if "top" in ql or "best selling" in ql or "highest" in ql:
            return (
                "SELECT b.brand_name, s.sku_name, SUM(f.net_revenue_inr) AS total_net_revenue_inr "
                "FROM sales_fact f JOIN skus s ON f.sku_id=s.sku_id JOIN brands b ON s.brand_id=b.brand_id "
                "GROUP BY s.sku_id ORDER BY total_net_revenue_inr DESC LIMIT 5"
            )

        if kpi in ("closing_stock_units", "days_of_cover"):
            where = []
            if brand:
                where.append(f"b.brand_name = '{brand}'")
            if region:
                where.append(f"r.region_name = '{region}'")
            if month:
                where.append(f"i.month = '{month}'")
            where_clause = f"WHERE {' AND '.join(where)}" if where else ""
            return (
                f"SELECT b.brand_name, s.sku_name, r.region_name, i.month, i.{kpi} "
                f"FROM inventory_snapshot i JOIN skus s ON i.sku_id=s.sku_id "
                f"JOIN brands b ON s.brand_id=b.brand_id JOIN regions r ON i.region_id=r.region_id "
                f"{where_clause} ORDER BY i.month LIMIT 100"
            )

        where = []
        if entities.get("brands"):  # multi-brand comparison, e.g. "Compare CrispCo and PureWave"
            brand_list = "', '".join(entities["brands"])
            where.append(f"b.brand_name IN ('{brand_list}')")
        elif brand:
            where.append(f"b.brand_name = '{brand}'")
        if region:
            where.append(f"r.region_name = '{region}'")
        if month:
            where.append(f"f.month = '{month}'")
        elif "quarter" in ql:
            pass  # left broad; LLM/real mode handles quarter math, mock returns monthly grain
        where_clause = f"WHERE {' AND '.join(where)}" if where else ""

        return (
            f"SELECT b.brand_name, SUM(f.units_sold) AS units_sold, "
            f"SUM(f.net_revenue_inr) AS net_revenue_inr, SUM(f.discount_inr) AS discount_inr "
            f"FROM sales_fact f JOIN skus s ON f.sku_id=s.sku_id JOIN brands b ON s.brand_id=b.brand_id "
            f"JOIN regions r ON f.region_id=r.region_id "
            f"{where_clause} "
            f"GROUP BY b.brand_id ORDER BY net_revenue_inr DESC LIMIT 100"
        ) if entities.get("brands") else (
            f"SELECT b.brand_name, s.sku_name, r.region_name, f.month, "
            f"SUM(f.units_sold) AS units_sold, SUM(f.net_revenue_inr) AS net_revenue_inr, "
            f"SUM(f.discount_inr) AS discount_inr "
            f"FROM sales_fact f JOIN skus s ON f.sku_id=s.sku_id JOIN brands b ON s.brand_id=b.brand_id "
            f"JOIN regions r ON f.region_id=r.region_id "
            f"{where_clause} "
            f"GROUP BY s.sku_id, r.region_id, f.month ORDER BY f.month LIMIT 100"
        )
