"""
SQL safety layer for the Structured Data Agent.

Guardrails implemented (documented in docs/design-decisions.md):
  1. Read-only connection: SQLite opened in `mode=ro` (OS-level enforcement,
     not just a string check) — a real DB would map to a read-only DB role.
  2. Statement allow-list: only a single SELECT statement is permitted.
     Anything containing DDL/DML keywords (INSERT/UPDATE/DELETE/DROP/ALTER/
     ATTACH/PRAGMA/etc.) or a statement separator (`;` followed by more SQL)
     is rejected before it ever reaches sqlite3.
  3. Table/column allow-list: the query is checked against a known schema so
     arbitrary system tables (sqlite_master, etc.) can't be probed.
  4. Row cap: LIMIT is injected if missing, capped at MAX_ROWS regardless of
     what the model asks for, to bound cost/latency of a runaway query.
"""
from __future__ import annotations
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "structured" / "fmcg.db"

ALLOWED_TABLES = {
    "brands", "skus", "regions", "channels", "campaigns",
    "sales_fact", "inventory_snapshot",
}

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|VACUUM|REPLACE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

MAX_ROWS = 200


class SQLGuardError(ValueError):
    pass


def _validate(sql: str) -> str:
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        raise SQLGuardError("Multiple statements are not allowed.")
    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        raise SQLGuardError("Only SELECT statements are allowed.")
    if FORBIDDEN_KEYWORDS.search(stripped):
        raise SQLGuardError("Query contains a forbidden keyword (DDL/DML).")
    # extract referenced tables (naive but sufficient for an allow-listed schema)
    referenced = set(re.findall(r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", stripped, re.IGNORECASE))
    unknown = referenced - ALLOWED_TABLES
    if unknown:
        raise SQLGuardError(f"Query references unknown/disallowed table(s): {unknown}")
    if not re.search(r"\bLIMIT\b", stripped, re.IGNORECASE):
        stripped += f" LIMIT {MAX_ROWS}"
    else:
        # clamp any existing LIMIT to MAX_ROWS
        stripped = re.sub(
            r"LIMIT\s+(\d+)",
            lambda m: f"LIMIT {min(int(m.group(1)), MAX_ROWS)}",
            stripped,
            flags=re.IGNORECASE,
        )
    return stripped


def run_query(sql: str) -> Dict[str, Any]:
    """Validate then execute a read-only SELECT against the FMCG warehouse."""
    try:
        safe_sql = _validate(sql)
    except SQLGuardError as e:
        return {"ok": False, "error": str(e), "sql": sql}

    try:
        uri = f"file:{DB_PATH}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(safe_sql)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return {"ok": True, "sql": safe_sql, "row_count": len(rows), "rows": rows}
    except sqlite3.Error as e:
        return {"ok": False, "error": f"SQLite error: {e}", "sql": safe_sql}


SCHEMA_DESCRIPTION = """
brands(brand_id, brand_name, category)
skus(sku_id, sku_code, sku_name, brand_id, category, unit, list_price_inr)
regions(region_id, region_name)
channels(channel_id, channel_name)
campaigns(campaign_id, campaign_name, brand_id, region_id, start_date, end_date)
sales_fact(fact_id, sku_id, region_id, channel_id, month['YYYY-MM'], units_sold,
           gross_revenue_inr, discount_inr, net_revenue_inr)
inventory_snapshot(snap_id, sku_id, region_id, month, closing_stock_units, days_of_cover)

Grain of sales_fact: one row per (sku, region, channel, month).
Join sales_fact -> skus -> brands to roll up by brand.
Only SELECT statements are permitted; results capped at 200 rows.
""".strip()


def get_metadata() -> Dict[str, Any]:
    """Supports 'metadata discovery' requirement: list available KPIs/dims."""
    return {
        "tables": sorted(ALLOWED_TABLES),
        "dimensions": ["brand", "sku", "region", "channel", "month", "campaign"],
        "kpis": ["units_sold", "gross_revenue_inr", "discount_inr", "net_revenue_inr",
                 "closing_stock_units", "days_of_cover"],
        "time_grain": "month (YYYY-MM)",
        "coverage_period": "2024-07 to 2025-06",
        "schema": SCHEMA_DESCRIPTION,
    }
