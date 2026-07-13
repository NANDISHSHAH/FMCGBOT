"""Standardized, unit-aware formatting helpers for the final answer."""
from __future__ import annotations
from typing import Any, Dict, List

UNIT_LABELS = {
    "net_revenue_inr": "INR", "gross_revenue_inr": "INR", "discount_inr": "INR",
    "units_sold": "units", "closing_stock_units": "units", "days_of_cover": "days",
}


def fmt_number(value: Any, col: str) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    if col.endswith("_inr"):
        return f"₹{value:,.0f}"
    if col in ("days_of_cover",):
        return f"{value:.1f}"
    if isinstance(value, float):
        return f"{value:,.1f}"
    return f"{value:,}"


def rows_to_markdown_table(rows: List[Dict[str, Any]], max_rows: int = 15) -> str:
    if not rows:
        return "_No rows returned._"
    cols = list(rows[0].keys())
    header = "| " + " | ".join(c.replace("_", " ") for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(fmt_number(row[c], c) for c in cols) + " |")
    table = "\n".join(lines)
    if len(rows) > max_rows:
        table += f"\n\n_...and {len(rows) - max_rows} more row(s), truncated for readability._"
    return table
