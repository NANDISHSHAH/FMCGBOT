import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.sql_tool import run_query


def test_valid_select_runs():
    r = run_query("SELECT brand_name FROM brands LIMIT 3")
    assert r["ok"] is True
    assert r["row_count"] <= 3


def test_rejects_ddl():
    r = run_query("DROP TABLE brands")
    assert r["ok"] is False


def test_rejects_multiple_statements():
    r = run_query("SELECT 1; DROP TABLE brands")
    assert r["ok"] is False


def test_rejects_unknown_table():
    r = run_query("SELECT * FROM sqlite_master")
    assert r["ok"] is False


def test_row_cap_enforced():
    r = run_query("SELECT * FROM sales_fact LIMIT 999999")
    assert r["ok"] is True
    assert r["row_count"] <= 200


def test_rejects_insert_disguised_as_select_comment():
    r = run_query("SELECT 1; INSERT INTO brands VALUES (99,'x','y')")
    assert r["ok"] is False
