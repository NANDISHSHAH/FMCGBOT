"""
Coding sub-agent tool: executes short, restricted Python snippets for
calculations the other agents don't do natively — YoY/MoM growth, ratios,
weighted averages, unit conversions, etc.

Guardrails:
  - No imports allowed in the snippet (builtins-only + a small safe `math`
    and `statistics` namespace pre-injected).
  - `__builtins__` stripped down to a safe subset (no open/exec/eval/import).
  - Execution wrapped with a wall-clock timeout via a hard instruction count
    is not feasible in pure Python without threads/signals; here we use a
    signal-based timeout (POSIX) as a best-effort guard, documented as a
    known limitation for a Windows-hosted deployment.
"""
from __future__ import annotations
import math
import signal
import statistics
from typing import Any, Dict

SAFE_BUILTINS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "len": len, "sorted": sorted, "range": range, "enumerate": enumerate,
    "zip": zip, "float": float, "int": int, "str": str, "list": list,
    "dict": dict, "tuple": tuple, "bool": bool,
}

SAFE_GLOBALS = {
    "__builtins__": SAFE_BUILTINS,
    "math": math,
    "statistics": statistics,
}


class CodeExecutionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise CodeExecutionTimeout("Code execution exceeded 3s limit")


def run_code(snippet: str, timeout_s: int = 3) -> Dict[str, Any]:
    """
    Executes `snippet`, which must assign a variable named `result`.
    Returns {"ok": True, "result": ...} or {"ok": False, "error": ...}.
    """
    forbidden = ["import ", "__", "open(", "exec(", "eval(", "os.", "sys.", "subprocess"]
    lowered = snippet.lower()
    for f in forbidden:
        if f in lowered:
            return {"ok": False, "error": f"Snippet contains forbidden token: '{f.strip()}'"}

    local_vars: Dict[str, Any] = {}
    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        exec(snippet, dict(SAFE_GLOBALS), local_vars)
        if "result" not in local_vars:
            return {"ok": False, "error": "Snippet must assign a `result` variable."}
        return {"ok": True, "result": local_vars["result"]}
    except CodeExecutionTimeout as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def pct_growth(old: float, new: float) -> Dict[str, Any]:
    """Convenience helper the coding agent can call directly for common growth math."""
    if old == 0:
        return {"ok": False, "error": "Cannot compute growth from a zero base."}
    growth = (new - old) / old * 100
    return {"ok": True, "result": round(growth, 2)}
