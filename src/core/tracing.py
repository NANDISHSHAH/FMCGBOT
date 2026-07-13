"""
Lightweight tracing layer, modeled on how NVIDIA's NeMo Agent Toolkit (NAT)
instruments agentic workflows: every step (agent call, tool call, LLM call)
is captured as a span with name, duration, and metadata; spans nest under a
single trace per user turn. NAT can consume/replay this kind of span data via
its profiler and evaluation system.

Why not just depend on `nvidia-nat` directly for this? See
docs/design-decisions.md ("Why Agno + a NAT-shaped tracer, not the full NAT
CLI") -- short version: the full toolkit installs via git submodules + LFS
and targets NVIDIA NIM-served models by default, which is disproportionate
infra for a take-home prototype using the Gemini API. This module
reproduces NAT's *span shape* so the trace log below could be exported to a
real NAT deployment later with a thin adapter (see nat_workflow.yaml).
"""
from __future__ import annotations
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Span:
    name: str
    kind: str  # "agent" | "tool" | "llm" | "orchestrator"
    start_ts: float
    end_ts: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.end_ts is None:
            return 0.0
        return (self.end_ts - self.start_ts) * 1000


class Trace:
    """One Trace per user turn. Holds an ordered list of spans."""

    def __init__(self, turn_id: Optional[str] = None):
        self.turn_id = turn_id or str(uuid.uuid4())[:8]
        self.spans: List[Span] = []

    @contextmanager
    def span(self, name: str, kind: str, **metadata):
        s = Span(name=name, kind=kind, start_ts=time.time(), metadata=metadata)
        self.spans.append(s)
        try:
            yield s
        except Exception as e:
            s.error = str(e)
            raise
        finally:
            s.end_ts = time.time()

    def total_latency_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans if s.kind in ("agent", "tool", "orchestrator"))

    def llm_latency_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans if s.kind == "llm")

    def total_cost_usd(self) -> float:
        return sum(s.metadata.get("estimated_cost_usd", 0.0) for s in self.spans if s.kind == "llm")

    def total_tokens(self) -> Dict[str, int]:
        inp = sum(s.metadata.get("input_tokens", 0) for s in self.spans if s.kind == "llm")
        out = sum(s.metadata.get("output_tokens", 0) for s in self.spans if s.kind == "llm")
        return {"input_tokens": inp, "output_tokens": out}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "spans": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "duration_ms": round(s.duration_ms, 1),
                    "metadata": s.metadata,
                    "error": s.error,
                }
                for s in self.spans
            ],
            "total_latency_ms": round(self.total_latency_ms(), 1),
            "llm_latency_ms": round(self.llm_latency_ms(), 1),
            "total_cost_usd": round(self.total_cost_usd(), 6),
            "tokens": self.total_tokens(),
        }
