"""
Conversation memory for a session.

Requirements covered: multi-turn interaction, contextual follow-ups,
conversation context preservation, memory optimization for long-running
sessions.

Design: keep the last N raw turns verbatim (for faithful short-range
follow-ups: "what about last month instead?"), and roll anything older into
a running summary via the LLM so token usage doesn't grow unbounded across a
long session. This is the standard sliding-window + summarization pattern.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

RECENT_TURNS_KEPT_VERBATIM = 6


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionMemory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turns: List[Turn] = []
        self.rolling_summary: str = ""
        self.last_resolved_entities: Dict[str, Any] = {}
        # e.g. {"brand": "NutriOat", "region": "North", "month": "2024-11"}

    def add_user_turn(self, content: str):
        self.turns.append(Turn(role="user", content=content))

    def add_assistant_turn(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.turns.append(Turn(role="assistant", content=content, metadata=metadata or {}))

    def update_entities(self, **kwargs):
        """Track the last-mentioned brand/region/month/etc for follow-up resolution."""
        for k, v in kwargs.items():
            if v is not None:
                self.last_resolved_entities[k] = v

    def maybe_compress(self, llm_client=None):
        """If history is getting long, fold the oldest turns into rolling_summary."""
        if len(self.turns) <= RECENT_TURNS_KEPT_VERBATIM:
            return
        to_compress = self.turns[: -RECENT_TURNS_KEPT_VERBATIM]
        self.turns = self.turns[-RECENT_TURNS_KEPT_VERBATIM:]
        compress_text = "\n".join(f"{t.role}: {t.content}" for t in to_compress)

        if llm_client is None:
            # cheap non-LLM fallback: keep a truncated log rather than nothing
            self.rolling_summary = (self.rolling_summary + "\n" + compress_text)[-1500:]
            return

        result = llm_client.complete(
            system="Summarize this conversation excerpt in 2-3 sentences, "
                   "preserving any entities (brands, regions, months, KPIs) mentioned.",
            messages=[{"role": "user", "content": compress_text}],
            max_tokens=200,
            mock_responder=lambda s, m: "[mock summary] Earlier turns discussed FMCG sales/queries; "
                                         "entities preserved in last_resolved_entities.",
        )
        self.rolling_summary = (self.rolling_summary + "\n" + result.text).strip()

    def history_as_messages(self) -> List[Dict[str, str]]:
        msgs = []
        if self.rolling_summary:
            msgs.append({"role": "user", "content": f"[Conversation so far, summarized]: {self.rolling_summary}"})
        for t in self.turns:
            msgs.append({"role": t.role, "content": t.content})
        return msgs
