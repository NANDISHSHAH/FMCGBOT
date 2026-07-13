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
import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
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

        self._persistent = False
        self._db_path = self._resolve_db_path()
        self._init_persistence()
        self._load_from_store()

    @staticmethod
    def _resolve_db_path() -> Path:
        env_path = os.environ.get("SESSION_MEMORY_DB_PATH")
        if env_path:
            return Path(env_path)
        return Path(__file__).resolve().parents[2] / "data" / "session_memory.sqlite"

    def _init_persistence(self):
        backend = (os.environ.get("SESSION_MEMORY_BACKEND") or "sqlite").strip().lower()
        if backend in {"inmemory", "none", "off", "disabled"}:
            self._persistent = False
            return
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_state (
                        session_id TEXT PRIMARY KEY,
                        rolling_summary TEXT NOT NULL DEFAULT '',
                        entities_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_turns_session_id_id
                    ON session_turns(session_id, id)
                    """
                )
            self._persistent = True
        except Exception:
            self._persistent = False

    def _load_from_store(self):
        if not self._persistent:
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT rolling_summary, entities_json FROM session_state WHERE session_id = ?",
                    (self.session_id,),
                ).fetchone()
                if row:
                    self.rolling_summary = row[0] or ""
                    try:
                        entities = json.loads(row[1] or "{}")
                        if isinstance(entities, dict):
                            self.last_resolved_entities = entities
                    except Exception:
                        self.last_resolved_entities = {}

                rows = conn.execute(
                    "SELECT role, content, metadata_json FROM session_turns WHERE session_id = ? ORDER BY id ASC",
                    (self.session_id,),
                ).fetchall()
                self.turns = []
                for role, content, metadata_json in rows:
                    metadata = {}
                    try:
                        parsed = json.loads(metadata_json or "{}")
                        if isinstance(parsed, dict):
                            metadata = parsed
                    except Exception:
                        metadata = {}
                    self.turns.append(Turn(role=role, content=content, metadata=metadata))
        except Exception:
            # Keep operating in-memory if persistence read fails.
            self._persistent = False

    def _persist_state(self):
        if not self._persistent:
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO session_state(session_id, rolling_summary, entities_json)
                    VALUES(?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        rolling_summary=excluded.rolling_summary,
                        entities_json=excluded.entities_json
                    """,
                    (self.session_id, self.rolling_summary, json.dumps(self.last_resolved_entities)),
                )
        except Exception:
            self._persistent = False

    def _persist_turn(self, turn: Turn):
        if not self._persistent:
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO session_turns(session_id, role, content, metadata_json)
                    VALUES(?, ?, ?, ?)
                    """,
                    (self.session_id, turn.role, turn.content, json.dumps(turn.metadata or {})),
                )
        except Exception:
            self._persistent = False

    def _rewrite_turns(self):
        if not self._persistent:
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM session_turns WHERE session_id = ?", (self.session_id,))
                conn.executemany(
                    """
                    INSERT INTO session_turns(session_id, role, content, metadata_json)
                    VALUES(?, ?, ?, ?)
                    """,
                    [
                        (self.session_id, t.role, t.content, json.dumps(t.metadata or {}))
                        for t in self.turns
                    ],
                )
        except Exception:
            self._persistent = False

    def add_user_turn(self, content: str):
        turn = Turn(role="user", content=content)
        self.turns.append(turn)
        self._persist_turn(turn)

    def add_assistant_turn(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        turn = Turn(role="assistant", content=content, metadata=metadata or {})
        self.turns.append(turn)
        self._persist_turn(turn)

    def update_entities(self, **kwargs):
        """Track the last-mentioned brand/region/month/etc for follow-up resolution."""
        for k, v in kwargs.items():
            if v is not None:
                self.last_resolved_entities[k] = v
        self._persist_state()

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
            self._rewrite_turns()
            self._persist_state()
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
        self._rewrite_turns()
        self._persist_state()

    def clear(self):
        self.turns = []
        self.rolling_summary = ""
        self.last_resolved_entities = {}
        if self._persistent:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute("DELETE FROM session_turns WHERE session_id = ?", (self.session_id,))
                    conn.execute("DELETE FROM session_state WHERE session_id = ?", (self.session_id,))
            except Exception:
                self._persistent = False

    def history_as_messages(self) -> List[Dict[str, str]]:
        msgs = []
        if self.rolling_summary:
            msgs.append({"role": "user", "content": f"[Conversation so far, summarized]: {self.rolling_summary}"})
        for t in self.turns:
            msgs.append({"role": t.role, "content": t.content})
        return msgs
