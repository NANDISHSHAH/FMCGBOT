"""
Public entry point: QNAAgent wraps OrchestratorAgent + per-session memory so
callers (CLI, notebook, API) just do:

    agent = QNAAgent()
    result = agent.chat("How did NutriOat Gold do in North?", session_id="s1")
    print(result["answer"])
"""
from __future__ import annotations
from typing import Any, Dict, Optional

from src.core.orchestrator import OrchestratorAgent
from src.core.memory import SessionMemory
from src.core.llm_client import LLMClient, MOCK_MODE


class QNAAgent:
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        llm = LLMClient(model=model, provider=provider) if (model or provider) else LLMClient()
        self.orchestrator = OrchestratorAgent(llm)
        self._sessions: Dict[str, SessionMemory] = {}
        self.mock_mode = llm.mock

    def _get_session(self, session_id: str) -> SessionMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory(session_id)
        return self._sessions[session_id]

    def chat(self, text: str, session_id: str = "default") -> Dict[str, Any]:
        memory = self._get_session(session_id)
        return self.orchestrator.handle_turn(text, memory)

    def reset_session(self, session_id: str = "default"):
        memory = self._sessions.pop(session_id, None)
        if memory is None:
            memory = SessionMemory(session_id)
        memory.clear()
