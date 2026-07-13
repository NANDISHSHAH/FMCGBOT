"""Interactive Streamlit UI for FMCGQABOT.

Run with:
    streamlit run src/ui.py

Requires the optional UI dependencies (streamlit). Install with:
    uv pip install -e ".[ui]"

To use a non-default LLM provider (e.g. OpenAI), set environment variables
before launching the UI:

    export LLM_PROVIDER=openai
    export LLM_MODEL=gpt-4o
    export OPENAI_API_KEY=sk-...
    streamlit run src/ui.py
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import streamlit as st

from src.qna_agent import QNAAgent


PAGE_TITLE = "FMCGQABOT — Enterprise FMCG Q&A"
EXAMPLE_QUESTIONS = [
    "How did NutriOat Gold do in North during the festive campaign?",
    "How many units of NutriOat Gold did we sell in North in November 2024?",
    "Why did NutriOat Gold sales rise in North during the festive campaign, and was the discounting within policy?",
    "Compare CrispCo and PureWave total net revenue this year",
    "What was the growth in units sold for NutriOat Gold in North from July 2024 to November 2024?",
    "What is the current discount depth policy for campaigns?",
    "What is the industry benchmark for quick commerce penetration in FMCG?",
    "kitna revenue hua SunFresh ka South me?",
]


def init_agent() -> QNAAgent:
    """Lazy-initialize the shared QNAAgent instance."""
    if "agent" not in st.session_state:
        provider = os.environ.get("LLM_PROVIDER") or None
        model = os.environ.get("LLM_MODEL") or None
        st.session_state.agent = QNAAgent(model=model, provider=provider)
    return st.session_state.agent


def get_session_id() -> str:
    """Return the currently selected session id."""
    return st.session_state.get("session_id", "default")


def render_sidebar() -> None:
    """Render session controls and mode info in the sidebar."""
    with st.sidebar:
        st.title("FMCGQABOT")
        st.caption("Enterprise FMCG Q&A Agent")

        agent = init_agent()
        mode_color = "🟢" if not agent.mock_mode else "🟡"
        provider_label = agent.orchestrator.llm.provider.upper()
        st.markdown(f"{mode_color} **Mode:** {'Live (' + provider_label + ')' if not agent.mock_mode else 'Mock (deterministic)'}")

        st.divider()
        st.subheader("Session")

        sessions: List[str] = st.session_state.setdefault("sessions", ["default"])
        current = get_session_id()

        selected = st.selectbox(
            "Active session",
            options=sessions,
            index=sessions.index(current) if current in sessions else 0,
            key="session_select",
        )
        st.session_state.session_id = selected

        col1, col2 = st.columns(2)
        with col1:
            if st.button("New session", use_container_width=True):
                new_id = f"session_{len(sessions) + 1}"
                sessions.append(new_id)
                st.session_state.session_id = new_id
                st.rerun()
        with col2:
            if st.button("Reset session", use_container_width=True):
                agent.reset_session(get_session_id())
                st.session_state.pop(f"history_{get_session_id()}", None)
                st.rerun()

        st.divider()
        st.subheader("Example questions")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True, key=f"example_{q}"):
                st.session_state.pending_question = q
                st.rerun()


def get_history(session_id: str) -> List[Dict[str, Any]]:
    """Return the chat history for a session."""
    key = f"history_{session_id}"
    if key not in st.session_state:
        st.session_state[key] = []
    return st.session_state[key]


def ask_question(question: str) -> Dict[str, Any]:
    """Send a question to the agent and return the full result."""
    agent = init_agent()
    session_id = get_session_id()
    return agent.chat(question, session_id=session_id)


def render_message(role: str, content: str, result: Dict[str, Any] | None = None) -> None:
    """Render a single chat message with optional metadata expanders."""
    with st.chat_message(role):
        st.markdown(content)

        if result and role == "assistant":
            sources = result.get("sources") or []
            followups = result.get("suggested_followups") or []
            notes = result.get("validation_notes") or []
            trace = result.get("trace")

            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        ref = src.get("ref", src)
                        st.markdown(f"- {ref}")

            if followups:
                with st.expander("Suggested follow-ups"):
                    for fu in followups:
                        st.markdown(f"- {fu}")

            if notes:
                with st.expander("Validation notes"):
                    for note in notes:
                        st.markdown(f"- {note}")

            if trace:
                with st.expander("Trace"):
                    st.json(
                        {
                            "total_latency_ms": trace.get("total_latency_ms"),
                            "llm_latency_ms": trace.get("llm_latency_ms"),
                            "total_cost_usd": trace.get("total_cost_usd"),
                            "tokens": trace.get("tokens"),
                        }
                    )


def main() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🧴", layout="wide")
    render_sidebar()

    st.title(PAGE_TITLE)
    st.caption("Ask questions about sales, inventory, campaigns, market reports, and finance guidance.")

    session_id = get_session_id()
    history = get_history(session_id)

    # Render existing history
    for turn in history:
        render_message("user", turn["question"])
        render_message("assistant", turn["answer"], turn.get("result"))

    # Handle pending question from example buttons
    pending = st.session_state.pop("pending_question", None)

    # Chat input
    question = st.chat_input("Ask a question...", key="chat_input")
    if pending:
        question = pending

    if question:
        render_message("user", question)
        with st.spinner("Thinking..."):
            result = ask_question(question)
        answer = result.get("answer", "No answer returned.")
        render_message("assistant", answer, result)

        history.append({"question": question, "answer": answer, "result": result})
        st.session_state[f"history_{session_id}"] = history


if __name__ == "__main__":
    main()
