"""
NeMo Agent Toolkit (NAT) plugin for FMCGQABOT.

Registers the QNAAgent as a NAT workflow function so the project can be run
with:

    uv run nat run --config_file nat_workflow.yaml --input "How did NutriOat Gold do in North?"

To use a non-default provider (e.g. OpenAI), set environment variables before
running NAT:

    export LLM_PROVIDER=openai
    export LLM_MODEL=gpt-4o
    export OPENAI_API_KEY=sk-...
    uv run nat run --config_file nat_workflow.yaml --input "How did NutriOat Gold do in North?"
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


def _load_env_file_if_present() -> None:
    """Load .env key=value pairs into os.environ when not already set."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _infer_provider_and_model() -> tuple[str | None, str | None]:
    """Infer provider/model when explicit LLM_PROVIDER/LLM_MODEL are absent."""
    provider = os.environ.get("LLM_PROVIDER") or None
    model = os.environ.get("LLM_MODEL") or None

    if provider:
        return provider, model

    # Pick provider from available credentials to avoid defaulting to Gemini
    # when users only configured OPENAI_API_KEY in .env.
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", model or "gpt-4o"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", model or "claude-3-5-sonnet-latest"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini", model or "gemini-2.5-pro"

    return provider, model


class FMCGQABOTQNAWorkflowConfig(FunctionBaseConfig, name="fmcgqabot_qna_workflow"):
    """NAT workflow configuration for the FMCGQABOT Q&A agent."""
    session_id: str = Field(
        default="nat_session",
        description="Session ID used for multi-turn memory.",
    )


@register_function(config_type=FMCGQABOTQNAWorkflowConfig)
async def fmcgqabot_qna_workflow(config: FMCGQABOTQNAWorkflowConfig, builder: Builder):
    """Build the FMCGQABOT Q&A agent as a NAT-callable function."""
    _load_env_file_if_present()

    # Lazy import so the plugin module can be imported without the data files existing yet.
    from src.qna_agent import QNAAgent

    provider, model = _infer_provider_and_model()
    logger.info("NAT plugin loaded from %s with provider=%s model=%s", __file__, provider, model)
    agent = QNAAgent(model=model, provider=provider)

    async def _answer(question: str) -> str:
        """Answer a business question using the FMCGQABOT Q&A agent."""
        if not question or not question.strip():
            return "Please provide a question."
        try:
            # QNAAgent.chat is synchronous; run it in a thread pool.
            result = await asyncio.to_thread(agent.chat, question.strip(), config.session_id)
            answer = result.get("answer")
            return str(answer) if answer is not None else "No answer returned."
        except Exception:
            logger.exception("FMCGQABOT Q&A workflow failed")
            return "I encountered an error while processing your question. Please try again."

    yield FunctionInfo.from_fn(
        _answer,
        description=(
            "FMCGQABOT enterprise FMCG Q&A agent. Answers questions about "
            "synthetic sales/inventory data, market reports, finance notes, "
            "promo playbooks, and product launches."
        ),
    )
