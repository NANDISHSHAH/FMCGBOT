"""
Thin LLM client used across all agents.

Design decision: every sub-agent and the orchestrator talk to the LLM through
this single class, never directly to a provider SDK. That gives us one place to:
    - swap models/providers,
    - measure latency + (approximate) token/cost per call for the cost report,
    - fall back to a deterministic MOCK mode when no API key is set,
        so the whole pipeline (and the notebook) still runs end-to-end without
        live credentials -- at the cost of answer quality/generality in mock mode.
        This trade-off is documented in docs/cost-latency-tradeoffs.md.

Provider selection:
    - Pass provider="gemini"|"anthropic"|"openai" to LLMClient, or
    - Set the LLM_PROVIDER environment variable, or
    - Let the model name imply the provider (gemini-*, claude-*, gpt-*).
    - API keys are read from provider-specific env vars:
        GEMINI_API_KEY / GOOGLE_API_KEY
        ANTHROPIC_API_KEY
        OPENAI_API_KEY (and optional OPENAI_BASE_URL for compatible endpoints)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.core.llm_adapters import (
    BaseLLMAdapter,
    ADAPTER_REGISTRY,
    provider_for_model,
)

MOCK_MODE = not any(
    os.environ.get(name)
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    )
)

DEFAULT_MODEL = "gemini-2.5-pro"
FAST_MODEL = "gemini-2.0-flash"

# Provider-specific defaults when the caller supplies a provider but no model.
_PROVIDER_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-pro",
    "anthropic": "claude-3-5-sonnet-latest",
    "openai": "gpt-4o",
}

# Provider-specific fast-model defaults when the user does not override FAST_MODEL.
_PROVIDER_FAST_MODELS = {
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "openai": "gpt-4o-mini",
}


def _default_model_for(provider: Optional[str]) -> str:
    """Return a sensible default model for the given provider family."""
    if not provider:
        return DEFAULT_MODEL
    return _PROVIDER_DEFAULT_MODELS.get(provider.lower(), DEFAULT_MODEL)


def fast_model_for(provider: Optional[str] = None) -> str:
    """Return a sensible fast/cheap model for the given provider family."""
    if provider:
        return _PROVIDER_FAST_MODELS.get(provider.lower(), FAST_MODEL)
    env_provider = os.environ.get("LLM_PROVIDER")
    if env_provider:
        return _PROVIDER_FAST_MODELS.get(env_provider.lower(), FAST_MODEL)
    inferred_provider = provider_for_model(FAST_MODEL)
    return _PROVIDER_FAST_MODELS.get(inferred_provider, FAST_MODEL)


# Very rough public list-price approximations (USD / MTok) used ONLY to
# produce an illustrative cost estimate in traces -- NOT billing-accurate.
PRICE_PER_MTOK = {
    # Gemini
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.0-flash": {"input": 0.15, "output": 0.60},
    "gemini-3.5-pro": {"input": 2.5, "output": 12.0},
    "gemini-3.5-flash": {"input": 0.35, "output": 1.4},
    # Anthropic Claude (illustrative)
    "claude-opus-4": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-latest": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # OpenAI GPT (illustrative)
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o3": {"input": 10.0, "output": 40.0},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
}


def _rate_key(model: str) -> str:
    """Find the closest pricing key for a model name."""
    if model in PRICE_PER_MTOK:
        return model
    model_lower = model.lower()
    for key in PRICE_PER_MTOK:
        if model_lower.startswith(key.lower()):
            return key
    return DEFAULT_MODEL


@dataclass
class LLMCallResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    mock: bool
    estimated_cost_usd: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "mock": self.mock,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


class LLMClient:
    """
    Unified LLM client.  The actual provider call is delegated to a
    BaseLLMAdapter subclass, making it trivial to switch between Gemini,
    Anthropic Claude, and OpenAI without changing agent code.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        mock: Optional[bool] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = self._resolve_provider(provider, model or "")
        self.model = model or _default_model_for(self.provider)
        self.mock = MOCK_MODE if mock is None else mock
        self._adapter: Optional[BaseLLMAdapter] = None

        if not self.mock:
            adapter_cls = ADAPTER_REGISTRY.get(self.provider)
            if adapter_cls is None:
                raise ValueError(
                    f"Unknown LLM provider '{self.provider}'. "
                    f"Supported: {', '.join(sorted(ADAPTER_REGISTRY))}"
                )
            kwargs: Dict[str, Any] = {"model": self.model, "api_key": api_key}
            if self.provider == "openai" and base_url is not None:
                kwargs["base_url"] = base_url
            self._adapter = adapter_cls(**kwargs)

    @staticmethod
    def _resolve_provider(provider: Optional[str], model: str) -> str:
        if provider:
            return provider.lower()
        env_provider = os.environ.get("LLM_PROVIDER")
        if env_provider:
            return env_provider.lower()
        return provider_for_model(model)

    def _estimate_cost(self, in_tok: int, out_tok: int) -> float:
        rates = PRICE_PER_MTOK.get(_rate_key(self.model), PRICE_PER_MTOK[DEFAULT_MODEL])
        return (in_tok / 1_000_000) * rates["input"] + (out_tok / 1_000_000) * rates["output"]

    def complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        mock_responder: Optional[callable] = None,
    ) -> LLMCallResult:
        """
        messages: [{"role": "user"|"assistant", "content": "..."}]
        mock_responder: optional callable(system, messages) -> str used to
            produce a realistic-looking canned answer in MOCK mode. If not
            given, a generic fallback template is used.
        """
        start = time.time()
        if self.mock:
            text = mock_responder(system, messages) if mock_responder else (
                "[MOCK MODE — no LLM API key set] "
                "This is a placeholder response; wire up a real key to get live generations."
            )
            # crude token estimate: ~4 chars/token
            in_tok = sum(len(m["content"]) for m in messages) // 4 + len(system) // 4
            out_tok = len(text) // 4
            latency_ms = (time.time() - start) * 1000 + 5  # mock still has ~5ms overhead
            return LLMCallResult(
                text=text, model=f"{self.model}-MOCK", input_tokens=in_tok,
                output_tokens=out_tok, latency_ms=latency_ms, mock=True,
                estimated_cost_usd=0.0,
            )

        if self._adapter is None:
            raise RuntimeError("Adapter not initialized but mock mode is disabled.")

        result = self._adapter.complete(system=system, messages=messages, max_tokens=max_tokens)
        result.mock = False
        result.estimated_cost_usd = self._estimate_cost(result.input_tokens, result.output_tokens)
        return result
