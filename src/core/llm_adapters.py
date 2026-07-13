"""
Provider-specific LLM adapters.

The LLMClient in src.core.llm_client is provider-agnostic and delegates to one
of these adapters.  Adding a new provider means implementing BaseLLMAdapter and
registering it in LLMClient._adapter_for_provider — no caller code changes.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Dict, Any, Optional

if TYPE_CHECKING:
    from src.core.llm_client import LLMCallResult


class BaseLLMAdapter(ABC):
    """Abstract adapter: every concrete provider implements this interface."""

    name: str = "base"

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self._client: Any = None

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
    ) -> "LLMCallResult":
        """Call the provider and return a normalized LLMCallResult."""
        ...

    def _client_for(self, module_name: str, client_cls_name: str, env_key: str):
        """Lazy-load a provider SDK client, falling back to env var API keys."""
        if self._client is not None:
            return self._client
        try:
            mod = __import__(module_name, fromlist=[client_cls_name])
        except ImportError as exc:
            raise ImportError(
                f"Provider '{self.name}' requires the '{module_name}' package. "
                f"Install it (e.g. pip install {module_name}) or choose a different provider."
            ) from exc
        cls = getattr(mod, client_cls_name)
        key = self.api_key or os.environ.get(env_key)
        self._client = cls(api_key=key)
        return self._client


class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini / Google GenAI adapter."""

    name = "gemini"

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(
            model,
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        )

    def complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
    ) -> "LLMCallResult":
        from src.core.llm_client import LLMCallResult

        client = self._client_for("google.genai", "Client", "GEMINI_API_KEY")
        prompt = self._messages_to_input(messages)
        start = time.time()
        resp = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": system,
                "max_output_tokens": max_tokens,
            },
        )
        latency_ms = (time.time() - start) * 1000
        text = getattr(resp, "text", None) or getattr(resp, "output_text", "")
        usage = getattr(resp, "usage_metadata", None) or getattr(resp, "usage", None)
        in_tok = (
            getattr(usage, "prompt_token_count", 0) or getattr(usage, "total_input_tokens", 0)
            if usage else 0
        )
        out_tok = (
            getattr(usage, "candidates_token_count", 0) or getattr(usage, "total_output_tokens", 0)
            if usage else 0
        )
        return LLMCallResult(
            text=text,
            model=self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            mock=False,
        )

    @staticmethod
    def _messages_to_input(messages: List[Dict[str, str]]) -> str:
        lines = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)


class AnthropicAdapter(BaseLLMAdapter):
    """Anthropic Claude adapter."""

    name = "anthropic"

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(
            model,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    def complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
    ) -> "LLMCallResult":
        from src.core.llm_client import LLMCallResult

        client = self._client_for("anthropic", "Anthropic", "ANTHROPIC_API_KEY")
        start = time.time()
        resp = client.messages.create(
            model=self.model,
            system=system,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )
        latency_ms = (time.time() - start) * 1000
        text = "\n".join(block.text for block in resp.content if block.type == "text")
        usage = resp.usage
        in_tok = getattr(usage, "input_tokens", 0)
        out_tok = getattr(usage, "output_tokens", 0)
        return LLMCallResult(
            text=text,
            model=self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            mock=False,
        )


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI GPT adapter (also works with OpenAI-compatible endpoints)."""

    name = "openai"

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(
            model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")

    def complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
    ) -> "LLMCallResult":
        from src.core.llm_client import LLMCallResult

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Provider 'openai' requires the 'openai' package. "
                "Install it (e.g. pip install openai) or choose a different provider."
            ) from exc

        if self._client is None:
            kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)

        chat_messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        chat_messages.extend(messages)

        start = time.time()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=chat_messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )
        latency_ms = (time.time() - start) * 1000
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0)
        out_tok = getattr(usage, "completion_tokens", 0)
        return LLMCallResult(
            text=text,
            model=self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            mock=False,
        )


ADAPTER_REGISTRY: Dict[str, type[BaseLLMAdapter]] = {
    "gemini": GeminiAdapter,
    "google": GeminiAdapter,
    "anthropic": AnthropicAdapter,
    "claude": AnthropicAdapter,
    "openai": OpenAIAdapter,
}


def provider_for_model(model: str) -> str:
    """Infer a provider from a model name string."""
    model_lower = model.lower()
    if model_lower.startswith("gemini-") or model_lower.startswith("models/gemini-"):
        return "gemini"
    if model_lower.startswith("claude-"):
        return "anthropic"
    if model_lower.startswith("gpt-") or model_lower.startswith("o1") or model_lower.startswith("o3"):
        return "openai"
    return "gemini"  # default preserves backward compatibility
