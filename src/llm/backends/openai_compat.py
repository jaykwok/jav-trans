"""OpenAI-compatible backend adapter.

Provider-specific streaming (including the Responses API and the micu Grok
shape) remains in the translation engine for now. This adapter deliberately
contains no second transport implementation, preventing Chat and Responses
behavior from diverging between direct backend use and normal jobs.
"""

from __future__ import annotations

import os
from typing import Callable

from llm.backends.base import BaseTranslationBackend


class OpenAICompatBackend(BaseTranslationBackend):
    def name(self) -> str:
        return "openai"

    def cache_identity(self) -> str:
        model_name = os.getenv("LLM_MODEL_NAME", "").strip()
        base_url = self._normalize_base_url(
            os.getenv("OPENAI_COMPATIBILITY_BASE_URL", "").strip()
        )
        return f"openai:{base_url or 'default'}:{model_name}"

    def supports_json_schema(self) -> bool:
        return "deepseek" not in os.getenv("LLM_MODEL_NAME", "").lower()

    def supports_reasoning(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    @staticmethod
    def _normalize_base_url(base_url: str | None) -> str | None:
        normalized = (base_url or "").strip().rstrip("/")
        if not normalized:
            return None
        lower = normalized.lower()
        for suffix in ("/chat/completions", "/responses", "/completions", "/models"):
            if lower.endswith(suffix):
                normalized = normalized[: -len(suffix)].rstrip("/")
                lower = normalized.lower()
                break
        if lower.endswith("/v1") or "/v1/" in lower:
            return normalized
        return f"{normalized}/v1"

    def chat_completion(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 384000,
        response_format: dict | None = None,
        stream: bool = True,
        reasoning_effort: str | None = None,
        api_format: str | None = None,
        expected_count: int = 0,
        cancel_event=None,
        on_progress: Callable[[dict], None] | None = None,
        on_usage: Callable[[dict], None] | None = None,
    ) -> str:
        # The canonical OpenAI transport is streaming. ``stream`` remains in
        # the cross-backend signature but direct calls still return final text.
        del stream
        self._raise_if_cancelled(cancel_event)

        from llm import translator

        kwargs = {
            "messages": messages,
            "expected_count": expected_count,
            "on_progress": on_progress,
            "reasoning_effort": reasoning_effort,
            "on_usage": on_usage,
            "cancel_event": cancel_event,
            "response_schema": response_format,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        effective_format = (
            api_format or os.getenv("LLM_API_FORMAT", "chat") or "chat"
        ).strip().lower()
        if effective_format == "responses":
            return translator._chat_responses(**kwargs)
        return translator._chat_completions(**kwargs)
