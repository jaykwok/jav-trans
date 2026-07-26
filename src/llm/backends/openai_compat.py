# OpenAI-compatible translation backend

import json
import os
import re
import threading
import time
from typing import Callable

from openai import OpenAI

from llm.backends.base import BaseTranslationBackend


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


class TranslationCancelledError(RuntimeError):
    pass


class RetryableTranslationFormatError(RuntimeError):
    pass


def _cancel_requested(cancel_event: threading.Event | None) -> bool:
    try:
        return bool(cancel_event is not None and cancel_event.is_set())
    except Exception:
        return False


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if _cancel_requested(cancel_event):
        raise TranslationCancelledError("任务已取消")


class OpenAICompatBackend(BaseTranslationBackend):
    """OpenAI 兼容 API 后端（DeepSeek、OpenAI、Azure 等）"""

    def __init__(self):
        self._client: OpenAI | None = None
        self._client_key: tuple[str, str] = ("", "")
        self._client_lock = threading.Lock()

    def name(self) -> str:
        return "openai"

    def supports_json_schema(self) -> bool:
        model_name = os.getenv("LLM_MODEL_NAME", "").strip()
        # DeepSeek 只支持 json_object，不支持 json_schema
        return "deepseek" not in model_name.lower()

    def supports_reasoning(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def _get_client(self) -> OpenAI:
        """获取或创建 OpenAI 客户端"""
        current_key = os.getenv("API_KEY", "").strip() or None
        current_url = self._normalize_base_url(
            os.getenv("OPENAI_COMPATIBILITY_BASE_URL", "").strip()
        )
        key_tuple = (current_key or "", current_url or "")

        with self._client_lock:
            if self._client is None or key_tuple != self._client_key:
                self._client = OpenAI(api_key=current_key, base_url=current_url)
                self._client_key = key_tuple
        return self._client

    def _normalize_base_url(self, base_url: str | None) -> str | None:
        """标准化 base URL"""
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

    def _is_deepseek_model(self, model_name: str | None) -> bool:
        return "deepseek" in (model_name or "").lower()

    def _chat_response_format(self, model_name: str | None, schema: dict) -> dict:
        """构造 response_format 参数"""
        if self._is_deepseek_model(model_name):
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "subtitle_translations",
                "strict": True,
                "schema": schema,
            },
        }

    def _is_retryable_api_error(self, exc: Exception) -> bool:
        """判断是否为可重试的 API 错误"""
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
        if status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True

        name = type(exc).__name__.lower()
        return any(
            marker in name
            for marker in (
                "ratelimit",
                "timeout",
                "connection",
                "serviceunavailable",
                "internalserver",
                "protocol",
            )
        )

    def _is_stream_interrupted_error(self, exc: Exception) -> bool:
        """判断是否为流中断错误"""
        message = str(exc).lower()
        name = type(exc).__name__.lower()
        return (
            "protocol" in name
            or "incomplete chunked read" in message
            or "peer closed connection" in message
            or "incomplete message body" in message
        )

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
        cancel_event = None,
        on_progress: Callable[[dict], None] | None = None,
        on_usage: Callable[[dict], None] | None = None,
    ) -> str:
        """执行 chat completion 请求"""
        _raise_if_cancelled(cancel_event)

        model_name = os.getenv("LLM_MODEL_NAME", "").strip()
        if not model_name:
            raise RuntimeError("请先在「翻译设置」中获取并选择翻译模型，再开始任务")

        api_format = os.getenv("LLM_API_FORMAT", "chat").strip().lower()
        if api_format == "responses":
            return self._chat_responses(
                messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                cancel_event=cancel_event,
                on_progress=on_progress,
                on_usage=on_usage,
            )

        request = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
        }

        if response_format:
            request["response_format"] = self._chat_response_format(
                model_name, response_format
            )

        if reasoning_effort:
            request["reasoning_effort"] = reasoning_effort
            request["extra_body"] = {"thinking": {"type": "enabled"}}

        if max_tokens > 0:
            request["max_tokens"] = max_tokens

        if stream:
            request["stream_options"] = {"include_usage": True}

        try:
            response_stream = self._create_chat_completion(request, cancel_event)
        except Exception as exc:
            if "stream_options" not in request or "stream_options" not in str(exc):
                raise
            request = dict(request)
            request.pop("stream_options", None)
            response_stream = self._create_chat_completion(request, cancel_event)

        return self._consume_chat_stream(
            response_stream,
            cancel_event=cancel_event,
            on_progress=on_progress,
            on_usage=on_usage,
        )

    def _create_chat_completion(self, request: dict, cancel_event):
        """创建 chat completion（带重试）"""
        last_error: Exception | None = None
        max_retries = 4

        for attempt in range(max_retries):
            _raise_if_cancelled(cancel_event)
            try:
                return self._get_client().chat.completions.create(**request)
            except Exception as exc:
                last_error = exc
                if not self._is_retryable_api_error(exc):
                    raise

                if attempt < max_retries - 1:
                    delay = min(20.0, 1.5 * (2**attempt))
                    self._interruptible_sleep(delay, cancel_event)

        if last_error is not None:
            raise last_error
        raise RuntimeError("chat completion failed without an exception")

    def _consume_chat_stream(
        self,
        response_stream,
        *,
        cancel_event,
        on_progress,
        on_usage,
    ) -> str:
        """消费 chat stream"""
        finish_reason = None
        reasoning_chars = 0
        content_parts = []
        last_emit = 0.0
        debounce_s = 0.25

        def maybe_emit(payload: dict, *, force: bool = False) -> None:
            nonlocal last_emit
            now = time.monotonic()
            if not force and now - last_emit < debounce_s:
                return
            last_emit = now
            self._emit_progress(on_progress, payload)

        try:
            for chunk in response_stream:
                _raise_if_cancelled(cancel_event)
                self._emit_usage(on_usage, self._extract_usage(chunk))

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                reasoning_content = getattr(delta, "reasoning_content", None)
                if reasoning_content:
                    reasoning_chars += len(reasoning_content)
                    maybe_emit({"phase": "thinking", "reasoning_chars": reasoning_chars})

                if hasattr(delta, "content") and delta.content:
                    content_parts.append(delta.content)
                    maybe_emit({"phase": "translating", "content_chars": len("".join(content_parts))})

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                _raise_if_cancelled(cancel_event)

        except Exception as exc:
            if self._is_retryable_api_error(exc):
                raise RetryableTranslationFormatError(
                    f"LLM stream interrupted: {type(exc).__name__}: {exc}"
                ) from exc
            raise

        _raise_if_cancelled(cancel_event)

        if finish_reason == "length":
            raise RuntimeError(
                "LLM response was cut off by max_tokens; increase TRANSLATION_MAX_TOKENS."
            )

        final_content = "".join(content_parts).strip()
        if not final_content:
            raise RetryableTranslationFormatError("LLM returned empty content.")

        maybe_emit({"phase": "done"}, force=True)
        return final_content

    def _chat_responses(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        reasoning_effort: str | None,
        cancel_event,
        on_progress,
        on_usage,
    ) -> str:
        """OpenAI Responses API（未完全实现，保留接口）"""
        raise NotImplementedError("Responses API not yet implemented in refactored backend")

    def _extract_usage(self, chunk) -> dict:
        """提取 token 使用统计"""
        usage = getattr(chunk, "usage", None)
        if usage is None:
            return {}

        cached_tokens = getattr(
            getattr(usage, "prompt_tokens_details", None), "cached_tokens", None
        )
        cache_hit_tokens = getattr(usage, "prompt_cache_hit_tokens", None)
        cache_miss_tokens = getattr(usage, "prompt_cache_miss_tokens", None)

        return {
            "cached_tokens": cached_tokens,
            "cache_hit_tokens": cache_hit_tokens,
            "cache_miss_tokens": cache_miss_tokens,
        }

    def _interruptible_sleep(self, total_s: float, cancel_event) -> None:
        """可中断的 sleep"""
        remaining = max(0.0, float(total_s))
        while remaining > 0:
            if _cancel_requested(cancel_event):
                return
            sleep_for = min(0.1, remaining)
            time.sleep(sleep_for)
            remaining -= sleep_for
