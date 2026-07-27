# Translation backend base classes

from abc import ABC, abstractmethod
from typing import Callable

from llm.errors import TranslationCancelledError


class BaseTranslationBackend(ABC):
    """翻译后端抽象基类"""

    @abstractmethod
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
        cancel_event = None,
        on_progress: Callable[[dict], None] | None = None,
        on_usage: Callable[[dict], None] | None = None,
    ) -> str:
        """执行翻译请求，返回完整内容"""
        pass

    def cache_identity(self) -> str:
        """Stable identity included in translation cache signatures."""
        return self.name()

    def supports_json_schema(self) -> bool:
        """是否支持 JSON schema 约束"""
        return False

    def supports_reasoning(self) -> bool:
        """是否支持 reasoning/thinking"""
        return False

    def supports_streaming(self) -> bool:
        """是否支持流式输出"""
        return True

    @abstractmethod
    def name(self) -> str:
        """后端标识名称"""
        pass

    def _emit_progress(
        self,
        on_progress: Callable[[dict], None] | None,
        payload: dict,
    ) -> None:
        """发送进度事件"""
        if on_progress is None:
            return
        try:
            on_progress(payload)
        except Exception:
            pass

    def _emit_usage(
        self,
        on_usage: Callable[[dict], None] | None,
        usage_data: dict,
    ) -> None:
        """发送 token 使用统计"""
        if on_usage is None or not usage_data:
            return
        try:
            on_usage(usage_data)
        except Exception:
            pass

    @staticmethod
    def _raise_if_cancelled(cancel_event) -> None:
        try:
            cancelled = cancel_event is not None and cancel_event.is_set()
        except Exception:
            cancelled = False
        if cancelled:
            raise TranslationCancelledError("任务已取消")
