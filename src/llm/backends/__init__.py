# Translation backend registry and factory

from typing import Protocol, Any
import os


class TranslationBackend(Protocol):
    """翻译后端协议"""

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
        on_progress = None,
        on_usage = None,
    ) -> str:
        """执行翻译请求，返回完整内容"""
        ...

    def supports_json_schema(self) -> bool:
        """是否支持 JSON schema 约束"""
        ...

    def supports_reasoning(self) -> bool:
        """是否支持 reasoning/thinking"""
        ...

    def supports_streaming(self) -> bool:
        """是否支持流式输出"""
        ...

    def name(self) -> str:
        """后端标识名称"""
        ...


_BACKEND_REGISTRY: dict[str, type] = {}


def register_backend(name: str, backend_class: type) -> None:
    """注册翻译后端"""
    _BACKEND_REGISTRY[name] = backend_class


def get_backend(name: str | None = None) -> Any:
    """获取翻译后端实例"""
    if name is None:
        name = os.getenv("TRANSLATION_BACKEND", "openai").strip().lower()

    if name not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown translation backend: {name}. "
            f"Available: {list(_BACKEND_REGISTRY.keys())}"
        )

    backend_class = _BACKEND_REGISTRY[name]
    return backend_class()


def list_backends() -> list[str]:
    """列出所有已注册的后端"""
    return list(_BACKEND_REGISTRY.keys())


# 自动注册内置后端
def _register_builtin_backends():
    """注册内置后端"""
    try:
        from llm.backends.openai_compat import OpenAICompatBackend
        register_backend("openai", OpenAICompatBackend)
    except ImportError:
        pass

    try:
        from llm.backends.local_model import LocalModelBackend
        register_backend("local", LocalModelBackend)
    except ImportError:
        pass


_register_builtin_backends()
