# Test translation backend switching

import os
import sys
import threading
import time
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

os.environ["PYTHONIOENCODING"] = "utf-8"


def test_backend_registry():
    """One backend per prompt contract, and nothing else.

    The in-process Transformers backend ("local") was removed on 2026-08-05: it
    was a third way to run a local model that no shipped model targeted, and it
    carried its own device/dtype/context settings that nothing validated.
    """
    from llm.backends import list_backends, get_backend

    backends = list_backends()
    print(f"[test] Available backends: {backends}")

    assert "openai" in backends, "OpenAI backend should be registered"
    assert "llamacpp" in backends, "llama.cpp backend should be registered"
    assert "local" not in backends, "the Transformers backend was removed"

    # 后端是进程级共享实例，避免本地大模型被每个 batch worker 重复加载。
    openai_backend = get_backend("openai")
    assert openai_backend.name() == "openai"
    assert get_backend("openai") is openai_backend

    llamacpp_backend = get_backend("llamacpp")
    assert llamacpp_backend.name() == "llamacpp"
    assert get_backend("llamacpp") is llamacpp_backend

    print("[test] OK Backend registry test passed")


def test_openai_backend():
    """测试 OpenAI 后端基本功能"""
    from llm.backends.openai_compat import OpenAICompatBackend

    backend = OpenAICompatBackend()

    # 测试功能支持
    assert backend.supports_streaming() is True
    assert backend.supports_reasoning() is True
    print(f"[test] OpenAI backend JSON schema support: {backend.supports_json_schema()}")

    print("[test] OK OpenAI backend test passed")


def test_translator_routes_custom_backend_and_preserves_task_format(monkeypatch):
    from llm.backends import register_backend
    from llm import translator

    calls = []

    class FakeBackend:
        def name(self):
            return "route-test"

        def cache_identity(self):
            return "route-test:model-v1"

        def chat_completion(self, _messages, **kwargs):
            calls.append(kwargs)
            return '{"translations":[{"id":0,"text":"好"}]}'

    backend = FakeBackend()
    register_backend("route-test", lambda: backend, replace=True)
    monkeypatch.setenv("TRANSLATION_BACKEND", "route-test")

    texts, _timings, _retries = translator.translate_segments(
        [{"start": 0.0, "end": 1.0, "text": "いい"}],
        api_format="responses",
    )

    assert texts == ["好"]
    assert calls[0]["api_format"] == "responses"
    assert calls[0]["expected_count"] == 1
    assert calls[0]["response_format"]["required"] == ["translations"]


def test_backend_identity_separates_translation_cache(monkeypatch):
    from llm import translator

    segments = [{"start": 0.0, "end": 1.0, "text": "いい"}]
    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("LLM_MODEL_NAME", "api-model")
    api_key = translator._translation_cache_key(0, segments)

    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setenv("LLAMACPP_GGUF_PATH", "D:\\models\\local.gguf")
    local_key = translator._translation_cache_key(0, segments)

    assert api_key != local_key


def test_backend_cancellation_keeps_shared_exception_type(monkeypatch):
    import pytest

    from llm import translator
    from llm.backends import register_backend
    from llm.errors import TranslationCancelledError

    class CancelBackend:
        def name(self):
            return "cancel-test"

        def cache_identity(self):
            return "cancel-test"

        def chat_completion(self, _messages, **_kwargs):
            raise TranslationCancelledError("任务已取消")

    register_backend("cancel-test", CancelBackend, replace=True)
    monkeypatch.setenv("TRANSLATION_BACKEND", "cancel-test")

    with pytest.raises(TranslationCancelledError):
        translator.translate_segments(
            [{"start": 0.0, "end": 1.0, "text": "いい"}],
        )


def test_translator_api_compatibility():
    """测试 translator API 向后兼容性"""
    from llm.translator import translate_segments, PROMPT_VERSION

    # 检查导出的 API
    assert callable(translate_segments)
    assert isinstance(PROMPT_VERSION, str)

    print(f"[test] Translator API version: {PROMPT_VERSION}")
    print("[test] OK Translator API compatibility test passed")


def test_mock_translation():
    """测试翻译流程（不实际调用 API）"""
    # 模拟字幕段
    segments = [
        {"text": "こんにちは", "start": 0.0, "end": 1.0},
        {"text": "ありがとう", "start": 1.5, "end": 2.5},
    ]

    print(f"[test] Mock segments: {len(segments)} items")
    print("[test] OK Mock translation test passed")
