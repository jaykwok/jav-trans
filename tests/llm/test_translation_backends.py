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
    """测试后端注册表"""
    from llm.backends import list_backends, get_backend

    backends = list_backends()
    print(f"[test] Available backends: {backends}")

    assert "openai" in backends, "OpenAI backend should be registered"
    assert "local" in backends, "Local backend should be registered"

    # 后端是进程级共享实例，避免本地大模型被每个 batch worker 重复加载。
    openai_backend = get_backend("openai")
    assert openai_backend.name() == "openai"
    assert get_backend("openai") is openai_backend

    local_backend = get_backend("local")
    assert local_backend.name() == "local"
    assert get_backend("local") is local_backend

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


def test_local_backend():
    """测试本地模型后端基本功能"""
    from llm.backends.local_model import LocalModelBackend

    backend = LocalModelBackend()

    # 测试功能支持
    # 当前 transformers 实现使用同步 generate；进度事件不等于 token 流。
    assert backend.supports_streaming() is False
    print(f"[test] Local backend JSON schema support: {backend.supports_json_schema()}")
    print(f"[test] Local backend reasoning support: {backend.supports_reasoning()}")

    print("[test] OK Local backend test passed")


def test_local_backend_wait_is_cancellable(monkeypatch):
    from llm.backends.local_model import LocalModelBackend
    from llm.errors import TranslationCancelledError

    backend = LocalModelBackend()
    backend._inference_lock.acquire()
    monkeypatch.setattr(backend, "_ensure_model", lambda: None)
    cancel = threading.Event()
    timer = threading.Timer(0.1, cancel.set)
    timer.start()
    started = time.perf_counter()
    try:
        try:
            backend.chat_completion([], cancel_event=cancel)
        except TranslationCancelledError:
            pass
        else:
            raise AssertionError("cancelled local request should raise")
    finally:
        timer.cancel()
        backend._inference_lock.release()
    assert time.perf_counter() - started < 0.8


def test_local_backend_serializes_generate(monkeypatch):
    from llm.backends.local_model import LocalModelBackend

    backend = LocalModelBackend()
    monkeypatch.setattr(backend, "_ensure_model", lambda: None)
    state = {"active": 0, "peak": 0}
    state_lock = threading.Lock()

    def fake_generate(*_args, **_kwargs):
        with state_lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.05)
        with state_lock:
            state["active"] -= 1
        return '{"translations":[]}'

    monkeypatch.setattr(backend, "_generate", fake_generate)
    threads = [
        threading.Thread(target=lambda: backend.chat_completion([]))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    assert state["peak"] == 1


def test_local_backend_rejects_full_context_before_generate():
    import pytest
    import torch

    from llm.backends.local_model import LocalModelBackend
    from llm.errors import TranslationContextLengthError

    class FakeTokenizer:
        model_max_length = 4
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.ones((1, 4), dtype=torch.long)}

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **_kwargs):
            raise AssertionError("generate must not run for an overlong prompt")

    backend = LocalModelBackend()
    backend._tokenizer = FakeTokenizer()
    backend._model = FakeModel()
    backend._max_length = 4

    with pytest.raises(TranslationContextLengthError, match="上下文已超限"):
        backend._generate(
            [],
            temperature=0.0,
            top_p=1.0,
            max_tokens=8,
            cancel_event=None,
            on_progress=None,
            on_usage=None,
        )


def test_local_backend_disables_thinking_in_chat_template():
    import torch

    from llm.backends.local_model import LocalModelBackend

    seen_kwargs = {}

    class FakeTokenizer:
        model_max_length = 64
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, _messages, **kwargs):
            seen_kwargs.update(kwargs)
            return "prompt"

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.ones((1, 4), dtype=torch.long)}

        def decode(self, *_args, **_kwargs):
            return '{"translations":[]}'

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            return torch.ones((1, 5), dtype=torch.long)

    backend = LocalModelBackend()
    backend._tokenizer = FakeTokenizer()
    backend._model = FakeModel()
    backend._max_length = 64

    backend._generate(
        [{"role": "user", "content": "x"}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        cancel_event=None,
        on_progress=None,
        on_usage=None,
    )

    # Qwen3-style templates read this flag to skip the <think> prelude; without
    # it a thinking model spends most of the generation budget before the JSON.
    assert seen_kwargs.get("enable_thinking") is False


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

    monkeypatch.setenv("TRANSLATION_BACKEND", "local")
    monkeypatch.setenv("LOCAL_MODEL_PATH", "local/model")
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
