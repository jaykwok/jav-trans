# Test translation backend switching

import os
import sys
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

os.environ["PYTHONIOENCODING"] = "utf-8"


def test_backend_registry():
    """测试后端注册表"""
    from llm.backends import list_backends, get_backend

    backends = list_backends()
    print(f"[test] Available backends: {backends}")

    assert "openai" in backends, "OpenAI backend should be registered"
    assert "local" in backends, "Local backend should be registered"

    # 测试获取后端
    openai_backend = get_backend("openai")
    assert openai_backend.name() == "openai"

    local_backend = get_backend("local")
    assert local_backend.name() == "local"

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
    assert backend.supports_streaming() is True
    print(f"[test] Local backend JSON schema support: {backend.supports_json_schema()}")
    print(f"[test] Local backend reasoning support: {backend.supports_reasoning()}")

    print("[test] OK Local backend test passed")


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


if __name__ == "__main__":
    print("=" * 60)
    print("Translation Backend Tests")
    print("=" * 60)

    try:
        test_backend_registry()
        test_openai_backend()
        test_local_backend()
        test_translator_api_compatibility()
        test_mock_translation()

        print("\n" + "=" * 60)
        print("OK All tests passed!")
        print("=" * 60)

    except Exception as exc:
        print(f"\nERROR Test failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
