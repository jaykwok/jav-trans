"""Sakura/GalTransl prompt profile + managed llama.cpp server backend."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm import profiles, translator
from llm.profiles import sakura_galtransl as sakura
from llm.backends import list_backends
from llm.backends.llamacpp_server import (
    LlamaCppServerBackend,
    resolve_gguf_model_path,
    resolve_server_executable,
)
from llm.errors import RetryableTranslationFormatError


_PROFILE_ENV_KEYS = (
    "TRANSLATION_PROMPT_PROFILE",
    "LLAMACPP_MODEL_FILE",
    "LLAMACPP_MODEL_REPO",
    "LLAMACPP_GGUF_PATH",
    "LLAMACPP_SERVER_PATH",
    "LLM_MODEL_NAME",
    "SAKURA_BATCH_SIZE",
    "SAKURA_WORKERS",
    "SAKURA_HISTORY_LINES",
    "TRANSLATION_BACKEND",
)


@pytest.fixture(autouse=True)
def _clean_profile_env(monkeypatch):
    for key in _PROFILE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


# --- registry -----------------------------------------------------------------


def test_llamacpp_backend_registered():
    assert "llamacpp" in list_backends()


# --- prompt contract ----------------------------------------------------------


def test_sakura_system_prompt_is_the_model_card_text():
    assert sakura.GALTRANSL_SYSTEM_PROMPT == (
        "你是一个视觉小说翻译模型，可以通顺地使用给定的术语表以指定的风格将日文翻译成简体中文，"
        "并联系上下文正确使用人称代词，注意不要混淆使役态和被动态的主语和宾语，"
        "不要擅自添加原文中没有的特殊符号，也不要擅自增加或减少换行。"
    )
    assert sakura.SAKURA_TEMPERATURE == 0.3
    assert sakura.SAKURA_TOP_P == 0.8


def test_sakura_messages_full_structure():
    messages = sakura.build_sakura_messages(
        ["こんにちは", "ダメ…もう…"],
        glossary_text="ちんぽ-肉棒\nチンポ-肉棒",
        history_lines=["你好", "住手"],
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == sakura.GALTRANSL_SYSTEM_PROMPT
    user = messages[1]["content"]
    assert user.startswith("历史翻译：你好\n住手\n")
    assert "参考以下术语表（可为空，格式为src->dst #备注）：" in user
    assert "ちんぽ->肉棒" in user and "チンポ->肉棒" in user
    assert user.endswith(
        "将下面的文本从日文翻译成简体中文：\nこんにちは\nダメ…もう…"
    )


def test_sakura_messages_without_history_or_glossary():
    user = sakura.build_sakura_messages(["こんにちは"])[1]["content"]
    assert "历史翻译" not in user
    assert "参考以下术语表" in user  # 模板要求该行恒在（可为空）
    assert user.endswith("こんにちは")


def test_sakura_messages_flatten_multiline_source():
    user = sakura.build_sakura_messages(["こん\nにちは", "はい"])[1]["content"]
    # 源行内部的换行会撑破行数合同，必须被拍平成单行
    tail = user.rsplit("：\n", 1)[1]
    assert tail.splitlines() == ["こん にちは", "はい"]


def test_parse_sakura_response_exact_lines():
    assert sakura.parse_sakura_response("你好\n不行…", 2) == ["你好", "不行…"]


def test_parse_sakura_response_strips_code_fence():
    assert sakura.parse_sakura_response("```\n你好\n```", 1) == ["你好"]


def test_parse_sakura_response_mismatch_raises():
    with pytest.raises(RetryableTranslationFormatError):
        sakura.parse_sakura_response("只有一行", 2)
    with pytest.raises(RetryableTranslationFormatError):
        sakura.parse_sakura_response("一\n二\n三", 2)


def test_glossary_projection_uses_arrow_format():
    assert sakura.glossary_to_sakura("ちんぽ-肉棒\nおちんちん-肉棒") == (
        "ちんぽ->肉棒\nおちんちん->肉棒"
    )
    assert sakura.glossary_to_sakura("") == ""


# --- profile detection --------------------------------------------------------


def test_profile_auto_detects_galtransl_model(monkeypatch):
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    assert profiles.select_profile().id != "sakura_galtransl"
    monkeypatch.setenv("LLAMACPP_MODEL_FILE", "Sakura-Galtransl-7B-v3.7.gguf")
    assert profiles.select_profile().id == "sakura_galtransl"


def test_profile_auto_ignores_llamacpp_defaults_for_other_backends(monkeypatch):
    # The built-in LLAMACPP_MODEL_FILE default is a Sakura file; while the
    # openai backend is selected it must not flip the profile.
    monkeypatch.setenv("LLAMACPP_MODEL_FILE", "Sakura-Galtransl-7B-v3.7.gguf")
    assert profiles.select_profile().id != "sakura_galtransl"


def test_profile_auto_detects_via_llm_model_name(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_NAME", "sakura-14b-qwen3-v1.5")
    assert profiles.select_profile().id == "sakura_galtransl"


def test_profile_forced_on_and_off(monkeypatch):
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "sakura")
    assert profiles.select_profile().id == "sakura_galtransl"
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "json")
    monkeypatch.setenv("LLAMACPP_MODEL_FILE", "Sakura-Galtransl-7B-v3.7.gguf")
    assert profiles.select_profile().id != "sakura_galtransl"


def test_prompt_version_marker_isolates_cache(monkeypatch):
    plain = translator._effective_prompt_version()
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "sakura")
    marked = translator._effective_prompt_version()
    assert marked != plain
    assert marked == "sakura_galtransl@v3"
    assert plain == f"json@{translator.PROMPT_VERSION}"


# --- server/model resolution --------------------------------------------------


def test_resolve_server_executable_missing_is_actionable(monkeypatch):
    monkeypatch.setattr("llm.backends.llamacpp_server.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError) as excinfo:
        resolve_server_executable()
    message = str(excinfo.value)
    assert "winget install llama.cpp" in message
    assert "releases" in message


def test_resolve_server_executable_accepts_directory(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("LLAMACPP_SERVER_PATH", str(tmp_path))
    assert resolve_server_executable() == str(exe)


def test_resolve_server_executable_bad_explicit_path(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMACPP_SERVER_PATH", str(tmp_path / "nope.exe"))
    with pytest.raises(RuntimeError):
        resolve_server_executable()


def test_resolve_gguf_explicit_path(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"")
    monkeypatch.setenv("LLAMACPP_GGUF_PATH", str(model))
    assert resolve_gguf_model_path() == str(model)


def test_resolve_gguf_missing_config_is_actionable(monkeypatch):
    with pytest.raises(RuntimeError) as excinfo:
        resolve_gguf_model_path()
    assert "GGUF" in str(excinfo.value)


def test_build_command_shape(monkeypatch):
    monkeypatch.setenv("LLAMACPP_CTX_SIZE", "4096")
    monkeypatch.setenv("LLAMACPP_PARALLEL", "2")
    backend = LlamaCppServerBackend()
    command = backend._build_command("llama-server.exe", "D:\\m.gguf", 12345)
    assert command[0] == "llama-server.exe"
    assert command[command.index("-m") + 1] == "D:\\m.gguf"
    assert command[command.index("--port") + 1] == "12345"
    assert command[command.index("-c") + 1] == "8192"  # ctx * parallel
    assert command[command.index("-np") + 1] == "2"
    assert "--no-webui" in command
    assert command[command.index("--host") + 1] == "127.0.0.1"


def test_cache_identity_reflects_model(monkeypatch):
    backend = LlamaCppServerBackend()
    monkeypatch.setenv("LLAMACPP_MODEL_REPO", "SakuraLLM/Sakura-GalTransl-7B-v3.7")
    monkeypatch.setenv("LLAMACPP_MODEL_FILE", "Sakura-Galtransl-7B-v3.7.gguf")
    assert (
        backend.cache_identity()
        == "llamacpp:SakuraLLM/Sakura-GalTransl-7B-v3.7/Sakura-Galtransl-7B-v3.7.gguf"
    )
    monkeypatch.setenv("LLAMACPP_GGUF_PATH", "D:\\models\\custom-q4.gguf")
    assert backend.cache_identity() == "llamacpp:custom-q4.gguf"


# --- translator routing -------------------------------------------------------


class _FakeSakuraBackend:
    def __init__(self, reply_builder):
        self._reply_builder = reply_builder
        self.calls: list[list[dict]] = []

    def chat_completion(self, messages, **kwargs):
        self.calls.append(messages)
        source_block = messages[1]["content"].rsplit("：\n", 1)[1]
        return self._reply_builder(source_block.splitlines())

    def cache_identity(self) -> str:
        return "fake-sakura"

    def name(self) -> str:
        return "fake"


def _segments(count: int) -> list[dict]:
    return [
        {"text": f"セリフ{i}", "start": float(i), "end": float(i) + 1.0}
        for i in range(count)
    ]


def test_translate_segments_routes_to_sakura(monkeypatch):
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "sakura")
    monkeypatch.setenv("SAKURA_BATCH_SIZE", "8")
    fake = _FakeSakuraBackend(lambda lines: "\n".join(f"译-{line}" for line in lines))
    monkeypatch.setattr(translator, "get_backend", lambda name=None: fake)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(20),
        max_workers=1,
        cache_path="",
        glossary="",
    )
    assert zh_texts == [f"译-セリフ{i}" for i in range(20)]
    # 20 行 / batch 8 = 3 个批次；没有 JSON 修复批追加请求
    assert len(fake.calls) == 3
    assert retry_events == []
    assert [t["mode"] for t in timings] == ["line_batch"] * 3
    assert all(t["prompt_profile"] == "sakura_galtransl@v3" for t in timings)
    for messages in fake.calls:
        assert messages[0]["content"] == sakura.GALTRANSL_SYSTEM_PROMPT


def test_translate_segments_sakura_history_flows_between_batches(monkeypatch):
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "sakura")
    monkeypatch.setenv("SAKURA_BATCH_SIZE", "4")
    monkeypatch.setenv("SAKURA_WORKERS", "1")
    fake = _FakeSakuraBackend(lambda lines: "\n".join(f"译-{line}" for line in lines))
    monkeypatch.setattr(translator, "get_backend", lambda name=None: fake)

    translator.translate_segments(_segments(8), max_workers=1, cache_path="", glossary="")
    assert len(fake.calls) == 2
    first_user = fake.calls[0][1]["content"]
    second_user = fake.calls[1][1]["content"]
    assert "历史翻译" not in first_user
    assert second_user.startswith("历史翻译：")
    assert "译-セリフ3" in second_user.split("参考以下术语表")[0]


def test_translate_segments_sakura_falls_back_per_line(monkeypatch):
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "sakura")
    monkeypatch.setenv("SAKURA_BATCH_SIZE", "4")

    def reply(lines: list[str]) -> str:
        if len(lines) > 1:
            return "只有一行"  # 批量请求持续违反行数合同
        return f"译-{lines[0]}"

    fake = _FakeSakuraBackend(reply)
    monkeypatch.setattr(translator, "get_backend", lambda name=None: fake)

    zh_texts, timings, _ = translator.translate_segments(
        _segments(4), max_workers=1, cache_path="", glossary=""
    )
    assert zh_texts == [f"译-セリフ{i}" for i in range(4)]
    # 2 次失败的整批尝试 + 4 次逐行回退
    assert len(fake.calls) == 6
    assert timings[0]["request_count"] == 5  # 1 次批请求计数 + 4 次逐行


def test_translate_segments_sakura_writes_line_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "sakura")
    monkeypatch.setenv("SAKURA_BATCH_SIZE", "4")
    cache_path = str(tmp_path / "translation_cache.jsonl")
    fake = _FakeSakuraBackend(lambda lines: "\n".join(f"译-{line}" for line in lines))
    monkeypatch.setattr(translator, "get_backend", lambda name=None: fake)

    translator.translate_segments(
        _segments(4), max_workers=1, cache_path=cache_path, glossary=""
    )
    first_calls = len(fake.calls)
    assert first_calls == 1

    # 第二轮：批缓存精确命中，零请求
    translator.translate_segments(
        _segments(4), max_workers=1, cache_path=cache_path, glossary=""
    )
    assert len(fake.calls) == first_calls
