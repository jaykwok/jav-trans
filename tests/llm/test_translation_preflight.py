"""The translation stage refuses to start on a config it cannot use."""

from __future__ import annotations

import pytest

from core import stage_errors
from llm import preflight
from llm.backends import openai_compat


@pytest.fixture(autouse=True)
def _openai_backend(monkeypatch):
    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")


def test_a_complete_config_reports_nothing():
    assert preflight.translation_config_problems() == []
    preflight.require_translation_config()


@pytest.mark.parametrize(
    ("cleared", "expected"),
    [
        ("API_KEY", stage_errors.MISSING_API_KEY),
        ("OPENAI_COMPATIBILITY_BASE_URL", stage_errors.MISSING_BASE_URL),
        ("LLM_MODEL_NAME", stage_errors.MISSING_MODEL),
    ],
)
def test_each_missing_setting_names_itself(monkeypatch, cleared, expected):
    monkeypatch.setenv(cleared, "")
    assert preflight.translation_config_problems() == [expected]
    with pytest.raises(RuntimeError) as excinfo:
        preflight.require_translation_config()
    assert str(excinfo.value) == expected


def test_every_missing_setting_is_reported_at_once(monkeypatch):
    for name in ("API_KEY", "OPENAI_COMPATIBILITY_BASE_URL", "LLM_MODEL_NAME"):
        monkeypatch.setenv(name, "")
    problems = preflight.translation_config_problems()
    assert problems == [
        stage_errors.MISSING_API_KEY,
        stage_errors.MISSING_BASE_URL,
        stage_errors.MISSING_MODEL,
    ]
    # One raise listing all three, so a user does not fix them one run at a time.
    with pytest.raises(RuntimeError) as excinfo:
        preflight.require_translation_config()
    assert str(excinfo.value).count("\n") == 2


@pytest.mark.parametrize("backend", ["llamacpp", "local"])
def test_other_backends_are_left_to_their_own_messages(monkeypatch, backend):
    monkeypatch.setenv("TRANSLATION_BACKEND", backend)
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("LLM_MODEL_NAME", "")
    for name in ("LLAMACPP_GGUF_PATH", "LLAMACPP_MODEL_FILE", "LLAMACPP_MODEL_REPO"):
        monkeypatch.delenv(name, raising=False)
    assert preflight.translation_config_problems() == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLAMACPP_MODEL_REPO", "SakuraLLM/Sakura-GalTransl-7B-v3.7"),
        ("LLAMACPP_MODEL_FILE", "Sakura-Galtransl-7B-v3.7.gguf"),
        ("LLAMACPP_GGUF_PATH", r"D:\models\Sakura-Galtransl-7B-v3.7.gguf"),
    ],
)
def test_a_retired_gguf_still_pinned_in_env_is_caught_at_submit(monkeypatch, name, value):
    """v1.0 shipped this model as the llamacpp default. Left in `.env` it now
    answers a contract it cannot speak, which costs a whole video of API-free
    but useless output instead of raising."""
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    for cleared in ("LLAMACPP_GGUF_PATH", "LLAMACPP_MODEL_FILE", "LLAMACPP_MODEL_REPO"):
        monkeypatch.delenv(cleared, raising=False)
    monkeypatch.setenv(name, value)
    problems = preflight.translation_config_problems()
    assert problems == [preflight.RETIRED_GGUF_MODEL]
    assert "翻译设置" in problems[0]


def test_the_shipped_gguf_default_is_accepted(monkeypatch):
    from core.config import DEFAULT_SETTINGS

    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.delenv("LLAMACPP_GGUF_PATH", raising=False)
    monkeypatch.setenv("LLAMACPP_MODEL_REPO", DEFAULT_SETTINGS["LLAMACPP_MODEL_REPO"])
    monkeypatch.setenv("LLAMACPP_MODEL_FILE", DEFAULT_SETTINGS["LLAMACPP_MODEL_FILE"])
    assert preflight.translation_config_problems() == []


def test_the_openai_client_asks_before_the_sdk_can_complain(monkeypatch):
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setattr(openai_compat, "_CLIENT", None)
    monkeypatch.setattr(openai_compat, "_CLIENT_KEY", ("", ""))

    def fail(*args, **kwargs):
        raise AssertionError("the SDK client must not be constructed")

    monkeypatch.setattr(openai_compat, "OpenAI", fail)
    with pytest.raises(RuntimeError) as excinfo:
        openai_compat._get_client()
    assert str(excinfo.value) == stage_errors.MISSING_API_KEY
