"""Profile selection: pin > backend-scoped auto-detection > json default."""

from llm import profiles


def _clear(monkeypatch):
    for name in (
        "TRANSLATION_PROMPT_PROFILE",
        "TRANSLATION_BACKEND",
        "LLAMACPP_GGUF_PATH",
        "LLAMACPP_MODEL_FILE",
        "LLAMACPP_MODEL_REPO",
        "LLM_MODEL_NAME",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_is_json(monkeypatch):
    _clear(monkeypatch)
    assert profiles.select_profile().id == "json"


def test_pin_overrides_detection(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "sakura_galtransl")
    assert profiles.select_profile().id == "sakura_galtransl"


def test_pin_aliases(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "sakura")
    assert profiles.select_profile().id == "sakura_galtransl"
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "off")
    assert profiles.select_profile().id == "json"
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "none")
    assert profiles.select_profile().id == "json"


def test_unknown_pin_falls_back_to_json(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "does-not-exist")
    assert profiles.select_profile().id == "json"


def test_llamacpp_detects_from_gguf_path(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setenv(
        "LLAMACPP_GGUF_PATH", r"D:\models\Sakura-GalTransl-7B-v3.7-Q4_K_M.gguf"
    )
    assert profiles.select_profile().id == "sakura_galtransl"


def test_openai_backend_only_reads_model_name(monkeypatch):
    # The hijack lesson: a sakura-flavored GGUF path must never flip the
    # profile while requests actually go to an API model.
    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv(
        "LLAMACPP_GGUF_PATH", r"D:\models\Sakura-GalTransl-7B-v3.7-Q4_K_M.gguf"
    )
    monkeypatch.setenv("LLM_MODEL_NAME", "glm-4.7")
    assert profiles.select_profile().id == "json"
    monkeypatch.setenv("LLM_MODEL_NAME", "sakura-14b-qwen2.5")
    assert profiles.select_profile().id == "sakura_galtransl"


def test_other_backends_need_explicit_pin(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_BACKEND", "local")
    monkeypatch.setenv("LLM_MODEL_NAME", "sakura-14b-qwen2.5")
    assert profiles.select_profile().id == "json"


def test_register_profile_rejects_duplicate_id():
    import pytest

    json_profile = profiles.get_profile("json")
    with pytest.raises(ValueError):
        profiles.register_profile(json_profile, match_tokens=())
