"""Profile selection: pin > backend-scoped auto-detection > json default.

Only the JSON contract ships since the Sakura/GalTransl profile was removed, so
the detection machinery is exercised through a profile registered for the test.
Keeping it covered is the point: it is the seam a new model family plugs into,
and its one hard rule (never read another backend's model config) was learned
from a real hijack.
"""

import pytest

from llm import profiles
from llm.profiles.base import TranslationProfile


class _FakeProfile(TranslationProfile):
    id = "fake_family"
    version = "1"

    def build_messages(self, segments, *, ids, ctx):
        return [{"role": "user", "content": "x"}]

    def parse_response(self, text, *, ids):
        return {}


@pytest.fixture
def detectable(monkeypatch):
    """Register a profile that auto-detection can find, then take it away."""
    profile = _FakeProfile()
    monkeypatch.setitem(profiles._REGISTRY, profile.id, profile)
    monkeypatch.setitem(profiles._MATCH_TOKENS, profile.id, ("fakefamily",))
    return profile


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


def test_only_unified_json_contract_is_registered():
    """Both API models and local Hy-MT3-7B share the unified JSON batch
    contract; model-specific profiles can still register via register_profile."""
    assert profiles.list_profiles() == ["json"]


def test_neither_contract_can_return_an_empty_translation():
    """What actually retired the Sakura profile. `hymt2` is line-oriented like
    it was, so this is the property that must not come back with it."""
    for profile_id in profiles.list_profiles():
        profile = profiles.get_profile(profile_id)
        try:
            parsed = profile.parse_response("", ids=[0])
        except Exception:
            continue
        assert not any(value == "" for value in parsed.values()), profile_id


def test_the_local_default_model_selects_the_unified_json_contract(monkeypatch):
    """Hy-MT3-7B local default shares the unified JSON batch contract."""
    from core.config import DEFAULT_SETTINGS

    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setenv("LLAMACPP_MODEL_REPO", DEFAULT_SETTINGS["LLAMACPP_MODEL_REPO"])
    monkeypatch.setenv("LLAMACPP_MODEL_FILE", DEFAULT_SETTINGS["LLAMACPP_MODEL_FILE"])
    assert profiles.select_profile().id == "json"


def test_a_gguf_name_never_hijacks_the_api_backend(monkeypatch, detectable):
    """Scoping regression: reading `LLAMACPP_MODEL_FILE` while the openai
    backend is selected must never flip an API model to another contract."""
    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("LLAMACPP_MODEL_FILE", "fakefamily-7b-Q4_K_M.gguf")
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-5.2")
    assert profiles.select_profile().id == "json"


def test_pin_overrides_detection(monkeypatch, detectable):
    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setenv("LLAMACPP_MODEL_FILE", "fakefamily-7b.gguf")
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "json")
    assert profiles.select_profile().id == "json"


def test_pin_aliases(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "off")
    assert profiles.select_profile().id == "json"
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "none")
    assert profiles.select_profile().id == "json"


def test_unknown_pin_falls_back_to_json(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "does-not-exist")
    assert profiles.select_profile().id == "json"


def test_llamacpp_detects_from_gguf_path(monkeypatch, detectable):
    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setenv("LLAMACPP_GGUF_PATH", r"D:\models\FakeFamily-7B-Q4_K_M.gguf")
    assert profiles.select_profile().id == "fake_family"


def test_openai_backend_only_reads_model_name(monkeypatch, detectable):
    # The hijack lesson: a GGUF path left over from the local backend must never
    # flip the profile while requests actually go to an API model.
    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("LLAMACPP_GGUF_PATH", r"D:\models\FakeFamily-7B-Q4_K_M.gguf")
    monkeypatch.setenv("LLM_MODEL_NAME", "glm-4.7")
    assert profiles.select_profile().id == "json"
    monkeypatch.setenv("LLM_MODEL_NAME", "fakefamily-14b")
    assert profiles.select_profile().id == "fake_family"


def test_other_backends_need_explicit_pin(monkeypatch, detectable):
    _clear(monkeypatch)
    monkeypatch.setenv("TRANSLATION_BACKEND", "local")
    monkeypatch.setenv("LLM_MODEL_NAME", "fakefamily-14b")
    assert profiles.select_profile().id == "json"


def test_register_profile_rejects_duplicate_id():
    json_profile = profiles.get_profile("json")
    with pytest.raises(ValueError):
        profiles.register_profile(json_profile, match_tokens=())
