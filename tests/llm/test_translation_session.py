"""Both passes of one film's translation belong to one run.

The batch pass and the repair pass have to agree about the profile that shapes
the prompt and the cache key, the instance that answers, the tier the base pass
paid for, and the glossary and language the text was produced under. None of
that is a per-pass decision, and it used to be eight parallel arguments handed
across at one call site - correct only because that call site happened to pass
the right values.

The neighbouring case is what makes this concrete rather than tidy: the profile
used to be selected twice, the second selection disagreed with the first, and
the cache key then described a prompt the request had not used.
"""

from __future__ import annotations

import pytest

from llm import repair as repair_module
from llm import request_config
from llm import settings as llm_settings
from llm import token_budget
from llm.profiles import get_profile
from llm.session import TranslationSession


def _session(**overrides) -> TranslationSession:
    fields = {
        "backend_name": "openai",
        "profile": get_profile("json"),
        "batch_size": 24,
        "max_workers": 4,
        "cache_path": "",
        "target_lang": "简体中文",
        "glossary": "",
        "character_reference": "",
        "reasoning_effort": "low",
    }
    fields.update(overrides)
    return TranslationSession(**fields)


def test_the_repair_tier_is_derived_from_what_the_base_pass_paid(monkeypatch):
    monkeypatch.setattr(llm_settings, "TRANSLATION_REPAIR_REASONING_EFFORT", "")

    # The base tier, floored at `low`: thinking-off has to escalate, and above
    # that the escalation was measured and did not pay for itself.
    assert _session(reasoning_effort="none").repair_reasoning_effort == "low"
    assert _session(reasoning_effort="low").repair_reasoning_effort == "low"
    assert _session(reasoning_effort="high").repair_reasoning_effort == "high"


def test_a_pinned_repair_tier_still_wins(monkeypatch):
    monkeypatch.setattr(llm_settings, "TRANSLATION_REPAIR_REASONING_EFFORT", "high")

    assert _session(reasoning_effort="none").repair_reasoning_effort == "high"


def test_the_tier_is_the_one_the_run_carried_not_the_one_settings_now_say(monkeypatch):
    """A settings change mid-film must not change what the repair escalates
    *from*: the base pass has already been paid for at its own tier."""
    session = _session(reasoning_effort="none")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    monkeypatch.setattr(llm_settings, "TRANSLATION_REPAIR_REASONING_EFFORT", "")

    assert session.reasoning_effort == "none"
    assert session.repair_reasoning_effort == "low"  # floored, not "high"


def test_both_passes_key_the_learned_ceiling_the_same_way(monkeypatch):
    session = _session(backend_name="openai")

    assert session.endpoint_identity() == token_budget.endpoint_identity("openai")


def test_the_profile_is_bound_for_the_pass_and_released_after():
    session = _session()
    assert request_config.current_profile() is None

    with session.bound():
        assert request_config.current_profile() is session.profile

    assert request_config.current_profile() is None


def test_a_nested_pass_gives_the_outer_one_back():
    outer = _session(profile=get_profile("json"))
    inner = _session(profile=get_profile("hymt2"))

    with outer.bound():
        with inner.bound():
            assert request_config.current_profile() is inner.profile
        assert request_config.current_profile() is outer.profile


def test_the_repair_pass_takes_the_run_not_loose_arguments():
    """It is a keyword-only `session`, so there is no way to hand it a profile
    from one run and a batch size from another."""
    import inspect

    parameters = inspect.signature(repair_module.apply_repair_pass).parameters

    assert "session" in parameters
    for gone in ("profile", "batch_size", "reasoning_effort", "target_lang", "glossary"):
        assert gone not in parameters


def test_a_session_cannot_change_under_a_pass():
    session = _session()

    with pytest.raises(Exception):
        session.reasoning_effort = "high"
