"""A budget the endpoint will not honour has to be audible, not silent.

`min(ceiling, budget)` is silent by construction: the request just goes out
smaller than asked for, and if the reply is then cut off the truncation
escalation cannot raise it back past the same line. There are two checks
because neither covers the other - this one runs at startup and knows only the
configuration, and `translator._max_tokens_budget` runs per request and sees the
budget the source text actually produced.
"""
from __future__ import annotations

import pytest

from llm import max_tokens_limits, preflight, settings as llm_settings, translator

_BASE_URL = "https://example.test/v1"
_MODEL = "test-model"


@pytest.fixture(autouse=True)
def _json_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSLATION_PROMPT_PROFILE", "json")
    monkeypatch.setenv(
        "TRANSLATION_MAX_TOKENS_CACHE_PATH", str(tmp_path / "limits.json")
    )
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", _BASE_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", _MODEL)
    monkeypatch.setattr(translator, "selected_backend_name", lambda *_a, **_k: "openai")
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    monkeypatch.setattr(llm_settings, "TRANSLATION_BATCH_SIZE", 200)
    monkeypatch.setattr(llm_settings, "LLM_REASONING_EFFORT", "low")
    translator._clamp_warned.clear()


def test_an_endpoint_that_has_said_nothing_is_quiet():
    # Nothing is being clamped, so there is nothing to report - the request will
    # go out at whatever the batch computes.
    assert preflight.translation_budget_warnings() == []


def test_a_ceiling_below_the_configured_floor_is_reported():
    # 200 cues at the default reasoning allowance need ~37k before a single
    # character of source text is added, so an endpoint capped at 20k cannot
    # serve this configuration at all.
    max_tokens_limits.record_exact_ceiling(_BASE_URL, _MODEL, 20000)

    warnings = preflight.translation_budget_warnings()

    assert len(warnings) == 1
    assert "20000" in warnings[0]
    assert "TRANSLATION_MAX_TOKENS" in warnings[0]


def test_a_ceiling_above_the_floor_is_quiet():
    max_tokens_limits.record_exact_ceiling(_BASE_URL, _MODEL, 131072)

    assert preflight.translation_budget_warnings() == []


def test_the_startup_check_does_not_consume_the_runtime_warning(capsys):
    # It used to call the warning-emitting variant, which printed at startup and
    # then marked the endpoint as already-warned - so the per-request check, the
    # one that sees the real source text, never said anything at all.
    max_tokens_limits.record_exact_ceiling(_BASE_URL, _MODEL, 20000)

    assert preflight.translation_budget_warnings()
    assert "[WARN]" not in capsys.readouterr().out

    translator._max_tokens_budget(50000)
    assert "[WARN]" in capsys.readouterr().out


def test_warnings_never_join_the_blocking_problems():
    max_tokens_limits.record_exact_ceiling(_BASE_URL, _MODEL, 1024)

    assert preflight.translation_budget_warnings()
    assert preflight.translation_config_problems("llamacpp") == []
