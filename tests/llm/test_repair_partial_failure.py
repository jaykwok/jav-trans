"""What the repair pass returns when it only partly succeeded.

Returning the Japanese source as the translation is the 10.1% untranslated-film
regression, so the pass ends on a gate that refuses to hand back a lingering
source echo. That gate was applied to the repaired array while the failure path
returned the *original* one - two different arrays, so the check proved nothing
about the value the caller got. A first request that fixed an echo, followed by
any later request-level failure, put that echo straight back and threw away a
repair that had already been generated and billed.

Verification, return value and cache are one array now, and ids the detectors
still flag are reported rather than hidden behind a rollback.
"""

from __future__ import annotations

import json

import pytest

from llm import repair
from llm import settings as llm_settings
from llm.errors import BackendLeaseInvalidatedError, ContentPolicyRefusalError
from llm.profiles import get_profile
from llm.session import TranslationSession

# Cue 0's "translation" is its own source - an exact echo, the failure this
# whole pass exists for. Cue 1 still carries kana, which is a correlate rather
# than a proof, and is what the second request is asked to fix.
_SEGMENTS = [
    {"text": "今日はいい天気ですね。", "start": 0, "end": 2},
    {"text": "明日は公園に行きます。", "start": 2, "end": 4},
]
_ORIGINAL = ["今日はいい天気ですね。", "明天去公園です。"]
_FIXED_FIRST = "今天天气真好。"


@pytest.fixture(autouse=True)
def _repair_budget(monkeypatch):
    monkeypatch.setattr(llm_settings, "TRANSLATION_REPAIR_MAX_IDS", 100)
    monkeypatch.setattr(llm_settings, "TRANSLATION_BATCH_MAX_REQUESTS", 20)


def _session(**overrides) -> TranslationSession:
    """The run both passes belong to. The repair pass takes this, not eight
    arguments a caller could pass inconsistently."""
    fields = {
        "backend_name": "openai",
        "profile": get_profile("json"),
        "batch_size": 1,
        "max_workers": 1,
        "cache_path": "",
        "target_lang": "zh",
        "glossary": "",
        "character_reference": "",
        "reasoning_effort": "low",
    }
    fields.update(overrides)
    return TranslationSession(**fields)


def _run(chat, texts, cache_writer=None):
    return repair.apply_repair_pass(
        list(_SEGMENTS),
        list(texts),
        chat=chat,
        session=_session(),
        cache_writer=cache_writer,
    )


def _fix_first_then_fail():
    """One repair that works, then a transport that stops working."""
    calls: list[str] = []

    def chat(*_args, **kwargs):
        calls.append(kwargs["reasoning_effort"])
        if len(calls) == 1:
            return json.dumps(
                {"translations": [{"id": 0, "text": _FIXED_FIRST}]},
                ensure_ascii=False,
            )
        raise RuntimeError("synthetic transport failure")

    return chat, calls


def test_a_later_failure_never_puts_the_japanese_source_back():
    chat, calls = _fix_first_then_fail()

    result, timing = _run(chat, _ORIGINAL)

    assert result[0] == _FIXED_FIRST  # the repair that was paid for survives
    assert timing["mode"] == "translation_repair_failed"
    assert len(calls) == 3  # stage 1 per cue, then one escalated attempt
    # The contract the gate exists to hold, asserted against what was actually
    # returned rather than against an array nobody sees.
    assert not repair._has_source_echo(
        repair._repair_source_text(_SEGMENTS[0]), result[0]
    )


def test_the_failure_record_says_what_is_still_wrong():
    chat, _calls = _fix_first_then_fail()

    _result, timing = _run(chat, _ORIGINAL)

    # Cue 0 was fixed, cue 1 never was. Reporting both as missing was the same
    # rollback story told in the timings.
    assert timing["segment_count"] == 1
    assert timing["missing_indexes"] == [1]
    assert timing["candidate_count"] == 2


def test_a_repair_that_survives_the_failure_is_still_written_to_the_cache():
    # It was generated and billed; a later request failing says nothing about
    # it, and a re-run must not buy it a second time.
    chat, _calls = _fix_first_then_fail()
    written: list[list[str]] = []

    result, _timing = _run(chat, _ORIGINAL, cache_writer=written.append)

    assert written == [result]


class TestATerminalErrorIsNotARepairFailure:
    """Escalating a tier is a recovery action, so it needs a reason to work.

    Content refusal, a lease that is gone and cancellation all say the next
    request cannot succeed either. Caught by `except Exception`, they were
    answered with a second request at a higher reasoning tier and then reported
    as `translation_repair_failed` - a stated terminal condition rewritten as a
    generic one, with the escalated request paid for on the way.
    """

    def test_a_content_refusal_is_asked_exactly_once(self):
        calls: list[str] = []

        def chat(*_args, **kwargs):
            calls.append(kwargs["reasoning_effort"])
            raise ContentPolicyRefusalError("refused on content policy (code=cyber_policy)")

        with pytest.raises(ContentPolicyRefusalError):
            _run(chat, _ORIGINAL)

        # One request, and the type the backend raised - not a tier escalation,
        # not a RuntimeError about echoes that the refusal never let us fix.
        assert calls == ["none"]

    def test_an_invalidated_lease_stops_the_pass(self):
        calls: list[str] = []

        def chat(*_args, **kwargs):
            calls.append(kwargs["reasoning_effort"])
            raise BackendLeaseInvalidatedError("the instance this task held is gone")

        with pytest.raises(BackendLeaseInvalidatedError):
            _run(chat, _ORIGINAL)

        assert len(calls) == 1

    def test_what_was_already_repaired_is_still_written(self):
        # Stopping early must not cost what was already generated and billed.
        calls: list[str] = []

        def chat(*_args, **kwargs):
            calls.append(kwargs["reasoning_effort"])
            if len(calls) == 1:
                return json.dumps(
                    {"translations": [{"id": 0, "text": _FIXED_FIRST}]},
                    ensure_ascii=False,
                )
            raise ContentPolicyRefusalError("refused on content policy")

        written: list[list[str]] = []
        with pytest.raises(ContentPolicyRefusalError):
            _run(chat, _ORIGINAL, cache_writer=written.append)

        assert len(calls) == 2
        assert written and written[0][0] == _FIXED_FIRST


def test_a_lingering_echo_still_fails_the_run_loudly():
    # The other half of the contract: the gate is outside the handler on
    # purpose, so a provider error cannot be a way past it.
    def chat(*_args, **_kwargs):
        raise RuntimeError("synthetic transport failure")

    with pytest.raises(RuntimeError, match="still echoes the Japanese source"):
        _run(chat, _ORIGINAL)
