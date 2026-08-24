"""A reply cut off at its own budget must not cost the whole film.

`TRANSLATION_OUTPUT_CHAR_RATIO` sizes every request from its source characters,
so `max_tokens` is an arithmetic bound on a legitimate translation rather than a
context limit. Hitting it means one of two things and the transport cannot tell
them apart: the bound was too tight for this batch, or the model is looping.

It used to mean neither - it meant the job died. sample-b (2026-08-13) failed
with 1,310 of 1,701 cues already translated and paid for, and the message sent
the reader to `TRANSLATION_MAX_TOKENS`, a ceiling that `min()` had already made
irrelevant: the profile budget for a 54-cue batch was 12,794 against a 384,000
ceiling, so raising the ceiling could not have changed anything.
"""
from __future__ import annotations

import pytest

from llm import settings as llm_settings
from llm import translator
from llm.errors import (
    ResponseTruncatedError,
    RetryableTranslationError,
    TranslationError,
)


_REPLY = '{"translations":[{"id":0,"text":"你好"}]}'


@pytest.fixture(autouse=True)
def _openai_backend(monkeypatch):
    monkeypatch.setattr(translator, "selected_backend_name", lambda: "openai")


def _install(monkeypatch, transport) -> None:
    """The retry ladder lives above the transport, which is now only Responses.

    This used to patch both surfaces: patching one of two made these tests pass
    or fail according to the developer's own `.env`, and they started hitting
    the live API the day the default moved.
    """
    monkeypatch.setattr(translator, "_chat_responses", transport)


def _chat(**kwargs) -> str:
    return translator._chat(
        [{"role": "user", "content": "x"}],
        expected_count=1,
        response_schema=None,
        **kwargs,
    )


class _Transport:
    """Records the budget of every request and truncates the first `n`."""

    def __init__(self, truncate_first: int, limit_from_budget: bool = True) -> None:
        self.truncate_first = truncate_first
        self.limit_from_budget = limit_from_budget
        self.budgets: list[int] = []

    def __call__(self, *_args, max_tokens: int = 0, **_kwargs) -> str:
        self.budgets.append(max_tokens)
        if len(self.budgets) <= self.truncate_first:
            raise ResponseTruncatedError(
                f"cut off at {max_tokens} output tokens.", limit=max_tokens
            )
        return _REPLY


class TestEscalation:
    def test_a_tight_budget_is_retried_once_at_the_factor(self, monkeypatch):
        transport = _Transport(truncate_first=1)
        _install(monkeypatch, transport)
        monkeypatch.setattr(llm_settings, "TRANSLATION_TRUNCATION_RETRY_FACTOR", 2.0)

        assert _chat(max_tokens=500) == _REPLY
        assert transport.budgets == [500, 1000]

    def test_the_escalation_is_reported_so_the_case_is_diagnosable(self, monkeypatch):
        """The failure that prompted this could not be classified afterwards,
        because nothing recorded which limit bound or how close the reply got."""
        transport = _Transport(truncate_first=1)
        _install(monkeypatch, transport)
        monkeypatch.setattr(llm_settings, "TRANSLATION_TRUNCATION_RETRY_FACTOR", 2.0)

        events: list[dict] = []
        _chat(max_tokens=500, on_progress=events.append)

        truncations = [item for item in events if item.get("phase") == "output_truncated"]
        assert len(truncations) == 1
        assert truncations[0]["limit"] == 500
        assert truncations[0]["retry_limit"] == 1000

    def test_only_one_escalation_so_a_runaway_stays_bounded(self, monkeypatch):
        transport = _Transport(truncate_first=99)
        _install(monkeypatch, transport)
        monkeypatch.setattr(llm_settings, "TRANSLATION_TRUNCATION_RETRY_FACTOR", 2.0)

        with pytest.raises(ResponseTruncatedError):
            _chat(max_tokens=500)
        assert transport.budgets == [500, 1000]

    def test_no_retry_when_the_ceiling_leaves_no_room(self, monkeypatch):
        """A budget already at `TRANSLATION_MAX_TOKENS` cannot be raised, so
        reissuing would be the identical request the old comment ruled out."""
        transport = _Transport(truncate_first=99)
        _install(monkeypatch, transport)
        monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 500)
        monkeypatch.setattr(llm_settings, "TRANSLATION_TRUNCATION_RETRY_FACTOR", 2.0)

        with pytest.raises(ResponseTruncatedError):
            _chat(max_tokens=500)
        assert transport.budgets == [500]

class TestMessage:
    def test_the_final_message_names_the_knob_that_actually_bound(self, monkeypatch):
        transport = _Transport(truncate_first=99)
        _install(monkeypatch, transport)
        monkeypatch.setattr(llm_settings, "TRANSLATION_TRUNCATION_RETRY_FACTOR", 2.0)

        with pytest.raises(ResponseTruncatedError) as excinfo:
            _chat(max_tokens=500)
        message = str(excinfo.value)
        assert "TRANSLATION_OUTPUT_CHAR_RATIO" in message
        # The old text sent everyone to a ceiling `min()` had already discarded.
        assert "increase TRANSLATION_MAX_TOKENS" not in message
        assert excinfo.value.limit == 1000


class TestContract:
    def test_truncation_is_not_a_generic_retryable(self):
        """The generic retry path reissues the identical request. Truncation
        must not enter it, or a runaway is paid for `TRANSLATION_API_RETRIES`
        times at the same budget before failing anyway."""
        error = ResponseTruncatedError("x", limit=1)
        assert isinstance(error, TranslationError)
        assert not isinstance(error, RetryableTranslationError)

    def test_the_limit_travels_with_the_error(self):
        assert ResponseTruncatedError("x", limit=4096).limit == 4096
