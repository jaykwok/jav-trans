"""`TRANSLATION_MAX_TOKENS` is a fallback, not a claim about the endpoint.

The real ceiling belongs to `(endpoint, model)` and only the endpoint knows it:
the same 384000 that the default OpenRouter deployment accepts is rejected
outright by an endpoint capping the parameter at 131072. So the number here is
what gets sent until a refusal teaches a better one, and the refusal is worth
learning from because it costs a round trip and produces no tokens.

What is learned is kept typed. A refusal that names its ceiling is the ceiling;
a value found by halving is only a value known to work, and treating the second
as the first pins an endpoint below its real capability for as long as the entry
lives.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm import max_tokens_limits, translator
from llm import settings as llm_settings
from llm.backends import openai_compat
from llm.errors import (
    MaxTokensRejectedError,
    ResponseTruncatedError,
    RetryableTranslationFormatError,
)

_BASE_URL = "https://example.test/v1"
_MODEL = "test-model"


@pytest.fixture(autouse=True)
def _isolated_limits(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "TRANSLATION_MAX_TOKENS_CACHE_PATH", str(tmp_path / "limits.json")
    )
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", _BASE_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", _MODEL)
    monkeypatch.setattr(translator, "selected_backend_name", lambda *_a, **_k: "openai")
    translator._clamp_warned.clear()


def _limits():
    return max_tokens_limits.load_limits(_BASE_URL, _MODEL)


def _failed_event(message: str, code: str = "invalid_request"):
    return SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(error=SimpleNamespace(code=code, message=message)),
    )


def _ok_stream():
    return iter(
        [
            SimpleNamespace(
                type="response.output_text.delta",
                delta='{"translations":[{"id":0,"text":"甲"}]}',
            ),
            SimpleNamespace(
                type="response.completed", response=SimpleNamespace(output=[])
            ),
        ]
    )


def _truncated_stream():
    """A reply that ran out of budget rather than one refused over the number."""
    return iter(
        [
            SimpleNamespace(
                type="response.incomplete",
                response=SimpleNamespace(
                    incomplete_details={"reason": "max_output_tokens"}
                ),
            )
        ]
    )


_MESSAGES = [
    {"role": "system", "content": "json"},
    {"role": "user", "content": "translate"},
]


def test_reads_the_ceiling_out_of_the_refusal_and_retries_once(monkeypatch):
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 384000)
    sent: list[int] = []

    def fake_create_response(request):
        sent.append(request["max_output_tokens"])
        if request["max_output_tokens"] > 131072:
            return iter([_failed_event("max_tokens参数非法：限制数值范围[1,131072]")])
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)

    output = translator._chat(_MESSAGES, expected_count=1)

    assert json.loads(output)["translations"][0]["text"] == "甲"
    assert sent == [384000, 131072]
    # Named by the endpoint, so it is the ceiling and it clamps.
    assert _limits().exact_ceiling == 131072
    assert translator._max_tokens_budget(384000) == 131072


def test_an_http_400_before_the_stream_is_classified_too(monkeypatch):
    # Endpoints that validate up front answer 400 while the stream is still
    # being opened, never emitting a `response.failed` frame. That path used to
    # bypass the classifier entirely and surface as a raw SDK error.
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 384000)
    sent: list[int] = []

    class _BadRequest(Exception):
        def __init__(self):
            super().__init__("Error code: 400")
            self.code = "invalid_request_error"
            self.message = "max_tokens参数非法：限制数值范围[1,131072]"

    def fake_create_response(request):
        sent.append(request["max_output_tokens"])
        if request["max_output_tokens"] > 131072:
            raise _BadRequest()
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)

    translator._chat(_MESSAGES, expected_count=1)

    assert sent == [384000, 131072]
    assert _limits().exact_ceiling == 131072


def test_a_halved_value_is_a_lower_bound_not_a_ceiling(monkeypatch):
    # The endpoint's real ceiling here is 50000, and it never says so.
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    sent: list[int] = []

    def fake_create_response(request):
        budget = request["max_output_tokens"]
        sent.append(budget)
        if budget > 50000:
            return iter([_failed_event("max_output_tokens is out of range")])
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)
    translator._chat(_MESSAGES, expected_count=1)

    assert sent == [65536, 32768]
    learned = _limits()
    assert learned.exact_ceiling is None
    assert learned.rejected_at == 65536
    assert learned.known_good == 32768

    # The point of keeping the two apart: the next request that wants more is
    # not pinned at 32768, it probes the middle of the bracket. Pinning would
    # have cost this endpoint a third of its real capability for 30 days.
    assert translator._max_tokens_budget(65536) == (32768 + 65536) // 2
    # A request that fits under the known-good bound is not probed at all.
    assert translator._max_tokens_budget(20000) == 20000


def test_gives_up_after_the_second_halving(monkeypatch):
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    sent: list[int] = []

    def fake_create_response(request):
        sent.append(request["max_output_tokens"])
        return iter([_failed_event("max_tokens is out of range")])

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)

    with pytest.raises(MaxTokensRejectedError):
        translator._chat(_MESSAGES, expected_count=1)

    assert sent == [65536, 32768, 16384]
    # And nothing is written down. Every one of those was refused, so no budget
    # on this endpoint has actually been seen to generate - and entries no
    # longer expire, so an uncorroborated bracket would be permanent. The next
    # run pays three refused round trips again, which produce no tokens.
    assert not _limits().known_anything


def test_a_named_ceiling_beats_the_configured_fallback(monkeypatch):
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    max_tokens_limits.record_exact_ceiling(_BASE_URL, _MODEL, 131072)
    sent: list[int] = []

    def fake_create_response(request):
        sent.append(request["max_output_tokens"])
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)
    translator._chat(_MESSAGES, expected_count=1)

    # The fallback is only for endpoints that have said nothing.
    assert sent == [131072]


def test_a_first_try_success_learns_nothing(monkeypatch):
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    monkeypatch.setattr(openai_compat, "_create_response", lambda request: _ok_stream())

    translator._chat(_MESSAGES, expected_count=1)

    # Sending 65536 and being answered says the ceiling is *at least* that, not
    # where it is - and writing the cache on every batch to record an assumption
    # is not worth the I/O.
    assert not _limits().known_anything


def test_a_failure_that_is_not_about_max_tokens_stays_retryable(monkeypatch):
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda request: iter([_failed_event("upstream blip", code="server_error")]),
    )

    with pytest.raises(RetryableTranslationFormatError):
        translator._chat(_MESSAGES, expected_count=1)


def test_learned_limits_do_not_expire(monkeypatch):
    """The cache key is `(base_url, model)`, and inside one key the number holds.

    A model that gets a different cap is a different key - deployments ship with
    their limit and are retired with it rather than reconfigured. The 30-day
    expiry this replaces was guarding against a provider quietly raising the cap
    on an id already in use, and charging for that guard by discarding
    everything learned, every month, on every endpoint.
    """
    max_tokens_limits.record_exact_ceiling(_BASE_URL, _MODEL, 131072)
    a_year_later = max_tokens_limits.time.time() + 365 * 24 * 3600
    monkeypatch.setattr(max_tokens_limits.time, "time", lambda: a_year_later)

    assert _limits().exact_ceiling == 131072
    # Still merges rather than accumulates: a later refusal contradicting it
    # retires it, which is now the only thing that can.
    max_tokens_limits.record_rejection(_BASE_URL, _MODEL, 65536)
    learned = _limits()
    assert learned.exact_ceiling is None
    assert learned.rejected_at == 65536


def test_an_unknown_endpoint_gets_exactly_what_was_asked_for(monkeypatch):
    # The fallback stands in for a *missing* budget, never for a ceiling. When
    # it capped explicit budgets too, a batch whose computed budget already sat
    # at the fallback could be truncated and then had no escalation left: the
    # retry was clamped straight back to the number that had just been cut off.
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    sent: list[int] = []

    def fake_create_response(request):
        sent.append(request["max_output_tokens"])
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)
    translator._chat(_MESSAGES, expected_count=1, max_tokens=100000)

    assert sent == [100000]
    # No budget of its own still means the fallback.
    assert translator._max_tokens_budget(None) == 65536


def test_a_rejection_retires_a_contradicted_known_good(monkeypatch):
    # An endpoint whose cap moved down, or a hand-edited file: two numbers that
    # cannot both be true make the bisection nonsense, so the contradicted one
    # goes rather than being kept.
    max_tokens_limits.record_success(_BASE_URL, _MODEL, 60000)
    max_tokens_limits.record_rejection(_BASE_URL, _MODEL, 40000)

    learned = _limits()
    assert learned.rejected_at == 40000
    assert learned.known_good is None


def test_local_backend_keeps_the_configured_ceiling(monkeypatch):
    # Nothing to learn from and nothing to fall back to: a local model cannot
    # refuse a `max_tokens`, so the setting is the runaway backstop it has
    # always been and a caller-supplied budget may only lower it.
    monkeypatch.setattr(translator, "selected_backend_name", lambda *_a, **_k: "llamacpp")
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    max_tokens_limits.record_exact_ceiling(_BASE_URL, _MODEL, 131072)

    assert translator._endpoint_identity() is None
    assert translator._max_tokens_budget(384000) == 65536
    assert translator._max_tokens_budget(1024) == 1024


def test_a_corrupt_cache_cannot_fail_a_translation(monkeypatch, tmp_path):
    # A capability cache is an optimisation. Anything unreadable in it must cost
    # at most a re-probe, never the film.
    path = tmp_path / "limits.json"
    path.write_text(
        json.dumps(
            {
                "schema": max_tokens_limits.LIMITS_SCHEMA,
                "limits": {
                    key: {"updated_at": "not-a-number", "exact_ceiling": "nonsense"}
                    for key in (f"junk{index}" for index in range(20))
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    monkeypatch.setattr(openai_compat, "_create_response", lambda request: _ok_stream())

    assert not _limits().known_anything
    assert translator._max_tokens_budget(None) == 65536
    assert translator._chat(_MESSAGES, expected_count=1)
    # Pruning walks every entry's timestamp, including the unparseable ones.
    max_tokens_limits.record_exact_ceiling(_BASE_URL, _MODEL, 131072)
    assert _limits().exact_ceiling == 131072


def test_the_truncation_retry_does_not_restart_the_probe_ladder(monkeypatch):
    """Retry budget, fallback and probe count belong to the call, not the ladder.

    Endpoint ceiling 66000 inside a `[65536, 131072)` bracket, and 65536 is not
    enough to finish the reply. With the counters living inside the ladder, the
    escalation got three fresh probes and a fresh known-good fallback, and since
    the ladder only walks *down* it spent them arriving back at 65536 - a full
    generation, billed, cut off in exactly the same place.
    """
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 131072)
    monkeypatch.setattr(llm_settings, "TRANSLATION_TRUNCATION_RETRY_FACTOR", 2.0)
    max_tokens_limits.record_rejection(_BASE_URL, _MODEL, 131072)
    max_tokens_limits.record_success(_BASE_URL, _MODEL, 65536)
    sent: list[int] = []

    def fake_create_response(request):
        budget = request["max_output_tokens"]
        sent.append(budget)
        if budget > 66000:
            return iter([_failed_event("max_output_tokens out of range")])
        return _truncated_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)

    with pytest.raises(MaxTokensRejectedError):
        translator._chat(_MESSAGES, expected_count=1, max_tokens=131072)

    assert sent == [98304, 81920, 73728, 65536, 69632]
    # The one that generates is the one that must not repeat.
    assert sent.count(65536) == 1


def test_the_truncation_message_names_the_budget_that_was_sent(monkeypatch):
    # The retry aims at one number and the ladder may step down from it before
    # anything is generated. Reporting the number it aimed at describes a
    # request that never happened - and hands the wrong `limit` to the caller.
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    monkeypatch.setattr(llm_settings, "TRANSLATION_TRUNCATION_RETRY_FACTOR", 2.0)
    sent: list[int] = []

    def fake_create_response(request):
        budget = request["max_output_tokens"]
        sent.append(budget)
        if budget > 15000:
            return iter(
                [_failed_event(f"max_tokens {budget} exceeds the limit of 15000")]
            )
        return _truncated_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)

    with pytest.raises(ResponseTruncatedError) as excinfo:
        translator._chat(_MESSAGES, expected_count=1, max_tokens=10000)

    assert sent == [10000, 20000, 15000]
    assert excinfo.value.limit == 15000
    assert "15000 tokens" in str(excinfo.value)


def test_a_success_retires_the_refusal_it_disproves():
    # The mirror of the rejection rule. Ranking the two by strength instead of
    # by age kept the refusal that had just been contradicted and threw away the
    # number this call had watched work.
    max_tokens_limits.record_rejection(_BASE_URL, _MODEL, 40000)
    max_tokens_limits.record_success(_BASE_URL, _MODEL, 30000)
    max_tokens_limits.record_success(_BASE_URL, _MODEL, 50000)

    learned = _limits()
    assert learned.known_good == 50000
    assert learned.rejected_at is None


def test_a_success_above_a_cached_ceiling_retires_it():
    max_tokens_limits.record_exact_ceiling(_BASE_URL, _MODEL, 40000)
    max_tokens_limits.record_success(_BASE_URL, _MODEL, 50000)

    learned = _limits()
    assert learned.exact_ceiling is None
    assert learned.known_good == 50000


def test_naming_the_parameter_is_not_refusing_the_output_ceiling():
    kind = openai_compat._max_tokens_refusal_kind
    assert kind("max_tokens参数非法：限制数值范围[1,131072]") == "output"
    assert kind("max_output_tokens is out of range") == "output"
    # Real, and satisfiable by a smaller budget - but the number moves with the
    # prompt, so it is this request's ceiling and not the endpoint's.
    assert kind("input tokens plus max_output_tokens must be <= 131072") == "request"
    # A floor: halving walks away from the only value that would have worked.
    assert kind("max_tokens must be at least 100000") == ""
    assert kind("max_output_tokens is not supported for this model") == ""
    # A provider may label a plain ceiling "unsupported value". The bound is
    # what is actionable, so it wins over the label.
    assert kind("Unsupported value: max_tokens must be <= 131072") == "output"
    # Says the magnitude was the problem without saying by how much. Still
    # learnable: halving is the right move and the message does claim too large.
    assert kind("max_tokens is out of range") == "output"
    assert kind("max_output_tokens 65536 exceeds what this model allows") == "output"
    # Claims nothing at all. Halve, but do not write a month-long bracket out of
    # a message that never said the number was too big.
    assert kind("max_tokens is invalid") == "unclear"


def test_an_unquantified_refusal_is_retried_but_not_learned(monkeypatch):
    """Persisting needs positive evidence that the number was too large.

    The costs are asymmetric. Not learning costs one refused round trip on the
    next run, and a refused request generates nothing. Learning a bracket that
    was never claimed clamps every budget to a midpoint for 30 days, and a
    clamped budget is paid for in generated tokens the first time a reply is cut
    off at it.
    """
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    sent: list[int] = []

    def fake_create_response(request):
        budget = request["max_output_tokens"]
        sent.append(budget)
        if budget > 40000:
            return iter([_failed_event("max_tokens is invalid")])
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)
    translator._chat(_MESSAGES, expected_count=1, max_tokens=65536)

    assert sent == [65536, 32768]
    learned = _limits()
    assert learned.rejected_at is None
    assert learned.exact_ceiling is None


def test_the_retry_log_carries_the_provider_message(monkeypatch, capsys):
    # The classifier is built from a handful of observed wordings, and the ones
    # it cannot place are the ones worth collecting. A refusal that is then
    # retried successfully never reaches an exception anyone reads, so the words
    # have to be in the retry line itself.
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)

    def fake_create_response(request):
        if request["max_output_tokens"] > 40000:
            return iter([_failed_event("max_tokens is invalid")])
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)
    translator._chat(_MESSAGES, expected_count=1, max_tokens=65536)

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if "endpoint rejected max_tokens" in line
    )
    assert "max_tokens is invalid" in line
    assert "not learned" in line


def test_a_refusal_is_staged_until_a_smaller_budget_generates(monkeypatch):
    """Nothing expires, so nothing gets written on the strength of wording alone.

    The endpoint's first answer is a refusal, and until something generates
    under it the only evidence that it was a *ceiling* is that the message
    sounded like one. That is a phrase list's opinion, and it would now be
    permanent. So the refusal steers this call and stays off disk until the
    endpoint behaves like a ceiling: refuse high, generate low.
    """
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    staged: list[int] = []

    def fake_create_response(request):
        budget = request["max_output_tokens"]
        staged.append(budget)
        if budget > 40000:
            return iter([_failed_event("max_output_tokens out of range")])
        # Read from inside the ladder, after the refusal and before the reply:
        # the refusal is steering (the retry is 32768, not 65535) but has not
        # been written down.
        assert not _limits().known_anything
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)
    translator._chat(_MESSAGES, expected_count=1, max_tokens=65536)

    assert staged == [65536, 32768]
    learned = _limits()
    assert learned.rejected_at == 65536
    assert learned.known_good == 32768


def test_an_uncorroborated_refusal_is_never_written_down(monkeypatch):
    # Same first request, but the smaller budget is refused too, so this call
    # never sees the endpoint generate anything. One bad classification would
    # otherwise pin this endpoint at 16384 with no expiry to undo it.
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)

    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda request: iter([_failed_event("max_output_tokens out of range")]),
    )

    with pytest.raises(MaxTokensRejectedError):
        translator._chat(_MESSAGES, expected_count=1, max_tokens=65536)

    assert not _limits().known_anything


def test_corroboration_does_not_promote_a_prompt_sized_refusal(monkeypatch):
    # Once a call is convinced, later refusals go straight to disk - but the
    # combined input+output limit still must not, or the flush would launder it
    # into a permanent endpoint fact.
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    sent: list[int] = []

    def fake_create_response(request):
        budget = request["max_output_tokens"]
        sent.append(budget)
        if budget > 40000:
            return iter([_failed_event("max_output_tokens out of range")])
        if budget > 20000:
            return iter(
                [_failed_event("input tokens plus max_output_tokens must be <= 131072")]
            )
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)
    translator._chat(_MESSAGES, expected_count=1, max_tokens=65536)

    assert sent == [65536, 32768, 16384]
    learned = _limits()
    # 65536 was corroborated by the reply at 16384 and is kept; 32768 was
    # refused for this prompt's size and is not.
    assert learned.rejected_at == 65536
    assert learned.known_good == 16384


def test_a_first_truncation_is_a_floor_even_with_nothing_else_known(monkeypatch):
    """An accepted budget is evidence whether or not anything was refused yet.

    Memory and disk used to be gated together, so an unknown endpoint learned
    nothing from a reply that ran out of room. Ceiling 60000, reply needs 50000,
    caller asks for 40000: the escalation doubles to 80000, gets refused, and
    the bisection - starting from no floor at all - computes 40000 again. That
    is correctly blocked for being a budget already outgrown, so the request
    fails at an endpoint where 60000 would have finished it.
    """
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    monkeypatch.setattr(llm_settings, "TRANSLATION_TRUNCATION_RETRY_FACTOR", 2.0)
    sent: list[int] = []

    def fake_create_response(request):
        budget = request["max_output_tokens"]
        sent.append(budget)
        if budget > 60000:
            return iter([_failed_event("max_output_tokens out of range")])
        if budget < 50000:
            return _truncated_stream()
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)

    output = translator._chat(_MESSAGES, expected_count=1, max_tokens=40000)

    assert json.loads(output)["translations"][0]["text"] == "甲"
    # 40000 is the floor the truncation established, 80000 the refused ceiling,
    # 60000 the midpoint between them.
    assert sent == [40000, 80000, 60000]
    assert _limits().known_good == 60000


def test_a_refusal_only_retires_the_successes_it_disproves():
    # 40000 refused says everything about the 50000 that was accepted and
    # nothing about the 30000 that was. Dropping the pair together left no floor
    # at all, and the next bisection restarted at 20000 - under a budget this
    # endpoint had already generated at, which is paid for in tokens when the
    # reply is then cut off rather than in a free refusal.
    max_tokens_limits.record_success(_BASE_URL, _MODEL, 50000)
    max_tokens_limits.record_success(_BASE_URL, _MODEL, 30000)
    max_tokens_limits.record_rejection(_BASE_URL, _MODEL, 40000)

    learned = _limits()
    assert learned.rejected_at == 40000
    assert learned.known_good == 30000
    assert max_tokens_limits.budget_for(learned, 65536) == (30000 + 40000) // 2


def test_the_surviving_floor_is_kept_in_memory_too():
    # The same rule on the in-call state, which is what a running batch bisects
    # against - fixing only the on-disk representation would leave the call that
    # made the observations working from the worse one.
    limits = max_tokens_limits.EndpointLimits()
    limits = max_tokens_limits.merge_observation(limits, success=50000)
    limits = max_tokens_limits.merge_observation(limits, success=30000)
    limits = max_tokens_limits.merge_observation(limits, rejection=40000)

    assert limits.known_good == 30000
    assert limits.rejected_at == 40000


def test_a_refusal_retires_the_recent_success_too(monkeypatch):
    # The second slot must not become a way for a contradicted floor to survive
    # the merge that dropped it.
    max_tokens_limits.record_rejection(_BASE_URL, _MODEL, 65536)
    max_tokens_limits.record_success(_BASE_URL, _MODEL, 50000)
    max_tokens_limits.record_success(_BASE_URL, _MODEL, 30000)
    max_tokens_limits.record_rejection(_BASE_URL, _MODEL, 20000)

    learned = _limits()
    assert learned.rejected_at == 20000
    assert learned.known_good is None


def test_a_floor_never_enters_the_halving_ladder(monkeypatch):
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    sent: list[int] = []

    def fake_create_response(request):
        sent.append(request["max_output_tokens"])
        return iter([_failed_event("max_tokens must be at least 100000")])

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)

    with pytest.raises(RetryableTranslationFormatError):
        translator._chat(_MESSAGES, expected_count=1, max_tokens=65536)

    # No ladder, and nothing written down: three smaller requests would each get
    # the same answer, and `rejected_at=16384` would then be a fact about an
    # endpoint that never refused 16384.
    assert sent == [65536]
    assert not _limits().known_anything


def test_a_combined_input_output_limit_is_retried_but_not_learned(monkeypatch):
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    sent: list[int] = []

    def fake_create_response(request):
        budget = request["max_output_tokens"]
        sent.append(budget)
        if budget > 30000:
            return iter(
                [_failed_event("input tokens plus max_output_tokens must be <= 131072")]
            )
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)
    translator._chat(_MESSAGES, expected_count=1, max_tokens=65536)

    # Bisected inside the call, because a smaller budget really does fit.
    assert sent == [65536, 32768, 16384]
    learned = _limits()
    # But nothing about a ceiling survives it. Learning 131072 as `exact_ceiling`
    # would clamp every shorter batch of the film by this prompt's arithmetic.
    assert learned.exact_ceiling is None
    assert learned.rejected_at is None


def test_accepted_limit_only_reads_an_unambiguous_bound():
    parse = openai_compat._accepted_max_tokens
    assert parse("max_tokens参数非法：限制数值范围[1,131072]", 384000) == 131072
    assert parse("max_tokens 384000 exceeds the limit of 131072", 384000) == 131072
    assert parse("max_output_tokens must be <= 131072", 384000) == 131072
    assert parse("max_tokens: maximum of 131072", 384000) == 131072
    # A request id is not a token ceiling. Taking the largest smaller integer
    # anywhere in the message used to learn 12345 from this.
    assert parse("max_tokens invalid; request id req_12345", 384000) is None
    assert parse("max_tokens is invalid", 384000) is None
    assert parse("max_tokens must be at least 500000", 384000) is None
    # A floor is not a ceiling. Reading this one would cap the endpoint at 100.
    assert parse("max_tokens must be greater than 100", 384000) is None


def test_a_clamped_budget_is_reported_once_per_endpoint(monkeypatch, capsys):
    # The startup check only sees configuration; a budget driven over the line
    # by the actual source text shows up here and nowhere else. Once per
    # endpoint, because a line per batch would be noise on a 1500-cue film.
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    max_tokens_limits.record_exact_ceiling(_BASE_URL, _MODEL, 20000)

    assert translator._max_tokens_budget(50000) == 20000
    assert translator._max_tokens_budget(50000) == 20000

    warnings = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "[WARN]" in line and "50000" in line
    ]
    assert len(warnings) == 1


def test_budget_for_is_pure_arithmetic():
    budget_for = max_tokens_limits.budget_for
    # Nothing known means nothing to clamp with. Resolving "no preference" into
    # a number is the caller's job, so no fallback is passed in at all.
    nothing = max_tokens_limits.EndpointLimits()
    assert budget_for(nothing, 384000) == 384000
    assert budget_for(nothing, 40000) == 40000

    exact = max_tokens_limits.EndpointLimits(exact_ceiling=131072)
    assert budget_for(exact, 384000) == 131072

    bracket = max_tokens_limits.EndpointLimits(rejected_at=65536, known_good=32768)
    assert budget_for(bracket, 384000) == (32768 + 65536) // 2
    assert budget_for(bracket, 30000) == 30000

    # Converged: the room still unclaimed is not worth a rejected round trip.
    tight = max_tokens_limits.EndpointLimits(rejected_at=50000, known_good=49500)
    assert budget_for(tight, 384000) == 49500


def test_a_successful_probe_advances_the_bracket(monkeypatch):
    # The bracket has to move on a *win* too. Recording only after a refusal
    # left a midpoint that succeeded first try unrecorded, so every later
    # request probed the same number again and never converged.
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    max_tokens_limits.record_rejection(_BASE_URL, _MODEL, 65536)
    max_tokens_limits.record_success(_BASE_URL, _MODEL, 32768)
    sent: list[int] = []

    def fake_create_response(request):
        sent.append(request["max_output_tokens"])
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)

    translator._chat(_MESSAGES, expected_count=1, max_tokens=65536)
    midpoint = (32768 + 65536) // 2
    assert sent == [midpoint]
    assert _limits().known_good == midpoint

    translator._chat(_MESSAGES, expected_count=1, max_tokens=65536)
    assert sent[-1] == (midpoint + 65536) // 2


def test_a_truncated_reply_records_the_budget_the_endpoint_accepted(monkeypatch):
    """Generating until the budget runs out is the endpoint *taking* the number.

    Only a clean return used to record it, so a probe that came back truncated
    left the bracket where it was - and the truncation escalation, which sizes
    itself from that same bracket, recomputed the identical midpoint and then
    refused to reissue because the retry was not larger than the limit that had
    just bound. One truncation, no escalation, batch lost.
    """
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    monkeypatch.setattr(llm_settings, "TRANSLATION_TRUNCATION_RETRY_FACTOR", 2.0)
    max_tokens_limits.record_rejection(_BASE_URL, _MODEL, 65536)
    max_tokens_limits.record_success(_BASE_URL, _MODEL, 32768)
    budgets: list[int] = []

    def fake_responses(*_args, max_tokens: int = 0, **_kwargs):
        budgets.append(max_tokens)
        if len(budgets) == 1:
            raise ResponseTruncatedError(f"cut off at {max_tokens}", limit=max_tokens)
        return '{"translations":[{"id":0,"text":"甲"}]}'

    monkeypatch.setattr(translator, "_chat_responses", fake_responses)

    translator._chat(_MESSAGES, expected_count=1, max_tokens=65536)

    midpoint = (32768 + 65536) // 2
    # Second request off the *narrowed* bracket. Without the record it was
    # `[49152]` alone: same midpoint, `retry <= limit`, straight to the failure.
    assert budgets == [midpoint, (midpoint + 65536) // 2]
    assert _limits().known_good == (midpoint + 65536) // 2


def test_a_fresh_refusal_retires_the_ceiling_it_contradicts():
    # Two numbers that cannot both be true, and here the tie-break is age: the
    # refusal happened seconds ago, the ceiling is up to 30 days of cache.
    max_tokens_limits.record_exact_ceiling(_BASE_URL, _MODEL, 65536)
    max_tokens_limits.record_rejection(_BASE_URL, _MODEL, 60000)

    learned = _limits()
    assert learned.exact_ceiling is None
    assert learned.rejected_at == 60000


def test_a_lowered_cap_is_re_bracketed_instead_of_walked_down(monkeypatch):
    """The endpoint's cap moved below what the cache says it named.

    Keeping the stale ceiling made every retry clamp to just under the value
    that had just been refused: 60000, 59999, 59998, fail - three round trips
    that could not have gone anywhere, on an endpoint that would have answered
    at 30000.
    """
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    max_tokens_limits.record_exact_ceiling(_BASE_URL, _MODEL, 65536)
    sent: list[int] = []

    def fake_create_response(request):
        budget = request["max_output_tokens"]
        sent.append(budget)
        if budget > 40000:
            return iter([_failed_event("max_output_tokens out of range")])
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)
    translator._chat(_MESSAGES, expected_count=1, max_tokens=60000)

    assert sent == [60000, 30000]
    learned = _limits()
    assert learned.exact_ceiling is None
    assert learned.rejected_at == 60000
    assert learned.known_good == 30000


def test_the_last_try_goes_to_a_budget_already_known_to_work(monkeypatch):
    """Running out of probes must not fail a request that could still be sent.

    Real ceiling 33000 inside a `[32768, 65536)` bracket: the ladder spends all
    three probes narrowing towards it and every one is refused, while a value
    the endpoint has already answered at sits unused in `known_good`. The extra
    request is not a probe - it is the translation, at the last budget known to
    work - so it is worth one round trip past the probe budget.
    """
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    max_tokens_limits.record_rejection(_BASE_URL, _MODEL, 65536)
    max_tokens_limits.record_success(_BASE_URL, _MODEL, 32768)
    sent: list[int] = []

    def fake_create_response(request):
        budget = request["max_output_tokens"]
        sent.append(budget)
        if budget > 33000:
            return iter([_failed_event("max_output_tokens out of range")])
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)

    output = translator._chat(_MESSAGES, expected_count=1, max_tokens=65536)

    assert json.loads(output)["translations"][0]["text"] == "甲"
    assert sent == [49152, 40960, 36864, 32768]


def test_an_unwritable_cache_cannot_turn_the_ladder_into_a_crawl(monkeypatch):
    """The cache is best effort; the probe ladder is not allowed to be.

    Every refusal used to be recomputed from what could be read back, so a cache
    that cannot be written read back as "nothing known" - and the next step from
    nothing known is `sent - 1`. Requests went 65536, 65535, 65534 and failed on
    an endpoint whose real cap was 40000.
    """
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)

    def unwritable(_payload):
        raise OSError("read-only file system")

    monkeypatch.setattr(max_tokens_limits, "_write_payload", unwritable)
    sent: list[int] = []

    def fake_create_response(request):
        budget = request["max_output_tokens"]
        sent.append(budget)
        if budget > 40000:
            return iter([_failed_event("max_output_tokens out of range")])
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)
    translator._chat(_MESSAGES, expected_count=1, max_tokens=65536)

    assert sent == [65536, 32768]
    # Nothing survived the call, which is the whole point of the case: the
    # ladder held its own state and the request completed anyway.
    assert not _limits().known_anything


def test_a_rejected_probe_never_drops_below_a_known_good_budget(monkeypatch):
    # Halving off the refused value would land under a budget already proven to
    # work; the next step has to come from the narrowed bracket instead.
    monkeypatch.setattr(translator, "TRANSLATION_MAX_TOKENS", 65536)
    max_tokens_limits.record_rejection(_BASE_URL, _MODEL, 65536)
    max_tokens_limits.record_success(_BASE_URL, _MODEL, 32768)
    sent: list[int] = []

    def fake_create_response(request):
        budget = request["max_output_tokens"]
        sent.append(budget)
        if budget > 40000:
            return iter([_failed_event("max_output_tokens out of range")])
        return _ok_stream()

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)
    translator._chat(_MESSAGES, expected_count=1, max_tokens=65536)

    midpoint = (32768 + 65536) // 2
    assert sent[0] == midpoint
    # Blind halving would have sent 24576, below the 32768 already known good.
    assert sent[1] == (32768 + midpoint) // 2
    assert all(value >= 32768 for value in sent[1:])
