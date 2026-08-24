from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from llm import translator
from llm.backends import openai_compat


class FakeClock:
    def __init__(self, step: float = 0.3) -> None:
        self.current = 100.0
        self.step = step

    def monotonic(self) -> float:
        self.current += self.step
        return self.current


def _chunk(*, reasoning: str | None = None, content: str | None = None, finish=None):
    delta = SimpleNamespace()
    if reasoning is not None:
        delta.reasoning_content = reasoning
    if content is not None:
        delta.content = content
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice])


def _stream(reasoning_parts: list[str], content_parts: list[str]):
    for part in reasoning_parts:
        yield _chunk(reasoning=part)
    for index, part in enumerate(content_parts):
        finish = "stop" if index == len(content_parts) - 1 else None
        yield _chunk(content=part, finish=finish)


def _broken_stream(exc: Exception):
    raise exc
    yield


def _usage_chunk(**usage_fields):
    return SimpleNamespace(choices=[], usage=SimpleNamespace(**usage_fields))


def _response_event(event_type: str, **kwargs):
    return SimpleNamespace(type=event_type, **kwargs)


def _response_stream(
    reasoning_parts: list[str], content_parts: list[str], *, usage=None
):
    """A streamed Responses call: reasoning deltas, text deltas, completion."""
    events = [
        _response_event("response.reasoning_summary_text.delta", delta=part)
        for part in reasoning_parts
    ]
    events += [
        _response_event("response.output_text.delta", delta=part)
        for part in content_parts
    ]
    completed = SimpleNamespace(output=[])
    if usage is not None:
        completed.usage = usage
    events.append(_response_event("response.completed", response=completed))
    return iter(events)


def _requested_ids_from_messages(messages) -> list[int]:
    match = re.search(r"requested_ids\s*=\s*(\[[^\]]*\])", messages[1]["content"])
    assert match is not None, messages[1]["content"]
    return json.loads(match.group(1))


def test_progress_reasoning_translating_done(monkeypatch):
    events: list[dict] = []
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek-v4-pro")
    monkeypatch.setattr(openai_compat.time, "monotonic", FakeClock(0.3).monotonic)
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda _request: _response_stream(
            ["思考", "继续"],
            ['{"translations":[{"i', 'd":0,"text":"甲"},{"id":1,"text":"乙"}]}'],
        ),
    )

    output = translator._chat(
        [{"role": "user", "content": "json"}],
        expected_count=2,
        on_progress=events.append,
    )

    assert output == '{"translations":[{"id":0,"text":"甲"},{"id":1,"text":"乙"}]}'
    phases = [event["phase"] for event in events]
    assert phases[:2] == ["thinking", "thinking"]
    assert "translating" in phases
    assert phases[-1] == "done"
    assert events[-1] == {"phase": "done", "translated": 2, "expected": 2}


def test_expected_zero_does_not_crash(monkeypatch):
    events: list[dict] = []
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek-v4-pro")
    monkeypatch.setattr(openai_compat.time, "monotonic", FakeClock(0.3).monotonic)
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda _request: _response_stream([], ['{"translations":[]}']),
    )

    output = translator._chat(
        [{"role": "user", "content": "json"}],
        expected_count=0,
        on_progress=events.append,
    )

    assert output == '{"translations":[]}'
    assert events[-1] == {"phase": "done", "translated": 0, "expected": 0}


def test_a_generic_endpoint_gets_the_strict_schema(monkeypatch):
    requests: list[dict] = []
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-5.5")
    monkeypatch.setenv(
        "OPENAI_COMPATIBILITY_BASE_URL", "https://api.openai.example/v1"
    )
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda request: requests.append(request) or _response_stream(
            [],
            ['{"translations":[{"id":0,"text":"甲"},{"id":1,"text":"乙"}]}'],
        ),
    )

    output = translator._chat(
        [{"role": "user", "content": "json"}],
        expected_count=2,
    )

    assert output == '{"translations":[{"id":0,"text":"甲"},{"id":1,"text":"乙"}]}'
    request = requests[0]
    assert request["stream"] is True
    assert request["reasoning"] == {"effort": "high"}
    text_format = request["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["name"] == "subtitle_translations"
    assert text_format["strict"] is True
    schema = text_format["schema"]
    assert schema["required"] == ["translations"]
    assert schema["properties"]["translations"]["items"]["required"] == ["id", "text"]
    assert "tools" not in request
    assert "web_search_options" not in request
    assert "include_reasoning" not in request
    assert request["max_output_tokens"] == translator.TRANSLATION_MAX_TOKENS
    assert request["temperature"] == translator.TRANSLATION_TEMPERATURE
    assert request["top_p"] == translator.TRANSLATION_TOP_P


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.deepseek.com",
        "https://openrouter.ai/api/v1",
        "https://relay.example/v1",
    ],
)
@pytest.mark.parametrize("effort", ["none", "low", "high"])
def test_every_tier_reaches_the_wire_verbatim_on_every_endpoint(
    monkeypatch, base_url: str, effort: str
):
    """The reason Chat Completions was retired, stated as a test. There the
    thinking axis was two fields with a per-provider spelling (`thinking.type`
    for DeepSeek, `reasoning.enabled` for OpenRouter) and an unknown one was
    dropped rather than refused - which is how a tier silently bills as its
    opposite. Responses has one field, and it is the tier name itself."""
    requests: list[dict] = []
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", base_url)
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda request: requests.append(request)
        or _response_stream([], ['{"translations":[{"id":0,"text":"甲"}]}']),
    )

    output = translator._chat(
        [{"role": "user", "content": "json"}],
        expected_count=1,
        reasoning_effort=effort,
    )

    assert output == '{"translations":[{"id":0,"text":"甲"}]}'
    assert requests[0]["reasoning"] == {"effort": effort}
    assert "thinking" not in (requests[0].get("extra_body") or {})


def test_deepseek_through_openrouter_gets_a_schema_its_own_api_cannot_take(
    monkeypatch,
):
    requests: list[dict] = []
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv(
        "OPENAI_COMPATIBILITY_BASE_URL", "https://openrouter.ai/api/v1"
    )
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda request: requests.append(request) or _response_stream(
            [],
            ['{"translations":[]}'],
        ),
    )

    output = translator._chat(
        [{"role": "user", "content": "json"}],
        expected_count=0,
    )

    assert output == '{"translations":[]}'
    assert requests[0]["text"]["format"]["type"] == "json_schema"
    # ...and by default nothing constrains which upstream serves it. Measured
    # 2026-08-24: `require_parameters` 404s any model that does not declare the
    # `structured_outputs` capability, including one that had just translated a
    # full film through this endpoint, so it cannot be the default.
    assert "provider" not in (requests[0].get("extra_body") or {})


def test_pinning_json_schema_demands_a_provider_that_can_enforce_it(monkeypatch):
    """The pin is for callers who want the schema enforced rather than merely
    obeyed. It fails loudly when no upstream qualifies, and `stage_errors` turns
    that 404 into the two remedies (change model, or drop to json_object)."""
    requests: list[dict] = []
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv(
        "OPENAI_COMPATIBILITY_BASE_URL", "https://openrouter.ai/api/v1"
    )
    monkeypatch.setattr(openai_compat.llm_settings, "LLM_STRUCTURED_OUTPUT", "json_schema")
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda request: requests.append(request)
        or _response_stream([], ['{"translations":[]}']),
    )

    translator._chat([{"role": "user", "content": "json"}], expected_count=0)

    assert requests[0]["text"]["format"]["type"] == "json_schema"
    assert requests[0]["extra_body"]["provider"] == {"require_parameters": True}


def test_the_provider_constraint_is_openrouter_only(monkeypatch):
    """`provider` is OpenRouter's own extension; a stricter server rejects it."""
    requests: list[dict] = []
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-5.5")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://api.openai.example/v1")
    monkeypatch.setattr(openai_compat.llm_settings, "LLM_STRUCTURED_OUTPUT", "json_schema")
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda request: requests.append(request)
        or _response_stream([], ['{"translations":[]}']),
    )

    translator._chat([{"role": "user", "content": "json"}], expected_count=0)

    assert requests[0]["text"]["format"]["type"] == "json_schema"
    assert "provider" not in (requests[0].get("extra_body") or {})


def test_a_relay_that_cannot_take_a_strict_schema_can_pin_json_object(monkeypatch):
    """The endpoint behind a private relay domain is undetectable, so the only
    honest answer is a switch. Pinning also drops the OpenRouter provider
    constraint, which exists solely to protect the strict schema."""
    requests: list[dict] = []
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://relay.example/v1")
    monkeypatch.setattr(openai_compat.llm_settings, "LLM_STRUCTURED_OUTPUT", "json_object")
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda request: requests.append(request)
        or _response_stream([], ['{"translations":[]}']),
    )

    translator._chat([{"role": "user", "content": "json"}], expected_count=0)

    assert requests[0]["text"]["format"] == {"type": "json_object"}
    assert openai_compat.OpenAICompatBackend().supports_json_schema() is False
    assert "provider" not in (requests[0].get("extra_body") or {})


def test_a_lookalike_host_is_not_the_official_deepseek_endpoint(monkeypatch):
    """Prefix matching read `https://api.deepseek.com.example/v1` as DeepSeek,
    which would drop every request on that endpoint to free-form JSON."""
    monkeypatch.setattr(openai_compat.llm_settings, "LLM_STRUCTURED_OUTPUT", "")

    assert openai_compat._is_official_deepseek_base_url("https://api.deepseek.com")
    assert openai_compat._is_official_deepseek_base_url("https://api.deepseek.com/beta")
    assert not openai_compat._is_official_deepseek_base_url(
        "https://api.deepseek.com.example/v1"
    )
    assert not openai_compat._is_official_deepseek_base_url(
        "https://openrouter.ai/api/v1"
    )
    assert openai_compat._is_openrouter_base_url("https://openrouter.ai/api/v1")
    assert not openai_compat._is_openrouter_base_url("https://openrouter.ai.example/v1")


def test_glossary_request_uses_its_own_schema(monkeypatch, tmp_path):
    requests: list[dict] = []
    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-5.5")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://api.openai.example/v1")
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda request: requests.append(request)
        or _response_stream([], ['{"terms":[{"ja":"先生","zh":"老师"}]}']),
    )

    terms = translator.extract_global_glossary(
        ["先生、ありがとう"],
        str(tmp_path / "glossary.json"),
    )

    assert terms == [{"ja": "先生", "zh": "老师"}]
    text_format = requests[0]["text"]["format"]
    assert text_format["name"] == "translation_glossary"
    assert text_format["schema"]["required"] == ["terms"]


def test_openai_compat_base_url_defaults_to_v1():
    assert (
        translator._normalize_openai_compat_base_url("https://openrouter.ai/api/v1")
        == "https://openrouter.ai/api/v1"
    )
    assert (
        translator._normalize_openai_compat_base_url("https://api.ticketpro.cc")
        == "https://api.ticketpro.cc/v1"
    )
    assert (
        translator._normalize_openai_compat_base_url("https://api.deepseek.com/")
        == "https://api.deepseek.com"
    )
    # DeepSeek's other documented base URL. Appending /v1 to it would build a
    # path that exists on no host.
    assert (
        translator._normalize_openai_compat_base_url("https://api.deepseek.com/beta")
        == "https://api.deepseek.com/beta"
    )
    assert (
        translator._normalize_openai_compat_base_url("https://api.ticketpro.cc/v1")
        == "https://api.ticketpro.cc/v1"
    )
    assert (
        translator._normalize_openai_compat_base_url(
            "https://api.ticketpro.cc/v1/chat/completions"
        )
        == "https://api.ticketpro.cc/v1"
    )


def test_reports_openai_and_deepseek_cache_usage(monkeypatch):
    usage_events: list[dict] = []
    requests: list[dict] = []
    monkeypatch.setenv("LLM_MODEL_NAME", "DeepSeek-V4-Pro")
    monkeypatch.setenv(
        "OPENAI_COMPATIBILITY_BASE_URL", "https://api.deepseek.com"
    )
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda request: requests.append(request) or _response_stream(
            [],
            ['{"translations":[]}'],
            usage=SimpleNamespace(
                prompt_tokens_details=SimpleNamespace(cached_tokens=128),
                prompt_cache_hit_tokens=96,
                prompt_cache_miss_tokens=32,
            ),
        ),
    )

    output = translator._chat(
        [{"role": "user", "content": "json"}],
        expected_count=0,
        on_usage=usage_events.append,
    )

    assert output == '{"translations":[]}'
    assert requests[0]["text"]["format"] == {"type": "json_object"}
    assert usage_events == [
        {
            "cached_tokens": 128,
            "cache_hit_tokens": 96,
            "cache_miss_tokens": 32,
        }
    ]


def test_top_tier_stream_protocol_error_falls_back_one_tier(monkeypatch):
    """A stream that dies mid-reasoning is the top tier's failure - it spends
    longest before emitting anything a parser can use. The fallback is derived
    from `REASONING_EFFORTS`, so renaming a tier cannot leave it pointing at a
    value the provider no longer accepts (which is what "medium" became)."""
    requests: list[dict] = []
    retry_events: list[dict] = []
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-5.5")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://api.openai.example/v1")

    class RemoteProtocolError(RuntimeError):
        pass

    def fake_create_response(request):
        requests.append(dict(request))
        if request["reasoning"]["effort"] == "high":
            return _broken_stream(
                RemoteProtocolError(
                    "peer closed connection without sending complete message body "
                    "(incomplete chunked read)"
                )
            )
        return _response_stream([], ['{"translations":[{"id":0,"text":"好"}]}'])

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)
    previous_retry_events = getattr(translator._RETRY_CONTEXT, "events", None)
    translator._RETRY_CONTEXT.events = retry_events
    try:
        output = translator._chat_with_reasoning(
            [{"role": "user", "content": "json"}],
            expected_count=1,
            reasoning_effort="high",
        )
    finally:
        if previous_retry_events is None:
            delattr(translator._RETRY_CONTEXT, "events")
        else:
            translator._RETRY_CONTEXT.events = previous_retry_events

    assert output == '{"translations":[{"id":0,"text":"好"}]}'
    assert [request["reasoning"]["effort"] for request in requests] == ["high", "low"]
    assert retry_events
    assert retry_events[0]["note"] == "fallback_reasoning_effort_low"


def test_responses_progress_translating_done(monkeypatch):
    events: list[dict] = []
    requests: list[dict] = []
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv(
        "OPENAI_COMPATIBILITY_BASE_URL", "https://openrouter.ai/api/v1"
    )
    monkeypatch.setattr(openai_compat.time, "monotonic", FakeClock(0.3).monotonic)

    def fake_create_response(request):
        requests.append(request)
        return iter(
            [
                _response_event(
                    "response.reasoning_summary_text.delta",
                    delta="思考",
                ),
                _response_event(
                    "response.output_text.delta",
                    delta='{"translations":[{"i',
                ),
                _response_event(
                    "response.output_text.delta",
                    delta='d":0,"text":"甲"},{"id":1,"text":"乙"}]}',
                ),
                _response_event("response.completed", response=SimpleNamespace(output=[])),
            ]
        )

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)

    output = translator._chat(
        [{"role": "system", "content": "json"}, {"role": "user", "content": "translate"}],
        expected_count=2,
        on_progress=events.append,
    )

    assert output == '{"translations":[{"id":0,"text":"甲"},{"id":1,"text":"乙"}]}'
    assert requests
    assert requests[0]["stream"] is True
    assert requests[0]["text"]["format"]["type"] == "json_schema"
    assert "extra_body" not in requests[0]
    assert requests[0]["reasoning"] == {"effort": "low"}
    assert requests[0]["input"][0]["role"] == "system"
    assert requests[0]["input"][0]["content"][0]["type"] == "input_text"
    assert requests[0]["max_output_tokens"] == translator.TRANSLATION_MAX_TOKENS
    phases = [event["phase"] for event in events]
    assert phases[0] == "thinking"
    assert "translating" in phases
    assert events[-1] == {"phase": "done", "translated": 2, "expected": 2}


def test_json_schema_support_depends_only_on_official_deepseek_url(monkeypatch):
    backend = openai_compat.OpenAICompatBackend()
    monkeypatch.setattr(openai_compat.llm_settings, "LLM_STRUCTURED_OUTPUT", "")
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv(
        "OPENAI_COMPATIBILITY_BASE_URL", "https://openrouter.ai/api/v1"
    )
    assert backend.supports_json_schema() is True

    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-5.5")
    monkeypatch.setenv(
        "OPENAI_COMPATIBILITY_BASE_URL", "https://api.deepseek.com/beta"
    )
    assert backend.supports_json_schema() is False


def test_grok_responses_uses_standard_openai_shape(monkeypatch):
    requests: list[dict] = []
    monkeypatch.setenv("LLM_MODEL_NAME", "grok-4.20-0309-non-reasoning")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://api.openai.example/v1")

    def fake_create_response(request):
        requests.append(request)
        return iter(
            [
                _response_event(
                    "response.output_text.delta",
                    delta='{"translations":[]}',
                ),
                _response_event("response.completed", response=SimpleNamespace(output=[])),
            ]
        )

    monkeypatch.setattr(openai_compat, "_create_response", fake_create_response)

    output = translator._chat(
        [{"role": "system", "content": "json"}, {"role": "user", "content": "translate"}],
        expected_count=0,
    )

    assert output == '{"translations":[]}'
    request = requests[0]
    assert request["stream"] is True
    assert request["input"][0]["role"] == "system"
    assert request["input"][0]["content"][0]["type"] == "input_text"
    assert request["reasoning"] == {"effort": "high"}
    text_format = request["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["name"] == "subtitle_translations"
    assert text_format["strict"] is True
    assert text_format["schema"]["required"] == ["translations"]
    assert (
        text_format["schema"]["properties"]["translations"]["items"]["required"]
        == ["id", "text"]
    )
    assert request["max_output_tokens"] == translator.TRANSLATION_MAX_TOKENS
    assert request["temperature"] == translator.TRANSLATION_TEMPERATURE
    assert request["top_p"] == translator.TRANSLATION_TOP_P
    assert "max_tokens" not in request


def test_debounce_limits_fast_reasoning_events(monkeypatch):
    events: list[dict] = []
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek-v4-pro")
    monkeypatch.setattr(openai_compat.time, "monotonic", FakeClock(0.05).monotonic)
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda _request: _response_stream(
            ["a", "b", "c", "d", "e", "f"],
            ['{"translations":[]}'],
        ),
    )

    translator._chat(
        [{"role": "user", "content": "json"}],
        expected_count=0,
        on_progress=events.append,
    )

    thinking_events = [event for event in events if event["phase"] == "thinking"]
    assert len(thinking_events) == 1


def test_translate_segments_emits_reset_on_retry(monkeypatch):
    events: list[dict] = []
    calls = {"count": 0}

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        calls["count"] += 1
        assert expected_count == 1
        if calls["count"] == 1:
            raise translator.RetryableTranslationFormatError("missing")
        return '{"translations":[{"id":0,"text":"好"}]}'

    monkeypatch.setattr(translator, "TRANSLATION_API_RETRIES", 2)
    monkeypatch.setattr(translator, "_request_backoff_sleep", lambda attempt, exc, **_kw: None)
    monkeypatch.setattr(translator, "_chat", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        [{"start": 0.0, "end": 1.0, "text": "いい"}],
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
        on_progress=events.append,
    )

    assert zh_texts == ["好"]
    assert retry_events == []
    assert timings[0]["segment_count"] == 1
    reset_events = [event for event in events if event["phase"] == "reset"]
    assert reset_events == [
        {"phase": "reset", "attempt": 0},
        {"phase": "reset", "attempt": 1},
    ]


def test_progress_callback_errors_do_not_break(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek-v4-pro")
    monkeypatch.setattr(openai_compat.time, "monotonic", FakeClock(0.3).monotonic)
    monkeypatch.setattr(
        openai_compat,
        "_create_response",
        lambda _request: _response_stream(["thinking"], ['{"translations":[]}']),
    )

    def broken_callback(_event):
        raise RuntimeError("ui failed")

    assert (
        translator._chat(
            [{"role": "user", "content": "json"}],
            expected_count=0,
            on_progress=broken_callback,
        )
        == '{"translations":[]}'
    )


def test_batched_progress_reaches_done_only_after_all_batches(monkeypatch):
    events: list[dict] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        ids = _requested_ids_from_messages(messages)
        if expected_count == 0:
            return '{"translations":[]}'
        if on_progress:
            on_progress(
                {
                    "phase": "done",
                    "translated": expected_count,
                    "expected": expected_count,
                }
            )
        return (
            '{"translations":['
            + ",".join(
                f'{{"id":{idx},"text":"zh-{idx}"}}'
                for idx in ids
            )
            + "]}"
        )

    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)

    zh_texts, _timings, _retry_events = translator.translate_segments(
        [
            {"start": float(index), "end": float(index) + 0.5, "text": f"ja-{index}"}
            for index in range(5)
        ],
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
        on_progress=events.append,
    )

    assert zh_texts == [f"zh-{index}" for index in range(5)]
    done_events = [event for event in events if event["phase"] == "done"]
    assert done_events == [
        {"phase": "done", "translated": 5, "expected": 5},
    ]
