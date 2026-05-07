from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm import translator


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


def _response_event(event_type: str, **kwargs):
    return SimpleNamespace(type=event_type, **kwargs)


def test_chat_progress_reasoning_translating_done(monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(translator.time, "monotonic", FakeClock(0.3).monotonic)
    monkeypatch.setattr(
        translator,
        "_create_chat_completion",
        lambda _request: _stream(
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
    monkeypatch.setattr(translator.time, "monotonic", FakeClock(0.3).monotonic)
    monkeypatch.setattr(
        translator,
        "_create_chat_completion",
        lambda _request: _stream([], ['{"translations":[]}']),
    )

    output = translator._chat(
        [{"role": "user", "content": "json"}],
        expected_count=0,
        on_progress=events.append,
    )

    assert output == '{"translations":[]}'
    assert events[-1] == {"phase": "done", "translated": 0, "expected": 0}


def test_responses_progress_translating_done(monkeypatch):
    events: list[dict] = []
    requests: list[dict] = []
    monkeypatch.setenv("LLM_API_FORMAT", "responses")
    monkeypatch.setattr(translator.time, "monotonic", FakeClock(0.3).monotonic)

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

    monkeypatch.setattr(translator, "_create_response", fake_create_response)

    output = translator._chat(
        [{"role": "system", "content": "json"}, {"role": "user", "content": "translate"}],
        expected_count=2,
        on_progress=events.append,
    )

    assert output == '{"translations":[{"id":0,"text":"甲"},{"id":1,"text":"乙"}]}'
    assert requests
    assert requests[0]["stream"] is True
    assert requests[0]["text"] == {"format": {"type": "json_object"}}
    assert requests[0]["input"][0]["role"] == "system"
    assert requests[0]["input"][0]["content"][0]["type"] == "input_text"
    assert requests[0]["max_output_tokens"] == translator.TRANSLATION_MAX_TOKENS
    phases = [event["phase"] for event in events]
    assert phases[0] == "thinking"
    assert "translating" in phases
    assert events[-1] == {"phase": "done", "translated": 2, "expected": 2}


def test_debounce_limits_fast_reasoning_events(monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(translator.time, "monotonic", FakeClock(0.05).monotonic)
    monkeypatch.setattr(
        translator,
        "_create_chat_completion",
        lambda _request: _stream(
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

    def fake_chat(messages, expected_count=0, on_progress=None):
        calls["count"] += 1
        assert expected_count == 1
        if calls["count"] == 1:
            raise translator.RetryableTranslationFormatError("missing")
        return '{"translations":[{"id":0,"text":"好"}]}'

    monkeypatch.setattr(translator, "TRANSLATION_API_RETRIES", 2)
    monkeypatch.setattr(translator, "_request_backoff_sleep", lambda attempt, exc: None)
    monkeypatch.setattr(translator, "_chat", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        [{"start": 0.0, "end": 1.0, "text": "いい"}],
        batch_size=100,
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
    monkeypatch.setattr(translator.time, "monotonic", FakeClock(0.3).monotonic)
    monkeypatch.setattr(
        translator,
        "_create_chat_completion",
        lambda _request: _stream(["thinking"], ['{"translations":[]}']),
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
        content = messages[1]["content"]
        ids = [
            int(line.split('"id": ')[1].split(",", 1)[0])
            for line in content.splitlines()
            if '"id": ' in line
        ]
        start = min(ids)
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
                for idx in range(start, start + expected_count)
            )
            + "]}"
        )

    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)

    zh_texts, _timings, _retry_events = translator.translate_segments(
        [
            {"start": float(index), "end": float(index) + 0.5, "text": f"ja-{index}"}
            for index in range(5)
        ],
        batch_size=2,
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

