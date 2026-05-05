from __future__ import annotations

import threading
from collections import defaultdict

import pytest

from llm import translator
def _segments(count: int) -> list[dict]:
    return [
        {"start": float(index), "end": float(index) + 1.0, "text": f"ja-{index}"}
        for index in range(count)
    ]


def _mock_json(start: int, count: int) -> str:
    items = [
        {"id": index, "text": f"zh-{index}"}
        for index in range(start, start + count)
    ]
    import json

    return json.dumps({"translations": items}, ensure_ascii=False)


def test_split_into_batches():
    assert len(translator._split_into_batches(_segments(0), 200)) == 0
    assert len(translator._split_into_batches(_segments(199), 200)) == 1
    assert len(translator._split_into_batches(_segments(200), 200)) == 1
    assert len(translator._split_into_batches(_segments(201), 200)) == 2
    assert len(translator._split_into_batches(_segments(450), 200)) == 3


def test_translate_segments_one_shot_when_below_threshold(monkeypatch):
    calls: list[int] = []

    def fake_chat(messages, expected_count=0, on_progress=None):
        calls.append(expected_count)
        return _mock_json(0, expected_count)

    monkeypatch.setattr(translator, "_chat", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(100),
        batch_size=200,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert calls == [100]
    assert retry_events == []
    assert len(zh_texts) == 100
    assert zh_texts[0] == "zh-0"
    assert zh_texts[-1] == "zh-99"
    assert timings[0]["mode"] == "oneshot_full_context"


def test_translate_segments_uses_task_character_reference(monkeypatch):
    system_prompts: list[str] = []

    def fake_chat(messages, expected_count=0, on_progress=None):
        system_prompts.append(messages[0]["content"])
        return _mock_json(0, expected_count)

    monkeypatch.setattr(translator, "CHARACTER_FULL_NAME_REFERENCE", "Env Name")
    monkeypatch.setattr(translator, "_chat", fake_chat)

    translator.translate_segments(
        _segments(1),
        batch_size=10,
        max_workers=1,
        cache_path="",
        target_lang="简体中文",
        glossary="",
        character_reference="Task Name",
    )

    assert system_prompts
    assert "Task Name" in system_prompts[0]
    assert "Env Name" not in system_prompts[0]


def test_translate_segments_batched(monkeypatch):
    calls: list[tuple[int, int]] = []
    lock = threading.Lock()

    def fake_chat(messages, expected_count=0, on_progress=None):
        content = messages[1]["content"]
        start = min(int(part.split('"id": ')[1].split(",", 1)[0]) for part in content.splitlines() if '"id": ' in part)
        with lock:
            calls.append((start, expected_count))
        return _mock_json(start, expected_count)

    monkeypatch.setattr(translator, "_chat", fake_chat)

    zh_texts, timings, retry_events = translator.translate_segments(
        _segments(450),
        batch_size=200,
        max_workers=3,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert sorted(calls) == [(0, 200), (200, 200), (400, 50)]
    assert retry_events == []
    assert len(zh_texts) == 450
    assert zh_texts[:3] == ["zh-0", "zh-1", "zh-2"]
    assert zh_texts[199:202] == ["zh-199", "zh-200", "zh-201"]
    assert zh_texts[-1] == "zh-449"
    assert [item["segment_count"] for item in timings[:3]] == [200, 200, 50]
    assert timings[-1]["mode"] == "batched_full_context_total"


def test_aggregated_progress_callback(monkeypatch):
    events: list[dict] = []
    current = {"value": 100.0}

    def fake_monotonic():
        current["value"] += 0.3
        return current["value"]

    monkeypatch.setattr(translator.time, "monotonic", fake_monotonic)
    callbacks, _ = translator._make_aggregated_progress_callback(4, 450, events.append)

    callbacks[0]({"phase": "thinking", "reasoning_chars": 10})
    callbacks[1]({"phase": "thinking", "reasoning_chars": 40})
    callbacks[2]({"phase": "thinking", "reasoning_chars": 25})
    callbacks[3]({"phase": "thinking", "reasoning_chars": 30})

    thinking_events = [event for event in events if event["phase"] == "thinking"]
    assert thinking_events[-1]["reasoning_chars"] == 40

    callbacks[0]({"phase": "done", "translated": 100, "expected": 100})
    callbacks[1]({"phase": "done", "translated": 100, "expected": 100})
    callbacks[2]({"phase": "done", "translated": 100, "expected": 100})
    callbacks[3]({"phase": "done", "translated": 150, "expected": 150})

    done_events = [event for event in events if event["phase"] == "done"]
    assert done_events == [{"phase": "done", "translated": 450, "expected": 450}]


def test_batch_retry_isolation(monkeypatch):
    attempts: defaultdict[int, int] = defaultdict(int)

    def fake_chat(messages, expected_count=0, on_progress=None):
        content = messages[1]["content"]
        start = min(int(part.split('"id": ')[1].split(",", 1)[0]) for part in content.splitlines() if '"id": ' in part)
        attempts[start] += 1
        if start == 200 and attempts[start] < 2:
            raise translator.RetryableTranslationFormatError("batch failed once")
        return _mock_json(start, expected_count)

    monkeypatch.setattr(translator, "TRANSLATION_API_RETRIES", 2)
    monkeypatch.setattr(translator, "_request_backoff_sleep", lambda attempt, exc: None)
    monkeypatch.setattr(translator, "_chat", fake_chat)

    zh_texts, _timings, retry_events = translator.translate_segments(
        _segments(450),
        batch_size=200,
        max_workers=3,
        cache_path="",
        target_lang="简体中文",
        glossary="",
    )

    assert attempts[0] == 1
    assert attempts[200] == 2
    assert attempts[400] == 1
    assert retry_events == []
    assert zh_texts[200] == "zh-200"
    assert zh_texts[-1] == "zh-449"

