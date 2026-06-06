from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from llm import translator


def _read_cache_jsonl(path) -> dict:
    return translator._load_translation_cache(path)


def _segments(count: int) -> list[dict]:
    return [
        {"start": float(index), "end": float(index) + 1.0, "text": f"ja-{index}"}
        for index in range(count)
    ]


def _mock_json(start: int, count: int) -> str:
    return json.dumps(
        {
            "translations": [
                {"id": index, "text": f"zh-{index}"}
                for index in range(start, start + count)
            ]
        },
        ensure_ascii=False,
    )


def _batch_start_from_messages(messages) -> int:
    match = re.search(r"requested_ids\s*=\s*(\[[^\]]*\])", messages[1]["content"])
    assert match is not None, messages[1]["content"]
    ids = json.loads(match.group(1))
    return min(ids) if ids else 0


def test_save_cache_entry_concurrent_jsonl_intact(tmp_path):
    cache_path = tmp_path / "translation_cache.jsonl"
    lock = threading.Lock()

    def write(index: int) -> None:
        translator._save_cache_entry(
            cache_path,
            str(index),
            [f"zh-{index}-0", f"zh-{index}-1"],
            lock,
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(write, range(3)))

    data = _read_cache_jsonl(cache_path)
    assert sorted(data) == ["0", "1", "2"]
    assert data["0"] == ["zh-0-0", "zh-0-1"]
    assert data["1"] == ["zh-1-0", "zh-1-1"]
    assert data["2"] == ["zh-2-0", "zh-2-1"]
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 3


def test_batched_translation_skips_cached_batches(monkeypatch, tmp_path):
    cache_path = tmp_path / "translation_cache.jsonl"
    segments = _segments(4)
    cache_key = translator._translation_cache_key(
        0,
        segments[:2],
        glossary="",
        target_lang="简体中文",
        character_reference="",
    )
    cache_path.write_text(
        json.dumps({"key": cache_key, "value": ["cached-0", "cached-1"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    calls: list[tuple[int, int]] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        start = _batch_start_from_messages(messages)
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        calls.append((start, expected_count))
        return _mock_json(start, expected_count)

    monkeypatch.setattr(translator, "_chat", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)

    zh_texts, timings, retry_events = translator.translate_segments(
        segments,
        max_workers=2,
        cache_path=str(cache_path),
        target_lang="简体中文",
        glossary="",
        character_reference="",
    )

    assert calls == [(2, 2)]
    assert retry_events == []
    assert zh_texts == ["cached-0", "cached-1", "zh-2", "zh-3"]
    assert any(item["mode"] == "translation_cache_hit" for item in timings)
    assert any(item["mode"] == "batched_full_context" for item in timings)
    assert timings[-1]["cache_hit_count"] == 1

    saved = _read_cache_jsonl(cache_path)
    assert saved[cache_key] == ["cached-0", "cached-1"]
    assert saved[
        translator._translation_cache_key(
            1,
            segments[2:],
            glossary="",
            target_lang="简体中文",
            character_reference="",
        )
    ] == ["zh-2", "zh-3"]
    assert cache_path.exists()


def test_load_translation_cache_warns_on_corrupt_jsonl(tmp_path, capsys):
    cache_path = tmp_path / "translation_cache.jsonl"
    cache_path.write_text("{not-json}\n", encoding="utf-8")

    assert translator._load_translation_cache(cache_path) == {}

    captured = capsys.readouterr()
    assert "[WARN] translation cache JSONL load failed" in captured.out
    assert str(cache_path) in captured.out


def test_translation_memory_reuses_text_when_timing_changes(monkeypatch, tmp_path):
    cache_path = tmp_path / "translation_cache.jsonl"
    first_segments = [
        {"start": 0.0, "end": 1.0, "text": "もっと来て"},
        {"start": 1.0, "end": 2.0, "text": "そこ触って"},
    ]
    changed_timing = [
        {"start": 0.0, "end": 1.5, "text": "もっと来て"},
        {"start": 1.5, "end": 2.3, "text": "そこ触って"},
    ]
    calls: list[tuple[int, int]] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        start = _batch_start_from_messages(messages)
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        calls.append((start, expected_count))
        texts = ["再靠近点", "摸那里"]
        return json.dumps(
            {
                "translations": [
                    {"id": start + offset, "text": texts[start + offset]}
                    for offset in range(expected_count)
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "_chat", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 1)

    first_texts, _, _ = translator.translate_segments(
        first_segments,
        max_workers=1,
        cache_path=str(cache_path),
        target_lang="简体中文",
        glossary="",
        character_reference="",
    )

    calls.clear()
    second_texts, timings, _ = translator.translate_segments(
        changed_timing,
        max_workers=1,
        cache_path=str(cache_path),
        target_lang="简体中文",
        glossary="",
        character_reference="",
    )

    assert first_texts == ["再靠近点", "摸那里"]
    assert second_texts == first_texts
    assert calls == []
    assert any(item["mode"] == "translation_memory_hit" for item in timings)
    assert timings[-1]["cache_hit_count"] == 0
    assert timings[-1]["translation_memory_hit_count"] == 2


def test_single_request_translation_uses_text_memory_when_timing_changes(monkeypatch, tmp_path):
    cache_path = tmp_path / "translation_cache.jsonl"
    first_segments = [
        {"start": 0.0, "end": 1.0, "text": "もっと来て"},
        {"start": 1.0, "end": 2.0, "text": "そこ触って"},
    ]
    changed_timing = [
        {"start": 0.0, "end": 1.4, "text": "もっと来て"},
        {"start": 1.4, "end": 2.4, "text": "そこ触って"},
    ]
    calls: list[int] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        calls.append(expected_count)
        return json.dumps(
            {
                "translations": [
                    {"id": 0, "text": "再靠近点"},
                    {"id": 1, "text": "摸那里"},
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "_chat", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 10)

    first_texts, _, _ = translator.translate_segments(
        first_segments,
        max_workers=1,
        cache_path=str(cache_path),
        target_lang="简体中文",
        glossary="",
        character_reference="",
    )

    calls.clear()
    second_texts, timings, _ = translator.translate_segments(
        changed_timing,
        max_workers=1,
        cache_path=str(cache_path),
        target_lang="简体中文",
        glossary="",
        character_reference="",
    )

    assert first_texts == ["再靠近点", "摸那里"]
    assert second_texts == first_texts
    assert calls == []
    assert timings[0]["mode"] == "translation_memory_hit"
    assert timings[0]["translation_memory_hit_count"] == 2


def test_translation_memory_partial_hit_requests_only_missing_ids(monkeypatch, tmp_path):
    cache_path = tmp_path / "translation_cache.jsonl"
    cached_key = translator._translation_memory_key(
        "もっと来て",
        glossary="",
        target_lang="简体中文",
    )
    translator._save_memory_entries(
        cache_path,
        [(cached_key, "再靠近点")],
        threading.Lock(),
    )
    segments = [
        {"start": 0.0, "end": 1.0, "text": "もっと来て"},
        {"start": 1.0, "end": 2.0, "text": "そこ触って"},
        {"start": 2.0, "end": 3.0, "text": "気持ちいい"},
    ]
    calls: list[tuple[list[int], int]] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        match = re.search(r"requested_ids\s*=\s*(\[[^\]]*\])", messages[1]["content"])
        assert match is not None
        ids = json.loads(match.group(1))
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        calls.append((ids, expected_count))
        texts = {1: "摸那里", 2: "好爽"}
        return json.dumps(
            {"translations": [{"id": idx, "text": texts[idx]} for idx in ids]},
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "_chat", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)

    zh_texts, timings, _ = translator.translate_segments(
        segments,
        max_workers=1,
        cache_path=str(cache_path),
        target_lang="简体中文",
        glossary="",
        character_reference="",
    )

    assert zh_texts == ["再靠近点", "摸那里", "好爽"]
    assert calls == [([1], 1), ([2], 1)]
    batch_timing = next(item for item in timings if item["mode"] == "batched_full_context")
    assert batch_timing["requested_ids"] == [1]
    assert batch_timing["translation_memory_hit_count"] == 1
    assert batch_timing["cache_hit_type"] == "mixed"


def test_translation_memory_miss_when_glossary_changes(monkeypatch, tmp_path):
    cache_path = tmp_path / "translation_cache.jsonl"
    first_memory_key = translator._translation_memory_key(
        "もっと来て",
        glossary="健太-Kenta",
        target_lang="简体中文",
    )
    second_memory_key = translator._translation_memory_key(
        "そこ触って",
        glossary="健太-Kenta",
        target_lang="简体中文",
    )
    translator._save_memory_entries(
        cache_path,
        [(first_memory_key, "再靠近点"), (second_memory_key, "摸那里")],
        threading.Lock(),
    )
    calls: list[int] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        calls.append(expected_count)
        return _mock_json(_batch_start_from_messages(messages), expected_count)

    monkeypatch.setattr(translator, "_chat", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 1)

    zh_texts, timings, _ = translator.translate_segments(
        [
            {"start": 0.0, "end": 1.0, "text": "もっと来て"},
            {"start": 1.0, "end": 2.0, "text": "そこ触って"},
        ],
        max_workers=1,
        cache_path=str(cache_path),
        target_lang="简体中文",
        glossary="小那海-Aya",
        character_reference="",
    )

    assert zh_texts == ["zh-0", "zh-1"]
    assert calls == [1, 1]
    assert timings[-1]["translation_memory_hit_count"] == 0


def test_translation_cache_key_changes_with_character_reference():
    segments = _segments(2)

    first = translator._translation_cache_key(
        0,
        segments,
        glossary="",
        target_lang="简体中文",
        character_reference="Task Name A",
    )
    second = translator._translation_cache_key(
        0,
        segments,
        glossary="",
        target_lang="简体中文",
        character_reference="Task Name B",
    )

    assert first != second

