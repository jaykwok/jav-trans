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
    cache_path = tmp_path / "translation_cache.json"
    migrated_path = cache_path.with_suffix(".jsonl")
    segments = _segments(4)
    cache_key = translator._translation_cache_key(
        0,
        segments[:2],
        glossary="",
        target_lang="简体中文",
    )
    cache_path.write_text(
        json.dumps({cache_key: ["cached-0", "cached-1"]}, ensure_ascii=False),
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

    zh_texts, timings, retry_events = translator.translate_segments(
        segments,
        batch_size=2,
        max_workers=2,
        cache_path=str(cache_path),
        target_lang="简体中文",
        glossary="",
    )

    assert calls == [(2, 2)]
    assert retry_events == []
    assert zh_texts == ["cached-0", "cached-1", "zh-2", "zh-3"]
    assert any(item["mode"] == "translation_cache_hit" for item in timings)
    assert any(item["mode"] == "batched_full_context" for item in timings)
    assert timings[-1]["cache_hit_count"] == 1

    saved = _read_cache_jsonl(migrated_path)
    assert saved[cache_key] == ["cached-0", "cached-1"]
    assert saved[
        translator._translation_cache_key(
            1,
            segments[2:],
            glossary="",
            target_lang="简体中文",
        )
    ] == ["zh-2", "zh-3"]
    assert migrated_path.exists()
    assert not cache_path.exists()


def test_load_translation_cache_warns_on_corrupt_jsonl(tmp_path, capsys):
    cache_path = tmp_path / "translation_cache.jsonl"
    cache_path.write_text("{not-json}\n", encoding="utf-8")

    assert translator._load_translation_cache(cache_path) == {}

    captured = capsys.readouterr()
    assert "[WARN] translation cache JSONL load failed" in captured.out
    assert str(cache_path) in captured.out


def test_load_translation_cache_warns_on_corrupt_legacy_json(tmp_path, capsys):
    cache_path = tmp_path / "translation_cache.json"
    cache_path.write_text("{not-json}", encoding="utf-8")

    assert translator._load_translation_cache(cache_path) == {}

    captured = capsys.readouterr()
    assert "[WARN] translation cache JSON load failed" in captured.out
    assert str(cache_path) in captured.out


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

