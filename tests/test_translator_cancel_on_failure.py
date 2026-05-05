import json
import time

import pytest

from llm import translator
def _segments(count: int) -> list[dict]:
    return [
        {"start": float(index), "end": float(index) + 1.0, "text": f"ja-{index}"}
        for index in range(count)
    ]


def _batch_start_from_messages(messages: list[dict]) -> int:
    ids = []
    for line in messages[1]["content"].splitlines():
        if '"id": ' in line:
            ids.append(int(line.split('"id": ')[1].split(",", 1)[0]))
    return min(ids)


def test_batched_translation_cancels_pending_futures_on_failure(monkeypatch):
    calls: list[int] = []

    def fake_chat(messages, expected_count=0, on_progress=None):
        start = _batch_start_from_messages(messages)
        calls.append(start)
        if start == 0:
            raise RuntimeError("boom")
        time.sleep(5.0)
        return json.dumps(
            {
                "translations": [
                    {"id": index, "text": f"zh-{index}"}
                    for index in range(start, start + expected_count)
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "_chat", fake_chat)

    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="boom"):
        translator.translate_segments(
            _segments(8),
            batch_size=2,
            max_workers=1,
            cache_path="",
            target_lang="简体中文",
            glossary="",
        )
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0
    assert calls == [0]

