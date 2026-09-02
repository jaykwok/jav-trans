from __future__ import annotations

import json
import re
import threading

from core import events
from llm import translator


def _segments(count: int) -> list[dict]:
    return [
        {"start": float(index), "end": float(index) + 1.0, "text": f"ja-{index}"}
        for index in range(count)
    ]


def _requested_ids_from_messages(messages) -> list[int]:
    content = messages[1]["content"]
    match = re.search(r"requested_ids\s*=\s*(\[[^\]]*\])", content)
    assert match is not None, content
    return json.loads(match.group(1))


def test_worker_threads_see_the_job_id_the_caller_had(monkeypatch):
    """run_batched dispatches _engine_chat from a ThreadPoolExecutor pool. A
    fresh worker thread has no `core.events` thread-local of its own, so a
    model_download event fired from inside chat (e.g. the llamacpp backend's
    first-call GGUF download) would otherwise carry an empty job_id -- which
    the frontend silently drops instead of showing a progress bar."""
    seen_job_ids: list[str] = []
    lock = threading.Lock()

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        with lock:
            seen_job_ids.append(events._current_job_id())
        requested_ids = _requested_ids_from_messages(messages)
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        start = min(requested_ids)
        items = [
            {"id": index, "text": f"zh-{index}"}
            for index in range(start, start + expected_count)
        ]
        return json.dumps({"translations": items}, ensure_ascii=False)

    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)

    events.set_current_job_id("job-caller")
    try:
        translator.translate_segments(
            _segments(8),
            max_workers=4,
            cache_path="",
            target_lang="简体中文",
            glossary="",
        )
    finally:
        events.set_current_job_id("")

    assert seen_job_ids, "fake_chat was never called"
    assert set(seen_job_ids) == {"job-caller"}


def test_worker_threads_do_not_leak_a_stale_job_id_across_calls(monkeypatch):
    """A pool worker thread is reused across many jobs over the process
    lifetime; the propagation must overwrite on every call, not just seed an
    empty thread-local once."""
    seen_job_ids: list[str] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        seen_job_ids.append(events._current_job_id())
        requested_ids = _requested_ids_from_messages(messages)
        items = [{"id": index, "text": f"zh-{index}"} for index in requested_ids]
        return json.dumps({"translations": items}, ensure_ascii=False)

    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)

    for job_id in ("job-a", "job-b"):
        events.set_current_job_id(job_id)
        try:
            translator.translate_segments(
                _segments(2),
                max_workers=1,
                cache_path="",
                target_lang="简体中文",
                glossary="",
            )
        finally:
            events.set_current_job_id("")

    assert seen_job_ids == ["job-a", "job-b"]
