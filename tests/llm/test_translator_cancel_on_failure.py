import json
import re
import threading
import time

import pytest

from llm import engine as engine_module, translator
from llm.errors import TranslationCancelledError
def _segments(count: int) -> list[dict]:
    return [
        {"start": float(index), "end": float(index) + 1.0, "text": f"ja-{index}"}
        for index in range(count)
    ]


def _batch_start_from_messages(messages: list[dict]) -> int:
    match = re.search(r"requested_ids\s*=\s*(\[[^\]]*\])", messages[1]["content"])
    assert match is not None, messages[1]["content"]
    ids = json.loads(match.group(1))
    return min(ids) if ids else 0


def test_batched_translation_cancels_pending_futures_on_failure(monkeypatch):
    calls: list[int] = []

    def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
        start = _batch_start_from_messages(messages)
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
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
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)

    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="boom"):
        translator.translate_segments(
            _segments(8),
            max_workers=1,
            cache_path="",
            target_lang="简体中文",
            glossary="",
        )
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0
    assert calls == [0]


def test_batched_translation_stops_in_flight_batches_on_failure(monkeypatch):
    # Cancelling a future only drops it while queued and `shutdown(wait=False)`
    # never interrupts a worker mid-request, so a batch already streaming used
    # to run to completion against a paid API after the job it belonged to had
    # been reported failed. The workers now share an abort flag, so a sibling
    # sees the failure at its next cancel check.
    started = threading.Event()
    observed_abort = threading.Event()
    ran_to_completion = threading.Event()

    def fake_chat(messages, expected_count=0, on_progress=None, cancel_event=None, **_kwargs):
        start = _batch_start_from_messages(messages)
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        if start == 0:
            # Let the sibling get well inside its request before failing.
            started.wait(2.0)
            raise RuntimeError("boom")
        started.set()
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                observed_abort.set()
                raise TranslationCancelledError("aborted")
            time.sleep(0.01)
        ran_to_completion.set()
        raise AssertionError("in-flight batch was never told to stop")

    monkeypatch.setattr(translator, "_chat", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)

    with pytest.raises(RuntimeError, match="boom"):
        translator.translate_segments(
            _segments(4),
            max_workers=2,
            cache_path="",
            target_lang="简体中文",
            glossary="",
        )

    assert observed_abort.wait(2.0), "sibling batch never saw the abort flag"
    assert not ran_to_completion.is_set()


def test_the_failure_survives_a_lower_numbered_sibling_standing_down(monkeypatch):
    # Batches are harvested in index order, so when batch 1 is the one that
    # fails, batch 0's abort is the first exception the main thread sees. It
    # used to be the one that propagated, which reported a cancellation for a
    # failure nobody could then read - and `cancelled` is not even the status a
    # failed job should land in.
    aborted = threading.Event()
    sibling_running = threading.Event()

    # Both futures have to land in the *same* `done` set for the ordering to
    # bite - otherwise the main thread wakes on the failure alone and never sees
    # the cancellation at all. Left to chance that happens sometimes; holding the
    # main thread off for a moment makes it happen every time.
    real_wait = engine_module.wait

    def unhurried_wait(*args, **kwargs):
        time.sleep(0.25)
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(engine_module, "wait", unhurried_wait)

    def fake_chat(messages, expected_count=0, on_progress=None, cancel_event=None, **_kwargs):
        start = _batch_start_from_messages(messages)
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        if start != 0:
            # Fail only once batch 0 is genuinely in flight; a batch that never
            # started raises nothing and would not exercise the ordering.
            sibling_running.wait(2.0)
            raise RuntimeError("the-real-root-cause")
        sibling_running.set()
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                aborted.set()
                raise translator.TranslationCancelledError("aborted")
            time.sleep(0.01)
        raise AssertionError("sibling was never told to stop")

    monkeypatch.setattr(translator, "_chat", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)

    with pytest.raises(RuntimeError, match="the-real-root-cause"):
        translator.translate_segments(
            _segments(4),
            max_workers=2,
            cache_path="",
            target_lang="简体中文",
            glossary="",
        )

    assert aborted.is_set(), "batch 0 should have stood down, not finished"


def test_batches_already_paid_for_are_committed_by_their_own_worker(
    monkeypatch, tmp_path
):
    # The cache write used to happen on the main thread after a future was
    # harvested, so a failure sorting first discarded every sibling that had
    # already come back - work that was generated, billed, and then bought again
    # by the retry. Sweeping the finished futures before unwinding only narrows
    # that window; a batch can always finish just after the sweep. The worker
    # commits its own batch instead, so a reply is durable from the moment it
    # lands, even with the abort already set.
    #
    # The commit is not ordered against the caller's unwind, and deliberately
    # so: the failure path does not join the pool, because a sibling blocked on
    # a socket read cannot be made to stand down promptly. So the entry can
    # appear a moment after `translate_segments` raises - what matters is that
    # it appears at all, which the harvest-side design could never manage.
    saw_abort = threading.Event()

    def fake_chat(messages, expected_count=0, on_progress=None, cancel_event=None, **_kwargs):
        ids = json.loads(
            re.search(r"requested_ids\s*=\s*(\[[^\]]*\])", messages[1]["content"]).group(1)
        )
        if expected_count == 0:
            return json.dumps({"translations": []}, ensure_ascii=False)
        if min(ids) != 0:
            raise RuntimeError("boom")
        # Batch 0's reply lands *after* the abort is already up, which is the
        # ordering that used to throw it away: the check that used to sit
        # between this return and the parse stood the batch down and dropped a
        # reply the run had already been billed for.
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                saw_abort.set()
                break
            time.sleep(0.01)
        return json.dumps(
            {"translations": [{"id": index, "text": f"zh-{index}"} for index in ids]},
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "_chat", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 2)

    cache_path = tmp_path / "translation_cache.jsonl"
    with pytest.raises(RuntimeError, match="boom"):
        translator.translate_segments(
            _segments(4),
            max_workers=2,
            cache_path=str(cache_path),
            target_lang="简体中文",
            glossary="",
        )

    deadline = time.perf_counter() + 5.0
    written = ""
    while time.perf_counter() < deadline:
        written = cache_path.read_text(encoding="utf-8") if cache_path.exists() else ""
        if "zh-0" in written:
            break
        time.sleep(0.01)

    assert saw_abort.is_set(), "batch 0 replied before the abort - nothing was tested"
    assert "zh-0" in written and "zh-1" in written

