import threading
import time

import pytest

from llm import translator


def test_translate_segments_cancelled_before_work_returns_fast():
    cancel_event = threading.Event()
    cancel_event.set()

    started = time.perf_counter()
    with pytest.raises(translator.TranslationCancelledError):
        translator.translate_segments(
            [{"start": 0.0, "end": 1.0, "text": "いい"}],
            max_workers=1,
            cache_path="",
            target_lang="简体中文",
            glossary="",
            cancel_event=cancel_event,
        )

    assert time.perf_counter() - started < 1.0


def test_batched_translation_cancel_does_not_wait_for_running_api_calls(monkeypatch):
    cancel_event = threading.Event()
    api_started = threading.Event()
    release_api = threading.Event()
    finished = threading.Event()
    outcome: list[BaseException] = []

    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 1)
    monkeypatch.setattr(translator, "TRANSLATION_PREFIX_WARMUP", False)

    def blocking_chat(*_args, **_kwargs):
        api_started.set()
        release_api.wait(timeout=5.0)
        return '[{"id": 0, "translation": "x"}]'

    monkeypatch.setattr(translator, "_chat_with_reasoning", blocking_chat)

    def run_translation() -> None:
        try:
            translator.translate_segments(
                [
                    {"start": 0.0, "end": 1.0, "text": "一"},
                    {"start": 1.0, "end": 2.0, "text": "二"},
                ],
                max_workers=2,
                cache_path="",
                target_lang="简体中文",
                glossary="",
                cancel_event=cancel_event,
            )
        except BaseException as exc:
            outcome.append(exc)
        finally:
            finished.set()

    caller = threading.Thread(target=run_translation)
    caller.start()
    assert api_started.wait(timeout=2.0)
    cancel_event.set()
    assert finished.wait(timeout=1.0)
    assert len(outcome) == 1
    assert isinstance(outcome[0], translator.TranslationCancelledError)

    release_api.set()
    caller.join(timeout=2.0)
