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
            batch_size=100,
            max_workers=1,
            cache_path="",
            target_lang="简体中文",
            glossary="",
            cancel_event=cancel_event,
        )

    assert time.perf_counter() - started < 1.0
