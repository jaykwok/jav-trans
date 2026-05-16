"""AJ-8: _interruptible_sleep respects cancel_event."""
import threading
import time

from llm.translator import _interruptible_sleep


def test_interruptible_sleep_cancels_early():
    cancel = threading.Event()
    timer = threading.Timer(0.2, cancel.set)
    timer.start()
    start = time.perf_counter()
    _interruptible_sleep(5.0, cancel)
    elapsed = time.perf_counter() - start
    timer.cancel()
    assert elapsed < 0.8, f"Expected cancel within 0.8s, took {elapsed:.2f}s"


def test_interruptible_sleep_completes_without_cancel():
    cancel = threading.Event()
    start = time.perf_counter()
    _interruptible_sleep(0.3, cancel)
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.3, f"Should sleep at least 0.3s, got {elapsed:.2f}s"


def test_interruptible_sleep_no_cancel_event():
    start = time.perf_counter()
    _interruptible_sleep(0.2, None)
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.2
