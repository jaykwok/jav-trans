from __future__ import annotations

from types import SimpleNamespace

import pytest

from whisper.qc import evaluate_asr_text_results_qc


def test_context_leak_skipped_when_backend_ignores_contexts():
    def forbidden_context_check(_text: str) -> bool:
        raise AssertionError("context leak callback should not be called")

    report = evaluate_asr_text_results_qc(
        [{"index": 1, "start": 0.0, "end": 3.0}],
        [{"text": "小那海あや。小那海あや。", "duration": 3.0}],
        is_context_leak=forbidden_context_check,
        backend=SimpleNamespace(accepts_contexts=False),
    )

    assert report["context_leak_check"] == "skipped(backend_ignores_contexts)"
    assert report["repetition_check"] == "on"
    assert report["repetition_threshold"] == 10


def test_context_leak_runs_when_backend_accepts_contexts():
    calls: list[str] = []

    def context_check(text: str) -> bool:
        calls.append(text)
        return True

    report = evaluate_asr_text_results_qc(
        [{"index": 1, "start": 0.0, "end": 3.0}],
        [{"text": "context leak text", "duration": 3.0}],
        is_context_leak=context_check,
        backend=SimpleNamespace(accepts_contexts=True),
    )

    assert calls == ["context leak text"]
    assert report["items"][0]["metrics"]["context_leak"] is True


def test_context_leak_callback_failure_would_surface_without_skip():
    def forbidden_context_check(_text: str) -> bool:
        raise RuntimeError("called")

    with pytest.raises(RuntimeError, match="called"):
        evaluate_asr_text_results_qc(
            [{"index": 1, "start": 0.0, "end": 3.0}],
            [{"text": "context leak text", "duration": 3.0}],
            is_context_leak=forbidden_context_check,
            backend=SimpleNamespace(accepts_contexts=True),
        )


