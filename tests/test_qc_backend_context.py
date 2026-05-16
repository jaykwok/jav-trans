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


def test_generation_error_metadata_is_counted():
    report = evaluate_asr_text_results_qc(
        [{"index": 1, "start": 0.0, "end": 2.0}],
        [
            {
                "text": "",
                "duration": 2.0,
                "asr_generation": {
                    "backend": "anime-whisper",
                    "error_kind": "overflow",
                    "error_detail": "decoder_input_ids exceeds max_target_positions",
                },
            }
        ],
        backend=SimpleNamespace(accepts_contexts=True),
    )

    assert report["generation_error_count"] == 1
    assert report["generation_overflow_count"] == 1
    assert report["timeout_count"] == 0
    assert report["items"][0]["reasons"][0] == "generation_overflow"
    assert report["items"][0]["metrics"]["generation"]["error_kind"] == "overflow"


def test_generation_counters_present_when_qc_disabled(monkeypatch):
    monkeypatch.setenv("ASR_QC_ENABLED", "0")

    report = evaluate_asr_text_results_qc(
        [{"index": 1, "start": 0.0, "end": 2.0}],
        [{"text": "ok", "duration": 2.0}],
        backend=SimpleNamespace(accepts_contexts=True),
    )

    assert report["enabled"] is False
    assert report["generation_error_count"] == 0
    assert report["generation_overflow_count"] == 0
    assert report["timeout_count"] == 0
    assert report["quarantined_count"] == 0


def test_quarantined_timeout_counts_as_timeout_and_quarantine():
    report = evaluate_asr_text_results_qc(
        [{"index": 1, "start": 0.0, "end": 2.0}],
        [
            {
                "text": "",
                "duration": 2.0,
                "asr_generation": {
                    "error_kind": "timeout",
                    "failure_kind": "timeout",
                    "error_detail": "circuit breaker",
                },
            }
        ],
        backend=SimpleNamespace(accepts_contexts=True),
    )

    assert report["generation_error_count"] == 1
    assert report["timeout_count"] == 1
    assert report["quarantined_count"] == 1
    assert report["recoverable_count"] == 1


