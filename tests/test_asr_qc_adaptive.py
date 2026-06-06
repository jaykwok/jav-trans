from __future__ import annotations

from asr.qc import collect_adaptive_precision_review, evaluate_asr_text_results_qc


def test_qc_policy_keeps_repetition_metadata():
    report = evaluate_asr_text_results_qc(
        [{"index": 1, "start": 0.0, "end": 3.0}],
        [{"text": "小那海あや。小那海あや。", "duration": 3.0}],
    )

    assert report["repetition_check"] == "on"
    assert report["repetition_threshold"] == 10


def test_generation_error_metadata_is_counted():
    report = evaluate_asr_text_results_qc(
        [{"index": 1, "start": 0.0, "end": 2.0}],
        [
            {
                "text": "",
                "duration": 2.0,
                "asr_generation": {
                    "backend": "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame",
                    "error_kind": "overflow",
                    "error_detail": "decoder_input_ids exceeds max_target_positions",
                },
            }
        ],
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
    )

    assert report["generation_error_count"] == 1
    assert report["timeout_count"] == 1
    assert report["quarantined_count"] == 1
    assert report["reject_count"] == 1


def test_adaptive_precision_review_marks_signal_reject(monkeypatch):
    chunks = [{"index": 1, "start": 0.0, "end": 2.0}]
    text_results = [
        {
            "text": "幻覚テキスト",
            "raw_text": "幻覚テキスト",
            "duration": 2.0,
            "avg_logprob": -1.5,
            "no_speech_prob": 0.1,
            "compression_ratio": 1.2,
            "log": [],
        }
    ]
    report = evaluate_asr_text_results_qc(
        chunks,
        text_results,
    )

    filtered, updated_report, log = collect_adaptive_precision_review(
        chunks,
        text_results,
        report,
    )

    assert filtered == text_results
    assert updated_report["review_uncertain_enabled"] is True
    assert updated_report["review_uncertain_count"] == 1
    assert updated_report["review_uncertain_items"][0]["text_preview"] == "幻覚テキスト"
    assert "ASR Adaptive Precision review chunk 1" in log[0]


def test_adaptive_precision_review_keeps_signal_reject_text(monkeypatch):
    chunks = [{"index": 1, "start": 0.0, "end": 2.0}]
    text_results = [
        {
            "text": "幻覚テキスト",
            "raw_text": "幻覚テキスト",
            "duration": 2.0,
            "avg_logprob": -1.5,
            "no_speech_prob": 0.1,
            "compression_ratio": 1.2,
            "log": [],
        }
    ]
    report = evaluate_asr_text_results_qc(
        chunks,
        text_results,
    )

    filtered, updated_report, log = collect_adaptive_precision_review(
        chunks,
        text_results,
        report,
    )

    assert filtered == text_results
    assert updated_report["review_uncertain_count"] == 1
    assert updated_report["review_uncertain_items"][0]["text_preview"] == "幻覚テキスト"
    assert updated_report["review_uncertain_items"][0]["original_text"] == "幻覚テキスト"
    assert "ASR Adaptive Precision review chunk 1" in log[0]


def test_adaptive_precision_filter_keeps_low_risk_low_logprob():
    chunks = [{"index": 1, "start": 0.0, "end": 10.0}]
    text_results = [
        {
            "text": "ふふ……そんなこと、恥ずかしいこと……",
            "raw_text": "ふふ……そんなこと、恥ずかしいこと……",
            "duration": 10.0,
            "avg_logprob": -0.84,
            "no_speech_prob": 0.00001,
            "compression_ratio": 0.8,
        }
    ]
    report = evaluate_asr_text_results_qc(
        chunks,
        text_results,
        is_low_value_text=lambda _text: False,
    )

    filtered, updated_report, log = collect_adaptive_precision_review(
        chunks,
        text_results,
        report,
    )

    assert report["precision_policy"] == "adaptive"
    assert report["reject_count"] == 0
    assert filtered == text_results
    assert updated_report["review_uncertain_count"] == 0
    assert log == []


def test_adaptive_precision_review_marks_hard_density_reject(monkeypatch):
    text = "もしも" * 120
    chunks = [{"index": 1, "start": 0.0, "end": 7.84}]
    text_results = [
        {
            "text": text,
            "raw_text": text,
            "duration": 7.84,
            "avg_logprob": -0.75,
            "no_speech_prob": 0.00001,
            "compression_ratio": 0.8,
        }
    ]
    report = evaluate_asr_text_results_qc(
        chunks,
        text_results,
    )

    filtered, updated_report, log = collect_adaptive_precision_review(
        chunks,
        text_results,
        report,
    )

    assert report["reject_count"] == 1
    assert "abnormal_char_density" in report["items"][0]["reasons"]
    assert filtered == text_results
    assert updated_report["review_uncertain_count"] == 1
    assert log
    assert "ASR Adaptive Precision review chunk 1" in log[0]

