from __future__ import annotations

import pytest

from whisper.qc import check_logprob_quality


def test_ok_when_all_signals_good():
    result = {"avg_logprob": -0.3, "no_speech_prob": 0.1, "compression_ratio": 1.2}
    qc = check_logprob_quality(result)
    assert qc["verdict"] == "ok"
    assert qc["reason"] is None


def test_reject_on_high_no_speech_prob(monkeypatch):
    monkeypatch.setenv("ASR_QC_NOSPEECH_THRESHOLD", "0.6")
    result = {"avg_logprob": -0.3, "no_speech_prob": 0.8, "compression_ratio": 1.2}
    qc = check_logprob_quality(result)
    assert qc["verdict"] == "reject"
    assert "no_speech" in qc["reason"]


def test_warn_on_low_avg_logprob(monkeypatch):
    monkeypatch.setenv("ASR_QC_LOGPROB_THRESHOLD", "-1.0")
    # compression_ratio within threshold so only logprob fires
    result = {"avg_logprob": -1.5, "no_speech_prob": 0.1, "compression_ratio": 1.2}
    qc = check_logprob_quality(result)
    assert qc["verdict"] == "warn"
    assert "logprob" in qc["reason"]


def test_reject_on_high_compression_ratio(monkeypatch):
    monkeypatch.setenv("ASR_QC_COMPRESSION_THRESHOLD", "2.4")
    result = {"avg_logprob": -0.5, "no_speech_prob": 0.1, "compression_ratio": 2.8}
    qc = check_logprob_quality(result)
    # high compression may be warn or reject depending on impl; either is non-ok
    assert qc["verdict"] in {"warn", "reject"}
    assert "compression" in qc["reason"]


def test_none_signals_all_skipped():
    result = {"avg_logprob": None, "no_speech_prob": None, "compression_ratio": None}
    qc = check_logprob_quality(result)
    assert qc["verdict"] == "ok"
    assert qc["reason"] is None


def test_reject_takes_priority_over_warn():
    result = {"avg_logprob": -1.5, "no_speech_prob": 0.9, "compression_ratio": 3.0}
    qc = check_logprob_quality(result)
    assert qc["verdict"] == "reject"
    assert "no_speech" in qc["reason"]


def test_partial_none_signals_checked():
    # no_speech_prob is None but avg_logprob is bad → warn
    result = {"avg_logprob": -2.0, "no_speech_prob": None, "compression_ratio": None}
    qc = check_logprob_quality(result)
    assert qc["verdict"] == "warn"
    assert "logprob" in qc["reason"]
