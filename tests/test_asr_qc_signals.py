from __future__ import annotations

from pathlib import Path

from whisper.qc import check_logprob_quality, evaluate_asr_chunk_qc


def test_ok_when_all_signals_good():
    result = {"avg_logprob": -0.3, "no_speech_prob": 0.1, "compression_ratio": 1.2}
    qc = check_logprob_quality(result)
    assert qc["verdict"] == "ok"
    assert qc["reason"] is None


def test_reject_on_high_no_speech_prob(monkeypatch):
    monkeypatch.setenv("ASR_QC_ADAPTIVE_HARD_NOSPEECH_THRESHOLD", "0.6")
    result = {"avg_logprob": -0.3, "no_speech_prob": 0.8, "compression_ratio": 1.2}
    qc = check_logprob_quality(result)
    assert qc["verdict"] == "reject"
    assert "no_speech" in qc["reason"]


def test_adaptive_precision_rejects_very_low_avg_logprob(monkeypatch):
    monkeypatch.setenv("ASR_QC_ADAPTIVE_BASE_LOGPROB", "-1.0")
    result = {
        "avg_logprob": -1.5,
        "no_speech_prob": 0.1,
        "compression_ratio": 1.2,
        "duration_s": 2.0,
        "compact_chars": 6,
        "chars_per_sec": 3.0,
        "text": "本当だ",
    }
    qc = check_logprob_quality(result)
    assert qc["verdict"] == "reject"
    assert "logprob" in qc["reason"]


def test_adaptive_precision_keeps_low_risk_low_logprob_dialogue():
    result = {
        "avg_logprob": -0.84,
        "no_speech_prob": 0.00001,
        "compression_ratio": 0.8,
        "duration_s": 10.0,
        "compact_chars": 36,
        "chars_per_sec": 3.6,
        "text": "ふふ……そんなこと、恥ずかしいこと……",
        "max_repeat": {"run": 1, "ratio": 0.0},
    }

    qc = check_logprob_quality(result)

    assert qc["verdict"] == "ok"
    assert qc["metrics"]["precision_policy"] == "adaptive"
    assert qc["metrics"]["drop_uncertain_enabled"] is False
    assert qc["metrics"]["logprob_threshold"] < -0.84
    assert qc["metrics"]["adaptive"]["semantic_text_relax"] > 0


def test_adaptive_precision_drop_uncertain_is_opt_in(monkeypatch):
    monkeypatch.setenv("ASR_QC_DROP_UNCERTAIN", "1")
    qc = check_logprob_quality(
        {"avg_logprob": -0.3, "no_speech_prob": 0.1, "compression_ratio": 1.2}
    )
    assert qc["metrics"]["drop_uncertain_enabled"] is True


def test_adaptive_precision_hard_rejects_abnormal_density():
    result = {
        "avg_logprob": -0.75,
        "no_speech_prob": 0.00001,
        "compression_ratio": 0.8,
        "duration_s": 7.84,
        "compact_chars": 579,
        "chars_per_sec": 73.8,
        "text": "もしも" * 120,
        "max_repeat": {"run": 1, "ratio": 0.0},
    }

    qc = check_logprob_quality(result)

    assert qc["verdict"] == "reject"
    assert "abnormal_char_density" in qc["reason"]


def test_adaptive_density_uses_adaptive_threshold_env(monkeypatch):
    monkeypatch.setenv("ASR_QC_ADAPTIVE_HARD_MAX_CHARS_PER_SEC", "80.0")
    result = {
        "avg_logprob": -0.3,
        "no_speech_prob": 0.00001,
        "compression_ratio": 0.8,
        "duration_s": 7.84,
        "compact_chars": 579,
        "chars_per_sec": 73.8,
        "text": "もしも" * 120,
        "max_repeat": {"run": 1, "ratio": 0.0},
    }

    qc = check_logprob_quality(result)

    assert qc["verdict"] == "ok"


def test_legacy_density_threshold_removed_from_source():
    root = Path(__file__).resolve().parents[1]
    checked = [
        root / "src/whisper/qc.py",
        root / "src/core/config.py",
        root / ".env.example",
    ]

    assert all("ASR_QC_MAX_CHARS_PER_SEC" not in path.read_text() for path in checked)


def test_reject_on_high_compression_ratio(monkeypatch):
    monkeypatch.setenv("ASR_QC_ADAPTIVE_HARD_COMPRESSION_THRESHOLD", "2.4")
    result = {"avg_logprob": -0.5, "no_speech_prob": 0.1, "compression_ratio": 2.8}
    qc = check_logprob_quality(result)
    assert qc["verdict"] == "reject"
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
    # no_speech_prob is None but avg_logprob is bad.
    result = {"avg_logprob": -2.0, "no_speech_prob": None, "compression_ratio": None}
    qc = check_logprob_quality(result)
    assert qc["verdict"] == "reject"
    assert "logprob" in qc["reason"]


def test_chunk_qc_exports_repetition_repair_suggestion():
    qc = evaluate_asr_chunk_qc(
        {"index": 0, "start": 0.0, "end": 6.0, "duration": 6.0},
        {
            "text": "あっ、あっ、あっ、あっ、あっ、あっ、あっ、あっ、",
            "raw_text": "あっ、あっ、あっ、あっ、あっ、あっ、あっ、あっ、",
            "duration": 6.0,
        },
    )

    assert qc["severity"] == "reject"
    assert "repeat_ngram_loop" in qc["reasons"]
    repair = qc["metrics"]["repetition_repair"]
    assert repair["action"] == "truncate_repetition"
    assert repair["changed"] is True
    assert repair["run"] >= 4
    assert len(repair["suggested_text"]) < len(qc["text_preview"])


def test_chunk_qc_exports_low_information_profile_without_dropping():
    qc = evaluate_asr_chunk_qc(
        {"index": 0, "start": 0.0, "end": 8.0, "duration": 8.0},
        {
            "text": "んー",
            "raw_text": "んー",
            "duration": 8.0,
        },
    )

    assert qc["metrics"]["low_information"]["level"] == "long_sparse"
    assert qc["metrics"]["low_information"]["action"] == "preserve_with_review"
    assert qc["ok"] is True
