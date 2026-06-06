from asr.qc import asr_qc_enabled, check_logprob_quality


def test_asr_qc_enabled_reads_env_at_runtime(monkeypatch):
    monkeypatch.setenv("ASR_QC_ENABLED", "1")
    assert asr_qc_enabled() is True

    monkeypatch.setenv("ASR_QC_ENABLED", "0")
    assert asr_qc_enabled() is False


def test_asr_qc_uncertain_text_is_review_only():
    qc = check_logprob_quality(
        {"avg_logprob": -0.3, "no_speech_prob": 0.1, "compression_ratio": 1.2}
    )
    assert qc["metrics"]["review_uncertain_enabled"] is True
