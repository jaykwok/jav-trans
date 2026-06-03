from asr.qc import asr_qc_drop_uncertain_enabled, asr_qc_enabled


def test_asr_qc_enabled_reads_env_at_runtime(monkeypatch):
    monkeypatch.setenv("ASR_QC_ENABLED", "1")
    assert asr_qc_enabled() is True

    monkeypatch.setenv("ASR_QC_ENABLED", "0")
    assert asr_qc_enabled() is False


def test_asr_qc_drop_uncertain_is_opt_in(monkeypatch):
    monkeypatch.delenv("ASR_QC_DROP_UNCERTAIN", raising=False)
    assert asr_qc_drop_uncertain_enabled() is False

    monkeypatch.setenv("ASR_QC_DROP_UNCERTAIN", "1")
    assert asr_qc_drop_uncertain_enabled() is True
