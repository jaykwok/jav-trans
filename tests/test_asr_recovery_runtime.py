from whisper.qc import asr_qc_enabled, asr_recovery_enabled


def test_asr_recovery_enabled_reads_env_at_runtime(monkeypatch):
    monkeypatch.setenv("ASR_RECOVERY_ENABLED", "1")
    assert asr_recovery_enabled() is True

    monkeypatch.setenv("ASR_RECOVERY_ENABLED", "0")
    assert asr_recovery_enabled() is False


def test_asr_qc_enabled_reads_env_at_runtime(monkeypatch):
    monkeypatch.setenv("ASR_QC_ENABLED", "1")
    assert asr_qc_enabled() is True

    monkeypatch.setenv("ASR_QC_ENABLED", "0")
    assert asr_qc_enabled() is False
