from whisper import pipeline


def test_chunk_config_reads_packing_env_at_runtime(monkeypatch):
    monkeypatch.setenv("ASR_CHUNK_PACKING_ENABLED", "0")
    assert pipeline._chunk_config()["packing_enabled"] is False

    monkeypatch.setenv("ASR_CHUNK_PACKING_ENABLED", "1")
    assert pipeline._chunk_config()["packing_enabled"] is True


def test_chunk_config_reads_drop_env_at_runtime(monkeypatch):
    monkeypatch.setenv("ASR_CHUNK_DROP_ENABLED", "0")
    assert pipeline._chunk_config()["drop_enabled"] is False

    monkeypatch.setenv("ASR_CHUNK_DROP_ENABLED", "1")
    monkeypatch.setenv("ASR_CHUNK_DROP_MIN_DURATION_S", "0.33")
    monkeypatch.setenv("ASR_CHUNK_DROP_RMS_DBFS", "-35.5")
    cfg = pipeline._chunk_config()

    assert cfg["drop_enabled"] is True
    assert cfg["drop_min_duration_s"] == 0.33
    assert cfg["drop_rms_dbfs"] == -35.5
