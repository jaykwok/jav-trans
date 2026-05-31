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


def test_chunk_config_reads_pre_asr_island_split_env_at_runtime(monkeypatch):
    monkeypatch.setenv("ASR_PRE_ASR_ISLAND_SPLIT_ENABLED", "1")
    monkeypatch.setenv("ASR_PRE_ASR_ISLAND_SPLIT_MIN_CORE_FRAMES", "300")
    monkeypatch.setenv("ASR_PRE_ASR_ISLAND_SPLIT_MIN_GAP_FRAMES", "12")
    monkeypatch.setenv("ASR_PRE_ASR_ISLAND_SPLIT_MIN_ISLAND_FRAMES", "4")
    monkeypatch.setenv("ASR_PRE_ASR_ISLAND_SPLIT_MAX_CHILDREN", "5")

    cfg = pipeline._chunk_config()

    assert cfg["pre_asr_island_split_enabled"] is True
    assert cfg["pre_asr_island_split_min_core_frames"] == 300
    assert cfg["pre_asr_island_split_min_gap_frames"] == 12
    assert cfg["pre_asr_island_split_min_island_frames"] == 4
    assert cfg["pre_asr_island_split_max_children"] == 5


def test_chunk_config_reads_pre_asr_valley_split_env_at_runtime(monkeypatch):
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_ENABLED", "1")
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_MIN_CORE_FRAMES", "360")
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_TARGET_CORE_FRAMES", "240")
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_MIN_VALLEY_FRAMES", "8")
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_MIN_CHILD_FRAMES", "30")
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_MAX_CHILDREN", "6")
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_THRESHOLD", "0.15")

    cfg = pipeline._chunk_config()

    assert cfg["pre_asr_valley_split_enabled"] is True
    assert cfg["pre_asr_valley_split_min_core_frames"] == 360
    assert cfg["pre_asr_valley_split_target_core_frames"] == 240
    assert cfg["pre_asr_valley_split_min_valley_frames"] == 8
    assert cfg["pre_asr_valley_split_min_child_frames"] == 30
    assert cfg["pre_asr_valley_split_max_children"] == 6
    assert cfg["pre_asr_valley_split_threshold"] == 0.15
