import pytest

from asr import pipeline


def test_chunk_config_reads_drop_env_at_runtime(monkeypatch):
    monkeypatch.setenv("BOUNDARY_DROP_LOW_ENERGY_ENABLED", "0")
    assert pipeline._boundary_config()["drop_enabled"] is False

    monkeypatch.setenv("BOUNDARY_DROP_LOW_ENERGY_ENABLED", "1")
    monkeypatch.setenv("BOUNDARY_DROP_LOW_ENERGY_MIN_DURATION_S", "0.33")
    monkeypatch.setenv("BOUNDARY_DROP_LOW_ENERGY_RMS_DBFS", "-35.5")
    cfg = pipeline._boundary_config()

    assert cfg["drop_enabled"] is True
    assert cfg["drop_min_duration_s"] == 0.33
    assert cfg["drop_rms_dbfs"] == -35.5


def test_chunk_config_reads_boundary_refiner_env_at_runtime(monkeypatch):
    monkeypatch.setenv("BOUNDARY_REFINER_ENABLED", "1")
    monkeypatch.setenv("BOUNDARY_REFINER_MODEL_PATH", "models/boundary-refiner.pt")
    monkeypatch.setenv("BOUNDARY_REFINER_BACKBONE", "transformers.Mamba2Model")
    monkeypatch.setenv("BOUNDARY_REFINER_DEVICE", "cuda")
    monkeypatch.setenv("BOUNDARY_REFINER_THRESHOLD", "0.63")

    cfg = pipeline._boundary_config()

    assert cfg["boundary_refiner_enabled"] is True
    assert cfg["boundary_refiner_model_path"] == "models/boundary-refiner.pt"
    assert cfg["boundary_refiner_backbone"] == "transformers.Mamba2Model"
    assert cfg["boundary_refiner_device"] == "cuda"
    assert cfg["boundary_refiner_threshold"] == 0.63


def test_chunk_config_rejects_non_mamba2_boundary_refiner(monkeypatch):
    monkeypatch.setenv("BOUNDARY_REFINER_BACKBONE", "tcn")

    with pytest.raises(ValueError, match="only transformers\\.Mamba2Model"):
        pipeline._boundary_config()


def test_chunk_config_rejects_mamba2_alias(monkeypatch):
    monkeypatch.setenv("BOUNDARY_REFINER_BACKBONE", "mamba2")

    with pytest.raises(ValueError, match="only transformers\\.Mamba2Model"):
        pipeline._boundary_config()


def test_chunk_config_rejects_torch_mamba2_alias(monkeypatch):
    monkeypatch.setenv("BOUNDARY_REFINER_BACKBONE", "torch_mamba2")

    with pytest.raises(ValueError, match="only transformers\\.Mamba2Model"):
        pipeline._boundary_config()


def test_chunk_config_reads_boundary_planner_env_at_runtime(monkeypatch):
    monkeypatch.setenv("BOUNDARY_FEATURE_FRAME_HOP_S", "0.04")
    monkeypatch.setenv("BOUNDARY_PLANNER_TARGET_CHUNK_S", "8.0")
    monkeypatch.setenv("BOUNDARY_PLANNER_MAX_CHUNK_S", "28.0")
    monkeypatch.setenv("BOUNDARY_PLANNER_MIN_CHUNK_S", "0.5")
    monkeypatch.setenv("BOUNDARY_PLANNER_START_WEIGHT", "1.7")
    monkeypatch.setenv("BOUNDARY_PLANNER_TARGET_PADDING_S", "1.5")
    monkeypatch.setenv("BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT", "12")
    monkeypatch.setenv("BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE", "128")
    monkeypatch.setenv("BOUNDARY_DP_CHUNK_BASE_COST", "0.02")
    monkeypatch.setenv("BOUNDARY_DP_OVER_TARGET_WEIGHT", "0.4")
    monkeypatch.setenv("BOUNDARY_DP_FAR_OVER_TARGET_WEIGHT", "1.8")
    monkeypatch.setenv("BOUNDARY_DP_UNDER_MIN_WEIGHT", "0.1")
    monkeypatch.setenv("BOUNDARY_DP_LONG_GAP_WEIGHT", "0.7")
    monkeypatch.setenv("BOUNDARY_DP_SPLIT_MERGE_WEIGHT", "0.2")

    cfg = pipeline._boundary_config()

    assert cfg["feature_frame_hop_s"] == 0.04
    assert cfg["boundary_planner_target_chunk_s"] == 8.0
    assert cfg["boundary_planner_max_chunk_s"] == 28.0
    assert cfg["boundary_planner_min_chunk_s"] == 0.5
    assert cfg["boundary_planner_start_weight"] == 1.7
    assert cfg["boundary_planner_target_padding_s"] == 1.5
    assert cfg["boundary_planner_max_splits_per_segment"] == 12
    assert cfg["boundary_planner_sequence_batch_size"] == 128
    assert cfg["boundary_dp_chunk_base_cost"] == 0.02
    assert cfg["boundary_dp_over_target_weight"] == 0.4
    assert cfg["boundary_dp_far_over_target_weight"] == 1.8
    assert cfg["boundary_dp_under_min_weight"] == 0.1
    assert cfg["boundary_dp_long_gap_weight"] == 0.7
    assert cfg["boundary_dp_split_merge_weight"] == 0.2
