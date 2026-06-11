from asr import pipeline


def test_chunk_config_reads_boundary_refiner_env_at_runtime(monkeypatch):
    monkeypatch.setenv("BOUNDARY_REFINER_MODEL_PATH", "src/boundary/checkpoints/boundary_refiner.pt")
    monkeypatch.setenv("BOUNDARY_REFINER_DEVICE", "cuda")

    cfg = pipeline._boundary_config()

    assert cfg["boundary_refiner_model_path"] == "src/boundary/checkpoints/boundary_refiner.pt"
    assert cfg["boundary_refiner_device"] == "cuda"
    assert "boundary_refiner_backbone" not in cfg

def test_chunk_config_reads_boundary_planner_env_at_runtime(monkeypatch):
    monkeypatch.setenv("BOUNDARY_FEATURE_FRAME_HOP_S", "0.04")
    monkeypatch.setenv("BOUNDARY_PLANNER_TARGET_CHUNK_S", "8.0")
    monkeypatch.setenv("BOUNDARY_PLANNER_MAX_CORE_CHUNK_S", "5.5")
    monkeypatch.setenv("BOUNDARY_PLANNER_MIN_CHUNK_S", "0.5")
    monkeypatch.setenv("BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT", "12")
    monkeypatch.setenv("BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE", "128")

    cfg = pipeline._boundary_config()

    assert cfg["feature_frame_hop_s"] == 0.04
    assert cfg["boundary_planner_target_chunk_s"] == 8.0
    assert cfg["boundary_planner_max_core_chunk_s"] == 5.5
    assert cfg["boundary_planner_min_chunk_s"] == 0.5
    assert cfg["boundary_planner_max_splits_per_segment"] == 12
    assert cfg["boundary_planner_sequence_batch_size"] == 128
