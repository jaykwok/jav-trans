import importlib
from pathlib import Path

import pytest

from helpers import ASR_06B_BACKEND, ASR_17B_BACKEND


def _reload_pipeline(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", ASR_06B_BACKEND)
    from asr import pipeline

    return importlib.reload(pipeline)


def test_chunk_config_reads_boundary_refiner_env_at_runtime(monkeypatch, tmp_path):
    checkpoint = tmp_path / "boundary_edge_refiner_v8_safe_tight.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf.pt"
    checkpoint.write_bytes(b"v8")
    monkeypatch.setenv(
        "BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        f"{ASR_06B_BACKEND}={checkpoint}",
    )
    monkeypatch.setenv("BOUNDARY_REFINER_DEVICE", "cuda")
    pipeline = _reload_pipeline(monkeypatch)
    monkeypatch.setattr(pipeline, "_boundary_refiner_runtime_adapter", lambda _path: "edge_sequence_v2")

    cfg = pipeline._boundary_config()

    assert cfg["boundary_refiner_model_path"] == str(
        checkpoint.resolve()
    )
    assert cfg["boundary_refiner_device"] == "cuda"
    assert "boundary_refiner_backbone" not in cfg


def test_chunk_config_reads_boundary_planner_env_at_runtime(monkeypatch, tmp_path):
    checkpoint = tmp_path / "boundary_edge_refiner_v8_safe_tight.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf.pt"
    checkpoint.write_bytes(b"v8")
    monkeypatch.setenv(
        "BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        f"{ASR_06B_BACKEND}={checkpoint}",
    )
    monkeypatch.setenv("BOUNDARY_FEATURE_FRAME_HOP_S", "0.04")
    monkeypatch.setenv("BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE", "128")
    pipeline = _reload_pipeline(monkeypatch)
    monkeypatch.setattr(pipeline, "_boundary_refiner_runtime_adapter", lambda _path: "edge_sequence_v2")

    cfg = pipeline._boundary_config()

    assert cfg["feature_frame_hop_s"] == 0.04
    assert "boundary_planner_target_chunk_s" not in cfg
    assert "boundary_planner_max_core_chunk_s" not in cfg
    assert "boundary_planner_min_chunk_s" not in cfg
    assert "boundary_planner_max_splits_per_segment" not in cfg
    assert cfg["boundary_planner_sequence_batch_size"] == 128


def test_chunk_config_rejects_boundary_refiner_repo_metadata_mismatch(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "boundary_edge_refiner_v8_safe_tight.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf.pt"
    torch.save(
        {
            "metadata": {
                "ptm_repo_id": ASR_06B_BACKEND,
                "runtime_adapter": "edge_sequence_v2",
            }
        },
        checkpoint,
    )
    monkeypatch.setenv("ASR_BACKEND", ASR_17B_BACKEND)
    monkeypatch.setenv(
        "BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        f"{ASR_17B_BACKEND}={checkpoint}",
    )
    from asr import pipeline

    pipeline = importlib.reload(pipeline)

    with pytest.raises(ValueError, match="does not match selected repo"):
        pipeline._boundary_config()
