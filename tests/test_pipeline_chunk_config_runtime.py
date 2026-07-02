from __future__ import annotations

import importlib
from pathlib import Path

from helpers import ASR_17B_BACKEND


def _reload_pipeline(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", ASR_17B_BACKEND)
    from asr import pipeline

    return importlib.reload(pipeline)


def test_boundary_config_reads_all_three_model_mappings(monkeypatch, tmp_path: Path) -> None:
    outer = tmp_path / "outer.pt"
    split = tmp_path / "split.pt"
    cut = tmp_path / "cut.pt"
    for path in (outer, split, cut):
        path.write_bytes(b"checkpoint")
    monkeypatch.setenv(
        "OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        f"{ASR_17B_BACKEND}={outer}",
    )
    monkeypatch.setenv(
        "SEMANTIC_SPLIT_MODEL_PATH_BY_REPO",
        f"{ASR_17B_BACKEND}={split}",
    )
    monkeypatch.setenv(
        "CUT_EDGE_REFINER_MODEL_PATH_BY_REPO",
        f"{ASR_17B_BACKEND}={cut}",
    )
    monkeypatch.setenv("OUTER_EDGE_REFINER_DEVICE", "cuda")
    monkeypatch.setenv("SEMANTIC_SPLIT_DEVICE", "cpu")
    monkeypatch.setenv("CUT_EDGE_REFINER_DEVICE", "cuda:1")
    pipeline = _reload_pipeline(monkeypatch)

    cfg = pipeline._boundary_config()

    assert cfg["outer_edge_refiner_model_path"] == str(outer.resolve())
    assert cfg["semantic_split_model_path"] == str(split.resolve())
    assert cfg["cut_edge_refiner_model_path"] == str(cut.resolve())
    assert cfg["outer_edge_refiner_device"] == "cuda"
    assert cfg["semantic_split_device"] == "cpu"
    assert cfg["cut_edge_refiner_device"] == "cuda:1"


def test_boundary_config_has_no_retired_planner_or_single_refiner_surface(monkeypatch) -> None:
    cfg = _reload_pipeline(monkeypatch)._boundary_config()

    assert "boundary_refiner_model_path" not in cfg
    assert not any(key.startswith("boundary_planner_") for key in cfg)
