from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from helpers import ASR_06B_BACKEND, ASR_17B_BACKEND


def _reload_pipeline(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", ASR_17B_BACKEND)
    from asr import pipeline

    return importlib.reload(pipeline)


def test_17b_boundary_config_fails_until_outer_v3_audit(monkeypatch) -> None:
    pipeline = _reload_pipeline(monkeypatch)
    with pytest.raises(RuntimeError, match="pending_outer_v3_audit"):
        pipeline._boundary_config()


def test_06b_boundary_config_fails_as_pending_binary_retrain(monkeypatch) -> None:
    monkeypatch.setenv("ASR_BACKEND", ASR_06B_BACKEND)
    from asr import pipeline

    pipeline = importlib.reload(pipeline)
    with pytest.raises(RuntimeError, match="pending_binary_retrain"):
        pipeline._boundary_config()


def test_boundary_config_has_no_retired_planner_or_single_refiner_surface(monkeypatch) -> None:
    pipeline = _reload_pipeline(monkeypatch)
    assert not hasattr(pipeline, "_boundary_refiner_checkpoint_path")
