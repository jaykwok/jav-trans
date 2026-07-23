from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tools.asr.cueqc import export_semantic_boundary_candidates as exporter


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = {
        "audio": [str(tmp_path / "audio.wav")],
        "output": str(tmp_path / "candidates.jsonl"),
        "boundary_audit_output": None,
        "split_feature_output": None,
        "speech_feature_output": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_split_feature_export_refuses_missing_runtime_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(exporter, "_build_processing_spans", lambda _audio: [])
    split_output = tmp_path / "split.npz"
    split_output.write_bytes(b"previous-valid-output")

    with pytest.raises(RuntimeError, match="training exporter is pending"):
        exporter.run(
            _args(tmp_path, split_feature_output=str(split_output))
        )

    assert split_output.read_bytes() == b"previous-valid-output"
    assert not list(tmp_path.glob(".split.*.npz"))


def test_source_bound_feature_outputs_reject_multiple_audio(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one --audio"):
        exporter.run(
            _args(
                tmp_path,
                audio=["a.wav", "b.wav"],
                speech_feature_output=str(tmp_path / "speech.npz"),
            )
        )
