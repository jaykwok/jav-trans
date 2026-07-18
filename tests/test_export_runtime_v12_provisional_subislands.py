from pathlib import Path
from types import SimpleNamespace

import pytest

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.boundary.ja import export_runtime_v12_provisional_subislands as exporter


def test_resolve_audio_path_prefers_manifest_relative(tmp_path: Path, monkeypatch) -> None:
    items_dir = tmp_path / "items"
    items_dir.mkdir()
    audio = items_dir / "audio.wav"
    audio.write_bytes(b"manifest")
    project_audio = tmp_path / "project" / "audio.wav"
    project_audio.parent.mkdir()
    project_audio.write_bytes(b"project")
    monkeypatch.setattr(exporter, "PROJECT_ROOT", tmp_path / "project")

    resolved = exporter.resolve_audio_path(
        value="audio.wav", items_path=items_dir / "items.jsonl"
    )

    assert resolved == audio.resolve()


def test_resolve_audio_path_falls_back_to_project_root(
    tmp_path: Path, monkeypatch
) -> None:
    items_dir = tmp_path / "items"
    items_dir.mkdir()
    audio = tmp_path / "project" / "datasets" / "audio.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"project")
    monkeypatch.setattr(exporter, "PROJECT_ROOT", tmp_path / "project")

    resolved = exporter.resolve_audio_path(
        value="datasets/audio.wav", items_path=items_dir / "items.jsonl"
    )

    assert resolved == audio.resolve()


def test_resolve_audio_path_rejects_missing_source(
    tmp_path: Path, monkeypatch
) -> None:
    items_dir = tmp_path / "items"
    items_dir.mkdir()
    monkeypatch.setattr(exporter, "PROJECT_ROOT", tmp_path / "project")

    with pytest.raises(FileNotFoundError, match="source audio not found"):
        exporter.resolve_audio_path(
            value="datasets/missing.wav", items_path=items_dir / "items.jsonl"
        )


def test_binary_split_validation_accepts_outer_only_when_no_cut_event() -> None:
    chunk = SimpleNamespace(
        boundary_contract_id=ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        boundary_source="outer_edge_refiner_v3",
        boundary_decision_source="outer_edge_refiner_v3",
        semantic_event_ids=[],
        primary_cut_candidates=[],
        weak_cut_candidates=[
            {"label": "continue", "p_cut": 0.2, "p_continue": 0.8, "p_unsure": 0.0}
        ],
    )

    exporter.validate_binary_split_chunk(chunk, sample_id="source")


def test_binary_split_validation_rejects_three_class_candidate() -> None:
    chunk = SimpleNamespace(
        boundary_contract_id=ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        boundary_source="outer_edge_refiner_v3",
        boundary_decision_source="outer_edge_refiner_v3",
        semantic_event_ids=[],
        primary_cut_candidates=[],
        weak_cut_candidates=[
            {"label": "unsure", "p_cut": 0.2, "p_continue": 0.3, "p_unsure": 0.5}
        ],
    )

    with pytest.raises(ValueError, match="non-binary label"):
        exporter.validate_binary_split_chunk(chunk, sample_id="source")
