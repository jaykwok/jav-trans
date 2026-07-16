from pathlib import Path

import pytest

from tools.boundary.ja import export_runtime_v10_provisional_subislands as exporter


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
