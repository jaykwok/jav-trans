from __future__ import annotations

import json
from pathlib import Path

from tools.datasets.reexport_joint_pre_asr_chunks import (
    collect_source_windows,
    reexport_windows,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _window(window_id: str, wav: Path, *, marker: str) -> dict:
    return {
        "schema": "joint_boundary_omni_source_window_v1",
        "window_id": window_id,
        "video_id": window_id.rsplit("-w", 1)[0],
        "audio_wav": str(wav),
        "duration_s": 75.0,
        "marker": marker,
    }


def test_collect_source_windows_later_dataset_wins(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    d1 = tmp_path / "v1"
    d2 = tmp_path / "v2"
    _write_jsonl(
        d1 / "source_windows.jsonl",
        [_window("a-w00", wav, marker="old"), _window("b-w00", wav, marker="old")],
    )
    _write_jsonl(
        d2 / "source_windows.jsonl",
        [_window("a-w00", wav, marker="new")],
    )

    rows, summary = collect_source_windows([d1, d2])

    by_id = {row["window_id"]: row for row in rows}
    assert len(rows) == 2
    assert by_id["a-w00"]["marker"] == "new"
    assert by_id["a-w00"]["source_dataset_name"] == "v2"
    assert by_id["b-w00"]["marker"] == "old"
    assert summary["input_window_count"] == 3
    assert summary["deduped_window_count"] == 2
    assert summary["overwritten_duplicate_window_count"] == 1


def test_reexport_windows_writes_updated_manifest(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    rows = [_window("a-w00", wav, marker="old")]

    def fake_exporter(*, wav_path: Path, feature_dir: Path) -> dict:
        feature_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "semantic_split_features.npz",
            "semantic_split_features.jsonl",
            "speech_sequence_features.npz",
            "pre_asr_candidates.jsonl",
            "boundary_audit.jsonl",
        ):
            (feature_dir / name).write_text("", encoding="utf-8")
        assert wav_path == wav.resolve()
        return {
            "semantic_split_features": str(feature_dir / "semantic_split_features.npz"),
            "semantic_split_metadata": str(feature_dir / "semantic_split_features.jsonl"),
            "speech_sequence_features": str(feature_dir / "speech_sequence_features.npz"),
            "pre_asr_candidates": str(feature_dir / "pre_asr_candidates.jsonl"),
            "boundary_audit": str(feature_dir / "boundary_audit.jsonl"),
            "span_count": 2,
            "candidate_count": 3,
            "resumed": False,
        }

    output_rows, summary = reexport_windows(
        rows,
        output_dir=tmp_path / "out",
        exporter=fake_exporter,
    )

    assert summary["window_count"] == 1
    assert summary["candidate_count"] == 3
    assert output_rows[0]["schema"] == "joint_pre_asr_chunk_reexport_v1"
    assert output_rows[0]["candidate_count"] == 3
    assert output_rows[0]["pre_asr_candidates"].endswith("pre_asr_candidates.jsonl")
    manifest_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "source_windows.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert manifest_rows == output_rows
