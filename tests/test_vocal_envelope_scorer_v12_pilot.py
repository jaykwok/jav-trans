from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import pytest

from boundary.ja.vocal_envelope_v12 import VOCAL_ENVELOPE_SCORER_V12_SOURCE_SCHEMA
from tools.boundary.ja.build_vocal_envelope_scorer_v12_pilot_manifest import (
    build_pilot_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _wav(path: Path, *, frames: int = 1600) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * frames)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_rows = []
    partition_rows = []
    specs = [
        ("train-a", "video-a", "train"),
        ("train-b", "video-b", "train"),
        ("val-a", "video-val", "val"),
        ("test-a", "video-test", "test"),
    ]
    for source_id, video_id, partition in specs:
        audio = tmp_path / f"{source_id}.wav"
        _wav(audio)
        source_rows.append(
            {
                "schema": "joint_boundary_omni_source_window_v1",
                "window_id": source_id,
                "video_id": video_id,
                "audio_wav": str(audio),
                "audio_wav_sha256": _sha256(audio),
                "duration_s": 0.1,
            }
        )
        partition_rows.append(
            {
                "schema": "candidate_island_scorer_v11_partition_manifest_v1",
                "boundary_serialization_contract_id": "boundary_acoustic_binary_v12",
                "source_id": source_id,
                "video_id": video_id,
                "partition": partition,
            }
        )
    source_windows = tmp_path / "source_windows.jsonl"
    partitions = tmp_path / "partitions.jsonl"
    _write_jsonl(source_windows, source_rows)
    _write_jsonl(partitions, partition_rows)
    return source_windows, partitions


def test_v12_pilot_reuses_identity_only_and_validates_audio(tmp_path: Path) -> None:
    source_windows, partitions = _fixture(tmp_path)
    output = tmp_path / "output"
    summary = build_pilot_manifest(
        source_windows=source_windows,
        partition_manifest=partitions,
        output_dir=output,
        train_count=2,
        heldout_count=2,
        seed=117,
    )
    assert summary["partition_counts"] == {"test": 1, "train": 2, "val": 1}
    assert summary["v11_truth_inherited"] is False
    rows = [
        json.loads(line)
        for line in (output / "source_manifest.jsonl").read_text().splitlines()
    ]
    assert all(row["schema"] == VOCAL_ENVELOPE_SCORER_V12_SOURCE_SCHEMA for row in rows)
    assert all(row["core_ids"] == [row["source_id"]] for row in rows)
    assert all(row["frame_count"] == 5 for row in rows)
    assert all(row["v11_span_inherited"] is False for row in rows)
    assert len({row["core_ids"][0] for row in rows}) == 4


def test_v12_pilot_rejects_audio_sha_and_partition_leak(tmp_path: Path) -> None:
    source_windows, partitions = _fixture(tmp_path)
    source_rows = [json.loads(line) for line in source_windows.read_text().splitlines()]
    source_rows[0]["audio_wav_sha256"] = "0" * 64
    _write_jsonl(source_windows, source_rows)
    with pytest.raises(ValueError, match="audio SHA mismatch"):
        build_pilot_manifest(
            source_windows=source_windows,
            partition_manifest=partitions,
            output_dir=tmp_path / "bad-sha",
            train_count=2,
            heldout_count=2,
        )

    partition_root = tmp_path / "partition"
    partition_root.mkdir()
    source_windows, partitions = _fixture(partition_root)
    source_rows = [json.loads(line) for line in source_windows.read_text().splitlines()]
    partition_rows = [json.loads(line) for line in partitions.read_text().splitlines()]
    partition_rows[-1]["video_id"] = partition_rows[-2]["video_id"]
    source_rows[-1]["video_id"] = partition_rows[-1]["video_id"]
    _write_jsonl(source_windows, source_rows)
    _write_jsonl(partitions, partition_rows)
    with pytest.raises(ValueError, match="video crosses partitions"):
        build_pilot_manifest(
            source_windows=source_windows,
            partition_manifest=partitions,
            output_dir=tmp_path / "leak",
            train_count=2,
            heldout_count=2,
        )


def test_v12_pilot_skips_a_train_identity_with_implausible_duration(
    tmp_path: Path,
) -> None:
    source_windows, partitions = _fixture(tmp_path)
    source_rows = [json.loads(line) for line in source_windows.read_text().splitlines()]
    bad_audio = Path(source_rows[0]["audio_wav"])
    _wav(bad_audio, frames=1600 * 20)
    source_rows[0]["audio_wav_sha256"] = _sha256(bad_audio)
    _write_jsonl(source_windows, source_rows)
    summary = build_pilot_manifest(
        source_windows=source_windows,
        partition_manifest=partitions,
        output_dir=tmp_path / "skip-bad-duration",
        train_count=1,
        heldout_count=2,
    )
    assert len(summary["selected_train_source_ids"]) == 1
    assert summary["rejected_identities"][0]["source_id"] == "train-a"
