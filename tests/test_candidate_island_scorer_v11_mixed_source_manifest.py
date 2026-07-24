from __future__ import annotations

import json
from pathlib import Path
import wave

import pytest

from tools.boundary.ja.select_candidate_island_scorer_v11_mixed_source_manifest import (
    CONTRACT_ID,
    SOURCE_SCHEMA,
    SOURCE_WINDOW_SCHEMA,
    select_manifest,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_wav(path: Path, *, frames: int = 3200) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * frames)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(source_id: str, video_id: str, audio: Path) -> dict:
    return {
        "schema": SOURCE_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "source_id": source_id,
        "video_id": video_id,
        "partition": "train",
        "audio": str(audio),
        "audio_sha256": _sha256(audio),
        "sample_rate": 16000,
        "sample_count": 3200,
        "duration_s": 0.2,
        "frame_count": 10,
        "frame_hop_s": 0.02,
        "teacher_only": True,
        "training_manifest_allowed": False,
    }


def _window(source_id: str, video_id: str, *, candidates: int) -> dict:
    return {
        "schema": SOURCE_WINDOW_SCHEMA,
        "window_id": source_id,
        "video_id": video_id,
        "candidate_count": candidates,
        "span_count": candidates,
        "source_start_s": 0.0,
        "source_end_s": 0.2,
    }


def test_select_manifest_keeps_included_and_prefers_richest_new_source(
    tmp_path: Path,
) -> None:
    audio_paths = {}
    for source_id in ("a-w00", "b-w00", "b-w01", "c-w00"):
        path = tmp_path / f"{source_id}.wav"
        _write_wav(path)
        audio_paths[source_id] = path
    sources = [
        _source("a-w00", "video-a", audio_paths["a-w00"]),
        _source("b-w00", "video-b", audio_paths["b-w00"]),
        _source("b-w01", "video-b", audio_paths["b-w01"]),
        _source("c-w00", "video-c", audio_paths["c-w00"]),
    ]
    windows = [
        _window("a-w00", "video-a", candidates=1),
        _window("b-w00", "video-b", candidates=3),
        _window("b-w01", "video-b", candidates=9),
        _window("c-w00", "video-c", candidates=5),
    ]
    source_manifest = tmp_path / "sources.jsonl"
    source_windows = tmp_path / "windows.jsonl"
    include_manifest = tmp_path / "include.jsonl"
    _write_jsonl(source_manifest, sources)
    _write_jsonl(source_windows, windows)
    _write_jsonl(include_manifest, [{"source_id": "a-w00"}])

    summary = select_manifest(
        source_manifest=source_manifest,
        source_windows=source_windows,
        include_manifest=include_manifest,
        exclude_manifest=None,
        new_video_count=2,
        output_dir=tmp_path / "out",
    )

    assert summary["source_count"] == summary["video_count"] == 3
    assert set(summary["new_selected_source_ids"]) == {"b-w01", "c-w00"}
    output = [
        json.loads(line)
        for line in (tmp_path / "out" / "mixed_source_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["source_id"] for row in output} == {"a-w00", "b-w01", "c-w00"}
    assert len({row["video_id"] for row in output}) == 3
    assert summary["teacher_output_used_as_truth"] is False
    assert summary["training_manifest_allowed"] is False


def test_select_manifest_rejects_two_included_sources_from_one_video(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_wav(first)
    _write_wav(second)
    sources = [
        _source("a-w00", "video-a", first),
        _source("a-w01", "video-a", second),
    ]
    source_manifest = tmp_path / "sources.jsonl"
    source_windows = tmp_path / "windows.jsonl"
    include_manifest = tmp_path / "include.jsonl"
    _write_jsonl(source_manifest, sources)
    _write_jsonl(
        source_windows,
        [
            _window("a-w00", "video-a", candidates=1),
            _window("a-w01", "video-a", candidates=2),
        ],
    )
    _write_jsonl(
        include_manifest,
        [{"source_id": "a-w00"}, {"source_id": "a-w01"}],
    )

    with pytest.raises(ValueError, match="one source per video"):
        select_manifest(
            source_manifest=source_manifest,
            source_windows=source_windows,
            include_manifest=include_manifest,
            exclude_manifest=None,
            new_video_count=0,
            output_dir=tmp_path / "out",
        )
