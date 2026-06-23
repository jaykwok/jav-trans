from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from boundary.ja import read_jsonl
from boundary.ja.train import endpoint_targets_from_record, time_span_to_frame_range
from tools.boundary.ja.build_scorer_v4_native_dataset import (
    DATASET_SCHEMA,
    build_scorer_v4_native_dataset,
    parse_args,
)


def _write_audio(path: Path, *, duration_s: float, frequency: float = 440.0, noise: bool = False) -> None:
    samples = max(1, int(round(16000 * duration_s)))
    if noise:
        rng = np.random.default_rng(7)
        audio = rng.normal(0.0, 0.02, samples).astype(np.float32)
    else:
        t = np.arange(samples, dtype=np.float32) / 16000.0
        audio = (0.05 * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)
    sf.write(path, audio, 16000)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _source_manifest(tmp_path: Path, name: str, *, rows: int = 8) -> Path:
    manifest_rows = []
    for index in range(rows):
        audio_path = tmp_path / f"{name}_{index}.wav"
        _write_audio(audio_path, duration_s=0.4 + index * 0.01, frequency=330.0 + index * 20.0)
        manifest_rows.append(
            {
                "audio": str(audio_path),
                "audio_id": f"{name}-{index}",
                "duration_s": 0.4 + index * 0.01,
                "source": name,
                "text": "あ",
            }
        )
    manifest_path = tmp_path / f"{name}.json"
    _write_manifest(manifest_path, manifest_rows)
    return manifest_path


def _negative_manifest(tmp_path: Path, *, rows: int = 6) -> Path:
    manifest_rows = []
    for index in range(rows):
        audio_path = tmp_path / f"negative_{index}.wav"
        _write_audio(audio_path, duration_s=1.0 + index * 0.05, noise=True)
        manifest_rows.append(
            {
                "audio": str(audio_path),
                "audio_id": f"negative-{index}",
                "duration_s": 1.0 + index * 0.05,
                "source": "background",
            }
        )
    manifest_path = tmp_path / "negative.json"
    _write_manifest(manifest_path, manifest_rows)
    return manifest_path


def test_build_scorer_v4_native_dataset_labels_separate_heads(tmp_path: Path):
    nsfw = _source_manifest(tmp_path, "anime_nsfw")
    sfw = _source_manifest(tmp_path, "anime_sfw")
    galgame = _source_manifest(tmp_path, "galgame")
    negative = _negative_manifest(tmp_path)
    args = parse_args(
        [
            "--count",
            "12",
            "--seed",
            "3",
            "--source",
            f"anime_nsfw=40={nsfw}",
            "--source",
            f"anime_sfw=40={sfw}",
            "--source",
            f"galgame=20={galgame}",
            "--negative-manifest",
            str(negative),
            "--speech-label-dilation-s",
            "0.02",
            "--short-gap-max-s",
            "0.08",
            "--negative-min-s",
            "0.8",
            "--negative-max-s",
            "1.0",
            "--context-gap-min-s",
            "0.2",
            "--context-gap-max-s",
            "0.3",
        ]
    )

    summary = build_scorer_v4_native_dataset(
        source_specs=args.source,
        negative_manifests=args.negative_manifest,
        output_dir=tmp_path / "dataset",
        count=args.count,
        seed=args.seed,
        args=args,
    )

    assert summary["schema"] == DATASET_SCHEMA
    assert summary["native_example_type_counts"] == {
        "mixed_contrast": 2,
        "positive_speech_timeline": 5,
        "pure_hard_negative": 3,
        "split_stress": 2,
    }
    records = read_jsonl(tmp_path / "dataset" / "scorer_v4_native_labels.jsonl")
    by_type: dict[str, list] = {}
    for record in records:
        by_type.setdefault(record.boundary_metadata["native_example_type"], []).append(record)

    pure_negative = by_type["pure_hard_negative"][0]
    _starts, _ends, pure_drop, pure_split = endpoint_targets_from_record(
        pure_negative,
        frame_count=len(pure_negative.speech_frames),
        boundary_radius_frames=0,
        split_boundary_radius_frames=1,
    )
    assert sum(pure_negative.speech_frames) == 0
    assert float(np.mean(pure_drop)) >= 0.95
    assert float(np.max(pure_split)) == 0.0

    mixed = by_type["mixed_contrast"][0]
    _starts, _ends, mixed_drop, _split = endpoint_targets_from_record(
        mixed,
        frame_count=len(mixed.speech_frames),
        boundary_radius_frames=0,
        split_boundary_radius_frames=1,
    )
    actual = mixed.boundary_metadata["actual_speech_segments"][0]
    start, end = time_span_to_frame_range(
        actual["start"],
        actual["end"],
        frame_count=len(mixed.speech_frames),
        frame_hop_s=mixed.frame_hop_s,
    )
    assert start < end
    assert float(np.max(mixed_drop[start:end])) == 0.0

    split_stress = by_type["split_stress"][0]
    _starts, _ends, split_drop, split_points = endpoint_targets_from_record(
        split_stress,
        frame_count=len(split_stress.speech_frames),
        boundary_radius_frames=0,
        split_boundary_radius_frames=1,
    )
    assert split_stress.boundary_metadata["cut_point_segments"]
    assert split_stress.boundary_metadata["cut_drop_zones"] == []
    assert float(np.max(split_points)) == 1.0
    assert float(np.max(split_drop)) == 0.0

    seen_sources: dict[str, str] = {}
    for record in records:
        partition = record.boundary_metadata["source_partition"]
        for source_id in record.boundary_metadata["source_audio_ids"]:
            if not source_id:
                continue
            assert seen_sources.setdefault(source_id, partition) == partition

    splits = json.loads((tmp_path / "dataset" / "scorer_v4_native_splits.json").read_text(encoding="utf-8"))
    assert sorted(index for rows in splits.values() for index in rows) == list(range(len(records)))
