from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from boundary.ja import read_jsonl
from boundary.ja.train import scorer_v7_targets_from_record, time_span_to_frame_range
from tools.boundary.ja.build_scorer_v7_native_dataset import (
    DATASET_SCHEMA,
    DEFAULT_MIX,
    build_scorer_v7_native_dataset,
    build_positive_example,
    build_pure_negative_example,
    load_negative_rows,
    load_source_groups,
    parse_args,
)
from tools.boundary.ja.summarize_scorer_checkpoint_by_dataset import iter_frame_limited_batches


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


def test_build_scorer_v7_native_dataset_labels_speech_and_split_heatmap(tmp_path: Path):
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
            "--type-mix",
            "long_speech_chain=0.10",
            "--type-mix",
            "positive_speech_timeline=0.25",
            "--type-mix",
            "pure_hard_negative=0.25",
            "--type-mix",
            "mixed_contrast=0.20",
            "--type-mix",
            "split_stress=0.20",
            "--speech-label-dilation-s",
            "0.02",
            "--short-gap-max-s",
            "0.08",
            "--touch-gap-prob",
            "0.0",
            "--negative-min-s",
            "0.8",
            "--negative-max-s",
            "1.0",
            "--context-gap-min-s",
            "0.2",
            "--context-gap-max-s",
            "0.3",
            "--long-chain-min-duration-s",
            "2.5",
            "--long-chain-max-duration-s",
            "4.0",
        ]
    )

    summary = build_scorer_v7_native_dataset(
        source_specs=args.source,
        negative_manifests=args.negative_manifest,
        output_dir=tmp_path / "dataset",
        count=args.count,
        seed=args.seed,
        args=args,
    )

    assert summary["schema"] == DATASET_SCHEMA
    assert summary["native_example_type_counts"] == {
        "long_speech_chain": 1,
        "mixed_contrast": 2,
        "positive_speech_timeline": 3,
        "pure_hard_negative": 3,
        "split_stress": 3,
    }
    assert set(summary["long_chain_gap_bucket_counts"]) == {
        "micro_20_80ms",
        "medium_250_600ms",
        "short_80_250ms",
        "touch",
    }
    assert summary["long_chain_utterance_count_min"] >= 6
    records = read_jsonl(tmp_path / "dataset" / "scorer_v7_native_labels.jsonl")
    by_type: dict[str, list] = {}
    for record in records:
        by_type.setdefault(record.boundary_metadata["native_example_type"], []).append(record)

    pure_negative = by_type["pure_hard_negative"][0]
    pure_speech, pure_split = scorer_v7_targets_from_record(
        pure_negative,
        frame_count=len(pure_negative.speech_frames),
        split_boundary_radius_frames=1,
    )
    assert float(np.max(pure_speech)) == 0.0
    assert float(np.max(pure_split)) == 0.0

    mixed = by_type["mixed_contrast"][0]
    mixed_speech, _split = scorer_v7_targets_from_record(
        mixed,
        frame_count=len(mixed.speech_frames),
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
    assert float(np.min(mixed_speech[start:end])) == 1.0

    split_stress = by_type["split_stress"][0]
    _speech, split_points = scorer_v7_targets_from_record(
        split_stress,
        frame_count=len(split_stress.speech_frames),
        split_boundary_radius_frames=1,
    )
    assert split_stress.boundary_metadata["cut_point_segments"]
    assert float(np.max(split_points)) >= 0.9
    assert float(np.min(split_points[split_points > 0.0])) < 1.0

    long_chain = by_type["long_speech_chain"][0]
    long_speech, long_split = scorer_v7_targets_from_record(
        long_chain,
        frame_count=len(long_chain.speech_frames),
        split_boundary_radius_frames=1,
    )
    long_boundaries = long_chain.boundary_metadata["cut_point_segments"]
    assert len(long_chain.boundary_metadata["actual_speech_segments"]) >= 6
    assert len(long_boundaries) >= 5
    assert {item["gap_bucket"] for item in long_boundaries} >= {
        "micro_20_80ms",
        "medium_250_600ms",
        "short_80_250ms",
        "touch",
    }
    assert long_chain.boundary_metadata["disable_implicit_gap_drop"] is True
    assert float(np.max(long_split)) >= 0.9
    assert float(np.max(long_speech)) == 1.0

    seen_sources: dict[str, str] = {}
    for record in records:
        partition = record.boundary_metadata["source_partition"]
        for source_id in record.boundary_metadata["source_audio_ids"]:
            if not source_id:
                continue
            assert seen_sources.setdefault(source_id, partition) == partition

    splits = json.loads((tmp_path / "dataset" / "scorer_v7_native_splits.json").read_text(encoding="utf-8"))
    assert sorted(index for rows in splits.values() for index in rows) == list(range(len(records)))


def test_default_mix_keeps_bgspan_anchor_without_longchain():
    assert DEFAULT_MIX == {
        "long_speech_chain": 0.0,
        "positive_speech_timeline": 0.40,
        "pure_hard_negative": 0.25,
        "mixed_contrast": 0.20,
        "split_stress": 0.15,
    }


def test_dataset_summary_tool_uses_frame_limited_batches():
    rows = [
        {"frame_count": 100},
        {"frame_count": 120},
        {"frame_count": 900},
        {"frame_count": 130},
        {"frame_count": 140},
    ]

    batches = list(iter_frame_limited_batches(rows, batch_size=8, max_batch_frames=512))

    assert [[row["frame_count"] for row in batch] for batch in batches] == [
        [100, 120],
        [900],
        [130, 140],
    ]
    for batch in batches:
        projected_padding_frames = len(batch) * max(int(row["frame_count"]) for row in batch)
        if len(batch) > 1:
            assert projected_padding_frames <= 512


def test_audited_low_info_vocalization_manifest_is_speech_negative_only(tmp_path: Path):
    audio_path = tmp_path / "audited_low_info_vocal.wav"
    _write_audio(audio_path, duration_s=0.12, frequency=220.0)
    manifest_path = tmp_path / "audited_negative.json"
    _write_manifest(
        manifest_path,
        [
            {
                "audio": str(audio_path),
                "audio_id": "audited-low-info-vocal",
                "duration_s": 0.12,
                "source": "audited_low_info_vocalization",
                "reason": "manual_drop_no_subtitle_value",
                "reason_tags": ["breath", "moan", "low_info_vocalization"],
                "audit_label": "drop",
            }
        ],
    )
    args = parse_args(
        [
            "--count",
            "1",
            "--negative-manifest",
            str(manifest_path),
            "--negative-min-s",
            "0.12",
            "--negative-max-s",
            "0.12",
        ]
    )
    negative_rows, skipped = load_negative_rows([str(manifest_path)])

    assert skipped == []
    _audio, record, detail = build_pure_negative_example(
        index=0,
        negative_rows=negative_rows,
        args=args,
        rng=random.Random(11),
        np_rng=np.random.default_rng(11),
    )

    speech, split_points = scorer_v7_targets_from_record(
        record,
        frame_count=len(record.speech_frames),
        split_boundary_radius_frames=1,
    )
    assert detail["negative_source_label"] == "audited_low_info_vocalization"
    assert detail["negative_reason_tags"] == ["breath", "moan", "low_info_vocalization"]
    assert record.boundary_metadata["negative_source"]["negative_audit_label"] == "drop"
    assert float(np.max(speech)) == 0.0
    assert float(np.max(split_points)) == 0.0


def test_short_positive_anchor_is_not_marked_drop_by_duration(tmp_path: Path):
    source_path = tmp_path / "short_positive.wav"
    _write_audio(source_path, duration_s=0.06, frequency=440.0)
    manifest_path = tmp_path / "short_positive.json"
    _write_manifest(
        manifest_path,
        [
            {
                "audio": str(source_path),
                "audio_id": "short-positive-anchor",
                "duration_s": 0.06,
                "source": "anime_sfw",
                "text": "はい",
            }
        ],
    )
    args = parse_args(
        [
            "--count",
            "1",
            "--source",
            f"anime_sfw=1={manifest_path}",
            "--min-speech-s",
            "0.05",
            "--max-speech-s",
            "0.10",
            "--speech-label-dilation-s",
            "0.0",
        ]
    )
    source_groups, skipped = load_source_groups(
        args.source,
        min_duration_s=args.min_speech_s,
        max_duration_s=args.source_max_duration_s,
    )

    assert skipped == []
    _audio, record, _detail = build_positive_example(
        index=0,
        source_groups=source_groups,
        args=args,
        rng=random.Random(7),
    )
    speech, split_points = scorer_v7_targets_from_record(
        record,
        frame_count=len(record.speech_frames),
        split_boundary_radius_frames=1,
    )

    assert record.duration_s <= 0.07
    assert float(np.max(speech)) == 1.0
    assert float(np.max(split_points)) == 0.0
