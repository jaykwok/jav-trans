from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.boundary.build_refiner_gap_dataset import GapDatasetConfig, build_gap_dataset
from tools.boundary.build_refiner_frame_sequence_dataset import (
    FrameSequenceConfig,
    build_frame_sequence_dataset,
)
from tools.boundary.build_weighted_source_manifest import build_weighted_manifest
from tools.boundary import train_refiner as train_refiner_module
from tools.boundary.train_refiner import TrainRefinerConfig, train_refiner
from boundary.ja import write_jsonl
from boundary.sequence_features import FRAME_SEQUENCE_FEATURE_SCHEMA


def test_build_refiner_gap_dataset_from_boundary_metadata(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    feature_manifest = tmp_path / "feature_manifest.json"
    write_jsonl(
        labels_path,
        [
            {
                "audio_id": "sample-1",
                "source": "unit",
                "duration_s": 4.0,
                "text": "",
                "teacher_segments": {"supervised": [{"start": 0.0, "end": 4.0, "score": 1.0}]},
                "frame_hop_s": 0.02,
                "speech_frames": [1] * 200,
                "label_quality": "supervised",
                "boundary_metadata": {
                    "actual_speech_segments": [
                        {"start": 0.0, "end": 1.0},
                        {"start": 1.02, "end": 2.0},
                        {"start": 3.0, "end": 4.0},
                    ],
                    "utterance_boundaries": [
                        {
                            "index": 0,
                            "boundary_type": "cut_point",
                            "gap_s": 0.02,
                        },
                        {
                            "index": 1,
                            "boundary_type": "gap_zone",
                            "gap_s": 1.0,
                        },
                    ],
                    "source_audio_ids": ["a", "b", "c"],
                },
            }
        ],
    )
    feature_manifest.write_text(
        json.dumps(
            [
                {
                    "audio_id": "sample-1",
                    "feature_path": "features/sample-1.npz",
                    "label_index": 0,
                    "frame_count": 200,
                    "frame_hop_s": 0.02,
                    "ptm_dim": 1024,
                    "mfcc_dim": 40,
                    "ptm": "qwen",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_jsonl = tmp_path / "gaps.jsonl"
    output_sequence_jsonl = tmp_path / "sequences.jsonl"

    summary = build_gap_dataset(
        labels_paths=[labels_path],
        feature_manifest_paths=[feature_manifest],
        output_jsonl=output_jsonl,
        output_sequence_jsonl=output_sequence_jsonl,
        config=GapDatasetConfig(
            synthetic_merge_positives_per_record=1,
            synthetic_merge_min_segment_s=0.8,
        ),
    )

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert summary["class_balance"] == {"merge_positive": 1, "split_negative": 2}
    assert {row["label_reason"] for row in rows} == {
        "split_cut_point",
        "split_gap_zone",
        "merge_synthetic_intra_island",
    }
    assert all(row["metadata"]["ptm_dim"] == 1024 for row in rows)
    sequence_rows = [
        json.loads(line)
        for line in output_sequence_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert summary["sequence_counts"] == {"sequences": 1, "sequence_items": 3}
    assert sequence_rows[0]["schema"] == "boundary_refiner_sequence_dataset_v1"
    assert sorted(sequence_rows[0]["sequence_labels"]) == [0, 0, 1]


def test_build_refiner_gap_dataset_requires_ptm_dim(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    feature_manifest = tmp_path / "feature_manifest.json"
    write_jsonl(
        labels_path,
        [
            {
                "audio_id": "sample-1",
                "source": "unit",
                "duration_s": 2.0,
                "text": "",
                "teacher_segments": {"supervised": [{"start": 0.0, "end": 2.0, "score": 1.0}]},
                "frame_hop_s": 0.02,
                "speech_frames": [1] * 100,
                "label_quality": "supervised",
                "boundary_metadata": {
                    "actual_speech_segments": [
                        {"start": 0.0, "end": 0.9},
                        {"start": 1.2, "end": 2.0},
                    ],
                },
            }
        ],
    )
    feature_manifest.write_text(
        json.dumps(
            [
                {
                    "audio_id": "sample-1",
                    "feature_path": "features/sample-1.npz",
                    "label_index": 0,
                    "frame_count": 100,
                    "frame_hop_s": 0.02,
                    "mfcc_dim": 40,
                    "ptm": "qwen",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ptm_dim"):
        build_gap_dataset(
            labels_paths=[labels_path],
            feature_manifest_paths=[feature_manifest],
            output_jsonl=tmp_path / "gaps.jsonl",
        )


def test_train_refiner_checkpoint_round_trip_from_gap_dataset(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")

    dataset = tmp_path / "gaps.jsonl"
    rows = []
    for index in range(8):
        merge = index % 2 == 0
        gap_s = 0.03 if merge else 0.8
        rows.append(
            {
                "schema": "boundary_refiner_gap_dataset_v2",
                "audio_id": f"sample-{index}",
                "source": "unit",
                "label_index": index,
                "gap_index": 0,
                "merge_target": merge,
                "label": 1 if merge else 0,
                "sequence_context_targets": [[0.0, 0.0] if merge else [0.5, 0.5]],
                "label_reason": "merge_test" if merge else "split_test",
                "feature_names": [
                    "gap_s",
                    "left_duration_s",
                    "right_duration_s",
                    "current_core_s",
                    "proposed_core_s",
                    "gap_merge_s",
                    "gap_ratio",
                    "proposed_over_target_s",
                    "left_score",
                    "right_score",
                    "valley_score_min",
                    "cut_score_max",
                    "gap_boundary_score",
                ],
                "refiner_input": {
                    "gap_s": gap_s,
                    "left_start": 0.0,
                    "left_end": 1.0,
                    "right_start": 1.0 + gap_s,
                    "right_end": 2.0 + gap_s,
                    "current_core_s": 1.0,
                    "proposed_core_s": 2.0 + gap_s,
                    "gap_merge_s": 1.5,
                    "left_score": 1.0,
                    "right_score": 1.0,
                    "valley_score_min": None,
                    "cut_score_max": None,
                    "gap_boundary_score": min(1.0, gap_s / 1.5),
                },
            }
        )
    dataset.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    metrics = train_refiner(
        dataset_paths=[dataset],
        output_dir=tmp_path / "train",
        config=TrainRefinerConfig(
            max_steps=2,
            batch_size=4,
            device="cpu",
            hidden_size=8,
            num_layers=1,
            state_size=4,
            num_heads=4,
            n_groups=2,
            chunk_size=4,
            log_interval_steps=0,
        ),
    )

    checkpoint = Path(metrics["checkpoint"])
    assert checkpoint.exists()
    assert metrics["loader_smoke"]["signature"]["schema"] == "boundary_refiner_v2"
    assert metrics["loader_smoke"]["signature"]["backbone"] == "transformers.Mamba2Model"


def test_train_refiner_accepts_sequence_dataset(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")

    feature_names = [
        "gap_s",
        "left_duration_s",
        "right_duration_s",
        "current_core_s",
        "proposed_core_s",
        "gap_merge_s",
        "gap_ratio",
        "proposed_over_target_s",
        "left_score",
        "right_score",
        "valley_score_min",
        "cut_score_max",
        "gap_boundary_score",
    ]
    dataset = tmp_path / "sequence.jsonl"
    rows = []
    for row_index in range(4):
        sequence = []
        labels = []
        for step_index in range(3):
            merge = (row_index + step_index) % 2 == 0
            gap_s = 0.04 if merge else 0.9
            sequence.append(
                [
                    gap_s,
                    1.0,
                    1.0,
                    1.0 + step_index,
                    2.0 + gap_s + step_index,
                    1.5,
                    min(1.0, gap_s / 1.5),
                    1.0 + gap_s,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    min(1.0, gap_s / 1.5),
                ]
            )
            labels.append(1 if merge else 0)
        rows.append(
            {
                "schema": "boundary_refiner_sequence_dataset_v2",
                "audio_id": f"seq-{row_index}",
                "feature_names": feature_names,
                "sequence_features": sequence,
                "sequence_labels": labels,
                "sequence_context_targets": [
                    [0.0, 0.0] if label else [0.5, 0.5]
                    for label in labels
                ],
            }
        )
    dataset.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    metrics = train_refiner(
        dataset_paths=[dataset],
        output_dir=tmp_path / "sequence-train",
        config=TrainRefinerConfig(
            max_steps=2,
            batch_size=2,
            device="cpu",
            hidden_size=8,
            num_layers=1,
            state_size=4,
            num_heads=4,
            n_groups=2,
            chunk_size=4,
            log_interval_steps=0,
        ),
    )

    assert Path(metrics["checkpoint"]).exists()
    assert metrics["train_items"] + metrics["val_items"] == 12
    assert metrics["class_counts"] == {"merge_positive": 6, "split_negative": 6}


def test_train_refiner_uses_streaming_tensor_loader(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")

    feature_names = [
        "gap_s",
        "left_duration_s",
        "right_duration_s",
        "current_core_s",
        "proposed_core_s",
        "gap_merge_s",
        "gap_ratio",
        "proposed_over_target_s",
        "left_score",
        "right_score",
        "valley_score_min",
        "cut_score_max",
        "gap_boundary_score",
    ]
    rows = []
    for row_index in range(6):
        merge = row_index % 2 == 0
        gap_s = 0.04 if merge else 0.9
        rows.append(
            {
                "schema": "boundary_refiner_sequence_dataset_v2",
                "audio_id": f"stream-{row_index}",
                "feature_names": feature_names,
                "sequence_features": [
                    [
                        gap_s,
                        1.0,
                        1.0,
                        1.0,
                        2.0 + gap_s,
                        1.5,
                        min(1.0, gap_s / 1.5),
                        1.0 + gap_s,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        min(1.0, gap_s / 1.5),
                    ]
                ],
                "sequence_labels": [1 if merge else 0],
                "sequence_context_targets": [[[0.0, 0.0] if merge else [0.5, 0.5]][0]],
            }
        )
    dataset = tmp_path / "streaming-sequence.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    original_iter = train_refiner_module._iter_dataset_rows
    calls = {"count": 0}

    def counting_iter(paths):
        calls["count"] += 1
        yield from original_iter(paths)

    monkeypatch.setattr(train_refiner_module, "_iter_dataset_rows", counting_iter)

    metrics = train_refiner(
        dataset_paths=[dataset],
        output_dir=tmp_path / "streaming-train",
        config=TrainRefinerConfig(
            max_steps=1,
            batch_size=2,
            device="cpu",
            hidden_size=8,
            num_layers=1,
            state_size=4,
            num_heads=4,
            n_groups=2,
            chunk_size=4,
            log_interval_steps=0,
        ),
    )

    assert Path(metrics["checkpoint"]).exists()
    assert metrics["train_items"] + metrics["val_items"] == 6
    assert calls["count"] == 2


def test_build_frame_sequence_dataset_trains_with_cached_features(tmp_path):
    pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")

    import numpy as np

    labels_path = tmp_path / "labels.jsonl"
    feature_manifest = tmp_path / "feature_manifest.json"
    feature_path = tmp_path / "sample-features.npz"
    ptm = np.arange(24, dtype=np.float32).reshape(6, 4) / 24.0
    mfcc = np.arange(12, dtype=np.float32).reshape(6, 2) / 12.0
    np.savez(feature_path, ptm=ptm, mfcc=mfcc)
    write_jsonl(
        labels_path,
        [
            {
                "audio_id": "sample-1",
                "source": "unit",
                "duration_s": 0.6,
                "text": "",
                "teacher_segments": {"supervised": [{"start": 0.0, "end": 0.6, "score": 1.0}]},
                "frame_hop_s": 0.1,
                "speech_frames": [1] * 6,
                "label_quality": "supervised",
                "boundary_metadata": {
                    "actual_speech_segments": [
                        {"start": 0.0, "end": 0.2},
                        {"start": 0.24, "end": 0.4},
                        {"start": 0.5, "end": 0.6},
                    ],
                    "utterance_boundaries": [
                        {
                            "index": 1,
                            "boundary_type": "gap_zone",
                            "gap_s": 0.1,
                        },
                    ],
                    "source_audio_ids": ["a", "a", "b"],
                },
            }
        ],
    )
    feature_manifest.write_text(
        json.dumps(
            [
                {
                    "audio_id": "sample-1",
                    "feature_path": str(feature_path),
                    "label_index": 0,
                    "frame_count": 6,
                    "frame_hop_s": 0.1,
                    "ptm_dim": 4,
                    "mfcc_dim": 2,
                    "ptm": "qwen",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_jsonl = tmp_path / "frame-sequences.jsonl"

    summary = build_frame_sequence_dataset(
        labels_paths=[labels_path],
        feature_manifest_paths=[feature_manifest],
        output_jsonl=output_jsonl,
        config=FrameSequenceConfig(
            left_context_s=0.2,
            right_context_s=0.2,
            max_ptm_dims=3,
            synthetic_merge_positives_per_record=1,
            synthetic_merge_min_segment_s=0.15,
        ),
    )
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]

    assert summary["class_balance"] == {"merge_positive": 2, "split_negative": 1}
    assert rows[0]["schema"] == "boundary_refiner_frame_sequence_dataset_v2"
    assert len(rows[0]["feature_names"]) == 36
    assert len(rows[0]["sequence_features"]) == 3
    assert len(rows[0]["sequence_context_targets"]) == 3

    metrics = train_refiner(
        dataset_paths=[output_jsonl],
        output_dir=tmp_path / "frame-sequence-train",
        config=TrainRefinerConfig(
            max_steps=2,
            batch_size=1,
            device="cpu",
            hidden_size=8,
            num_layers=1,
            state_size=4,
            num_heads=4,
            n_groups=2,
            chunk_size=4,
            log_interval_steps=0,
        ),
    )
    assert Path(metrics["checkpoint"]).exists()
    assert metrics["train_items"] + metrics["val_items"] == 3
    assert metrics["loader_smoke"]["decision"] is None
    assert metrics["loader_smoke"]["signature"]["metadata"]["runtime_adapter"] == "frame_sequence_v1"
    assert metrics["loader_smoke"]["signature"]["metadata"]["feature_schema"] == FRAME_SEQUENCE_FEATURE_SCHEMA
    assert metrics["loader_smoke"]["signature"]["metadata"]["feature_schema_hash"]
    assert rows[0]["feature_schema"] == FRAME_SEQUENCE_FEATURE_SCHEMA
    assert rows[0]["feature_schema_hash"]
    assert rows[0]["feature_signature"]["feature_schema"] == FRAME_SEQUENCE_FEATURE_SCHEMA
    assert torch.cuda.is_available() in {True, False}


def test_build_weighted_source_manifest_samples_requested_mix(tmp_path):
    nsfw_audio = tmp_path / "nsfw.wav"
    sfw_audio = tmp_path / "sfw.wav"
    nsfw_audio.write_bytes(b"RIFF")
    sfw_audio.write_bytes(b"RIFF")
    nsfw_manifest = tmp_path / "nsfw_manifest.json"
    sfw_manifest = tmp_path / "sfw_manifest.json"
    nsfw_manifest.write_text(
        json.dumps(
            [
                {
                    "audio": str(nsfw_audio),
                    "audio_id": "nsfw-1",
                    "duration_s": 1.5,
                    "source": "anime_nsfw",
                    "text": "a",
                },
                {
                    "audio": str(tmp_path / "missing.wav"),
                    "audio_id": "missing",
                    "duration_s": 1.5,
                    "source": "anime_nsfw",
                    "text": "ignored",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    sfw_manifest.write_text(
        json.dumps(
            [
                {
                    "audio": str(sfw_audio),
                    "audio_id": "sfw-1",
                    "duration_s": 2.0,
                    "source": "anime_sfw",
                    "text": "b",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_manifest = tmp_path / "mixed" / "source_manifest.json"
    summary = build_weighted_manifest(
        specs=[
            f"anime_nsfw=7={nsfw_manifest}",
            f"anime_sfw=3={sfw_manifest}",
        ],
        output_manifest=output_manifest,
        total_rows=10,
        seed=123,
    )

    rows = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert summary["group_counts"] == {"anime_nsfw": 7, "anime_sfw": 3}
    assert len(rows) == 10
    assert {row["source_mix_group"] for row in rows} == {"anime_nsfw", "anime_sfw"}
    assert all(row["source_mix_manifest"] for row in rows)
