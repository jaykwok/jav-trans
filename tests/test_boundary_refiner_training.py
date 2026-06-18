from __future__ import annotations

import json
from pathlib import Path

import pytest

from boundary.ja import write_jsonl
from boundary.sequence_features import (
    FRAME_SEQUENCE_FEATURE_SCHEMA,
    FrameSequenceFeatureConfig,
    feature_extraction_hash,
    feature_extraction_signature,
)
from tools.boundary import train_refiner as train_refiner_module
from tools.boundary.build_refiner_frame_sequence_dataset import (
    FrameSequenceConfig,
    build_frame_sequence_dataset,
)
from tools.boundary.build_weighted_source_manifest import build_weighted_manifest
from tools.boundary.train_refiner import TrainRefinerConfig, train_refiner


def _feature_names() -> list[str]:
    return [
        "gap_s",
        "left_duration_s",
        "right_duration_s",
        "proposed_core_s",
        "gap_reference_s",
        "gap_ratio",
        "left_ptm_mean_000",
        "gap_ptm_mean_000",
        "right_ptm_mean_000",
    ]


def _sequence_feature_metadata(feature_names: list[str]) -> dict:
    config = FrameSequenceFeatureConfig(max_ptm_dims=1, include_mfcc=False)
    return {
        "feature_schema": FRAME_SEQUENCE_FEATURE_SCHEMA,
        "feature_schema_hash": feature_extraction_hash(
            config=config,
            feature_names=feature_names,
        ),
        "feature_signature": feature_extraction_signature(
            config=config,
            feature_names=feature_names,
        ),
    }


def _write_v5_dataset(path: Path, *, rows: int = 4, items_per_row: int = 2) -> Path:
    feature_names = _feature_names()
    payloads = []
    for row_index in range(rows):
        sequence = []
        targets = []
        weights = []
        for item_index in range(items_per_row):
            gap_s = 0.2 + 0.1 * item_index
            sequence.append(
                [
                    gap_s,
                    1.0,
                    0.8,
                    2.0 + gap_s,
                    0.5,
                    gap_s / 0.5,
                    0.1 * row_index,
                    0.2 * item_index,
                    0.3,
                ]
            )
            targets.append([0.02 * (item_index + 1), -0.03 * (row_index + 1)])
            weights.append([1.0, 0.6])
        payloads.append(
            {
                "schema": "boundary_refiner_frame_sequence_dataset_v5",
                "audio_id": f"seq-{row_index}",
                "feature_names": feature_names,
                "sequence_features": sequence,
                "sequence_boundary_delta_targets": targets,
                "sequence_boundary_delta_weights": weights,
                **_sequence_feature_metadata(feature_names),
            }
        )
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in payloads) + "\n",
        encoding="utf-8",
    )
    return path


def _tiny_train_config(**overrides) -> TrainRefinerConfig:
    return TrainRefinerConfig(
        **{
            "max_steps": 1,
            "batch_size": 2,
            "device": "cpu",
            "hidden_size": 8,
            "num_layers": 1,
            "state_size": 4,
            "num_heads": 4,
            "n_groups": 2,
            "chunk_size": 4,
            "log_interval_steps": 0,
            **overrides,
        }
    )


def test_train_refiner_accepts_v5_delta_dataset(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")
    dataset = _write_v5_dataset(tmp_path / "sequence.jsonl", rows=4, items_per_row=3)

    metrics = train_refiner(
        dataset_paths=[dataset],
        output_dir=tmp_path / "sequence-train",
        config=_tiny_train_config(max_steps=2),
    )

    assert Path(metrics["checkpoint"]).exists()
    assert metrics["train_items"] + metrics["val_items"] == 12
    assert "class_counts" not in metrics
    assert metrics["train"]["start_delta_mae_s"] >= 0.0
    assert metrics["train"]["start_delta_error"]["mae_s"] >= 0.0
    assert metrics["train"]["end_delta_error"]["mae_s"] >= 0.0
    timing_policy = metrics["loader_smoke"]["signature"]["metadata"]["timing_policy"]
    assert timing_policy["delta_loss"] == "smooth_l1"
    assert timing_policy["start_delta_loss_weight"] == 1.0
    assert timing_policy["end_delta_loss_weight"] == 0.6
    smoke = metrics["loader_smoke"]["first_decision"]
    assert set(smoke) == {"source", "start_refine_delta_s", "end_refine_delta_s"}


def test_train_refiner_rejects_merge_era_fields(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")
    dataset = _write_v5_dataset(tmp_path / "sequence-with-old-labels.jsonl", rows=1, items_per_row=1)
    row = json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])
    row["sequence_labels"] = [1]
    row["merge_positive"] = 1
    dataset.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="old merge/split/context fields"):
        train_refiner(
            dataset_paths=[dataset],
            output_dir=tmp_path / "rejected-train",
            config=_tiny_train_config(),
        )


def test_train_refiner_can_initialize_v5_checkpoint_and_freeze_backbone(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")
    dataset = _write_v5_dataset(tmp_path / "init-sequence.jsonl")
    base_config = _tiny_train_config()

    first = train_refiner(
        dataset_paths=[dataset],
        output_dir=tmp_path / "base-train",
        config=base_config,
    )
    second = train_refiner(
        dataset_paths=[dataset],
        output_dir=tmp_path / "head-train",
        config=TrainRefinerConfig(
            **{
                **base_config.__dict__,
                "init_checkpoint": str(first["checkpoint"]),
                "freeze_backbone": True,
                "preserve_init_normalization": True,
            }
        ),
    )

    metadata = second["loader_smoke"]["signature"]["metadata"]
    assert metadata["freeze_backbone"] is True
    assert metadata["preserve_init_normalization"] is True
    assert metadata["normalization_source"] == "init_checkpoint"
    assert metadata["init_checkpoint"]["schema"] == "boundary_refiner_v5"


def test_train_refiner_uses_streaming_tensor_loader(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")
    dataset = _write_v5_dataset(tmp_path / "streaming-sequence.jsonl", rows=6, items_per_row=1)

    original_iter = train_refiner_module._iter_dataset_rows
    calls = {"count": 0}

    def counting_iter(paths):
        calls["count"] += 1
        yield from original_iter(paths)

    monkeypatch.setattr(train_refiner_module, "_iter_dataset_rows", counting_iter)

    metrics = train_refiner(
        dataset_paths=[dataset],
        output_dir=tmp_path / "streaming-train",
        config=_tiny_train_config(),
    )

    assert Path(metrics["checkpoint"]).exists()
    assert metrics["train_items"] + metrics["val_items"] == 6
    assert calls["count"] == 2


def test_train_refiner_reuses_v5_tensor_cache_without_jsonl_rescan(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")
    dataset = _write_v5_dataset(tmp_path / "cache-sequence.jsonl", rows=6, items_per_row=1)
    tensor_cache = tmp_path / "cache-sequence.tensor.pt"

    first = train_refiner(
        dataset_paths=[dataset],
        output_dir=tmp_path / "cache-train-first",
        config=_tiny_train_config(tensor_cache_path=str(tensor_cache)),
    )
    assert Path(first["checkpoint"]).exists()
    assert tensor_cache.exists()

    def fail_iter(paths):
        raise AssertionError("JSONL rows should not be read when tensor cache exists")
        yield from ()

    monkeypatch.setattr(train_refiner_module, "_iter_dataset_rows", fail_iter)
    second = train_refiner(
        dataset_paths=[dataset],
        output_dir=tmp_path / "cache-train-second",
        config=_tiny_train_config(tensor_cache_path=str(tensor_cache)),
    )
    assert second["train_items"] + second["val_items"] == 6


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
            synthetic_boundary_delta_jitter_s=0.2,
        ),
    )
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]

    assert "class_balance" not in summary
    assert rows[0]["schema"] == "boundary_refiner_frame_sequence_dataset_v5"
    assert "sequence_labels" not in rows[0]
    assert "gap_reference_s" in rows[0]["feature_names"]
    assert "gap_merge_s" not in rows[0]["feature_names"]
    assert len(rows[0]["sequence_features"]) == 1
    assert len(rows[0]["sequence_boundary_delta_targets"]) == 1
    assert rows[0]["sequence_boundary_delta_targets"][0] != [0.0, 0.0]

    metrics = train_refiner(
        dataset_paths=[output_jsonl],
        output_dir=tmp_path / "frame-sequence-train",
        config=_tiny_train_config(max_steps=2, batch_size=1),
    )
    assert Path(metrics["checkpoint"]).exists()
    assert metrics["train_items"] + metrics["val_items"] == 1
    assert metrics["loader_smoke"]["decision_count"] == 1
    assert metrics["loader_smoke"]["signature"]["metadata"]["runtime_adapter"] == "frame_sequence_v1"
    assert metrics["loader_smoke"]["signature"]["metadata"]["feature_schema"] == FRAME_SEQUENCE_FEATURE_SCHEMA
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
