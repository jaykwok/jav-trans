from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import wave

import numpy as np
import pytest

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.ja.model import (
    CANDIDATE_ISLAND_SCORER_V11_COMPACT_CAPACITY_PROFILE,
    CANDIDATE_ISLAND_SCORER_V11_COMPACT_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_SYNTHETIC_TRAIN_SOURCE_SCHEMA,
    load_speech_island_scorer_checkpoint,
)
from tools.boundary.ja.compile_candidate_island_scorer_v11_canonical import (
    HELDOUT_VERDICT_SCHEMA,
    PARTITION_SCHEMA,
    compile_canonical,
)
from tools.boundary.ja.compile_candidate_island_scorer_v11_features import (
    compile_features,
)
from tools.boundary.ja.train_candidate_island_scorer_v11 import (
    _cuda_warmup_rows,
    _pack_batches,
    _plan_training_batches,
    _resolve_training_device,
    _restore_model_and_adamw_after_warmup,
    run as train,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_wav(path: Path, frame_count: int) -> None:
    samples = frame_count * 320
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * samples)


def _canonical_fixture(tmp_path: Path, *, provenance: str = "human_confirmed_visible_dom"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    frame_count = 8
    source_rows = []
    partition_rows = []
    verdict_rows = []
    for index, partition in enumerate(("train", "val", "test")):
        source_id = f"video-{partition}-w00"
        video_id = f"video-{partition}"
        audio = tmp_path / f"{source_id}.wav"
        _write_wav(audio, frame_count)
        source_rows.append(
            {
                "schema": "joint_boundary_omni_source_window_v1",
                "window_id": source_id,
                "video_id": video_id,
                "audio_wav": str(audio),
                "audio_wav_sha256": _sha256(audio),
                "duration_s": frame_count * 0.02,
            }
        )
        partition_rows.append(
            {
                "schema": PARTITION_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "video_id": video_id,
                "partition": partition,
                "original_dataset_role": partition,
            }
        )
        if partition in {"val", "test"}:
            verdict_rows.append(
                {
                    "schema": HELDOUT_VERDICT_SCHEMA,
                    "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                    "source_id": source_id,
                    "partition": partition,
                    "frame_count": frame_count,
                    "frame_hop_s": 0.02,
                    "reviewed_full_source": True,
                    "verdict": "complete_with_target_inside_candidate",
                    "spans": [
                        {"label": "outside_candidate", "start_frame": 0, "end_frame": 2},
                        {"label": "inside_candidate", "start_frame": 2, "end_frame": 5},
                        {"label": "unsure", "start_frame": 5, "end_frame": 6},
                        {"label": "outside_candidate", "start_frame": 6, "end_frame": 8},
                    ],
                    "review_provenance": provenance,
                    "training_manifest_allowed": False,
                    "human_review_required": False,
                }
            )
    synthetic_audio = tmp_path / "synthetic-train.wav"
    _write_wav(synthetic_audio, frame_count)
    synthetic_path = tmp_path / "synthetic_train_sources.jsonl"
    _write_jsonl(
        synthetic_path,
        [
            {
                "schema": CANDIDATE_ISLAND_SCORER_V11_SYNTHETIC_TRAIN_SOURCE_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": "synthetic-train",
                "partition": "train",
                "source_kind": "semantic_composite_candidate",
                "synthetic_composite": True,
                "input_distribution": "train_exact_candidate_context_composite_v1",
                "audio": str(synthetic_audio),
                "audio_sha256": _sha256(synthetic_audio),
                "sample_rate": 16000,
                "sample_count": frame_count * 320,
                "duration_s": frame_count * 0.02,
                "frame_count": frame_count,
                "frame_hop_s": 0.02,
                "core_ids": ["synthetic-core"],
                "canonical_spans": [
                    {"label": "outside_candidate", "start_frame": 0, "end_frame": 2},
                    {"label": "inside_candidate", "start_frame": 2, "end_frame": 6},
                    {"label": "outside_candidate", "start_frame": 6, "end_frame": 8},
                ],
                "training_manifest_allowed": True,
            }
        ],
    )
    source_path = tmp_path / "source_windows.jsonl"
    partition_path = tmp_path / "partition_manifest.jsonl"
    verdict_path = tmp_path / "manual_verdicts.jsonl"
    real_train_outside_path = tmp_path / "real_train_outside_sources.jsonl"
    _write_jsonl(source_path, source_rows)
    _write_jsonl(partition_path, partition_rows)
    _write_jsonl(verdict_path, verdict_rows)
    train_source = next(row for row in source_rows if "-train-" in row["window_id"])
    _write_jsonl(
        real_train_outside_path,
        [
            {
                "schema": "candidate_island_scorer_v11_real_train_outside_source_v1",
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": train_source["window_id"],
                "video_id": train_source["video_id"],
                "partition": "train",
                "input_distribution": "real_workflow_source_window_gemini_asr_masked_v1",
                "synthetic_composite": False,
                "audio": train_source["audio_wav"],
                "audio_sha256": train_source["audio_wav_sha256"],
                "duration_s": frame_count * 0.02,
                "frame_count": frame_count,
                "frame_hop_s": 0.02,
                "core_ids": [f"real-train-outside-source::{train_source['window_id']}"],
                "canonical_spans": [
                    {"label": "outside_candidate", "start_frame": 0, "end_frame": 3},
                    {"label": "unsure", "start_frame": 3, "end_frame": 8},
                ],
                "annotation_provenance": "gemini_outside_complement_plus_1p7b_asr_empty_v1",
                "gemini_output_used_as_inside_truth": False,
                "asr_text_used_as_inside_truth": False,
                "asr_empty_used_without_gemini_outside": False,
                "unsure_training_label": -100,
                "training_manifest_allowed": True,
            }
        ],
    )
    return (
        synthetic_path,
        real_train_outside_path,
        source_path,
        partition_path,
        verdict_path,
    )


def test_v11_canonical_requires_all_human_full_source_truth(tmp_path: Path) -> None:
    synthetic_path, real_train_path, source_path, partition_path, verdict_path = _canonical_fixture(tmp_path)
    partial = tmp_path / "partial.jsonl"
    partial.write_text(verdict_path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="full-source truth for every frozen source"):
        compile_canonical(
            synthetic_train_sources=synthetic_path,
            real_train_outside_sources=real_train_path,
            source_windows=source_path,
            partition_manifest=partition_path,
            manual_verdicts=[partial],
            output_dir=tmp_path / "partial-output",
        )

    omni_synthetic, omni_real_train, omni_source, omni_partition, omni_verdict = _canonical_fixture(
        tmp_path / "omni", provenance="omni:qwen3.5-omni-plus"
    )
    with pytest.raises(ValueError, match="Omni-only"):
        compile_canonical(
            synthetic_train_sources=omni_synthetic,
            real_train_outside_sources=omni_real_train,
            source_windows=omni_source,
            partition_manifest=omni_partition,
            manual_verdicts=[omni_verdict],
            output_dir=tmp_path / "omni-output",
        )


def test_v11_canonical_requires_train_only_unique_synthetic_cores(tmp_path: Path) -> None:
    synthetic_path, real_train_path, source_path, partition_path, verdict_path = _canonical_fixture(tmp_path)
    rows = [json.loads(line) for line in synthetic_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["partition"] = "val"
    invalid_partition = tmp_path / "synthetic-val.jsonl"
    _write_jsonl(invalid_partition, rows)
    with pytest.raises(ValueError, match="not train-only"):
        compile_canonical(
            synthetic_train_sources=invalid_partition,
            real_train_outside_sources=real_train_path,
            source_windows=source_path,
            partition_manifest=partition_path,
            manual_verdicts=[verdict_path],
            output_dir=tmp_path / "invalid-partition-output",
        )

    rows[0]["partition"] = "train"
    duplicate = dict(rows[0])
    duplicate["source_id"] = "synthetic-train-duplicate"
    duplicate_core = tmp_path / "synthetic-duplicate-core.jsonl"
    _write_jsonl(duplicate_core, [rows[0], duplicate])
    with pytest.raises(ValueError, match="core identity is reused"):
        compile_canonical(
            synthetic_train_sources=duplicate_core,
            real_train_outside_sources=real_train_path,
            source_windows=source_path,
            partition_manifest=partition_path,
            manual_verdicts=[verdict_path],
            output_dir=tmp_path / "duplicate-core-output",
        )


def test_v11_canonical_rejects_gemini_inside_as_real_train_truth(tmp_path: Path) -> None:
    synthetic_path, real_train_path, source_path, partition_path, verdict_path = _canonical_fixture(tmp_path)
    rows = [json.loads(line) for line in real_train_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["gemini_output_used_as_inside_truth"] = True
    invalid = tmp_path / "gemini-inside-truth.jsonl"
    _write_jsonl(invalid, rows)
    with pytest.raises(ValueError, match="Gemini inside"):
        compile_canonical(
            synthetic_train_sources=synthetic_path,
            real_train_outside_sources=invalid,
            source_windows=source_path,
            partition_manifest=partition_path,
            manual_verdicts=[verdict_path],
            output_dir=tmp_path / "gemini-inside-output",
        )


def test_v11_canonical_clips_unavailable_review_grid_tail_to_wav(tmp_path: Path) -> None:
    synthetic_path, real_train_path, source_path, partition_path, verdict_path = _canonical_fixture(
        tmp_path
    )
    source_rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
    ]
    val_source = next(row for row in source_rows if "-val-" in row["window_id"])
    val_audio = Path(val_source["audio_wav"])
    _write_wav(val_audio, 7)
    val_source["audio_wav_sha256"] = _sha256(val_audio)
    _write_jsonl(source_path, source_rows)

    summary = compile_canonical(
        synthetic_train_sources=synthetic_path,
        real_train_outside_sources=real_train_path,
        source_windows=source_path,
        partition_manifest=partition_path,
        manual_verdicts=[verdict_path],
        output_dir=tmp_path / "clipped-canonical",
    )
    canonical_path = Path(summary["canonical_sources"])
    if not canonical_path.is_absolute():
        canonical_path = Path.cwd() / canonical_path
    rows = [
        json.loads(line)
        for line in canonical_path.read_text(encoding="utf-8").splitlines()
    ]
    clipped = next(row for row in rows if row["source_id"] == val_source["window_id"])
    assert clipped["reviewed_nominal_frame_count"] == 8
    assert clipped["audio_sample_count"] == 7 * 320
    assert clipped["frame_count"] == 7
    assert clipped["canonical_spans"][-1] == {
        "label": "outside_candidate",
        "start_frame": 6,
        "end_frame": 7,
        "start_s": 0.12,
        "end_s": 0.14,
    }
    assert clipped["audio_geometry_policy"] == (
        "trim_unavailable_review_grid_tail_to_decoded_audio_v1"
    )


def test_v11_real_train_ignores_only_subframe_audio_tail(tmp_path: Path) -> None:
    synthetic_path, real_train_path, source_path, partition_path, verdict_path = _canonical_fixture(tmp_path)
    source_rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()]
    real_rows = [json.loads(line) for line in real_train_path.read_text(encoding="utf-8").splitlines()]
    train_source = next(row for row in source_rows if "-train-" in row["window_id"])
    train_audio = Path(train_source["audio_wav"])
    with wave.open(str(train_audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * (8 * 320 + 5))
    digest = _sha256(train_audio)
    train_source["audio_wav_sha256"] = digest
    real_rows[0]["audio_sha256"] = digest
    _write_jsonl(source_path, source_rows)
    _write_jsonl(real_train_path, real_rows)

    summary = compile_canonical(
        synthetic_train_sources=synthetic_path,
        real_train_outside_sources=real_train_path,
        source_windows=source_path,
        partition_manifest=partition_path,
        manual_verdicts=[verdict_path],
        output_dir=tmp_path / "subframe-tail-canonical",
    )
    rows = [
        json.loads(line)
        for line in Path(summary["canonical_sources"]).read_text(encoding="utf-8").splitlines()
    ]
    real = next(row for row in rows if row["source_id"] == "video-train-w00")
    assert real["frame_count"] == real["reviewed_nominal_frame_count"] == 8
    assert real["audio_sample_count"] == 8 * 320 + 5
    assert real["audio_geometry_policy"] == "exact_or_subframe_audio_tail_ignored_v1"


def test_v11_canonical_accepts_signed_real_train_full_source_manual_truth(
    tmp_path: Path,
) -> None:
    synthetic_path, real_train_path, source_path, partition_path, verdict_path = (
        _canonical_fixture(tmp_path)
    )
    source_rows = [
        json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()
    ]
    partition_rows = [
        json.loads(line)
        for line in partition_path.read_text(encoding="utf-8").splitlines()
    ]
    source_id = "video-train-manual-w00"
    video_id = "video-train-manual"
    audio = tmp_path / f"{source_id}.wav"
    _write_wav(audio, 8)
    source_rows.append(
        {
            "schema": "joint_boundary_omni_source_window_v1",
            "window_id": source_id,
            "video_id": video_id,
            "audio_wav": str(audio),
            "audio_wav_sha256": _sha256(audio),
            "duration_s": 0.16,
        }
    )
    partition_rows.append(
        {
            "schema": PARTITION_SCHEMA,
            "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
            "source_id": source_id,
            "video_id": video_id,
            "partition": "train",
            "original_dataset_role": "train",
        }
    )
    _write_jsonl(source_path, source_rows)
    _write_jsonl(partition_path, partition_rows)
    real_train_manual = tmp_path / "real_train_manual_sources.jsonl"
    _write_jsonl(
        real_train_manual,
        [
            {
                "schema": "candidate_island_scorer_v11_real_train_manual_source_v1",
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "video_id": video_id,
                "partition": "train",
                "input_distribution": "real_workflow_source_window_human_full_source_v1",
                "audio": str(audio),
                "audio_sha256": _sha256(audio),
                "frame_count": 8,
                "frame_hop_s": 0.02,
                "core_ids": [f"real-train-manual-source::{source_id}"],
                "canonical_spans": [
                    {"label": "outside_candidate", "start_frame": 0, "end_frame": 2},
                    {"label": "inside_candidate", "start_frame": 2, "end_frame": 7},
                    {"label": "unsure", "start_frame": 7, "end_frame": 8},
                ],
                "annotation_provenance": "human_full_source_review",
                "teacher_output_used_as_truth": False,
                "unselected_source_label_inheritance": False,
                "unsure_training_label": -100,
                "reviewed_full_source": True,
                "training_manifest_allowed": True,
            }
        ],
    )

    summary = compile_canonical(
        synthetic_train_sources=synthetic_path,
        real_train_outside_sources=real_train_path,
        real_train_manual_sources=real_train_manual,
        source_windows=source_path,
        partition_manifest=partition_path,
        manual_verdicts=[verdict_path],
        output_dir=tmp_path / "manual-canonical",
    )
    assert summary["real_train_manual_source_count"] == 1
    assert summary["real_train_manual_inside_frames"] == 5
    assert summary["real_train_full_source_human_confirmed"] is True
    rows = [
        json.loads(line)
        for line in Path(summary["canonical_sources"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    manual = next(row for row in rows if row["source_id"] == source_id)
    assert manual["source_kind"] == "real_train_full_source_manual"
    assert manual["unsure_training_label"] == -100

def test_v11_feature_compile_and_random_init_cpu_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    synthetic_path, real_train_path, source_path, partition_path, verdict_path = _canonical_fixture(tmp_path)
    canonical_summary = compile_canonical(
        synthetic_train_sources=synthetic_path,
        real_train_outside_sources=real_train_path,
        source_windows=source_path,
        partition_manifest=partition_path,
        manual_verdicts=[verdict_path],
        output_dir=tmp_path / "canonical",
    )
    canonical_path = Path(canonical_summary["canonical_sources"])
    if not canonical_path.is_absolute():
        canonical_path = Path.cwd() / canonical_path
    canonical_rows = [json.loads(line) for line in canonical_path.read_text(encoding="utf-8").splitlines()]
    raw_rows = []
    rng = np.random.default_rng(117)
    for row in canonical_rows:
        feature_path = tmp_path / f"{row['source_id']}.npz"
        np.savez(
            feature_path,
            ptm=rng.normal(size=(8, 2048)).astype(np.float32),
            mfcc=rng.normal(size=(8, 40)).astype(np.float32),
            frame_hop_s=np.asarray([0.02], dtype=np.float32),
        )
        raw_rows.append(
            {
                "schema": CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": row["source_id"],
                "partition": row["partition"],
                "feature_extractor_schema": CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA,
                "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
                "feature_path": str(feature_path),
                "feature_sha256": _sha256(feature_path),
                "frame_count": 8,
                "frame_hop_s": 0.02,
                "ptm_dim": 2048,
                "mfcc_dim": 40,
            }
        )
    raw_manifest = tmp_path / "raw_features.jsonl"
    _write_jsonl(raw_manifest, raw_rows)
    feature_summary = compile_features(
        canonical_sources=canonical_path,
        raw_feature_manifest=raw_manifest,
        output_dir=tmp_path / "features",
    )
    assert feature_summary["partition_window_counts"] == {"test": 1, "train": 2, "val": 1}
    assert feature_summary["partition_supervised_window_counts"] == {
        "test": 1,
        "train": 2,
        "val": 1,
    }
    assert feature_summary["partition_ignored_only_window_counts"] == {
        "test": 0,
        "train": 0,
        "val": 0,
    }
    label_path = tmp_path / "features" / "source_labels" / "synthetic-train.npz"
    with np.load(label_path, allow_pickle=False) as labels:
        assert labels["training_labels"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
        assert labels["boundary_valid"].tolist()[5] is True
    real_label_path = tmp_path / "features" / "source_labels" / "video-train-w00.npz"
    with np.load(real_label_path, allow_pickle=False) as labels:
        assert labels["training_labels"].tolist() == [0, 0, 0, -100, -100, -100, -100, -100]

    args = argparse.Namespace(
        dataset_manifest=feature_summary["dataset_manifest"],
        feature_cache_gate=feature_summary["feature_cache_gate"],
        output_dir=str(tmp_path / "training"),
        variant="baseline",
        capacity_profile=CANDIDATE_ISLAND_SCORER_V11_COMPACT_CAPACITY_PROFILE,
        heatmap_weight=0.0,
        class_weight_outside=1.0,
        class_weight_inside=1.0,
        device="cpu",
        smoke=True,
        seed=117,
        epochs=1,
        max_steps=1,
        max_padded_frames=32,
        source_cache_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
    )
    result = train(args)
    assert result["training_steps"] == 1
    assert result["numeric_gate_pass"] is False
    assert result["promotion_allowed"] is False
    progress = json.loads(
        (tmp_path / "training" / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["schema"] == "candidate_island_scorer_v11_training_progress_v1"
    assert progress["status"] == "completed"
    assert progress["step"] == progress["total_steps"] == 1
    assert progress["checkpoint_sha256"] == result["checkpoint_sha256"]
    assert progress["metrics"] == result["metrics"]
    checkpoint = Path(result["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    bundle = load_speech_island_scorer_checkpoint(checkpoint, device="cpu")
    assert bundle.schema == CANDIDATE_ISLAND_SCORER_V11_COMPACT_SCHEMA
    assert bundle.model_config["capacity_profile"] == (
        CANDIDATE_ISLAND_SCORER_V11_COMPACT_CAPACITY_PROFILE
    )
    assert bundle.metadata["training_initialization"] == "random"


def test_v11_feature_compile_rejects_projected_ptm128(tmp_path: Path) -> None:
    synthetic_path, real_train_path, source_path, partition_path, verdict_path = _canonical_fixture(tmp_path)
    canonical_summary = compile_canonical(
        synthetic_train_sources=synthetic_path,
        real_train_outside_sources=real_train_path,
        source_windows=source_path,
        partition_manifest=partition_path,
        manual_verdicts=[verdict_path],
        output_dir=tmp_path / "canonical",
    )
    canonical_path = Path(canonical_summary["canonical_sources"])
    if not canonical_path.is_absolute():
        canonical_path = Path.cwd() / canonical_path
    rows = []
    for canonical in [json.loads(line) for line in canonical_path.read_text(encoding="utf-8").splitlines()]:
        feature_path = tmp_path / f"compact-{canonical['source_id']}.npz"
        np.savez(
            feature_path,
            ptm=np.zeros((8, 128), dtype=np.float32),
            mfcc=np.zeros((8, 40), dtype=np.float32),
            frame_hop_s=np.asarray([0.02], dtype=np.float32),
        )
        rows.append(
            {
                "schema": CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": canonical["source_id"],
                "partition": canonical["partition"],
                "feature_extractor_schema": CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA,
                "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
                "feature_path": str(feature_path),
                "feature_sha256": _sha256(feature_path),
                "frame_count": 8,
                "frame_hop_s": 0.02,
                "ptm_dim": 128,
                "mfcc_dim": 40,
            }
        )
    raw_manifest = tmp_path / "compact.jsonl"
    _write_jsonl(raw_manifest, rows)
    with pytest.raises(ValueError, match="projected/truncated PTM"):
        compile_features(
            canonical_sources=canonical_path,
            raw_feature_manifest=raw_manifest,
            output_dir=tmp_path / "features",
        )


def test_v11_training_resolves_bare_cuda_to_current_device(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert str(_resolve_training_device("cuda", torch)) == "cuda:3"
    assert str(_resolve_training_device("cuda:1", torch)) == "cuda:1"
    assert str(_resolve_training_device("cpu", torch)) == "cpu"


def test_v11_cuda_warmup_uses_longest_budgeted_batch() -> None:
    rows = [
        {"row_id": "short", "window_start_frame": 0, "window_end_frame": 300},
        {"row_id": "long-a", "window_start_frame": 0, "window_end_frame": 900},
        {"row_id": "medium", "window_start_frame": 0, "window_end_frame": 600},
        {"row_id": "long-b", "window_start_frame": 0, "window_end_frame": 850},
    ]

    selected = _cuda_warmup_rows(rows, max_padded_frames=2000)

    assert [row["row_id"] for row in selected] == ["long-a", "long-b"]
    assert max(row["window_end_frame"] for row in selected) * len(selected) <= 2000


def test_v11_actual_model_warmup_restores_fresh_adamw_step_zero() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(117)
    warmed = torch.nn.Linear(4, 2)
    initial_state = {
        key: value.detach().clone() for key, value in warmed.state_dict().items()
    }
    fresh = torch.nn.Linear(4, 2)
    fresh.load_state_dict(initial_state)
    warmed_optimizer = torch.optim.AdamW(warmed.parameters(), lr=1e-3, weight_decay=1e-4)
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_input = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    warmup_loss = warmed(warmup_input).square().mean()
    warmup_loss.backward()
    warmed_optimizer.step()

    _restore_model_and_adamw_after_warmup(warmed, warmed_optimizer, initial_state)

    assert all(
        bool(torch.count_nonzero(value) == 0)
        for state in warmed_optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    actual_input = torch.linspace(-1.0, 1.0, 20).reshape(5, 4)
    for model, optimizer in (
        (warmed, warmed_optimizer),
        (fresh, fresh_optimizer),
    ):
        optimizer.zero_grad(set_to_none=True)
        loss = model(actual_input).square().mean()
        loss.backward()
        optimizer.step()
    for warmed_parameter, fresh_parameter in zip(
        warmed.parameters(), fresh.parameters(), strict=True
    ):
        torch.testing.assert_close(warmed_parameter, fresh_parameter, rtol=0.0, atol=0.0)


def test_v11_planned_batches_match_legacy_seeded_epoch_order() -> None:
    rows = [
        {"row_id": f"row-{index}", "window_start_frame": 0, "window_end_frame": length}
        for index, length in enumerate((300, 900, 600, 850, 500, 700))
    ]
    legacy_rng = random.Random(117)
    expected = []
    for _epoch in range(3):
        shuffled = list(rows)
        legacy_rng.shuffle(shuffled)
        expected.append(_pack_batches(shuffled, max_padded_frames=2000))

    planned = _plan_training_batches(
        rows, epochs=3, max_padded_frames=2000, seed=117
    )

    assert [
        [[row["row_id"] for row in batch] for batch in epoch]
        for epoch in planned
    ] == [
        [[row["row_id"] for row in batch] for batch in epoch]
        for epoch in expected
    ]


def test_v11_planned_batches_skip_unsure_only_owner_windows_after_shuffle() -> None:
    rows = [
        {
            "row_id": f"row-{index}",
            "window_start_frame": 0,
            "window_end_frame": length,
            "definite_owner_frame_count": definite,
        }
        for index, (length, definite) in enumerate(
            ((300, 20), (900, 0), (600, 10), (850, 0), (500, 4), (700, 2))
        )
    ]
    legacy_rng = random.Random(117)
    expected = []
    for _epoch in range(3):
        shuffled = list(rows)
        legacy_rng.shuffle(shuffled)
        supervised = [row for row in shuffled if row["definite_owner_frame_count"] > 0]
        expected.append(_pack_batches(supervised, max_padded_frames=2000))

    planned = _plan_training_batches(
        rows, epochs=3, max_padded_frames=2000, seed=117
    )

    assert [
        [[row["row_id"] for row in batch] for batch in epoch]
        for epoch in planned
    ] == [
        [[row["row_id"] for row in batch] for batch in epoch]
        for epoch in expected
    ]
    assert all(
        row["definite_owner_frame_count"] > 0
        for epoch in planned
        for batch in epoch
        for row in batch
    )
