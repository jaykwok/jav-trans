from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.ja.model import (
    CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_SCHEMA,
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
from tools.boundary.ja.train_candidate_island_scorer_v11 import run as train


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
    source_path = tmp_path / "source_windows.jsonl"
    partition_path = tmp_path / "partition_manifest.jsonl"
    verdict_path = tmp_path / "manual_verdicts.jsonl"
    _write_jsonl(source_path, source_rows)
    _write_jsonl(partition_path, partition_rows)
    _write_jsonl(verdict_path, verdict_rows)
    return source_path, partition_path, verdict_path


def test_v11_canonical_requires_all_human_full_source_truth(tmp_path: Path) -> None:
    source_path, partition_path, verdict_path = _canonical_fixture(tmp_path)
    partial = tmp_path / "partial.jsonl"
    partial.write_text(verdict_path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="full-source truth for every frozen source"):
        compile_canonical(
            source_windows=source_path,
            partition_manifest=partition_path,
            manual_verdicts=[partial],
            output_dir=tmp_path / "partial-output",
        )

    omni_source, omni_partition, omni_verdict = _canonical_fixture(
        tmp_path / "omni", provenance="omni:qwen3.5-omni-plus"
    )
    with pytest.raises(ValueError, match="Omni-only"):
        compile_canonical(
            source_windows=omni_source,
            partition_manifest=omni_partition,
            manual_verdicts=[omni_verdict],
            output_dir=tmp_path / "omni-output",
        )


def test_v11_feature_compile_and_random_init_cpu_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    source_path, partition_path, verdict_path = _canonical_fixture(tmp_path)
    canonical_summary = compile_canonical(
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
    assert feature_summary["partition_window_counts"] == {"test": 1, "train": 1, "val": 1}
    label_path = tmp_path / "features" / "source_labels" / "video-train-w00.npz"
    with np.load(label_path, allow_pickle=False) as labels:
        assert labels["training_labels"].tolist() == [0, 0, 1, 1, 1, -100, 0, 0]
        assert labels["boundary_valid"].tolist()[5] is False

    args = argparse.Namespace(
        dataset_manifest=feature_summary["dataset_manifest"],
        feature_cache_gate=feature_summary["feature_cache_gate"],
        output_dir=str(tmp_path / "training"),
        variant="baseline",
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
        hidden_size=8,
        num_layers=1,
        state_size=4,
        num_heads=2,
        head_dim=8,
        n_groups=1,
        conv_kernel=2,
        chunk_size=2,
    )
    result = train(args)
    assert result["training_steps"] == 1
    assert result["numeric_gate_pass"] is False
    assert result["promotion_allowed"] is False
    checkpoint = Path(result["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    bundle = load_speech_island_scorer_checkpoint(checkpoint, device="cpu")
    assert bundle.schema == CANDIDATE_ISLAND_SCORER_V11_SCHEMA
    assert bundle.metadata["training_initialization"] == "random"


def test_v11_feature_compile_rejects_projected_ptm128(tmp_path: Path) -> None:
    source_path, partition_path, verdict_path = _canonical_fixture(tmp_path)
    canonical_summary = compile_canonical(
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
