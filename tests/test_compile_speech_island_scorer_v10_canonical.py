from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.boundary.ja.compile_speech_island_scorer_v10_canonical import (
    CANONICAL_LABELS,
    canonical_frame_labels,
    finalize_dataset,
    prepare_dataset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _audio(path: Path, *, samples: int = 1600) -> None:
    sf.write(path, np.zeros(samples, dtype=np.float32), 16000)


def _negative(tmp_path: Path, partition: str) -> dict:
    identity = f"preasr-{partition}-w00-chunk00000"
    audio = tmp_path / f"{identity}.wav"
    _audio(audio)
    return {
        "audio_id": identity,
        "audio": str(audio),
        "duration_s": 0.1,
        "sample_rate": 16000,
        "source": "omni_definite_drop",
        "source_partition": partition,
        "video_id": f"video-{partition}",
        "window_id": f"video-{partition}-w00",
        "omni_confidence": 0.95,
        "omni_flags": ["breathing"],
        "background_type": "breathing",
    }


def _speech(tmp_path: Path, partition: str, negative: dict) -> dict:
    source_id = f"speech-{partition}"
    audio = tmp_path / f"{source_id}.wav"
    _audio(audio)
    detail = {
        "audio_id": negative["audio_id"],
        "audio": negative["audio"],
        "background_type": "breathing",
        "omni_flags": ["breathing"],
    }
    return {
        "schema": "cueqc_v13_unique_core_composite_v1",
        "sample_id": source_id,
        "audio": str(audio),
        "sample_rate": 16000,
        "sample_count": 1600,
        "duration_s": 0.1,
        "source_partition": partition,
        "core_spans": [
            {
                "core_id": f"core-{partition}-a",
                "start_sample": 0,
                "end_sample": 640,
            },
            {
                "core_id": f"core-{partition}-b",
                "start_sample": 960,
                "end_sample": 1600,
            },
        ],
        "negative_unit_span": {
            "start_sample": 700,
            "end_sample": 900,
            "source": detail,
        },
        "inter_unit_gaps": {
            "left_start_sample": 640,
            "left_end_sample": 700,
            "right_start_sample": 900,
            "right_end_sample": 960,
            "sources": [detail, detail],
        },
        "additive_overlay": None,
        "label_contract": "new_runtime_chunk_intersection_with_exact_unique_semantic_core_spans_v1",
    }


def _prepare_fixture(tmp_path: Path) -> tuple[dict, Path]:
    negatives = [_negative(tmp_path, partition) for partition in ("train", "val", "test")]
    speech = [
        _speech(tmp_path, partition, negative)
        for partition, negative in zip(("train", "val", "test"), negatives, strict=True)
    ]
    speech_manifest = tmp_path / "speech.jsonl"
    negative_manifest = tmp_path / "negative.jsonl"
    _write_jsonl(speech_manifest, speech)
    _write_jsonl(negative_manifest, negatives)
    output = tmp_path / "prepared"
    summary = prepare_dataset(
        speech_manifest=speech_manifest,
        negative_manifest=negative_manifest,
        output_dir=output,
    )
    return summary, output


def test_prepare_uses_galgame_core_and_cueqc_drop_as_binary_supervision(
    tmp_path: Path,
) -> None:
    summary, output = _prepare_fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in (output / "canonical_sources.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["dataset"]["source_count"] == 6
    assert summary["dataset"]["core_count"] == 6
    assert summary["dataset"]["max_core_use_count"] == 1
    assert summary["training_ready"] is False
    assert {row["row_role"] for row in rows} == {"speech", "all_background"}
    speech = next(row for row in rows if row["source_id"] == "speech-train")
    assert [span["label"] for span in speech["canonical_spans"]] == [
        "speech",
        "background",
        "background",
        "background",
        "speech",
    ]
    assert speech["boundary_serialization_contract_id"] == (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    )


def test_exact_mixed_frame_maps_to_unsure_without_a_boundary_band() -> None:
    source = {
        "source_id": "mixed",
        "sample_rate": 1000,
        "sample_count": 60,
        "canonical_spans": [
            {"start_sample": 0, "end_sample": 25, "label": "speech"},
            {"start_sample": 25, "end_sample": 60, "label": "background"},
        ],
    }
    labels = canonical_frame_labels(source, frame_hop_s=0.02)
    assert labels.tolist() == [
        CANONICAL_LABELS["speech"],
        CANONICAL_LABELS["unsure"],
        CANONICAL_LABELS["background"],
    ]


def test_prepare_rejects_background_video_partition_leakage(tmp_path: Path) -> None:
    negatives = [_negative(tmp_path, partition) for partition in ("train", "val", "test")]
    negatives[1]["video_id"] = negatives[0]["video_id"]
    negative_manifest = tmp_path / "negative.jsonl"
    speech_manifest = tmp_path / "speech.jsonl"
    _write_jsonl(negative_manifest, negatives)
    _write_jsonl(
        speech_manifest,
        [
            _speech(tmp_path, partition, negative)
            for partition, negative in zip(
                ("train", "val", "test"), negatives, strict=True
            )
        ],
    )
    with pytest.raises(ValueError, match="video identity crosses"):
        prepare_dataset(
            speech_manifest=speech_manifest,
            negative_manifest=negative_manifest,
            output_dir=tmp_path / "out",
        )


def test_finalize_requires_raw_17b_features_and_replays_trainer_contract(
    tmp_path: Path,
) -> None:
    _summary, prepared = _prepare_fixture(tmp_path)
    sources = [
        json.loads(line)
        for line in (prepared / "canonical_sources.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    feature_rows = []
    for source in sources:
        frame_count = 5
        feature_path = tmp_path / f"{source['source_id']}.npz"
        np.savez_compressed(
            feature_path,
            ptm=np.zeros((frame_count, 2048), dtype=np.float32),
            mfcc=np.zeros((frame_count, 40), dtype=np.float32),
        )
        feature_rows.append(
            {
                "audio_id": source["source_id"],
                "feature_path": str(feature_path),
                "duration_s": source["duration_s"],
                "frame_hop_s": 0.02,
                "frame_count": frame_count,
                "ptm_dim": 2048,
                "mfcc_dim": 40,
                "ptm": QWEN_ASR_17B_REPO_ID,
            }
        )
    feature_manifest = tmp_path / "features.jsonl"
    _write_jsonl(feature_manifest, feature_rows)
    gate_summary = tmp_path / "manual_gate.json"
    gate_summary.write_text(
        json.dumps(
            {
                "schema": "speech_scorer_v10_canonical_manual_gate_v1",
                "canonical_sources_sha256": hashlib.sha256(
                    (prepared / "canonical_sources.jsonl").read_bytes()
                ).hexdigest(),
                "manual_gate_pass": True,
            }
        ),
        encoding="utf-8",
    )
    summary = finalize_dataset(
        canonical_sources=prepared / "canonical_sources.jsonl",
        feature_manifest=feature_manifest,
        manual_gate_summary=gate_summary,
        output_dir=tmp_path / "final",
    )
    assert summary["training_ready"] is True
    assert summary["promotion_ready"] is False
    assert summary["trainer_dataset"]["source_count"] == 6
    assert all(
        counts == {"speech_rows": 1, "all_background_rows": 1}
        for counts in summary["partition_label_presence"].values()
    )

    feature_rows[0]["ptm"] = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"
    _write_jsonl(feature_manifest, feature_rows)
    with pytest.raises(ValueError, match="1.7B PTM"):
        finalize_dataset(
            canonical_sources=prepared / "canonical_sources.jsonl",
            feature_manifest=feature_manifest,
            manual_gate_summary=gate_summary,
            output_dir=tmp_path / "rejected",
        )


def test_finalize_rejects_pending_manual_gate(tmp_path: Path) -> None:
    _summary, prepared = _prepare_fixture(tmp_path)
    gate_summary = tmp_path / "manual_gate.json"
    gate_summary.write_text(
        json.dumps(
            {
                "schema": "speech_scorer_v10_canonical_manual_gate_v1",
                "canonical_sources_sha256": hashlib.sha256(
                    (prepared / "canonical_sources.jsonl").read_bytes()
                ).hexdigest(),
                "manual_gate_pass": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="has not passed"):
        finalize_dataset(
            canonical_sources=prepared / "canonical_sources.jsonl",
            feature_manifest=tmp_path / "not-read.jsonl",
            manual_gate_summary=gate_summary,
            output_dir=tmp_path / "rejected",
        )
