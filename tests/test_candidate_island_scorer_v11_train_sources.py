from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import wave

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.ja.model import (
    CANDIDATE_ISLAND_SCORER_V11_SYNTHETIC_TRAIN_SOURCE_SCHEMA,
)
from tools.boundary.ja.build_candidate_island_scorer_v11_train_sources import build


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_wav(path: Path, samples: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * samples)


def _args(
    *, source_manifest: Path, background_inventory: Path, partition: Path,
    outside_consensus: Path, output: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        source_manifest=str(source_manifest),
        background_inventory=str(background_inventory),
        heldout_partition_manifest=str(partition),
        outside_consensus_manifest=str(outside_consensus),
        output_dir=str(output),
        vocal_source_count=1,
        outside_control_count=0,
        max_semantic_sources=None,
        seed=117,
    )


def test_build_v11_train_sources_preserves_candidate_and_brackets_outside(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.wav"
    core_a = tmp_path / "core-a.wav"
    core_b = tmp_path / "core-b.wav"
    outside = tmp_path / "outside.wav"
    vocal = tmp_path / "vocal.wav"
    _write_wav(candidate, 1600)
    _write_wav(core_a, 320)
    _write_wav(core_b, 320)
    _write_wav(outside, 640)
    _write_wav(vocal, 640)
    source_manifest = tmp_path / "composites.jsonl"
    _write_jsonl(
        source_manifest,
        [
            {
                "schema": "cueqc_v13_unique_core_composite_v1",
                "sample_id": "sample-1",
                "audio": str(candidate),
                "sample_rate": 16000,
                "sample_count": 1600,
                "source_partition": "train",
                "core_spans": [
                    {
                        "core_id": "core-a",
                        "source_audio": str(core_a),
                        "start_sample": 0,
                        "end_sample": 320,
                    },
                    {
                        "core_id": "core-b",
                        "source_audio": str(core_b),
                        "start_sample": 1280,
                        "end_sample": 1600,
                    },
                ],
                "inter_unit_gaps": {
                    "left_start_sample": 320,
                    "left_end_sample": 640,
                    "right_start_sample": 960,
                    "right_end_sample": 1280,
                    "sources": [
                        {
                            "audio_id": "preasr-train-video-w00-chunk00001",
                            "audio": str(outside),
                            "background_type": "noise",
                        },
                        {
                            "audio_id": "preasr-train-video-w00-chunk00002",
                            "audio": str(outside),
                            "background_type": "noise",
                        },
                    ],
                },
                "negative_unit_span": {
                    "start_sample": 640,
                    "end_sample": 960,
                    "source": {
                        "audio_id": "preasr-train-video-w00-chunk00003",
                        "audio": str(vocal),
                        "background_type": "breathing",
                    },
                },
                "additive_overlay": {
                    "source": {
                        "audio_id": "preasr-train-video-w00-chunk00004",
                        "audio": str(outside),
                        "background_type": "noise",
                    },
                    "mix": {"target_snr_db": 4.0},
                },
            }
        ],
    )
    background_inventory = tmp_path / "background.jsonl"
    common = {
        "schema": "speech_scorer_v10_canonical_source_v1",
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "partition": "train",
        "row_role": "all_background",
        "background_source_video_ids": ["train-background-video"],
    }
    _write_jsonl(
        background_inventory,
        [
            {
                **common,
                "source_id": "outside-1",
                "audio": str(outside),
                "background_type": "noise",
                "omni_flags": ["noise"],
            },
            {
                **common,
                "source_id": "vocal-1",
                "audio": str(vocal),
                "background_type": "breathing",
                "omni_flags": ["breathing"],
            },
        ],
    )
    partition = tmp_path / "partition.jsonl"
    _write_jsonl(
        partition,
        [
            {
                "schema": "candidate_island_scorer_v11_partition_manifest_v1",
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": "heldout-w00",
                "video_id": "heldout",
                "partition": "val",
            }
        ],
    )
    outside_consensus = tmp_path / "outside-consensus.jsonl"
    _write_jsonl(
        outside_consensus,
        [
            {
                "schema": "candidate_island_scorer_v11_outside_consensus_v1",
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": "outside-1",
                "decision": "clear_outside",
                "training_label": 0,
                "training_manifest_allowed": True,
            }
        ],
    )

    summary = build(
        _args(
            source_manifest=source_manifest,
            background_inventory=background_inventory,
            partition=partition,
            outside_consensus=outside_consensus,
            output=tmp_path / "output",
        )
    )
    manifest = Path(summary["synthetic_train_sources"])
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert summary["source_kind_counts"] == {
        "isolated_human_vocal_candidate": 1,
        "semantic_composite_candidate": 1,
    }
    assert summary["overlay_counts"] == {"disabled_repeated_overlay": 1}
    assert all(
        row["schema"] == CANDIDATE_ISLAND_SCORER_V11_SYNTHETIC_TRAIN_SOURCE_SCHEMA
        for row in rows
    )
    semantic = next(row for row in rows if row["source_kind"] == "semantic_composite_candidate")
    assert semantic["core_ids"] == ["core-a", "core-b"]
    assert [span["label"] for span in semantic["canonical_spans"]] == [
        "outside_candidate",
        "inside_candidate",
        "outside_candidate",
    ]
    inside = semantic["canonical_spans"][1]
    assert inside["end_frame"] - inside["start_frame"] == 5
    assert semantic["outside_brackets"]["left"]["crop_policy"] == (
        "natural_contiguous_crop_no_repeat_v1"
    )
    assert semantic["outside_brackets"]["left"]["output_sample_count"] == 640
    assert semantic["candidate_source"]["overlay"] is None
    assert semantic["candidate_source"]["overlay_policy"] == (
        "disabled_repeated_full_candidate_overlay_v1"
    )


def test_build_v11_train_sources_replaces_heldout_background_identity(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.wav"
    core_a = tmp_path / "core-a.wav"
    core_b = tmp_path / "core-b.wav"
    outside = tmp_path / "outside.wav"
    vocal = tmp_path / "vocal.wav"
    heldout_component = tmp_path / "heldout-component.wav"
    for path in (candidate, outside, vocal, heldout_component):
        _write_wav(path, 32000)
    _write_wav(core_a, 8000)
    _write_wav(core_b, 8000)
    source_manifest = tmp_path / "composites.jsonl"
    _write_jsonl(
        source_manifest,
        [
            {
                "schema": "cueqc_v13_unique_core_composite_v1",
                "sample_id": "sample-1",
                "audio": str(candidate),
                "sample_count": 32000,
                "source_partition": "train",
                "core_spans": [
                    {
                        "core_id": "core-a",
                        "source_audio": str(core_a),
                        "start_sample": 0,
                        "end_sample": 8000,
                    },
                    {
                        "core_id": "core-b",
                        "source_audio": str(core_b),
                        "start_sample": 24000,
                        "end_sample": 32000,
                    },
                ],
                "inter_unit_gaps": {
                    "left_start_sample": 8000,
                    "left_end_sample": 12000,
                    "right_start_sample": 20000,
                    "right_end_sample": 24000,
                    "sources": [
                        {
                            "audio_id": "preasr-heldout-w00-chunk00001",
                            "audio": str(heldout_component),
                            "background_type": "noise",
                        },
                        {
                            "audio_id": "preasr-train-only-w00-chunk00001",
                            "audio": str(outside),
                            "background_type": "noise",
                        },
                    ],
                },
                "negative_unit_span": {
                    "start_sample": 12000,
                    "end_sample": 20000,
                    "source": {
                        "audio_id": "preasr-train-only-w00-chunk00002",
                        "audio": str(vocal),
                        "background_type": "breathing",
                    },
                },
                "additive_overlay": None,
            }
        ],
    )
    background_inventory = tmp_path / "background.jsonl"
    base = {
        "schema": "speech_scorer_v10_canonical_source_v1",
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "partition": "train",
        "row_role": "all_background",
        "background_source_video_ids": ["train-only"],
    }
    _write_jsonl(
        background_inventory,
        [
            {**base, "source_id": "outside", "audio": str(outside), "background_type": "noise"},
            {**base, "source_id": "vocal", "audio": str(vocal), "background_type": "breathing"},
        ],
    )
    partition = tmp_path / "partition.jsonl"
    _write_jsonl(
        partition,
        [
            {
                "schema": "candidate_island_scorer_v11_partition_manifest_v1",
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": "heldout-w00",
                "video_id": "heldout",
                "partition": "test",
            }
        ],
    )
    outside_consensus = tmp_path / "outside-consensus.jsonl"
    _write_jsonl(
        outside_consensus,
        [
            {
                "schema": "candidate_island_scorer_v11_outside_consensus_v1",
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": "outside",
                "decision": "clear_outside",
                "training_label": 0,
                "training_manifest_allowed": True,
            }
        ],
    )

    summary = build(
        _args(
            source_manifest=source_manifest,
            background_inventory=background_inventory,
            partition=partition,
            outside_consensus=outside_consensus,
            output=tmp_path / "output",
        )
    )
    assert summary["heldout_component_replacement_count"] == 1
    rows = [
        json.loads(line)
        for line in Path(summary["synthetic_train_sources"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    semantic = next(row for row in rows if row["source_kind"] == "semantic_composite_candidate")
    left = semantic["candidate_source"]["internal_components"]["left_gap"]
    assert left["heldout_component_replaced"] is True
    assert "heldout" not in left["selected"]["audio_id"]
