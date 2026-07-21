from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.audits.evaluate_scorer_v10_prediction_audit import evaluate as evaluate_prediction
from tools.audits.generate_scorer_v10_prediction_audit_html import (
    SUMMARY_SCHEMA as PREDICTION_SUMMARY_SCHEMA,
    VERDICT_SCHEMA as PREDICTION_VERDICT_SCHEMA,
)
from tools.boundary.ja.apply_speech_island_scorer_v10_canonical_r4_repairs import (
    PLACEMENT_SCHEMA,
    SUMMARY_SCHEMA,
    apply_repairs,
    map_event_to_rendered_audio,
)
from tools.boundary.ja.compile_speech_island_scorer_v10_canonical import (
    canonical_frame_labels,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _source(
    source_id: str,
    partition: str,
    *,
    role: str,
    sample_count: int,
    spans: list[dict],
    core_ids: list[str] | None = None,
    background_id: str = "",
    background_source_ids: list[str] | None = None,
    audio: str = "",
) -> dict:
    return {
        "schema": "speech_scorer_v10_canonical_source_v1",
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": "speech_scorer_canonical_frames_v1",
        "source_id": source_id,
        "audio": audio or f"{source_id}.wav",
        "row_role": role,
        "partition": partition,
        "core_ids": core_ids or [],
        "background_id": background_id,
        "background_source_ids": background_source_ids or [],
        "background_source_video_ids": [f"video-{partition}"],
        "sample_rate": 16000,
        "sample_count": sample_count,
        "duration_s": sample_count / 16000,
        "input_distribution": "full_source_windows",
        "canonical_spans": spans,
    }


def test_map_source_event_handles_crop_and_tile() -> None:
    assert map_event_to_rendered_audio(
        event_start_sample=100,
        event_end_sample=300,
        source_sample_count=1000,
        source_offset_sample=200,
        rendered_start_sample=5000,
        rendered_end_sample=5400,
    ) == [
        {
            "tile_index": 0,
            "source_start_sample": 200,
            "source_end_sample": 300,
            "mapped_start_sample": 5000,
            "mapped_end_sample": 5100,
        }
    ]
    assert map_event_to_rendered_audio(
        event_start_sample=100,
        event_end_sample=300,
        source_sample_count=500,
        source_offset_sample=0,
        rendered_start_sample=1000,
        rendered_end_sample=2200,
    ) == [
        {
            "tile_index": 0,
            "source_start_sample": 100,
            "source_end_sample": 300,
            "mapped_start_sample": 1100,
            "mapped_end_sample": 1300,
        },
        {
            "tile_index": 1,
            "source_start_sample": 100,
            "source_end_sample": 300,
            "mapped_start_sample": 1600,
            "mapped_end_sample": 1800,
        },
        {
            "tile_index": 2,
            "source_start_sample": 100,
            "source_end_sample": 200,
            "mapped_start_sample": 2100,
            "mapped_end_sample": 2200,
        },
    ]


def test_apply_r4_repairs_preserves_identity_and_propagates_tiled_gap(
    tmp_path: Path,
) -> None:
    affected_audio = tmp_path / "affected.wav"
    affected_audio.write_bytes(b"affected-audio")
    rows = [
        _source(
            "val-composite",
            "val",
            role="speech",
            sample_count=1920,
            core_ids=["val-core-a", "val-core-b"],
            background_source_ids=["affected-bg"],
            spans=[
                {"start_sample": 0, "end_sample": 640, "label": "speech", "core_id": "val-core-a"},
                {"start_sample": 640, "end_sample": 1280, "label": "background", "background_id": "affected-bg"},
                {"start_sample": 1280, "end_sample": 1920, "label": "speech", "core_id": "val-core-b"},
            ],
        ),
        _source(
            "affected-control",
            "val",
            role="all_background",
            sample_count=500,
            background_id="affected-bg",
            background_source_ids=["affected-bg"],
            audio=str(affected_audio),
            spans=[
                {"start_sample": 0, "end_sample": 500, "label": "background", "background_id": "affected-bg"}
            ],
        ),
        _source(
            "val-background",
            "val",
            role="all_background",
            sample_count=320,
            background_id="val-bg",
            background_source_ids=["val-bg"],
            spans=[{"start_sample": 0, "end_sample": 320, "label": "background", "background_id": "val-bg"}],
        ),
    ]
    for partition in ("train", "test"):
        rows.extend(
            [
                _source(
                    f"{partition}-speech",
                    partition,
                    role="speech",
                    sample_count=320,
                    core_ids=[f"{partition}-core"],
                    spans=[{"start_sample": 0, "end_sample": 320, "label": "speech", "core_id": f"{partition}-core"}],
                ),
                _source(
                    f"{partition}-background",
                    partition,
                    role="all_background",
                    sample_count=320,
                    background_id=f"{partition}-bg",
                    background_source_ids=[f"{partition}-bg"],
                    spans=[{"start_sample": 0, "end_sample": 320, "label": "background", "background_id": f"{partition}-bg"}],
                ),
            ]
        )
    canonical = tmp_path / "canonical.jsonl"
    _write_jsonl(canonical, rows)

    audit_dir = tmp_path / "prediction_audit"
    audit_dir.mkdir()
    audit_id = "speech_deletion:val-composite"
    prediction_row = {
        "audit_id": audit_id,
        "source_id": "val-composite",
        "partition": "val",
        "row_role": "speech",
        "category": "speech_deletion",
        "frame_count": 6,
        "truth_spans": [{"label": "truth_speech", "start_frame": 0, "end_frame": 2, "start_s": 0.0, "end_s": 0.04}],
        "prediction_spans": [{"label": "model_speech", "start_frame": 1, "end_frame": 2, "start_s": 0.02, "end_s": 0.04}],
        "truth_drop_spans": [{"label": "truth_speech_model_background", "start_frame": 0, "end_frame": 1, "start_s": 0.0, "end_s": 0.02}],
    }
    _write_jsonl(audit_dir / "audit_manifest.jsonl", [prediction_row])
    (audit_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": PREDICTION_SUMMARY_SCHEMA,
                "review_item_count": 1,
                "category_counts": {"speech_deletion": 1},
                "audit_manifest": str(audit_dir / "audit_manifest.jsonl"),
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        audit_dir / "manual_verdicts.jsonl",
        [
            {
                "schema": PREDICTION_VERDICT_SCHEMA,
                "audit_id": audit_id,
                "source_id": "val-composite",
                "partition": "val",
                "row_role": "speech",
                "category": "speech_deletion",
                "verdict": "canonical_should_be_background",
            }
        ],
    )
    evaluate_prediction(
        audit_summary=audit_dir / "summary.json",
        audit_manifest=audit_dir / "audit_manifest.jsonl",
        manual_verdicts=audit_dir / "manual_verdicts.jsonl",
        output=audit_dir / "manual_gate.json",
    )

    repair_dir = tmp_path / "speech_repair"
    repair_dir.mkdir()
    events_path = repair_dir / "events.jsonl"
    event = {
        "schema": "speech_scorer_v10_background_speech_repair_event_v1",
        "event_id": "affected-control::event00",
        "event_index": 0,
        "source_id": "affected-control",
        "partition": "val",
        "background_id": "affected-bg",
        "start_sample": 0,
        "end_sample": 320,
        "core_id": "source-event-core",
    }
    _write_jsonl(events_path, [event])
    decisions_path = repair_dir / "decisions.jsonl"
    decisions_path.write_text("", encoding="utf-8")
    repair_gate = repair_dir / "manual_gate.json"
    repair_gate.write_text(
        json.dumps(
            {
                "schema": "speech_scorer_v10_background_speech_repair_manual_gate_v1",
                "manual_review_complete": True,
                "canonical_repair_ready": True,
                "unsure_count": 0,
                "boundary_followup_count": 0,
                "source_without_target_count": 0,
                "canonical_sources_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
                "repair_events": str(events_path),
                "repair_events_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
                "decisions": str(decisions_path),
                "decisions_sha256": hashlib.sha256(decisions_path.read_bytes()).hexdigest(),
                "repair_event_count": 1,
                "repair_source_count": 1,
            }
        ),
        encoding="utf-8",
    )
    composite_manifest = tmp_path / "composites.jsonl"
    _write_jsonl(
        composite_manifest,
        [
            {
                "sample_id": "val-composite",
                "sample_count": 1920,
                "source_partition": "val",
                "negative_unit_span": None,
                "inter_unit_gaps": {
                    "left_start_sample": 640,
                    "left_end_sample": 1280,
                    "right_start_sample": 1280,
                    "right_end_sample": 1280,
                    "sources": [
                        {
                            "audio_id": "affected-bg",
                            "audio": str(affected_audio),
                            "source_offset_s": 0.0,
                            "duration_s": 0.04,
                        },
                        {
                            "audio_id": "unused-bg",
                            "audio": "unused.wav",
                            "source_offset_s": 0.0,
                            "duration_s": 0.0,
                        },
                    ],
                },
                "additive_overlay": None,
            }
        ],
    )

    output = tmp_path / "output"
    summary = apply_repairs(
        canonical_sources=canonical,
        prediction_audit_dirs=[audit_dir],
        background_speech_repair_gate=repair_gate,
        composite_source_manifest=composite_manifest,
        output_dir=output,
    )
    corrected = {
        row["source_id"]: row
        for row in map(
            json.loads,
            (output / "canonical_sources.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    placements = list(
        map(
            json.loads,
            (output / "repair_placements.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    )

    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["source_identity_changed"] is False
    assert summary["partition_identity_changed"] is False
    assert summary["training_manifest_ready"] is False
    assert summary["dataset"]["max_core_use_count"] == 1
    assert corrected["affected-control"]["row_role"] == "speech"
    assert len(corrected["affected-control"]["core_ids"]) == 1
    assert canonical_frame_labels(corrected["val-composite"]).tolist()[0] == 0
    assert len(corrected["val-composite"]["core_ids"]) == 4
    assert all(row["schema"] == PLACEMENT_SCHEMA for row in placements)
    assert sum(row["core_registered"] for row in placements) == 3
