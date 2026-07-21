from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.audits.evaluate_scorer_v10_canonical_r4_replacement_audit import (
    evaluate,
)
from tools.audits.generate_scorer_v10_canonical_r4_replacement_audit_html import (
    ITEM_SCHEMA,
    MANUAL_VERDICT_SCHEMA,
    SUMMARY_SCHEMA as AUDIT_SUMMARY_SCHEMA,
)
from tools.boundary.ja.apply_speech_island_scorer_v10_canonical_r4_repairs import (
    CHANGED_SPAN_SCHEMA,
    PLACEMENT_SCHEMA,
    SUMMARY_SCHEMA as R4_SUMMARY_SCHEMA,
)
from tools.boundary.ja.apply_speech_island_scorer_v10_canonical_r5_verdicts import (
    SUMMARY_SCHEMA as R5_SUMMARY_SCHEMA,
    apply_verdicts,
)
from tools.boundary.ja.compile_speech_island_scorer_v10_canonical import (
    canonical_frame_labels,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _source(
    source_id: str,
    partition: str,
    *,
    role: str,
    spans: list[dict],
    core_ids: list[str] | None = None,
    background_id: str = "",
) -> dict:
    return {
        "schema": "speech_scorer_v10_canonical_source_v1",
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": "speech_scorer_canonical_frames_v1",
        "source_id": source_id,
        "audio": f"{source_id}.wav",
        "row_role": role,
        "partition": partition,
        "core_ids": core_ids or [],
        "background_id": background_id,
        "background_source_ids": [background_id] if background_id else [],
        "background_source_video_ids": [f"video-{source_id}"],
        "sample_rate": 16000,
        "sample_count": 640,
        "duration_s": 0.04,
        "input_distribution": "full_source_windows",
        "canonical_spans": spans,
    }


def _background_span(background_id: str, start: int = 0, end: int = 640) -> dict:
    return {
        "start_sample": start,
        "end_sample": end,
        "label": "background",
        "label_source": "synthetic_background",
        "background_id": background_id,
    }


def _speech_span(core_id: str, start: int, end: int) -> dict:
    return {
        "start_sample": start,
        "end_sample": end,
        "label": "speech",
        "label_source": "teacher_speech",
        "core_id": core_id,
    }


def _repair_span(
    *, placement_id: str, event_id: str, core_id: str, background_id: str
) -> dict:
    return {
        "start_sample": 320,
        "end_sample": 480,
        "label": "speech",
        "label_source": "manual_background_speech_repair_propagated_v1",
        "manual_original_label": "background",
        "origin_background_id": background_id,
        "core_id": core_id,
        "repair_event_id": event_id,
        "repair_placement_id": placement_id,
        "repair_role": "left_gap",
    }


def _placement(
    *,
    placement_id: str,
    event_id: str,
    source_id: str,
    target_source_id: str,
    partition: str,
    role: str,
    core_id: str,
    start: int,
    end: int,
) -> dict:
    return {
        "schema": PLACEMENT_SCHEMA,
        "placement_id": placement_id,
        "mapping_id": f"mapping::{placement_id}",
        "event_id": event_id,
        "event_core_id": f"event-core::{event_id}",
        "source_id": source_id,
        "background_id": f"background::{source_id}",
        "target_source_id": target_source_id,
        "partition": partition,
        "role": role,
        "tile_index": 0,
        "occurrence_index": 0,
        "mapped_start_sample": start,
        "mapped_end_sample": end,
        "mapped_start_s": start / 16000,
        "mapped_end_s": end / 16000,
        "placement_core_id": core_id,
        "background_label_change_ranges": [
            {"start_sample": start, "end_sample": end}
        ],
        "background_label_change_sample_count": end - start,
        "already_speech_sample_count": 0,
        "core_registered": True,
    }


def test_r5_applies_group_and_individual_replacement_verdicts(tmp_path: Path) -> None:
    baseline = [
        _source(
            "train-speech",
            "train",
            role="speech",
            core_ids=["train-core"],
            spans=[_speech_span("train-core", 0, 640)],
        ),
        _source(
            "train-background",
            "train",
            role="all_background",
            background_id="train-bg",
            spans=[_background_span("train-bg")],
        ),
        _source(
            "val-target",
            "val",
            role="speech",
            core_ids=["val-core"],
            spans=[
                _speech_span("val-core", 0, 320),
                _background_span("val-control-bg", 320, 640),
            ],
        ),
        _source(
            "val-control",
            "val",
            role="all_background",
            background_id="val-control-bg",
            spans=[_background_span("val-control-bg")],
        ),
        _source(
            "val-background",
            "val",
            role="all_background",
            background_id="val-bg",
            spans=[_background_span("val-bg")],
        ),
        _source(
            "test-target",
            "test",
            role="speech",
            core_ids=["test-core"],
            spans=[
                _speech_span("test-core", 0, 320),
                _background_span("test-control-bg", 320, 640),
            ],
        ),
        _source(
            "test-control",
            "test",
            role="all_background",
            background_id="test-control-bg",
            spans=[_background_span("test-control-bg")],
        ),
        _source(
            "test-background",
            "test",
            role="all_background",
            background_id="test-bg",
            spans=[_background_span("test-bg")],
        ),
    ]
    baseline_path = tmp_path / "baseline.jsonl"
    _write_jsonl(baseline_path, baseline)

    event_group = "test-control::event00"
    event_individual = "val-control::event00"
    test_control_placement = "test-control::control::event00"
    test_target_placement = "test-target::left-gap::event00"
    val_control_placement = "val-control::control::event00"
    val_target_placement = "val-target::left-gap::event00"
    core_ids = {
        test_control_placement: "repair-test-control",
        test_target_placement: "repair-test-target",
        val_control_placement: "repair-val-control",
        val_target_placement: "repair-val-target",
    }

    current = copy.deepcopy(baseline)
    current_by_id = {row["source_id"]: row for row in current}
    for source_id, placement_id, event_id, background_id in (
        (
            "test-control",
            test_control_placement,
            event_group,
            "test-control-bg",
        ),
        (
            "val-control",
            val_control_placement,
            event_individual,
            "val-control-bg",
        ),
    ):
        row = current_by_id[source_id]
        row["row_role"] = "speech"
        row["repaired_background_id"] = background_id
        row["background_id"] = ""
        row["core_ids"] = [core_ids[placement_id]]
        row["canonical_spans"] = [
            _background_span(background_id, 0, 320),
            _repair_span(
                placement_id=placement_id,
                event_id=event_id,
                core_id=core_ids[placement_id],
                background_id=background_id,
            ),
            _background_span(background_id, 480, 640),
        ]
        row["canonical_spans"][1]["repair_role"] = "control"
    for source_id, placement_id, event_id, background_id in (
        ("test-target", test_target_placement, event_group, "test-control-bg"),
        ("val-target", val_target_placement, event_individual, "val-control-bg"),
    ):
        row = current_by_id[source_id]
        row["core_ids"].append(core_ids[placement_id])
        row["canonical_spans"] = [
            row["canonical_spans"][0],
            _repair_span(
                placement_id=placement_id,
                event_id=event_id,
                core_id=core_ids[placement_id],
                background_id=background_id,
            ),
            _background_span(background_id, 480, 640),
        ]
    current_path = tmp_path / "r4-canonical.jsonl"
    _write_jsonl(current_path, current)

    placements = [
        _placement(
            placement_id=test_control_placement,
            event_id=event_group,
            source_id="test-control",
            target_source_id="test-control",
            partition="test",
            role="control",
            core_id=core_ids[test_control_placement],
            start=320,
            end=480,
        ),
        _placement(
            placement_id=test_target_placement,
            event_id=event_group,
            source_id="test-control",
            target_source_id="test-target",
            partition="test",
            role="left_gap",
            core_id=core_ids[test_target_placement],
            start=320,
            end=480,
        ),
        _placement(
            placement_id=val_control_placement,
            event_id=event_individual,
            source_id="val-control",
            target_source_id="val-control",
            partition="val",
            role="control",
            core_id=core_ids[val_control_placement],
            start=320,
            end=480,
        ),
        _placement(
            placement_id=val_target_placement,
            event_id=event_individual,
            source_id="val-control",
            target_source_id="val-target",
            partition="val",
            role="left_gap",
            core_id=core_ids[val_target_placement],
            start=320,
            end=480,
        ),
    ]
    placements_path = tmp_path / "placements.jsonl"
    _write_jsonl(placements_path, placements)
    changed_spans_path = tmp_path / "changed-spans.jsonl"
    _write_jsonl(
        changed_spans_path,
        [
            {
                "schema": CHANGED_SPAN_SCHEMA,
                "source_id": row["target_source_id"],
                "partition": row["partition"],
                "start_sample": row["mapped_start_sample"],
                "end_sample": row["mapped_end_sample"],
                "original_label": "background",
                "label": "speech",
                "label_source": "manual_background_speech_repair_propagated_v1",
                "core_id": row["placement_core_id"],
                "origin_core_id": "",
                "repair_event_id": row["event_id"],
                "repair_placement_id": row["placement_id"],
            }
            for row in placements
        ],
    )
    r4_summary_path = tmp_path / "r4-summary.json"
    r4_summary_path.write_text(
        json.dumps(
            {
                "schema": R4_SUMMARY_SCHEMA,
                "input_canonical_sources": str(baseline_path),
                "input_canonical_sources_sha256": _sha256(baseline_path),
                "canonical_sources": str(current_path),
                "canonical_sources_sha256": _sha256(current_path),
                "repair_placements": str(placements_path),
                "repair_placements_sha256": _sha256(placements_path),
                "changed_spans": str(changed_spans_path),
                "changed_spans_sha256": _sha256(changed_spans_path),
                "audio_bytes_changed": False,
            }
        ),
        encoding="utf-8",
    )

    audit_manifest_path = tmp_path / "audit-manifest.jsonl"
    audit_rows = [{**row, "schema": ITEM_SCHEMA, "item_id": row["placement_id"]} for row in placements]
    _write_jsonl(audit_manifest_path, audit_rows)
    repair_events_path = tmp_path / "repair-events.jsonl"
    _write_jsonl(repair_events_path, [{"event_id": event_group}, {"event_id": event_individual}])
    audit_summary_path = tmp_path / "audit-summary.json"
    audit_summary_path.write_text(
        json.dumps(
            {
                "schema": AUDIT_SUMMARY_SCHEMA,
                "canonical_summary": str(r4_summary_path),
                "canonical_summary_sha256": _sha256(r4_summary_path),
                "canonical_sources": str(current_path),
                "canonical_sources_sha256": _sha256(current_path),
                "repair_placements": str(placements_path),
                "repair_placements_sha256": _sha256(placements_path),
                "repair_events": str(repair_events_path),
                "repair_events_sha256": _sha256(repair_events_path),
                "audit_manifest": str(audit_manifest_path),
                "audit_manifest_sha256": _sha256(audit_manifest_path),
                "review_item_count": len(audit_rows),
            }
        ),
        encoding="utf-8",
    )
    verdict_by_id = {
        test_control_placement: "source_event_not_target",
        test_target_placement: "source_event_not_target",
        val_control_placement: "repair_speech_correct",
        val_target_placement: "not_target_after_render",
    }
    verdicts_path = tmp_path / "manual-verdicts.jsonl"
    _write_jsonl(
        verdicts_path,
        [
            {
                "schema": MANUAL_VERDICT_SCHEMA,
                "item_id": row["placement_id"],
                "placement_id": row["placement_id"],
                "event_id": row["event_id"],
                "source_id": row["source_id"],
                "target_source_id": row["target_source_id"],
                "partition": row["partition"],
                "role": row["role"],
                "core_registered": row["core_registered"],
                "verdict": verdict_by_id[row["placement_id"]],
            }
            for row in placements
        ],
    )
    gate_path = tmp_path / "manual-gate.json"
    evaluate(
        audit_summary=audit_summary_path,
        audit_manifest=audit_manifest_path,
        manual_verdicts=verdicts_path,
        output=gate_path,
    )

    output = tmp_path / "r5"
    result = apply_verdicts(
        r4_summary_path=r4_summary_path,
        replacement_gate_path=gate_path,
        output_dir=output,
    )
    corrected = {
        row["source_id"]: row
        for row in map(
            json.loads,
            (output / "canonical_sources.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }

    assert result["schema"] == R5_SUMMARY_SCHEMA
    assert result["replacement_resolution_pass"] is True
    assert result["source_event_rejection_count"] == 1
    assert result["placement_followup_count"] == 1
    assert result["accepted_repair_placement_count"] == 1
    assert result["rejected_repair_placement_count"] == 3
    assert result["rejected_registered_core_count"] == 3
    baseline_by_id = {row["source_id"]: row for row in baseline}
    assert corrected["test-control"] == baseline_by_id["test-control"]
    assert corrected["test-target"]["core_ids"] == ["test-core"]
    assert corrected["val-control"]["core_ids"] == ["repair-val-control"]
    assert corrected["val-target"]["core_ids"] == ["val-core"]
    assert canonical_frame_labels(corrected["test-target"]).tolist() == [1, 0]
    assert canonical_frame_labels(corrected["val-target"]).tolist() == [1, 0]
    assert result["training_manifest_ready"] is False
