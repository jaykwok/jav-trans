from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.audits.evaluate_scorer_v10_canonical_r4_replacement_audit import (
    RESULT_SCHEMA,
    evaluate,
)
from tools.audits.generate_scorer_v10_canonical_r4_replacement_audit_html import (
    MANUAL_VERDICT_SCHEMA,
    SUMMARY_SCHEMA,
    build_audit,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_canonical_r4_replacement_page_and_gate(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.linspace(-0.2, 0.2, 640, dtype=np.float32), 16000)
    canonical = tmp_path / "canonical.jsonl"
    _write_jsonl(
        canonical,
        [
            {
                "source_id": "target",
                "audio": str(audio),
                "partition": "test",
                "sample_rate": 16000,
                "duration_s": 0.04,
                "canonical_spans": [
                    {"start_sample": 0, "end_sample": 320, "label": "speech", "label_source": "repair"},
                    {"start_sample": 320, "end_sample": 640, "label": "background", "label_source": "background"},
                ],
            },
            {
                "source_id": "source",
                "audio": str(audio),
                "partition": "test",
                "sample_rate": 16000,
                "duration_s": 0.04,
                "canonical_spans": [{"start_sample": 0, "end_sample": 640, "label": "speech", "label_source": "repair"}],
            },
        ],
    )
    placements = tmp_path / "placements.jsonl"
    _write_jsonl(
        placements,
        [
            {
                "schema": "speech_scorer_v10_background_speech_repair_placement_v1",
                "placement_id": "placement-1",
                "mapping_id": "mapping-1",
                "event_id": "event-1",
                "event_core_id": "event-core",
                "source_id": "source",
                "background_id": "background-a",
                "target_source_id": "target",
                "partition": "test",
                "role": "left_gap",
                "tile_index": 0,
                "occurrence_index": 0,
                "mapped_start_sample": 0,
                "mapped_end_sample": 320,
                "mapped_start_s": 0.0,
                "mapped_end_s": 0.02,
                "placement_core_id": "placement-core",
                "background_label_change_ranges": [{"start_sample": 0, "end_sample": 320}],
                "background_label_change_sample_count": 320,
                "already_speech_sample_count": 0,
                "core_registered": True,
            }
        ],
    )
    dependency_mappings = tmp_path / "dependency_mappings.jsonl"
    _write_jsonl(
        dependency_mappings,
        [
            {
                "schema": "speech_scorer_v10_background_dependency_mapping_v1",
                "mapping_id": "mapping-1",
                "mapped_intervals": [
                    {
                        "tile_index": 0,
                        "source_start_sample": 160,
                        "source_end_sample": 480,
                        "mapped_start_sample": 0,
                        "mapped_end_sample": 320,
                    }
                ],
            }
        ],
    )
    events = tmp_path / "events.jsonl"
    _write_jsonl(
        events,
        [{"event_id": "event-1", "start_sample": 0, "end_sample": 640}],
    )
    gate = tmp_path / "repair_gate.json"
    gate.write_text(
        json.dumps(
            {
                "repair_events": str(events),
                "repair_events_sha256": hashlib.sha256(events.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    canonical_summary = tmp_path / "canonical_summary.json"
    canonical_summary.write_text(
        json.dumps(
            {
                "schema": "speech_scorer_v10_corrected_canonical_r4_summary_v1",
                "canonical_sources": str(canonical),
                "canonical_sources_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
                "repair_placements": str(placements),
                "repair_placements_sha256": hashlib.sha256(placements.read_bytes()).hexdigest(),
                "repair_placement_count": 1,
                "dependency_mappings": str(dependency_mappings),
                "dependency_mappings_sha256": hashlib.sha256(
                    dependency_mappings.read_bytes()
                ).hexdigest(),
                "background_speech_repair_gate": str(gate),
                "background_speech_repair_gate_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audit"
    build_audit(canonical_summary=canonical_summary, output_dir=output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "audit_manifest.jsonl").read_text(encoding="utf-8"))
    page = (output / "index.html").read_text(encoding="utf-8")

    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["review_item_count"] == 1
    assert summary["standalone_exact_clip_playback"] is True
    assert summary["standalone_clip_count"] >= 2
    assert manifest["source_occurrence_start_s"] == 0.01
    assert manifest["source_occurrence_end_s"] == 0.03
    assert manifest["source_occurrence_sample_count"] == 320
    assert "独立 WAV" in page
    assert "source 子段 → target occurrence" in page
    assert "仅 canonical 标签变化区间" in page
    assert "源事件非目标语音（整组撤销）" in page
    assert "完整 island" in page
    assert 'preload="none"' in page

    verdicts = output / "manual_verdicts.jsonl"
    _write_jsonl(
        verdicts,
        [
            {
                "schema": MANUAL_VERDICT_SCHEMA,
                "item_id": manifest["item_id"],
                "placement_id": manifest["placement_id"],
                "event_id": manifest["event_id"],
                "source_id": manifest["source_id"],
                "target_source_id": manifest["target_source_id"],
                "partition": manifest["partition"],
                "role": manifest["role"],
                "core_registered": manifest["core_registered"],
                "verdict": "repair_speech_correct",
            }
        ],
    )
    result = evaluate(
        audit_summary=output / "summary.json",
        audit_manifest=output / "audit_manifest.jsonl",
        manual_verdicts=verdicts,
        output=output / "manual_gate.json",
    )
    assert result["schema"] == RESULT_SCHEMA
    assert result["canonical_repair_pass"] is True
    assert result["feature_cache_relabel_allowed"] is True
    assert result["training_manifest_allowed"] is False

    invalid_verdicts = output / "manual_verdicts.invalid.jsonl"
    invalid = json.loads(verdicts.read_text(encoding="utf-8"))
    invalid["verdict"] = "source_event_not_target"
    _write_jsonl(invalid_verdicts, [invalid])
    invalid_result = evaluate(
        audit_summary=output / "summary.json",
        audit_manifest=output / "audit_manifest.jsonl",
        manual_verdicts=invalid_verdicts,
        output=output / "manual_gate.invalid.json",
    )
    assert invalid_result["manual_review_complete"] is True
    assert invalid_result["canonical_repair_pass"] is False
    assert invalid_result["source_event_repair_count"] == 1
    assert invalid_result["feature_cache_relabel_allowed"] is False
