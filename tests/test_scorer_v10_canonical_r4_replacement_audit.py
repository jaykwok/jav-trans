from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
    audio.write_bytes(b"RIFF-audit")
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
    events = tmp_path / "events.jsonl"
    _write_jsonl(
        events,
        [{"event_id": "event-1", "start_sample": 0, "end_sample": 320}],
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
    assert "直接播放/停止" in page
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
