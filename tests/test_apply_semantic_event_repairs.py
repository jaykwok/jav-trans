from __future__ import annotations

import json
from pathlib import Path

from tools.boundary.ja.apply_semantic_event_repairs import apply_repairs


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_repairs_add_semantic_boundary_and_keep_outer_blocker(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    verdicts = tmp_path / "verdicts.jsonl"
    output = tmp_path / "repaired.jsonl"
    _write(
        labels,
        [
            {
                "schema": "teacher",
                "sample_id": "s",
                "audio": "a.wav",
                "duration_s": 4.0,
                "source": "source",
                "reference_text": "どうしよう。タクシーを呼んでもらって、病院に行く",
                "text_units": [
                    {"unit_id": "u00", "text": "どうしよう。", "kind": "semantic"},
                    {
                        "unit_id": "u01",
                        "text": "タクシーを呼んでもらって、病院に行く",
                        "kind": "semantic",
                    },
                ],
                "semantic_events": [
                    {
                        "event_id": "e00",
                        "left_unit_id": "u00",
                        "right_unit_id": "u01",
                        "status": "matched",
                        "interval_start_s": 1.0,
                        "interval_end_s": 1.2,
                    }
                ],
            }
        ],
    )
    _write(
        verdicts,
        [
            {
                "schema": "semantic_event_repair_manual_verdict_v1",
                "repair_id": "outer",
                "sample_id": "s",
                "time_s": 0.2,
                "left_text": "（source 开头，无左侧语义单元）",
                "right_text": "どうしよう。",
                "decision": "outer_only",
                "note": "breath",
            },
            {
                "schema": "semantic_event_repair_manual_verdict_v1",
                "repair_id": "existing",
                "sample_id": "s",
                "time_s": 1.2,
                "left_text": "どうしよう。",
                "right_text": "タクシーを呼んでもらって、病院に行く",
                "decision": "semantic_split",
                "note": "",
            },
            {
                "schema": "semantic_event_repair_manual_verdict_v1",
                "repair_id": "new",
                "sample_id": "s",
                "time_s": 2.4,
                "left_text": "タクシーを呼んでもらって、",
                "right_text": "病院に行く",
                "decision": "semantic_split",
                "note": "",
            },
        ],
    )

    rows = apply_repairs(
        timeline_labels=labels, repair_verdicts=verdicts, output=output
    )

    row = rows[0]
    assert [unit["text"] for unit in row["text_units"]] == [
        "どうしよう。",
        "タクシーを呼んでもらって、",
        "病院に行く",
    ]
    assert [event["coarse_anchor_s"] for event in row["semantic_events"]] == [
        1.2,
        2.4,
    ]
    assert all(
        event["anchor_source"] == "manual_safe_candidate"
        for event in row["semantic_events"]
    )
    assert row["region_blocker_anchors"] == [
        {
            "time_s": 0.2,
            "scope": "outer_only",
            "repair_id": "outer",
            "note": "breath",
        }
    ]
