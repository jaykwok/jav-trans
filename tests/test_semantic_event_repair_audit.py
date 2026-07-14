from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audits.generate_semantic_event_repair_audit_html import (
    build_audit,
    build_repair_rows,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_repair_rows_require_manually_approved_safe_candidates(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    events = tmp_path / "events.jsonl"
    verdicts = tmp_path / "verdicts.jsonl"
    specs = tmp_path / "specs.jsonl"
    _write_jsonl(
        events,
        [
            {
                "event_key": "e",
                "sample_id": "s",
                "audio": str(audio),
                "projection_file_sha256": "p",
                "proposer_sha256": "q",
                "candidates": [
                    {
                        "candidate_id": "c",
                        "time_s": 1.0,
                        "proposer_probability": 0.5,
                        "left_audio": str(audio),
                        "right_audio": str(audio),
                        "tick_audio": str(audio),
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        verdicts,
        [{"event_key": "e", "candidates": [{"candidate_id": "c", "label": "safe"}]}],
    )
    _write_jsonl(
        specs,
        [
            {
                "repair_id": f"r{index}",
                "event_key": "e",
                "candidate_id": "c",
                "left_text": "left",
                "right_text": "right",
            }
            for index in range(5)
        ],
    )

    rows = build_repair_rows(events=events, verdicts=verdicts, specs=specs)

    assert len(rows) == 5
    assert all(row["time_s"] == 1.0 for row in rows)

    bad_verdicts = tmp_path / "bad.jsonl"
    _write_jsonl(
        bad_verdicts,
        [{"event_key": "e", "candidates": [{"candidate_id": "c", "label": "left_clipped"}]}],
    )
    with pytest.raises(ValueError, match="approved safe"):
        build_repair_rows(events=events, verdicts=bad_verdicts, specs=specs)


def test_repair_audit_explains_semantic_outer_and_continue_routes(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    rows = [
        {
            "repair_id": f"r{index}",
            "sample_id": "s",
            "source_event_key": "e",
            "candidate_id": f"c{index}",
            "time_s": float(index),
            "left_text": "left",
            "right_text": "right",
            "scope_hint": "hint",
            "reason": "reason",
            "full_audio": str(audio),
            "left_audio": str(audio),
            "right_audio": str(audio),
            "tick_audio": str(audio),
            "proposer_probability": 0.5,
            "projection_file_sha256": "p",
            "proposer_sha256": "q",
        }
        for index in range(5)
    ]

    page = build_audit(
        rows=rows, output_dir=tmp_path / "audit", update_latest=False
    ).read_text(encoding="utf-8")

    assert "新增/保留 Semantic Split" in page
    assert "声学安全但语义继续" in page
    assert "仅 Outer 边缘" in page
    assert "semantic_event_repair_manual_verdict_v1" in page
    assert "左侧硬截断" in page
    assert "右侧硬起播" in page
