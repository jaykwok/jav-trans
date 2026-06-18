from __future__ import annotations

import json
from pathlib import Path

from tools.boundary.export_cueqc_drop_hardcases import export_cueqc_drop_hardcases


def _label(
    sample_id: str,
    *,
    decision: str,
    tags: list[str] | None = None,
    text: str = "...",
    start: float = 0.0,
    end: float = 0.5,
    video_id: str = "AAA",
    chunk_index: int = 1,
) -> dict:
    return {
        "schema": "cueqc_false_drop_audit_label_v1",
        "dataset_id": "test-audit",
        "audit_id": sample_id,
        "sample_id": sample_id,
        "video_id": video_id,
        "video_label": "sample",
        "chunk_index": chunk_index,
        "start": start,
        "end": end,
        "text": text,
        "display_prob_drop": 0.91,
        "display_prob_keep": 0.09,
        "confidence": 0.91,
        "manual_decision": decision,
        "reason_tags": tags or [],
        "notes": "",
        "updated_at": "2026-06-17T00:00:00Z",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_export_cueqc_drop_hardcases_routes_and_deduplicates(tmp_path: Path):
    labels_a = tmp_path / "labels_a.jsonl"
    labels_b = tmp_path / "labels_b.jsonl"
    _write_jsonl(
        labels_a,
        [
            _label("drop-short", decision="drop_ok", tags=["vocalization"], text="あ...", end=0.4),
            _label("drop-dialogue", decision="drop_ok", tags=["dialogue"], text="待って", end=1.0),
            _label("keep-safety", decision="false_drop_keep", tags=["dialogue"], text="今日は", end=1.2),
        ],
    )
    _write_jsonl(
        labels_b,
        [
            _label("drop-short", decision="drop_ok", tags=["breath"], text="あ...", end=0.4),
            _label("drop-long", decision="drop_ok", tags=["environment"], text="...", end=3.0),
            _label("uncertain-safety", decision="uncertain", text="ん...", end=0.6),
        ],
    )

    summary = export_cueqc_drop_hardcases(
        label_paths=[labels_a, labels_b],
        output_dir=tmp_path / "out",
    )

    assert summary["counts"]["raw_label_rows"] == 6
    assert summary["counts"]["unique_label_items"] == 5
    assert summary["counts"]["duplicate_extra_rows"] == 1
    assert summary["counts"]["confirmed_drop_candidates"] == 3
    assert summary["counts"]["safety_holdout_items"] == 2
    assert summary["candidate_route_counts"] == {
        "speech_boundary_frame_negative_candidate": 3,
    }

    candidates = [
        json.loads(line)
        for line in (tmp_path / "out" / "cueqc_confirmed_drop_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_sample = {row["sample_id"]: row for row in candidates}
    assert by_sample["drop-short"]["candidate_route"] == "speech_boundary_frame_negative_candidate"
    assert by_sample["drop-short"]["source_label_count"] == 2
    assert by_sample["drop-short"]["reason_tags"] == ["breath", "vocalization"]
    assert by_sample["drop-dialogue"]["candidate_route"] == "speech_boundary_frame_negative_candidate"
    assert by_sample["drop-long"]["candidate_route"] == "speech_boundary_frame_negative_candidate"
    assert by_sample["drop-dialogue"]["reason_tags"] == ["dialogue"]
    assert by_sample["drop-long"]["reason_tags"] == ["environment"]
    assert all(
        row["hard_negative_status"] == "candidate_requires_audio_materialization"
        for row in candidates
    )

    safety = [
        json.loads(line)
        for line in (tmp_path / "out" / "cueqc_drop_safety_holdout.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["sample_id"] for row in safety} == {"keep-safety", "uncertain-safety"}
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "summary.md").exists()
