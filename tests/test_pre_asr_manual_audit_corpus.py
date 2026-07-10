from __future__ import annotations

import json

from tools.datasets.build_pre_asr_manual_audit_corpus import build_corpus


def _write_verdicts(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_manual_audit_corpus_deduplicates_and_excludes_conflicts(tmp_path):
    first = tmp_path / "audit-a" / "manual_verdicts.jsonl"
    second = tmp_path / "audit-b" / "manual_verdicts.jsonl"
    first.parent.mkdir()
    second.parent.mkdir()
    base = {
        "window_id": "video-a-w00",
        "video_id": "video-a",
        "candidate_id": "old-a",
        "chunk_index": 0,
        "start": 1.0,
        "end": 2.0,
    }
    _write_verdicts(
        first,
        [
            {**base, "verdict": "keep"},
            {
                **base,
                "candidate_id": "old-b",
                "start": 3.0,
                "end": 4.0,
                "verdict": "drop",
            },
        ],
    )
    _write_verdicts(
        second,
        [
            {**base, "verdict": "drop"},
            {
                **base,
                "candidate_id": "old-c",
                "start": 5.0,
                "end": 6.0,
                "verdict": "keep",
            },
        ],
    )

    rows, summary = build_corpus([first, second], holdout_percent=20)

    assert summary["source_verdict_rows"] == 4
    assert summary["unique_spans"] == 3
    assert summary["conflicting_span_count"] == 1
    assert rows[0]["label"] == "ambiguous_ignore"
    assert rows[0]["training_label_included"] is False
    assert len({row["manual_partition"] for row in rows}) == 1
    assert len({row["candidate_id"] for row in rows}) == 3
