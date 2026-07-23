from __future__ import annotations

import json
from pathlib import Path

from tools.asr.cueqc.select_cueqc_v13_drop_safety_audit import select


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_drop_safety_selection_uses_upstream_membership_not_pre_cueqc_inner(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "selected.jsonl"
    _write_jsonl(
        runtime,
        [
            {
                "subisland_id": "long",
                "sample_id": "source-long",
                "duration_s": 9.0,
                "pre_asr_candidate": {"scorer_speech_p90": 0.1},
            },
            {
                "subisland_id": "membership",
                "sample_id": "source-membership",
                "duration_s": 2.0,
                "pre_asr_candidate": {"scorer_speech_p90": 0.95},
            },
            {
                "subisland_id": "stale-inner",
                "sample_id": "source-stale-inner",
                "duration_s": 1.0,
                "pre_asr_candidate": {"scorer_speech_p90": 0.2},
                "inner_edge_prediction": {
                    "start_probabilities": {"semantic_target": 1.0},
                    "end_probabilities": {"semantic_target": 1.0},
                },
            },
        ],
    )
    _write_jsonl(
        labels,
        [
            {"subisland_id": item, "label": "drop"}
            for item in ("long", "membership", "stale-inner")
        ],
    )

    summary = select(
        runtime_chunks=runtime,
        labels=labels,
        output=output,
        per_axis=1,
    )

    selected = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["subisland_id"] for row in selected] == ["long", "membership"]
    assert summary["axes"] == [
        "longest_drop",
        "highest_upstream_candidate_membership",
    ]
