from __future__ import annotations

import json
from pathlib import Path

from tools.fusionvad_ja.analyze_valley_splits import main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_analyze_valley_splits_simulates_low_score_cut_and_risk_stats(tmp_path):
    cache = tmp_path / "vad-cache.json"
    _write_json(
        cache,
        {
            "runtime_vad_signature": {"frame_hop_s": 1.0},
            "processing_spans": [
                {
                    "start": 0.0,
                    "end": 12.0,
                    "split_reason": "tail",
                    "vad_segments": [{"start": 0.0, "end": 12.0, "score": 0.9}],
                },
                {
                    "start": 20.0,
                    "end": 24.0,
                    "split_reason": "tail",
                    "vad_segments": [{"start": 20.0, "end": 24.0, "score": 0.9}],
                },
            ],
        },
    )
    frame_scores = tmp_path / "scores.json"
    _write_json(
        frame_scores,
        {
            "frame_hop_s": 1.0,
            "scores": [0.9] * 5 + [0.05] * 2 + [0.9] * 30,
        },
    )
    diagnostics = tmp_path / "diagnostics.jsonl"
    _write_jsonl(
        diagnostics,
        [
            {
                "chunk_index": 0,
                "fallback_subtype": "vad_coarse_after_sentinel",
            }
        ],
    )
    output_dir = tmp_path / "out"

    assert main(
        [
            "--vad-cache",
            str(cache),
            "--frame-scores",
            str(frame_scores),
            "--diagnostics",
            str(diagnostics),
            "--output-dir",
            str(output_dir),
            "--min-core-frames",
            "8",
            "--target-core-frames",
            "5",
            "--min-valley-frames",
            "2",
            "--min-child-frames",
            "3",
            "--valley-threshold",
            "0.2",
        ]
    ) == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (output_dir / "valley_split_plan.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["original_chunk_count"] == 2
    assert summary["new_chunk_count"] == 3
    assert summary["split_chunk_count"] == 1
    assert summary["risk_vad_coarse_after_sentinel_count"] == 1
    assert summary["risk_split_count"] == 1
    assert rows[0]["valley_split"] is True
    assert rows[0]["children"][0]["split_policy"] == "r16_pre_asr_valley_v1"
    assert rows[1]["valley_split"] is False
