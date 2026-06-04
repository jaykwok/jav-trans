from __future__ import annotations

import json
from pathlib import Path

from tools.boundary.ja.plan_reward_boundary_segments import main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_reward_boundary_planner_splits_long_gap_crossing_segment_with_oracle_candidate(tmp_path):
    boundary = tmp_path / "boundary.jsonl"
    _write_jsonl(
        boundary,
        [
            {
                "audio_id": "clip",
                "duration_s": 12.0,
                "frame_hop_s": 1.0,
                "actual_speech_segments": [{"start": 0.0, "end": 3.0}, {"start": 8.0, "end": 11.0}],
            },
        ],
    )
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "audio_id": "clip",
                "duration_s": 12.0,
                "frame_hop_s": 1.0,
                "probabilities": {
                    "speech": [0.9] * 11 + [0.0],
                    "cut": [0.0] * 12,
                    "start": [0.0] * 12,
                    "end": [0.0] * 12,
                },
            },
        ],
    )
    output_dir = tmp_path / "out"

    assert main(
        [
            "--boundary-manifest",
            str(boundary),
            "--predictions",
            str(predictions),
            "--candidate-source",
            "oracle",
            "--use-truth-cost",
            "--speech-threshold",
            "0.5",
            "--pad-s",
            "0",
            "--merge-gap-s",
            "1",
            "--fallback-target-duration-s",
            "6",
            "--fallback-gap-overlap-s",
            "1",
            "--min-child-s",
            "1",
            "--split-penalty",
            "0",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    details = [
        json.loads(line)
        for line in (output_dir / "plan_details.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["baseline"]["predicted_segment_count"] == 1
    assert summary["baseline"]["predicted_gap_crossing_segment_count"] == 1
    assert summary["planned"]["predicted_segment_count"] == 2
    assert summary["planned"]["predicted_gap_crossing_segment_count"] == 0
    assert len(details[0]["planned_segments"]) == 2


def test_reward_boundary_planner_keeps_segment_without_candidates(tmp_path):
    boundary = tmp_path / "boundary.jsonl"
    _write_jsonl(
        boundary,
        [
            {
                "audio_id": "clip",
                "duration_s": 12.0,
                "frame_hop_s": 1.0,
                "actual_speech_segments": [{"start": 0.0, "end": 11.0}],
            },
        ],
    )
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "audio_id": "clip",
                "duration_s": 12.0,
                "frame_hop_s": 1.0,
                "probabilities": {
                    "speech": [0.9] * 11 + [0.0],
                    "cut": [0.0] * 12,
                    "start": [0.0] * 12,
                    "end": [0.0] * 12,
                },
            },
        ],
    )
    output_dir = tmp_path / "out"

    assert main(
        [
            "--boundary-manifest",
            str(boundary),
            "--predictions",
            str(predictions),
            "--candidate-source",
            "probability",
            "--speech-threshold",
            "0.5",
            "--pad-s",
            "0",
            "--merge-gap-s",
            "1",
            "--fallback-target-duration-s",
            "6",
            "--min-child-s",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["baseline"]["predicted_segment_count"] == 1
    assert summary["planned"]["predicted_segment_count"] == 1
