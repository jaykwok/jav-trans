from __future__ import annotations

import json
from pathlib import Path

from tools.fusionvad_ja.analyze_pre_asr_island_chunks import main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_pre_asr_island_analysis_uses_time_overlap_and_gap_reasons(tmp_path):
    cache = tmp_path / "vad-cache.json"
    _write_json(
        cache,
        {
            "processing_spans": [
                {
                    "start": 0.0,
                    "end": 12.0,
                    "split_reason": "tail",
                    "vad_segments": [
                        {"start": 0.5, "end": 2.0, "score": 0.9},
                        {"start": 4.0, "end": 5.0, "score": 0.8},
                        {"start": 8.0, "end": 10.0, "score": 0.7},
                    ],
                },
                {
                    "start": 20.0,
                    "end": 36.0,
                    "split_reason": "tail",
                    "vad_segments": [{"start": 21.0, "end": 35.0, "score": 0.9}],
                },
            ],
        },
    )
    diagnostics = tmp_path / "diag" / "diagnostics.jsonl"
    _write_jsonl(
        diagnostics,
        [
            {
                "chunk_index": 7,
                "start": 0.0,
                "end": 12.0,
                "alignment_quality": "vad_coarse",
                "fallback_subtype": "vad_coarse_after_sentinel",
                "failure_bucket": "vad_coarse_alignment",
                "failure_reasons": ["alignment_fallback", "alignment_sentinel"],
            },
            {
                "chunk_index": 8,
                "start": 20.0,
                "end": 36.0,
                "alignment_quality": "drop_or_review",
                "fallback_subtype": "asr_empty_text",
                "failure_bucket": "empty_text_for_chunk",
                "failure_reasons": ["empty_text_for_chunk"],
            },
        ],
    )
    output_dir = tmp_path / "out"

    assert main(
        [
            "--vad-cache",
            str(cache),
            "--diagnostics",
            str(diagnostics),
            "--output-dir",
            str(output_dir),
            "--long-chunk-s",
            "14",
            "--long-gap-s",
            "1.0",
        ]
    ) == 0

    rows = [
        json.loads(line)
        for line in (output_dir / "chunk_analysis.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert rows[0]["risk_reasons"] == [
        "multi_island_long_gap",
        "vad_coarse_after_sentinel",
    ]
    assert rows[0]["matched_chunk_indices"] == [7]
    assert rows[1]["risk_reasons"] == [
        "continuous_speech_no_internal_gap",
        "long_chunk",
        "asr_empty",
    ]
    assert summary["chunk_count"] == 2
    assert summary["risk_reason_counts"]["multi_island_long_gap"] == 1
    assert summary["risk_reason_counts"]["asr_empty"] == 1
