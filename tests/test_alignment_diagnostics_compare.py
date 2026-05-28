from __future__ import annotations

import json
from pathlib import Path

from tools.fusionvad_ja.compare_alignment_diagnostics import main, summarize_diagnostics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_summarize_diagnostics_builds_checkpoint_row(tmp_path):
    diag_dir = tmp_path / "diag"
    _write_json(
        diag_dir / "summary.json",
        {
            "case_count": 1,
            "chunk_count": 4,
            "output_segment_count": 3,
            "nonempty_chunk_count": 3,
            "fallback_chunk_count": 1,
            "failure_candidate_count": 2,
            "alignment_quality_counts": {"forced": 2, "vad_coarse": 1, "drop_or_review": 1},
            "fallback_type_counts": {"none": 3, "vad_coarse": 1},
            "failure_bucket_counts": {"vad_coarse_alignment": 1, "align_text_empty": 1},
        },
    )
    _write_jsonl(
        diag_dir / "failure_candidates.jsonl",
        [
            {"failure_bucket": "vad_coarse_alignment"},
            {"failure_bucket": "align_text_empty"},
        ],
    )

    row = summarize_diagnostics("checkpoint-x", diag_dir)

    assert row["label"] == "checkpoint-x"
    assert row["chunk_count"] == 4
    assert row["forced_ratio"] == 0.5
    assert row["fallback_chunk_ratio"] == 0.25
    assert row["failure_candidate_ratio"] == 0.5
    assert row["coarse_alignment_count"] == 1
    assert row["drop_or_review_count"] == 1
    assert row["failure_bucket_counts"]["align_text_empty"] == {"count": 1, "ratio": 0.25}


def test_compare_alignment_diagnostics_cli_writes_summary(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_json(first / "summary.json", {"chunk_count": 2, "alignment_quality_counts": {"forced": 2}})
    _write_json(second / "summary.json", {"chunk_count": 2, "alignment_quality_counts": {"drop_or_review": 1, "forced": 1}})
    output_dir = tmp_path / "out"

    assert main(
        [
            "--diagnostics",
            f"base={first}",
            "--diagnostics",
            f"full15000={second}",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    summary = json.loads((output_dir / "checkpoint_comparison.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (output_dir / "checkpoint_comparison_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    markdown = (output_dir / "checkpoint_comparison.md").read_text(encoding="utf-8")

    assert summary["run_count"] == 2
    assert {row["label"] for row in rows} == {"base", "full15000"}
    assert "| `base` |" in markdown
    assert "| `full15000` |" in markdown
