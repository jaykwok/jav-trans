from __future__ import annotations

import json
from pathlib import Path

from tools.fusionvad_ja.analyze_alignment_failure_subtypes import (
    analyze_rows,
    main,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_analyze_rows_groups_subtypes_and_recommends_routes():
    rows = [
        {
            "failure_candidate": True,
            "fallback_subtype": "vad_coarse_after_sentinel",
            "alignment_quality": "vad_coarse",
            "fallback_type": "vad_coarse",
            "failure_bucket": "vad_coarse_alignment",
            "duration_s": 12.0,
            "compact_chars": 30,
            "prealign_align_len": 30,
            "aligned_segment_count": 1,
            "analysis_text": "長い発話です",
        },
        {
            "failure_candidate": True,
            "fallback_subtype": "nonlexical_text",
            "alignment_quality": "nonlexical",
            "fallback_type": "vad_coarse",
            "failure_bucket": "nonlexical_text",
            "duration_s": 3.0,
            "compact_chars": 0,
            "prealign_align_len": 0,
            "aligned_segment_count": 0,
            "analysis_text": "...",
        },
        {
            "failure_candidate": False,
            "fallback_subtype": "none",
            "alignment_quality": "forced",
            "fallback_type": "none",
            "duration_s": 2.0,
        },
    ]

    summary, examples = analyze_rows(rows, examples_per_subtype=2)
    by_subtype = {item["subtype_group"]: item for item in summary["subtypes"]}

    assert summary["failure_rows"] == 2
    assert by_subtype["vad_coarse_after_sentinel"]["route"] == "aligner_robustness"
    assert by_subtype["nonlexical_text"]["route"] == "nonlexical_time_policy"
    assert {row["subtype_group"] for row in examples} == {
        "vad_coarse_after_sentinel",
        "nonlexical_text",
    }


def test_subtype_analysis_cli_writes_outputs(tmp_path):
    diagnostics = tmp_path / "diagnostics.jsonl"
    _write_jsonl(
        diagnostics,
        [
            {
                "failure_candidate": True,
                "fallback_subtype": "align_text_empty",
                "alignment_quality": "drop_or_review",
                "fallback_type": "proportional",
                "failure_bucket": "align_text_empty",
                "duration_s": 1.5,
                "compact_chars": 1,
                "prealign_align_len": 0,
                "analysis_text": "♪",
            }
        ],
    )
    output_dir = tmp_path / "out"

    assert main(["--diagnostics", str(diagnostics), "--output-dir", str(output_dir)]) == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "summary.md").read_text(encoding="utf-8")
    examples = [
        json.loads(line)
        for line in (output_dir / "subtype_examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["subtype_count"] == 1
    assert summary["subtypes"][0]["route"] == "prealign_policy"
    assert examples[0]["subtype_group"] == "align_text_empty"
    assert "## Subtype Routes" in markdown
