from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.audits.generate_outer_v2_directional_tail_audit_html import (
    build_audit,
    select_tail_rows,
)
from tools.audits.evaluate_outer_v2_directional_tail_audit import evaluate


def _row(audio_id: str, **errors: float) -> dict:
    return {
        "audio_id": audio_id,
        "start_inward_s": 0.0,
        "end_inward_s": 0.0,
        "start_outward_s": 0.0,
        "end_outward_s": 0.0,
        "start_absolute_s": 0.0,
        "end_absolute_s": 0.0,
        **errors,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_select_tail_rows_prioritizes_three_inward_and_two_outward() -> None:
    rows = [
        _row("in-1", start_inward_s=0.9),
        _row("in-2", end_inward_s=0.8),
        _row("in-3", start_inward_s=0.7),
        _row("out-1", start_outward_s=0.6),
        _row("out-2", end_outward_s=0.5),
        _row("other", start_absolute_s=0.4),
    ]

    selected = select_tail_rows(rows)

    assert [row["audio_id"] for row in selected] == [
        "in-1",
        "in-2",
        "in-3",
        "out-1",
        "out-2",
    ]
    assert len({row["audio_id"] for row in selected}) == 5


def test_build_audit_labels_overlapping_audio_as_alternative_edges(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(16000 * 4, dtype=np.float32), 16000)
    details = []
    synthetic = []
    for index in range(5):
        audio_id = f"sample-{index}"
        details.append(
            {
                **_row(
                    audio_id,
                    start_inward_s=0.5 - index * 0.05,
                    start_absolute_s=0.5 - index * 0.05,
                    end_outward_s=index * 0.05,
                    end_absolute_s=index * 0.05,
                ),
                "source_audio": str(source),
                "raw_end_s": 4.0,
                "truth_start_s": 1.0,
                "truth_end_s": 3.0,
                "predicted_start_s": 1.2,
                "predicted_end_s": 3.2,
                "start_signed_s": 0.2,
                "end_signed_s": 0.2,
                "start_action": "refined",
                "end_action": "refined",
                "abstain_reason": "",
            }
        )
        synthetic.append(
            {
                "audio_id": audio_id,
                "timeline_pattern": "010",
                "background_mix": None,
                "sources": [
                    {
                        "source_audio_id": f"core-{index}",
                        "source_text": f"台詞 {index}",
                    }
                ],
            }
        )
    details_path = tmp_path / "details.jsonl"
    synthetic_path = tmp_path / "synthetic.jsonl"
    _write_jsonl(details_path, details)
    _write_jsonl(synthetic_path, synthetic)

    index_path = build_audit(
        evaluation_details=details_path,
        synthetic_details=synthetic_path,
        output_dir=tmp_path / "audit",
    )

    html = index_path.read_text(encoding="utf-8")
    assert "不是相邻区间" in html
    assert "模型 start" in html
    assert "训练标签 start" in html
    assert "clipped_semantic" in html
    assert "includes_nonsemantic" in html
    assert len(list((tmp_path / "audit" / "audio").glob("*__model-kept.wav"))) == 5
    summary = json.loads((tmp_path / "audit" / "summary.json").read_text(encoding="utf-8"))
    assert summary["item_count"] == 5
    assert summary["selection_policy"] == "top3_max_inward_then_top2_max_outward_unique"


def test_evaluate_tail_audit_separates_model_and_training_target_errors(
    tmp_path: Path,
) -> None:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    (audit_dir / "summary.json").write_text(
        json.dumps({"audio_ids": ["a", "b"]}), encoding="utf-8"
    )
    _write_jsonl(
        audit_dir / "manual_verdicts.jsonl",
        [
            {
                "audio_id": "a",
                "start_model": "correct",
                "start_target": "includes_nonsemantic",
                "end_model": "correct",
                "end_target": "correct",
            },
            {
                "audio_id": "b",
                "start_model": "too_wide_nonsemantic",
                "start_target": "correct",
                "end_model": "correct",
                "end_target": "includes_nonsemantic",
            },
        ],
    )

    result = evaluate(audit_dir=audit_dir)

    assert result["complete"] is True
    assert result["known_model_clipping_count"] == 0
    assert result["model_too_wide_count"] == 1
    assert result["training_target_includes_nonsemantic_count"] == 2
    assert result["tail_clipping_gate_pass"] is True
    assert result["promotion_ready"] is False
