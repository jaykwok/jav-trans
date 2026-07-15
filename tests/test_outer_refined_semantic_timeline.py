from __future__ import annotations

import json

import pytest

from boundary.outer_refiner_v2 import PairedOuterEdgePrediction
from tools.audits.generate_outer_refined_source_audit_html import build_audit
from tools.audits.evaluate_outer_refined_source_audit import evaluate
from tools.boundary.ja.export_outer_refined_semantic_timeline import (
    rebase_timeline_row,
)


def _prediction(*, start_s: float, end_s: float) -> PairedOuterEdgePrediction:
    return PairedOuterEdgePrediction(
        raw_start_s=0.0,
        raw_end_s=5.0,
        start_s=start_s,
        end_s=end_s,
        start_action="refined",
        end_action="refined",
        abstain_reason="",
        start_probabilities={"semantic_target": 0.9},
        end_probabilities={"semantic_target": 0.9},
        class_probabilities=None,
    )


def test_rebase_timeline_preserves_source_and_uses_outer_local_coordinates() -> None:
    row = {
        "schema": "semantic_timeline_teacher_v1",
        "sample_id": "s",
        "audio": "source.wav",
        "duration_s": 5.0,
        "semantic_alignments": [
            {"unit_id": "u00", "status": "matched", "start_s": 1.2, "end_s": 2.0},
            {"unit_id": "u01", "status": "matched", "start_s": 2.4, "end_s": 4.0},
        ],
        "semantic_events": [
            {
                "event_id": "e00",
                "status": "matched",
                "interval_start_s": 2.0,
                "interval_end_s": 2.4,
            }
        ],
    }

    rebased = rebase_timeline_row(
        row,
        output_audio="outer.wav",
        prediction=_prediction(start_s=1.0, end_s=4.2),
        outer_checkpoint_sha256="sha",
    )

    assert rebased["audio_contract"] == "learned_outer_refined_island_v1"
    assert rebased["source_audio"] == "source.wav"
    assert rebased["source_span"] == {"start_s": 1.0, "end_s": 4.2}
    assert rebased["duration_s"] == 3.2
    assert rebased["semantic_alignments"][0]["start_s"] == pytest.approx(0.2)
    assert rebased["semantic_events"][0]["interval_start_s"] == 1.0
    assert rebased["outer_alignment_violations"] == []
    assert rebased["training_ready"] is True


def test_rebase_timeline_exposes_outer_tail_clipping() -> None:
    row = {
        "sample_id": "s",
        "audio": "source.wav",
        "duration_s": 5.0,
        "semantic_alignments": [
            {"unit_id": "u00", "status": "matched", "start_s": 0.8, "end_s": 4.5}
        ],
        "semantic_events": [],
    }

    rebased = rebase_timeline_row(
        row,
        output_audio="outer.wav",
        prediction=_prediction(start_s=1.0, end_s=4.0),
        outer_checkpoint_sha256="sha",
    )

    assert rebased["outer_alignment_violations"] == ["u00"]
    assert rebased["training_ready"] is False


def test_outer_source_audit_reviews_start_and_end_without_coarse_timeline_bias(
    tmp_path,
) -> None:
    source = tmp_path / "source.wav"
    refined = tmp_path / "outer.wav"
    source.write_bytes(b"source")
    refined.write_bytes(b"outer")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "source_audio": str(source),
                "audio": str(refined),
                "reference_text": "台詞",
                "source_duration_s": 5.0,
                "duration_s": 3.0,
                "source_span": {"start_s": 1.0, "end_s": 4.0},
                "outer_prediction": {
                    "start_action": "refined",
                    "end_action": "refined",
                },
                "outer_alignment_violations": ["u00"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    page = build_audit(labels=labels, output_dir=tmp_path / "audit")

    html = page.read_text(encoding="utf-8")
    assert "不显示旧切点或 Omni coarse timeline" in html
    assert "前缘 start" in html
    assert "后缘 end" in html
    assert "outer_refined_source_manual_verdict_v2" in html
    assert "violations=" not in html


def test_outer_source_gate_rejects_residual_nonsemantic_edge(tmp_path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"sample_id": "sample"}) + "\n", encoding="utf-8"
    )
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    (audit_dir / "manual_verdicts.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "start_model": "correct",
                "end_model": "too_wide_nonsemantic",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate(audit_dir=audit_dir, expected_labels=labels)

    assert result["zero_clipping_pass"] is True
    assert result["edge_cleanup_pass"] is False
    assert result["promotion_ready"] is False
