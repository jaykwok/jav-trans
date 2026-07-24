from __future__ import annotations

import json
from pathlib import Path

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.audits.generate_candidate_island_dual_evidence_ab_review import generate


def _write_review(
    root: Path,
    *,
    final_spans: list[dict],
    unsafe_spans: list[dict],
    unsafe_frames: int,
    outside_precision: float,
) -> Path:
    root.mkdir()
    row = {
        "source_id": "source-1",
        "partition": "test",
        "duration_s": 0.08,
        "frame_count": 4,
        "audio": "/audio/source-1.wav",
        "human_spans": [
            {"label": "inside_candidate", "start_frame": 0, "end_frame": 2, "start_s": 0.0, "end_s": 0.04},
            {"label": "outside_candidate", "start_frame": 2, "end_frame": 4, "start_s": 0.04, "end_s": 0.08},
        ],
        "protect_spans": [
            {"label": "protect", "start_frame": 0, "end_frame": 2, "start_s": 0.0, "end_s": 0.04}
        ],
        "remove_spans": [
            {"label": "remove", "start_frame": 2, "end_frame": 4, "start_s": 0.04, "end_s": 0.08}
        ],
        "final_spans": final_spans,
        "unsafe_outside_spans": unsafe_spans,
        "human_inside_frames": 2,
        "protect_recall": 1.0,
        "final_outside_precision": outside_precision,
        "supervised_ratio": 0.75,
        "conflict_frames": 0,
        "unsafe_outside_frames": unsafe_frames,
    }
    per_source = root / "per_source.jsonl"
    per_source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary = {
        "schema": "candidate_island_dual_evidence_review_summary_v1",
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "manifest_sha256": "same-manifest",
        "human_verdicts_sha256": "same-human",
        "per_source": str(per_source),
        "human_inside_frames": 2,
        "unsafe_outside_frames": unsafe_frames,
        "unsafe_outside_s": unsafe_frames * 0.02,
        "protect_recall": 1.0,
        "final_outside_precision": outside_precision,
        "supervised_ratio": 0.75,
        "conflict_ratio": 0.0,
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def test_dual_evidence_ab_review_uses_core_and_reports_retention(tmp_path: Path) -> None:
    base = _write_review(
        tmp_path / "base",
        final_spans=[
            {"label": "inside_candidate", "start_frame": 0, "end_frame": 2, "start_s": 0.0, "end_s": 0.04},
            {"label": "outside_candidate", "start_frame": 2, "end_frame": 4, "start_s": 0.04, "end_s": 0.08},
        ],
        unsafe_spans=[],
        unsafe_frames=0,
        outside_precision=1.0,
    )
    candidate = _write_review(
        tmp_path / "candidate",
        final_spans=[
            {"label": "outside_candidate", "start_frame": 0, "end_frame": 1, "start_s": 0.0, "end_s": 0.02},
            {"label": "inside_candidate", "start_frame": 1, "end_frame": 2, "start_s": 0.02, "end_s": 0.04},
            {"label": "unsure", "start_frame": 2, "end_frame": 4, "start_s": 0.04, "end_s": 0.08},
        ],
        unsafe_spans=[
            {"label": "unsafe", "start_frame": 0, "end_frame": 1, "start_s": 0.0, "end_s": 0.02}
        ],
        unsafe_frames=1,
        outside_precision=0.5,
    )

    summary = generate(
        base_review=base,
        candidate_review=candidate,
        output_dir=tmp_path / "output",
        base_name="Medium",
        candidate_name="High",
        update_nav=False,
    )

    assert summary["changed_frames"] == 3
    assert summary["base_metrics"]["true_speech_retention"] == 1.0
    assert summary["candidate_metrics"]["true_speech_retention"] == 0.5
    assert summary["true_speech_retention_gate"] == 0.95
    page = (tmp_path / "output" / "index.html").read_text(encoding="utf-8")
    assert "createAuditReviewCore" in page
    assert "formatAuditTimestamp" in page
    assert "equivalent_both_unacceptable" in page
    assert "Medium vs High" in page
