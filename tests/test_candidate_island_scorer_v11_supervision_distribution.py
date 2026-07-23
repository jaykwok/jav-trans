from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.boundary.ja.audit_candidate_island_scorer_v11_supervision_distribution import (
    audit_distribution,
)


CONTRACT = "boundary_acoustic_binary_v12"
SCHEMA = "candidate_island_scorer_v11_canonical_source_v1"


def _row(
    source_id: str,
    *,
    partition: str,
    source_kind: str | None,
    labels: list[tuple[str, int]],
    synthetic: bool = False,
) -> dict:
    cursor = 0
    spans = []
    for label, frames in labels:
        spans.append(
            {"label": label, "start_frame": cursor, "end_frame": cursor + frames}
        )
        cursor += frames
    row = {
        "schema": SCHEMA,
        "boundary_serialization_contract_id": CONTRACT,
        "source_id": source_id,
        "partition": partition,
        "frame_count": cursor,
        "canonical_spans": spans,
        "annotation_provenance": "fixture",
        "synthetic_composite": synthetic,
        "training_manifest_allowed": True,
    }
    if source_kind is not None:
        row["source_kind"] = source_kind
    return row


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_distribution_audit_exposes_same_source_mismatch(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    _write(
        canonical,
        [
            _row(
                "synthetic",
                partition="train",
                source_kind="semantic_composite_candidate",
                labels=[("inside_candidate", 8), ("outside_candidate", 2)],
                synthetic=True,
            ),
            _row(
                "dual-inside-only",
                partition="train",
                source_kind="real_train_full_source_calibrated_dual_evidence",
                labels=[("inside_candidate", 8), ("unsure", 2)],
            ),
            _row(
                "masked-outside-only",
                partition="train",
                source_kind="real_train_outside_masked",
                labels=[("outside_candidate", 3), ("unsure", 7)],
            ),
            _row(
                "heldout-mixed",
                partition="val",
                source_kind=None,
                labels=[("inside_candidate", 4), ("outside_candidate", 6)],
            ),
        ],
    )

    summary = audit_distribution(canonical_sources=canonical, output_dir=tmp_path / "out")

    assert summary["source_count"] == 4
    assert summary["source_level_mixed_supervision_mismatch"] is True
    assert summary["real_train_full_source_summary"]["source_count"] == 2
    assert (
        summary["real_train_full_source_summary"][
            "mixed_inside_outside_source_count"
        ]
        == 0
    )
    assert summary["heldout_real_full_source_summary"][
        "mixed_inside_outside_source_count"
    ] == 1
    assert summary["calibrated_dual_evidence_summary"][
        "zero_outside_source_count"
    ] == 1
    assert summary["decision"]["gpu_retrain_recommended"] is False
    assert (tmp_path / "out" / "source_stats.jsonl").is_file()


def test_distribution_audit_rejects_non_contiguous_truth(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    row = _row(
        "broken",
        partition="train",
        source_kind="real_train_full_source_calibrated_dual_evidence",
        labels=[("inside_candidate", 5), ("outside_candidate", 5)],
    )
    row["canonical_spans"][1]["start_frame"] = 4
    _write(canonical, [row])

    with pytest.raises(ValueError, match="non-contiguous canonical spans"):
        audit_distribution(canonical_sources=canonical, output_dir=tmp_path / "out")
