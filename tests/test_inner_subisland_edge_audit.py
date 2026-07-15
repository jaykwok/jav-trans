from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.boundary.ja.build_inner_subisland_edge_audit import (
    build_page,
    edge_features,
    edge_ownership,
)
from tools.boundary.ja.evaluate_inner_subisland_edge_audit import evaluate


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_edge_features_keep_full_ptm_and_recompute_position() -> None:
    ptm = np.zeros((3, 2048), dtype=np.float32)
    ptm[:, -1] = [1.0, 2.0, 3.0]
    mfcc = np.zeros((3, 40), dtype=np.float32)

    features = edge_features(ptm, mfcc)

    assert features.shape == (3, 2089)
    assert features[:, 2047].tolist() == [1.0, 2.0, 3.0]
    assert features[:, -1].tolist() == [0.0, 0.5, 1.0]


def test_only_internal_facing_edges_belong_to_inner() -> None:
    assert edge_ownership(start_s=0.0, end_s=1.0, source_duration_s=3.0) == {
        "start_requires_inner": False,
        "end_requires_inner": True,
    }
    assert edge_ownership(start_s=1.0, end_s=2.0, source_duration_s=3.0) == {
        "start_requires_inner": True,
        "end_requires_inner": True,
    }
    assert edge_ownership(start_s=2.0, end_s=3.0, source_duration_s=3.0) == {
        "start_requires_inner": True,
        "end_requires_inner": False,
    }


def test_page_separates_outer_owned_and_inner_edges(tmp_path: Path) -> None:
    page = build_page(
        rows=[
            {
                "sample_id": "s",
                "subisland_id": "s__s00",
                "audio": "audio.wav",
                "raw_start_s": 0.0,
                "raw_end_s": 1.0,
                "refined_start_s": 0.0,
                "refined_end_s": 0.8,
                "start_requires_inner": False,
                "end_requires_inner": True,
                "bootstrap_prediction": {
                    "start_action": "refined",
                    "end_action": "refined",
                    "abstain_reason": "",
                },
            }
        ],
        output_dir=tmp_path / "audit",
        update_latest=False,
    ).read_text(encoding="utf-8")

    assert "全局最外侧边缘仍归 Outer v2" in page
    assert "Outer owned" in page
    assert "截语音" in page
    assert "残留杂音" in page
    assert ".join('\\n')+'\\n'" in page


def test_evaluator_enforces_zero_clipping(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    verdicts = tmp_path / "verdicts.jsonl"
    _write(
        items,
        [
            {
                "schema": "inner_subisland_edge_audit_item_v1",
                "subisland_id": "s0",
                "start_requires_inner": False,
                "end_requires_inner": True,
                "bootstrap_prediction": {"end_action": "refined"},
            },
            {
                "schema": "inner_subisland_edge_audit_item_v1",
                "subisland_id": "s1",
                "start_requires_inner": True,
                "end_requires_inner": False,
                "bootstrap_prediction": {"start_action": "refined"},
            },
        ],
    )
    _write(
        verdicts,
        [
            {"schema": "inner_subisland_edge_manual_verdict_v1", "subisland_id": "s0", "end_verdict": "correct"},
            {"schema": "inner_subisland_edge_manual_verdict_v1", "subisland_id": "s1", "start_verdict": "clipped"},
        ],
    )

    summary = evaluate(items=items, verdicts=verdicts, output=tmp_path / "summary.json")

    assert summary["manual_gate_complete"] is True
    assert summary["zero_clipping_pass"] is False
    assert summary["bootstrap_inner_promotion_ready"] is False
