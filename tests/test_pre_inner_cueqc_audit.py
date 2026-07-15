from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.boundary.ja.build_pre_inner_cueqc_audit import (
    build_items,
    build_page,
    validate_partition,
)
from tools.boundary.ja.evaluate_pre_inner_cueqc_audit import evaluate


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_partition_must_be_contiguous() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        validate_partition(
            [
                {"sample_id": "s", "start_s": 0.0, "end_s": 1.0},
                {"sample_id": "s", "start_s": 1.2, "end_s": 2.0},
            ]
        )


def test_builder_keeps_whole_subisland_contract(tmp_path: Path) -> None:
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"RIFF")
    subislands = tmp_path / "subislands.jsonl"
    _write(
        subislands,
        [
            {"sample_id": "s", "subisland_id": "s__s00", "audio": str(audio), "start_s": 0.0, "end_s": 1.0, "duration_s": 1.0},
            {"sample_id": "s", "subisland_id": "s__s01", "audio": str(audio), "start_s": 1.0, "end_s": 2.0, "duration_s": 1.0},
        ],
    )

    rows = build_items(subislands=subislands, output_dir=tmp_path / "audit")

    assert len(rows) == 2
    assert rows[0]["decision_contract"] == "whole_provisional_subisland_content_before_inner_v1"


def test_page_explains_keep_even_when_edges_are_wide(tmp_path: Path) -> None:
    page = build_page(
        rows=[
            {
                "sample_id": "s",
                "subisland_id": "s__s00",
                "audio": "audio.wav",
                "start_s": 0.0,
                "end_s": 1.0,
                "duration_s": 1.0,
            }
        ],
        output_dir=tmp_path / "audit",
        update_latest=False,
    ).read_text(encoding="utf-8")

    assert "即使前后静音很宽" in page
    assert "整块没有目标语音" in page
    assert "运行时显式保留并进入 Inner" in page
    assert ".join('\\n')+'\\n'" in page


def test_evaluator_routes_drop_whole_and_preserves_unsure(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    verdicts = tmp_path / "verdicts.jsonl"
    _write(
        items,
        [
            {"schema": "pre_inner_cueqc_audit_item_v1", "sample_id": "s", "subisland_id": "keep", "audio": "audio.wav", "start_s": 0.0, "end_s": 1.0, "duration_s": 1.0},
            {"schema": "pre_inner_cueqc_audit_item_v1", "sample_id": "s", "subisland_id": "drop", "audio": "audio.wav", "start_s": 1.0, "end_s": 2.0, "duration_s": 1.0},
            {"schema": "pre_inner_cueqc_audit_item_v1", "sample_id": "s", "subisland_id": "unsure", "audio": "audio.wav", "start_s": 2.0, "end_s": 3.0, "duration_s": 1.0},
        ],
    )
    _write(
        verdicts,
        [
            {"schema": "pre_inner_cueqc_manual_verdict_v1", "subisland_id": "keep", "label": "keep"},
            {"schema": "pre_inner_cueqc_manual_verdict_v1", "subisland_id": "drop", "label": "drop"},
            {"schema": "pre_inner_cueqc_manual_verdict_v1", "subisland_id": "unsure", "label": "unsure"},
        ],
    )

    summary = evaluate(items=items, verdicts=verdicts, output_dir=tmp_path / "gate")

    assert summary["manual_gate_pass"] is True
    assert summary["model_training_ready"] is True
    assert summary["inner_input_count"] == 2
    assert summary["removed_span_count"] == 1
    assert summary["unsure_runtime_route"] == "preserve_then_inner"
