from __future__ import annotations

import json

import numpy as np
import pytest

from tools.boundary.ja.evaluate_galgame_semantic_core_text_audit import evaluate
from tools.boundary.ja.label_galgame_semantic_core_text_with_omni import (
    build_prompt,
    is_provider_data_inspection_rejection,
    load_excluded_candidate_audio_ids,
    select_audit_rows,
    select_candidates,
    validate_response,
)


def test_candidate_selection_is_unique_and_respects_exclusion() -> None:
    rows = [
        {"audio_id": f"a{index}", "audio": f"a{index}.ogg", "text": f"台詞{index}"}
        for index in range(20)
    ]
    selected = select_candidates(
        rows,
        count=10,
        seed=7,
        excluded_audio_ids={"a0", "a1", "a2"},
    )

    assert len(selected) == 10
    assert len({row["audio_id"] for row in selected}) == 10
    assert {row["audio_id"] for row in selected}.isdisjoint({"a0", "a1", "a2"})


def test_candidate_exclusion_manifest_reads_top_level_audio_ids(tmp_path) -> None:
    manifest = tmp_path / "prior-candidates.jsonl"
    manifest.write_text(
        '{"audio_id":"prior-a"}\n{"audio":"somewhere/prior-b.ogg"}\n',
        encoding="utf-8",
    )

    assert load_excluded_candidate_audio_ids([str(manifest)]) == {
        "prior-a",
        "prior-b",
    }


def test_provider_data_inspection_rejection_is_detected_without_text_rules() -> None:
    assert is_provider_data_inspection_rejection(
        RuntimeError('{"code":"data_inspection_failed"}')
    )
    assert not is_provider_data_inspection_rejection(RuntimeError("timeout"))


def test_invalid_teacher_schema_remains_a_value_error() -> None:
    _prompt, mapping = build_prompt([{"audio_id": "a", "text": "台詞"}])
    with pytest.raises(ValueError, match="ids/order"):
        validate_response({"items": []}, mapping)


def test_batch_response_requires_exact_short_id_order_and_labels() -> None:
    batch = [
        {"audio_id": "a", "text": "普通の台詞"},
        {"audio_id": "b", "text": "あんっ…"},
    ]
    _prompt, mapping = build_prompt(batch)

    validated = validate_response(
        {
            "items": [
                {"id": "c000", "label": "all_semantic"},
                {"id": "c001", "label": "contains_nonsemantic"},
            ]
        },
        mapping,
    )

    assert [(row["audio_id"], label) for row, label in validated] == [
        ("a", "all_semantic"),
        ("b", "contains_nonsemantic"),
    ]
    with pytest.raises(ValueError):
        validate_response(
            {"items": [{"id": "c001", "label": "all_semantic"}]}, mapping
        )


def test_audit_selection_includes_available_teacher_labels() -> None:
    rows = [
        {"audio_id": f"s{index}", "label": label}
        for index, label in enumerate(
            [
                "all_semantic",
                "contains_nonsemantic",
                "unsure",
                "all_semantic",
                "contains_nonsemantic",
                "all_semantic",
            ]
        )
    ]

    selected = select_audit_rows(rows, count=5, seed=3)

    assert len(selected) == 5
    assert {row["label"] for row in selected} == {
        "all_semantic",
        "contains_nonsemantic",
        "unsure",
    }


def test_manual_gate_requires_every_teacher_label_to_be_correct(tmp_path) -> None:
    items = tmp_path / "items.jsonl"
    verdicts = tmp_path / "verdicts.jsonl"
    output = tmp_path / "gate.json"
    item_rows = [
        {
            "schema": "galgame_semantic_core_text_teacher_v1",
            "audio_id": "semantic",
            "label": "all_semantic",
        },
        {
            "schema": "galgame_semantic_core_text_teacher_v1",
            "audio_id": "nonsemantic",
            "label": "contains_nonsemantic",
        },
    ]
    items.write_text(
        "".join(json.dumps(row) + "\n" for row in item_rows), encoding="utf-8"
    )
    verdict_rows = [
        {
            "schema": "galgame_semantic_core_text_manual_verdict_v1",
            "audio_id": row["audio_id"],
            "teacher_label": row["label"],
            "verdict": "correct",
            "note": "",
        }
        for row in item_rows
    ]
    verdicts.write_text(
        "".join(json.dumps(row) + "\n" for row in verdict_rows), encoding="utf-8"
    )

    summary = evaluate(items=items, verdicts=verdicts, output=output)

    assert summary["manual_gate_complete"] is True
    assert summary["teacher_gate_pass"] is True
    assert summary["correct_by_teacher_label"] == {
        "all_semantic": 1,
        "contains_nonsemantic": 1,
    }

    verdict_rows[1]["verdict"] = "wrong"
    verdicts.write_text(
        "".join(json.dumps(row) + "\n" for row in verdict_rows), encoding="utf-8"
    )
    failed = evaluate(items=items, verdicts=verdicts, output=output)
    assert failed["manual_gate_complete"] is True
    assert failed["teacher_gate_pass"] is False
