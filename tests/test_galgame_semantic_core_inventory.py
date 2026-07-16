from __future__ import annotations

import json

import pytest

from tools.boundary.ja.compile_galgame_semantic_core_inventory import (
    compile_inventory,
)


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _label(audio_id: str, label: str) -> dict:
    return {
        "schema": "galgame_semantic_core_text_teacher_v1",
        "prompt_version": "prompt-v1",
        "model": "omni",
        "audio_id": audio_id,
        "audio": f"{audio_id}.ogg",
        "duration_s": 1.0,
        "reference_text": f"台詞{audio_id}",
        "source": "galgame",
        "label": label,
    }


def test_inventory_contains_only_unique_all_semantic_cores(tmp_path) -> None:
    labels = tmp_path / "labels.jsonl"
    supplemental = tmp_path / "supplemental.jsonl"
    output = tmp_path / "inventory.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(
        labels,
        [
            _label("a", "all_semantic"),
            _label("b", "contains_nonsemantic"),
        ],
    )
    _write_jsonl(supplemental, [_label("c", "all_semantic")])

    summary = compile_inventory(
        labels=[labels, supplemental],
        output=output,
        summary_path=summary_path,
        count=2,
        seed=3,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {row["audio_id"] for row in rows} == {"a", "c"}
    assert {row["approval_label"] for row in rows} == {"all_semantic"}
    assert summary["unique_core_count"] == 2
    assert summary["max_core_use_count"] == 1
    assert summary["teacher_label_manifests"] == [str(labels), str(supplemental)]


def test_inventory_rejects_any_excluded_core_overlap(tmp_path) -> None:
    labels = tmp_path / "labels.jsonl"
    excluded = tmp_path / "prior.jsonl"
    _write_jsonl(labels, [_label("a", "all_semantic")])
    _write_jsonl(excluded, [{"audio_id": "a"}])

    with pytest.raises(ValueError, match="overlap excluded cores"):
        compile_inventory(
            labels=labels,
            output=tmp_path / "inventory.jsonl",
            summary_path=tmp_path / "summary.json",
            count=1,
            seed=3,
            excluded_candidate_manifests=[excluded],
        )


def test_inventory_rejects_source_ledger_overlap(tmp_path) -> None:
    labels = tmp_path / "labels.jsonl"
    source_ledger = tmp_path / "used-dataset.jsonl"
    _write_jsonl(labels, [_label("a", "all_semantic")])
    _write_jsonl(
        source_ledger,
        [{"boundary_metadata": {"source_audio_ids": ["a"]}}],
    )

    with pytest.raises(ValueError, match="overlap excluded cores"):
        compile_inventory(
            labels=labels,
            output=tmp_path / "inventory.jsonl",
            summary_path=tmp_path / "summary.json",
            count=1,
            seed=3,
            excluded_source_manifests=[source_ledger],
        )
