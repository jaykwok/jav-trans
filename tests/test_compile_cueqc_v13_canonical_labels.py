from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.asr.cueqc.compile_cueqc_v13_canonical_labels import compile_labels


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        "utf-8",
    )


def test_canonical_labels_keep_unsure_and_resolve_conflicts_before_training(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    manual = tmp_path / "manual.jsonl"
    output = tmp_path / "canonical.jsonl"
    chunks = [
        {
            "sample_id": "s",
            "subisland_id": item_id,
            "source_partition": "train",
            "audio": "s.wav",
            "start_s": index,
            "end_s": index + 1,
            "duration_s": 1.0,
        }
        for index, item_id in enumerate(("a", "b", "c"))
    ]
    _write(runtime, chunks)
    _write(
        teacher,
        [
            {"subisland_id": "a", "label": "keep"},
            {"subisland_id": "b", "label": "drop"},
            {"subisland_id": "b", "label": "keep"},
            {"subisland_id": "c", "label": "unsure"},
        ],
    )
    _write(manual, [{"subisland_id": "c", "label": "drop"}])

    summary = compile_labels(
        runtime_chunks=runtime,
        teacher_labels=teacher,
        manual_overrides=manual,
        output=output,
    )
    rows = [json.loads(line) for line in output.read_text("utf-8").splitlines()]

    assert [row["teacher_label"] for row in rows] == ["keep", "unsure", "unsure"]
    assert [row["label"] for row in rows] == ["keep", "unsure", "drop"]
    assert [row["training_label"] for row in rows] == [1, -100, 0]
    assert rows[1]["label_source"] == "duplicate_request_conflict_to_unsure"
    assert rows[2]["manual_override_applied"] is True
    assert summary["canonical_label_counts"] == {"drop": 1, "keep": 1, "unsure": 1}
    assert summary["teacher_unsure_ignored"] == 1


def test_canonical_labels_reject_stale_or_incomplete_exact_projection(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    exact = tmp_path / "exact.jsonl"
    output = tmp_path / "canonical.jsonl"
    _write(
        runtime,
        [
            {
                "sample_id": "s",
                "subisland_id": item_id,
                "audio": "s.wav",
                "start_s": index,
                "end_s": index + 1,
                "duration_s": 1.0,
            }
            for index, item_id in enumerate(("a", "b"))
        ],
    )
    _write(teacher, [{"subisland_id": item_id, "label": "keep"} for item_id in ("a", "b")])
    _write(exact, [{"subisland_id": "a", "label": "keep"}])

    with pytest.raises(ValueError, match="exact labels are incomplete; missing 1 chunks"):
        compile_labels(
            runtime_chunks=runtime,
            teacher_labels=teacher,
            exact_labels=exact,
            output=output,
        )
