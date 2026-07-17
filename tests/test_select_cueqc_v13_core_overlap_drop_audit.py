import json
from pathlib import Path

from tools.asr.cueqc.select_cueqc_v13_core_overlap_drop_audit import select


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_selects_largest_core_overlap_with_unique_sources(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    exact = tmp_path / "exact.jsonl"
    output = tmp_path / "selected.jsonl"
    _write(
        runtime,
        [
            {
                "sample_id": "a",
                "subisland_id": "a0",
                "duration_s": 1.0,
                "start_s": 0.0,
                "end_s": 1.0,
                "audio": "a.wav",
            },
            {
                "sample_id": "a",
                "subisland_id": "a1",
                "duration_s": 1.0,
                "start_s": 1.0,
                "end_s": 2.0,
                "audio": "a.wav",
            },
            {
                "sample_id": "b",
                "subisland_id": "b0",
                "duration_s": 1.0,
                "start_s": 0.0,
                "end_s": 1.0,
                "audio": "b.wav",
            },
        ],
    )
    _write(
        teacher,
        [
            {"subisland_id": "a0", "label": "drop", "confidence": 0.9},
            {"subisland_id": "a1", "label": "drop", "confidence": 0.9},
            {"subisland_id": "b0", "label": "drop", "confidence": 0.9},
        ],
    )
    _write(
        exact,
        [
            {"subisland_id": "a0", "semantic_core_overlaps": [{"overlap_samples": 12000}]},
            {"subisland_id": "a1", "semantic_core_overlaps": [{"overlap_samples": 15000}]},
            {"subisland_id": "b0", "semantic_core_overlaps": [{"overlap_samples": 8000}]},
        ],
    )

    summary = select(
        runtime_chunks=runtime,
        teacher_labels=teacher,
        exact_labels=exact,
        output=output,
        count=2,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary["selected_count"] == 2
    assert [row["subisland_id"] for row in rows] == ["a1", "b0"]
    assert rows[0]["exact_core_overlap_s"] == 0.9375
