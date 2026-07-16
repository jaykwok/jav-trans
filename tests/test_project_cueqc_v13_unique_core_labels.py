import json
from pathlib import Path

from tools.boundary.ja.project_cueqc_v13_unique_core_labels import project


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_project_labels_each_new_runtime_chunk_from_exact_core_overlap(
    tmp_path: Path,
) -> None:
    sources = tmp_path / "sources.jsonl"
    runtime = tmp_path / "runtime.jsonl"
    output = tmp_path / "labels.jsonl"
    _write(
        sources,
        [
            {
                "sample_id": "s",
                "source_partition": "train",
                "core_spans": [
                    {"core_id": "a", "start_sample": 0, "end_sample": 16000},
                    {"core_id": "b", "start_sample": 32000, "end_sample": 48000},
                ],
            }
        ],
    )
    _write(
        runtime,
        [
            {
                "sample_id": "s",
                "subisland_id": "s0",
                "audio": "s.wav",
                "start_s": 0.0,
                "end_s": 1.0,
                "duration_s": 1.0,
            },
            {
                "sample_id": "s",
                "subisland_id": "s1",
                "audio": "s.wav",
                "start_s": 1.0,
                "end_s": 2.0,
                "duration_s": 1.0,
            },
            {
                "sample_id": "s",
                "subisland_id": "s2",
                "audio": "s.wav",
                "start_s": 2.0,
                "end_s": 3.0,
                "duration_s": 1.0,
            },
        ],
    )

    summary = project(source_manifest=sources, runtime_chunks=runtime, output=output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert [row["label"] for row in rows] == ["keep", "drop", "keep"]
    assert summary["label_counts"] == {"drop": 1, "keep": 2}
    assert summary["parent_label_inheritance"] is False
