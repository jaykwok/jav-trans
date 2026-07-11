from __future__ import annotations

import json
from pathlib import Path

from tools.datasets.verify_pre_asr_label_run import verify_run


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    candidates = _write_jsonl(
        dataset / "features" / "w00" / "pre_asr_candidates.jsonl",
        [
            {"candidate_id": "preasr-w00-chunk00000", "sample_id": "preasr-w00-chunk00000"},
            {"candidate_id": "preasr-w00-chunk00001", "sample_id": "preasr-w00-chunk00001"},
        ],
    )
    _write_jsonl(
        dataset / "source_windows.jsonl",
        [
            {
                "window_id": "w00",
                "pre_asr_candidates": str(candidates),
            }
        ],
    )
    return dataset


def test_verify_run_reports_complete_label_closure(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    output = tmp_path / "output"
    _write_json(
        output / "joint_labels" / "w00.json",
        {
            "window_id": "w00",
            "pre_asr_labels": [
                {
                    "candidate_id": "preasr-w00-chunk00000",
                    "label": "definite_keep",
                    "training_label_included": True,
                },
                {
                    "candidate_id": "preasr-w00-chunk00001",
                    "label": "ambiguous_ignore",
                    "training_label_included": False,
                    "local_fallback": "api_moderation_reject_to_ambiguous_ignore",
                },
            ],
        },
    )

    summary = verify_run(dataset_dir=dataset, output_dir=output)

    assert summary["complete"] is True
    assert summary["expected_window_count"] == 1
    assert summary["pre_asr_label_count"] == 2
    assert summary["label_counts"] == {
        "ambiguous_ignore": 1,
        "definite_keep": 1,
    }
    assert summary["local_fallback_counts"] == {
        "api_moderation_reject_to_ambiguous_ignore": 1,
    }


def test_verify_run_reports_missing_candidate(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    output = tmp_path / "output"
    _write_json(
        output / "joint_labels" / "w00.json",
        {
            "window_id": "w00",
            "pre_asr_labels": [
                {
                    "candidate_id": "preasr-w00-chunk00000",
                    "label": "definite_drop",
                    "training_label_included": True,
                }
            ],
        },
    )

    summary = verify_run(dataset_dir=dataset, output_dir=output)

    assert summary["complete"] is False
    assert summary["missing_candidate_id_count"] == 1
    assert summary["missing_candidate_ids_sample"] == ["preasr-w00-chunk00001"]


def test_verify_run_ignores_stale_error_for_completed_window(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    output = tmp_path / "output"
    _write_json(
        output / "joint_labels" / "w00.json",
        {
            "window_id": "w00",
            "pre_asr_labels": [
                {
                    "candidate_id": "preasr-w00-chunk00000",
                    "label": "definite_drop",
                    "training_label_included": True,
                },
                {
                    "candidate_id": "preasr-w00-chunk00001",
                    "label": "definite_keep",
                    "training_label_included": True,
                },
            ],
        },
    )
    _write_json(output / "errors" / "w00.json", {"window_id": "w00"})

    summary = verify_run(dataset_dir=dataset, output_dir=output)

    assert summary["complete"] is True
    assert summary["error_count"] == 0
    assert summary["stale_error_count"] == 1
