from __future__ import annotations

import json
from pathlib import Path

from tools.audits.analyze_scorer_v10_fragmentation_distribution import analyze


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_fragmentation_distribution_separates_train_and_heldout(tmp_path: Path) -> None:
    canonical = []
    predictions = []
    for source_id, partition, overlay in (
        ("train-overlay", "train", True),
        ("val-clean", "val", False),
    ):
        canonical.append(
            {
                "source_id": source_id,
                "partition": partition,
                "row_role": "speech",
                "audio": f"{source_id}.wav",
                "core_ids": [f"core-{source_id}"],
                "boundary_serialization_contract_id": "boundary_acoustic_binary_v12",
                "additive_overlay": (
                    {
                        "source": {"background_type": "breathing"},
                        "mix": {"achieved_snr_db": 10.0},
                    }
                    if overlay
                    else None
                ),
            }
        )
        predictions.append(
            {
                "source_id": source_id,
                "partition": partition,
                "row_role": "speech",
                "truth_spans": [
                    {"label": "truth_speech", "start_frame": 0, "end_frame": 10}
                ],
                "prediction_spans": [
                    {"label": "model_speech", "start_frame": 0, "end_frame": 4},
                    {"label": "model_speech", "start_frame": 5, "end_frame": 10},
                ],
            }
        )

    output = tmp_path / "out"
    summary = analyze(
        predictions=_write_jsonl(tmp_path / "predictions.jsonl", predictions),
        canonical_sources=_write_jsonl(tmp_path / "canonical.jsonl", canonical),
        output_dir=output,
    )
    train_rows = [
        json.loads(line)
        for line in (output / "train_fragmentation_hardcases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    heldout_rows = [
        json.loads(line)
        for line in (output / "heldout_fragmentation_hardcases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["source_id"] for row in train_rows] == ["train-overlay"]
    assert [row["source_id"] for row in heldout_rows] == ["val-clean"]
    assert summary["train_hardcase_count"] == 1
    assert summary["heldout_hardcase_count"] == 1
    assert summary["hardcase_policy"]["heldout_never_enters_training"] is True
    overlay = next(
        row
        for row in summary["composition_fragmentation"]
        if row["partition"] == "train" and row["group"] == "overlay"
    )
    assert overlay["fragmentation_rate"] == 1.0
    assert summary["gap_distribution"]["train"]["gap_le_1_frame"] == 1
