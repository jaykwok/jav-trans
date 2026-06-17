from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.boundary.prepare_v51_training import prepare_training_prep


def _write_dataset(path: Path, rows: int = 2) -> None:
    feature_names = ["gap_s", "left_duration_s"]
    payloads = []
    for index in range(rows):
        axis = "right.start" if index % 2 == 0 else "left.end"
        payloads.append(
            {
                "schema": "boundary_refiner_frame_sequence_dataset_v5",
                "audio_id": f"pref-{index}",
                "source": "boundary_preference_v1",
                "feature_names": feature_names,
                "feature_dim": len(feature_names),
                "feature_schema": "frame_sequence_features_v1",
                "feature_schema_hash": "hash-test",
                "feature_signature": {"feature_schema": "frame_sequence_features_v1"},
                "sequence_features": [[0.2, 1.0]],
                "sequence_boundary_delta_targets": [[0.08, 0.0] if axis == "right.start" else [0.0, -0.08]],
                "sequence_boundary_delta_weights": [[1.0, 0.0] if axis == "right.start" else [0.0, 0.6]],
                "sequence_reasons": [f"preference_{axis}_challenger"],
                "gap_indexes": [index],
                "metadata": {
                    "video_id": "sample",
                    "axis": axis,
                    "winner": "challenger",
                },
            }
        )
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in payloads) + "\n",
        encoding="utf-8",
    )


def _write_checkpoint(path: Path) -> None:
    torch = pytest.importorskip("torch")
    torch.save(
        {
            "schema": "boundary_refiner_v5",
            "feature_names": ["gap_s", "left_duration_s"],
            "model_config": {
                "hidden_size": 128,
                "num_layers": 2,
                "state_size": 32,
                "num_heads": 4,
                "n_groups": 2,
                "chunk_size": 8,
                "bidirectional": True,
                "input_dim": 2,
                "output_dim": 2,
            },
            "metadata": {
                "runtime_adapter": "frame_sequence_v1",
                "feature_schema": "frame_sequence_features_v1",
                "feature_schema_hash": "hash-test",
            },
        },
        path,
    )


def test_prepare_v51_training_writes_reproducible_scripts(tmp_path: Path):
    dataset = tmp_path / "compiled_preference_v5.jsonl"
    checkpoint = tmp_path / "boundary_refiner.pt"
    output_dir = tmp_path / "prep"
    _write_dataset(dataset)
    _write_checkpoint(checkpoint)

    summary = prepare_training_prep(
        dataset_path=dataset,
        init_checkpoint=checkpoint,
        output_dir=output_dir,
        min_formal_preferences=3,
    )

    assert summary["dataset"]["row_count"] == 2
    assert summary["dataset"]["axis_counts"] == {"right.start": 1, "left.end": 1}
    assert summary["recommendations"]["formal_training_recommended"] is False
    assert summary["recommendations"]["pilot_training_possible"] is True
    assert summary["recommendations"]["replace_default_checkpoint"] is False
    assert (output_dir / "training_prep_summary.json").exists()
    assert (output_dir / "summary.md").exists()
    dry_run = (output_dir / "dry_run_1step_cpu.ps1").read_text(encoding="utf-8")
    formal = (output_dir / "train_preference_head.ps1").read_text(encoding="utf-8")
    assert "$env:PYTHONIOENCODING='utf-8'" in dry_run
    assert "--hidden-size 128" in dry_run
    assert "--num-layers 2" in dry_run
    assert "--state-size 32" in dry_run
    assert "--preserve-init-normalization" in formal
    assert "--freeze-backbone" in formal
    assert "$stamp = Get-Date -Format yyyyMMdd_HHmmss" in formal
