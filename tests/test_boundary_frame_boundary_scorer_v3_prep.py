from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from boundary.ja import TeacherSegment, build_supervised_record, write_jsonl
from tools.boundary.ja.prepare_frame_boundary_scorer_v3 import prepare_frame_boundary_scorer_v3


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            [{"audio_id": "clip", "audio": str(path.with_name("clip.wav")), "duration_s": 1.0}],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_frame_boundary_scorer_v3_prep_writes_scripts(tmp_path: Path):
    labels = tmp_path / "labels.jsonl"
    manifest = tmp_path / "manifest.json"
    record = replace(
        build_supervised_record(
            audio_id="clip",
            source="synthetic",
            duration_s=1.0,
            speech_segments=[TeacherSegment(0.1, 0.4), TeacherSegment(0.6, 0.9)],
            frame_hop_s=0.1,
        ),
        boundary_metadata={
            "cut_point_segments": [{"time_s": 0.5}],
            "cut_drop_zones": [{"start": 0.4, "end": 0.6}],
        },
    )
    write_jsonl(labels, [record])
    _write_manifest(manifest)

    summary = prepare_frame_boundary_scorer_v3(
        labels=labels,
        manifest=manifest,
        output_dir=tmp_path / "prep",
        device="cuda",
        batch_size=1,
        max_steps=2,
        positive_weight=3.0,
        negative_weight=4.0,
        cut_positive_weight=5.0,
        cut_negative_weight=2.0,
        cut_loss_weight=1.5,
        focal_gamma=1.0,
        runtime_profiles=["0.7,0.5,0.9"],
    )

    assert summary["schema"] == "speech_boundary_ja_frame_boundary_scorer_v3_prep"
    assert summary["label_summary"]["cut_point_segments"] == 1
    assert summary["label_summary"]["cut_drop_zones"] == 1
    assert summary["training_config"]["positive_weight"] == pytest.approx(3.0)
    assert summary["eval_config"]["device"] == "cpu"
    assert summary["eval_config"]["batch_size"] == 1
    assert summary["eval_config"]["runtime_profiles"] == ["0.7,0.5,0.9"]
    assert (tmp_path / "prep" / "build_feature_cache.ps1").exists()
    assert (tmp_path / "prep" / "train_frame_boundary_scorer_v3.ps1").exists()
    assert (tmp_path / "prep" / "evaluate_frame_boundary_scorer_v3.ps1").exists()
    assert "--positive-weight 3.0" in (tmp_path / "prep" / "train_frame_boundary_scorer_v3.ps1").read_text(
        encoding="utf-8"
    )
    eval_script = (tmp_path / "prep" / "evaluate_frame_boundary_scorer_v3.ps1").read_text(encoding="utf-8")
    assert "--device 'cpu'" in eval_script
    assert "--batch-size 1" in eval_script
    assert "--runtime-profile '0.7,0.5,0.9'" in eval_script


def test_frame_boundary_scorer_v3_prep_rejects_missing_cut_targets(tmp_path: Path):
    labels = tmp_path / "labels.jsonl"
    manifest = tmp_path / "manifest.json"
    write_jsonl(
        labels,
        [
            build_supervised_record(
                audio_id="clip",
                source="synthetic",
                duration_s=1.0,
                speech_segments=[TeacherSegment(0.1, 0.9)],
                frame_hop_s=0.1,
            )
        ],
    )
    _write_manifest(manifest)

    with pytest.raises(ValueError, match="cut_point_segments or cut_drop_zones"):
        prepare_frame_boundary_scorer_v3(
            labels=labels,
            manifest=manifest,
            output_dir=tmp_path / "prep",
        )
