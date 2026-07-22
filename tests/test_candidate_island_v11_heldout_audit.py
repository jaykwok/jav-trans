from __future__ import annotations

import json
from pathlib import Path
import wave

import pytest

import tools.audits.generate_candidate_island_v11_heldout_audit_html as audit_tool
from tools.audits.generate_candidate_island_v11_heldout_audit_html import (
    MANUAL_VERDICT_SCHEMA,
    build_audit,
)


def _write_wav(path: Path, *, seconds: int = 1) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 16000 * seconds)


def test_candidate_heldout_audit_freezes_partitions_and_uses_candidate_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(audit_tool, "update_audit_entrypoints", lambda **_kwargs: None)
    windows = []
    groups = []
    for index, (video_id, role) in enumerate(
        [("train-video", "train"), ("val-video", "val"), ("test-video", "val")]
    ):
        window_id = f"{video_id}-w00"
        audio = tmp_path / f"{window_id}.wav"
        _write_wav(audio)
        windows.append(
            {
                "schema": "joint_boundary_omni_source_window_v1",
                "window_id": window_id,
                "video_id": video_id,
                "audio_wav": str(audio),
                "duration_s": 1.0,
            }
        )
        groups.append({"audio_id": window_id, "dataset_role": role})
    source_windows = tmp_path / "source_windows.jsonl"
    source_windows.write_text(
        "".join(json.dumps(row) + "\n" for row in windows), encoding="utf-8"
    )
    feature_bundle = tmp_path / "features.pt"
    torch.save({"groups": groups}, feature_bundle)

    index = build_audit(
        source_windows=source_windows,
        feature_bundle=feature_bundle,
        val_video_ids=["val-video"],
        test_video_ids=["test-video"],
        output_dir=tmp_path / "audit",
    )

    page = index.read_text(encoding="utf-8")
    assert MANUAL_VERDICT_SCHEMA in page
    assert "inside_candidate" in page
    assert "outside_candidate" in page
    assert "添加 inside_candidate" in page
    assert "更新区间" in page
    assert 'class="range-start"' in page
    assert 'class="range-end"' in page
    assert "function updateRange(" in page
    assert "materialize_unreviewed_ranges" in page
    assert "complete_with_target_inside_candidate" in page
    assert "complete_all_outside_candidate" in page
    assert "complete_with_target_speech" not in page
    assert "speech_scorer_v10_full_source_span_manual_verdict_v1" not in page
    assert "body{margin:0;background:#f3f5f7" in page
    assert ".inside_candidate{background:var(--inside_candidate)" in page
    assert ".outside_candidate{background:var(--outside_candidate)" in page
    assert "body{margin:0;outside_candidate:" not in page
    assert ".inside_candidate{outside_candidate:" not in page
    manifest = [
        json.loads(line)
        for line in (tmp_path / "audit" / "audit_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["partition"] for row in manifest] == ["test", "val"]
    assert all(row["frame_count"] == 50 for row in manifest)
    partitions = [
        json.loads(line)
        for line in (tmp_path / "audit" / "partition_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["partition"] for row in partitions} == {"train", "val", "test"}
    summary = json.loads(
        (tmp_path / "audit" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["source_count"] == 2
    assert summary["training_manifest_allowed"] is False
    assert summary["model_output_used_as_annotation_seed"] is False
