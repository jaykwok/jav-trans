from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.audits.generate_multicore_composite_audit_html import build_audit
from tools.boundary.ja.build_multicore_composite_smoke import (
    SAMPLE_RATE,
    build_smoke,
)


def _write_wave(path: Path, *, frequency: float, duration_s: float = 2.0) -> None:
    time = np.arange(int(round(duration_s * SAMPLE_RATE)), dtype=np.float32) / SAMPLE_RATE
    audio = (0.08 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    labels = tmp_path / "semantic_labels.jsonl"
    label_rows = []
    for index in range(4):
        audio = tmp_path / f"core-source-{index}.wav"
        _write_wave(audio, frequency=220.0 + index * 30.0)
        label_rows.append(
            {
                "sample_id": f"source-{index}",
                "audio": str(audio),
                "text_units": [
                    {
                        "unit_id": "u00",
                        "text": f"semantic text {index}",
                        "kind": "semantic",
                    }
                ],
                "semantic_alignments": [
                    {
                        "unit_id": "u00",
                        "status": "matched",
                        "start_s": 0.1,
                        "end_s": 1.9,
                        "confidence": 1.0,
                    }
                ],
            }
        )
    labels.write_text(
        "".join(json.dumps(row) + "\n" for row in label_rows), encoding="utf-8"
    )

    negative_manifest = tmp_path / "negative.jsonl"
    negative_rows = []
    for audio_id, flag, frequency, partition in (
        ("a-music-heldout", "music", 70.0, "test"),
        ("music-a", "music", 90.0, "train"),
        ("music-b", "music", 120.0, "train"),
        ("vocal-a", "heavy_breathing+moaning", 160.0, "train"),
    ):
        audio = tmp_path / f"{audio_id}.wav"
        _write_wave(audio, frequency=frequency, duration_s=3.0)
        negative_rows.append(
            {
                "audio": str(audio),
                "audio_id": audio_id,
                "background_type": flag,
                "omni_flags": flag.split("+"),
                "source_partition": partition,
                "duration_s": 3.0,
            }
        )
    negative_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in negative_rows),
        encoding="utf-8",
    )

    gap_pool = tmp_path / "gaps.json"
    gap_pool.write_text(
        json.dumps({"durations_s": [0.1, 0.2, 0.4, 0.8, 1.2, 2.0]}),
        encoding="utf-8",
    )
    return labels, negative_manifest, gap_pool


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_multicore_smoke_covers_split_safe_abstain_and_continue(tmp_path: Path) -> None:
    labels, negative_manifest, gap_pool = _write_inputs(tmp_path)
    output = tmp_path / "smoke"
    summary = build_smoke(
        semantic_labels=labels,
        negative_manifest=negative_manifest,
        gap_duration_pool=gap_pool,
        output_dir=output,
        seed=7,
    )
    rows = _rows(output / "multicore_composite_smoke.jsonl")
    by_id = {row["sample_id"]: row for row in rows}

    assert summary["sample_count"] == 5
    assert summary["core_count_values"] == [1, 2, 3]
    assert summary["semantic_event_count"] == 5
    assert summary["inner_safe_count"] == 4
    assert summary["inner_abstain_count"] == 1
    assert summary["continue_control_count"] == 1
    assert summary["training_ready"] is False
    assert summary["negative_selection"]["music_audio_ids"] == ["music-a", "music-b"]

    assert len(by_id["mc02_music_gap_three_core"]["semantic_events"]) == 2
    overlap = by_id["mc04_overlap_abstain"]
    assert overlap["semantic_events"][0]["inner_target"]["status"] == "abstain"
    assert overlap["core_spans"][1]["start_sample"] < overlap["core_spans"][0]["end_sample"]
    control = by_id["mc05_single_core_bgm_switch_continue"]
    assert control["semantic_events"] == []
    assert control["continue_control"] is True

    for row in rows:
        assert row["duration_s"] == row["sample_count"] / row["sample_rate"]
        for core in row["core_spans"]:
            assert core["start_s"] == core["start_sample"] / SAMPLE_RATE
            assert core["end_s"] == core["end_sample"] / SAMPLE_RATE
        for gap in row["gap_spans"]:
            assert gap["start_s"] == gap["start_sample"] / SAMPLE_RATE
            assert gap["end_s"] == gap["end_sample"] / SAMPLE_RATE
        for event in row["semantic_events"]:
            assert event["representative_s"] == event["representative_sample"] / SAMPLE_RATE
            assert event["event_interval_start_s"] == (
                event["event_interval_start_sample"] / SAMPLE_RATE
            )
            assert event["event_interval_end_s"] == (
                event["event_interval_end_sample"] / SAMPLE_RATE
            )

    core_library = _rows(output / "semantic_core_library.jsonl")
    assert len(core_library) == 4
    assert all(row["duration_s"] == row["sample_count"] / SAMPLE_RATE for row in core_library)


def test_multicore_audit_keeps_split_and_inner_distinct(tmp_path: Path) -> None:
    labels, negative_manifest, gap_pool = _write_inputs(tmp_path)
    output = tmp_path / "smoke"
    build_smoke(
        semantic_labels=labels,
        negative_manifest=negative_manifest,
        gap_duration_pool=gap_pool,
        output_dir=output,
        seed=7,
    )
    page_path = build_audit(
        manifest=output / "multicore_composite_smoke.jsonl",
        output_dir=tmp_path / "audit",
    )
    page = page_path.read_text(encoding="utf-8")
    assert "multi-core composite smoke5" in page
    assert "Semantic Split" in page
    assert "Inner edge" in page
    assert "overlap abstain" in page
    assert "not_applicable" in page
    assert "semantic_split_multicore_composite_manual_verdict_v1" in page
    assert "播放事件上下文" in page
    assert page.count('"sample_id":') == 5
