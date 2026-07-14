from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
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


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
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
    snr_reference = tmp_path / "snr-reference.jsonl"
    snr_reference.write_text(
        "".join(
            json.dumps(
                {
                    "background_mix": {
                        "enabled": True,
                        "snr_db": snr_db,
                    }
                }
            )
            + "\n"
            for snr_db in (8.0, 11.0, 14.0, 17.0, 20.0, 22.0)
        )
        + json.dumps({"background_mix": {"snr_db": 9.0, "skipped": "low_rms"}})
        + "\n",
        encoding="utf-8",
    )
    return labels, negative_manifest, gap_pool, snr_reference


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_multicore_smoke_covers_split_safe_abstain_and_continue(tmp_path: Path) -> None:
    labels, overlay_manifest, gap_pool, snr_reference = _write_inputs(tmp_path)
    output = tmp_path / "smoke"
    summary = build_smoke(
        semantic_labels=labels,
        overlay_manifest=overlay_manifest,
        gap_duration_pool=gap_pool,
        snr_reference_manifest=snr_reference,
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
    assert summary["overlay_selection"]["music_audio_ids"] == ["music-a", "music-b"]
    assert summary["overlay_mode"] == "additive_full_duration"
    assert summary["all_semantic_cores_have_simultaneous_overlay"] is True
    assert summary["snr_quantiles"]["count"] == 6

    assert len(by_id["ov02_music_over_three_core_two_safe"]["semantic_events"]) == 2
    overlap = by_id["ov04_music_over_overlap_abstain"]
    assert overlap["semantic_events"][0]["inner_target"]["status"] == "abstain"
    assert overlap["core_spans"][1]["start_sample"] < overlap["core_spans"][0]["end_sample"]
    control = by_id["ov05_bgm_switch_over_single_core_continue"]
    assert control["semantic_events"] == []
    assert control["continue_control"] is True

    for row in rows:
        assert row["schema"] == "semantic_split_multicore_additive_overlay_smoke_v2"
        assert "audio" not in row
        assert "negative_manifest" not in row
        assert row["duration_s"] == row["sample_count"] / row["sample_rate"]
        assert row["overlay"]["mode"] == "additive_full_duration"
        assert row["overlay"]["start_sample"] == 0
        assert row["overlay"]["end_sample"] == row["sample_count"]
        assert row["overlay"]["semantic_timeline_effect"] == "none"
        assert row["overlay"]["overlay_transitions_create_semantic_events"] is False
        assert all(source["rendered_sample_count"] == row["sample_count"] for source in row["overlay"]["sources"])
        assert all(source["source_sample_count"] > 0 for source in row["overlay"]["sources"])
        assert row["overlay"]["mix"]["achieved_snr_db"] == pytest.approx(
            row["overlay"]["mix"]["target_snr_db"], abs=1e-5
        )
        clean, clean_rate = sf.read(row["clean_audio"], dtype="float32")
        overlay_audio, overlay_rate = sf.read(row["overlay_audio"], dtype="float32")
        mixed, mixed_rate = sf.read(row["mixed_audio"], dtype="float32")
        assert clean_rate == overlay_rate == mixed_rate == SAMPLE_RATE
        assert len(clean) == len(overlay_audio) == len(mixed) == row["sample_count"]
        assert np.max(np.abs(mixed - (clean + overlay_audio))) <= 3.0 / 32768.0
        assert np.sqrt(np.mean(np.square(overlay_audio.astype(np.float64)))) > 0.0
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
    labels, overlay_manifest, gap_pool, snr_reference = _write_inputs(tmp_path)
    output = tmp_path / "smoke"
    build_smoke(
        semantic_labels=labels,
        overlay_manifest=overlay_manifest,
        gap_duration_pool=gap_pool,
        snr_reference_manifest=snr_reference,
        output_dir=output,
        seed=7,
    )
    page_path = build_audit(
        manifest=output / "multicore_composite_smoke.jsonl",
        output_dir=tmp_path / "audit",
    )
    page = page_path.read_text(encoding="utf-8")
    assert "additive-overlay smoke5" in page
    assert "Clean composite" in page
    assert "Mixed（训练输入）" in page
    assert "Overlay-only" in page
    assert "语义完整可懂" in page
    assert "overlay 合格" in page
    assert "可听且无字幕语义" in page
    assert "不可听/过强/含清楚词语" in page
    assert "Semantic Split" in page
    assert "Inner edge" in page
    assert "overlap abstain" in page
    assert "not_applicable" in page
    assert "semantic_split_multicore_additive_overlay_manual_verdict_v2" in page
    assert "不重叠自适应上下文" in page
    assert "semantic_split_multicore_composite_manual_verdict_v1" not in page
    assert page.count('"sample_id":') == 5
    script = re.search(r"<script>([\s\S]*?)</script>", page)
    assert script is not None
    node = shutil.which("node")
    if node is not None:
        parsed = subprocess.run(
            [node, "--check", "-"],
            input=script.group(1),
            text=True,
            capture_output=True,
            check=False,
        )
        assert parsed.returncode == 0, parsed.stderr
