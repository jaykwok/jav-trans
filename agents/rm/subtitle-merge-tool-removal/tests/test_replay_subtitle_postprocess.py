from __future__ import annotations

import json
from pathlib import Path

from tools.subtitles.replay_subtitle_postprocess import build_summary


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_replay_subtitle_postprocess_compares_dense_merge(tmp_path: Path):
    bilingual = _write_json(
        tmp_path / "sample.bilingual.json",
        {
            "blocks": [
                {
                    "start": 0.0,
                    "end": 0.60,
                    "ja_text": "あ",
                    "zh_text": "啊",
                    "text": "あ",
                },
                {
                    "start": 0.68,
                    "end": 1.20,
                    "ja_text": "ん",
                    "zh_text": "嗯",
                    "text": "ん",
                },
                {
                    "start": 2.0,
                    "end": 10.0,
                    "ja_text": "あっ" * 8,
                    "zh_text": "啊" * 3,
                    "text": "あっ" * 8,
                },
            ]
        },
    )
    aligned = _write_json(
        tmp_path / "sample.aligned_segments.json",
        {"segments": [{"start": 0.0, "end": 1.0}]},
    )
    timings = _write_json(
        tmp_path / "sample.timings.json",
        {"asr_details": {"asr_qc": {"empty_text_for_speech_count": 0}}},
    )
    out_dir = tmp_path / "out"

    summary = build_summary(
        bilingual_path=bilingual,
        aligned_path=aligned,
        timings_path=timings,
        video_fps=29.97,
        output_dir=out_dir,
    )

    assert summary["before"]["block_count"] == 2
    assert summary["after"]["block_count"] == 2
    assert summary["delta"]["block_count"] == 0
    assert summary["after"]["dense_cue_merge_count"] == 0
    assert summary["after"]["quality"]["subtitle_overlap_count"] == 0
    assert summary["after"]["nonlexical_repetition"]["count"] == 1
    assert (out_dir / "before_blocks.json").exists()
    assert (out_dir / "after_blocks.json").exists()


def test_replay_subtitle_postprocess_can_use_aligned_segments_as_source(tmp_path: Path):
    bilingual = _write_json(tmp_path / "sample.bilingual.json", {"blocks": []})
    aligned = _write_json(
        tmp_path / "sample.aligned_segments.json",
        {
            "segments": [
                {"start": 0.0, "end": 0.60, "text": "あ"},
                {"start": 0.68, "end": 1.20, "text": "ん"},
            ]
        },
    )
    out_dir = tmp_path / "out-aligned"

    summary = build_summary(
        bilingual_path=bilingual,
        aligned_path=aligned,
        timings_path=None,
        video_fps=29.97,
        output_dir=out_dir,
        mode="srt",
        source="aligned",
    )

    assert summary["source"] == "aligned"
    assert summary["mode"] == "srt"
    assert summary["before"]["block_count"] == 1
    assert summary["after"]["block_count"] == 1
