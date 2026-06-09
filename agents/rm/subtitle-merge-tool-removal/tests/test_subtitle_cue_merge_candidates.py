from __future__ import annotations

import json
from pathlib import Path

from tools.subtitles.analyze_subtitle_cue_merge_candidates import build_summary


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_cue_merge_candidate_analysis_merges_consecutive_safe_cues(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SUBTITLE_MERGE_ADJACENT", "0")
    bilingual = _write_json(
        tmp_path / "sample.bilingual.json",
        {
            "blocks": [
                {
                        "start": 0.0,
                        "end": 0.7,
                        "ja_text": "あ",
                        "zh_text": "啊",
                    },
                    {
                        "start": 0.90,
                        "end": 1.48,
                        "ja_text": "ん",
                        "zh_text": "嗯",
                    },
                    {
                        "start": 1.65,
                        "end": 2.30,
                        "ja_text": "だめ",
                        "zh_text": "不行",
                    },
                ]
            },
    )
    out_dir = tmp_path / "out"

    summary = build_summary(
        bilingual_path=bilingual,
        timings_path=None,
        output_dir=out_dir,
        video_fps=29.97,
        min_score=0.5,
        max_gap_s=0.45,
        max_combined_s=4.8,
        max_text_units=34.0,
    )

    assert summary["before"]["block_count"] == 3
    assert summary["after"]["planner_merge_count"] == 2
    assert summary["after"]["block_count"] == 1
    assert summary["outputs"]["before_srt"].endswith("before.bilingual.srt")
    assert summary["outputs"]["planner_vtt"].endswith("planner.bilingual.vtt")
    assert (out_dir / "before.bilingual.srt").exists()
    assert (out_dir / "planner.bilingual.srt").exists()
    assert (out_dir / "before.bilingual.vtt").read_text(encoding="utf-8").startswith("WEBVTT")
    assert (out_dir / "planner.bilingual.vtt").exists()
    assert "speaker_change" not in summary["pair_analysis"]["dense_blocker_counts"]
    actions = json.loads((out_dir / "planner_actions.json").read_text(encoding="utf-8"))
    assert len(actions) == 2
    assert actions[0]["left_index"] == 0
    assert (out_dir / "summary.md").exists()


def test_cue_merge_candidate_analysis_can_use_aligned_segments_as_source(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("SUBTITLE_MERGE_ADJACENT", "0")
    bilingual = _write_json(tmp_path / "sample.bilingual.json", {"blocks": []})
    aligned = _write_json(
        tmp_path / "sample.aligned_segments.json",
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 0.55,
                    "text": "あ",
                    "source_chunk_index": 7,
                    "words": [
                        {
                            "word": "あ",
                            "start": 0.0,
                            "end": 0.55,
                            "source_chunk_index": 7,
                        }
                    ],
                },
                {
                    "start": 0.66,
                    "end": 1.20,
                    "text": "ん",
                    "source_chunk_index": 7,
                    "words": [
                        {
                            "word": "ん",
                            "start": 0.66,
                            "end": 1.20,
                            "source_chunk_index": 7,
                        }
                    ],
                },
            ]
        },
    )
    out_dir = tmp_path / "out-aligned"

    summary = build_summary(
        bilingual_path=bilingual,
        aligned_path=aligned,
        timings_path=None,
        output_dir=out_dir,
        video_fps=29.97,
        min_score=0.5,
        max_gap_s=0.45,
        max_combined_s=4.8,
        max_text_units=34.0,
        mode="srt",
        source="aligned",
    )

    assert summary["source"] == "aligned"
    assert summary["mode"] == "srt"
    assert summary["before"]["block_count"] == 2
    assert summary["after"]["planner_merge_count"] == 1
    assert summary["after"]["block_count"] == 1
    assert summary["outputs"]["planner_srt"].endswith("planner.ja.srt")
    assert (out_dir / "planner.ja.vtt").exists()
    actions = json.loads((out_dir / "planner_actions.json").read_text(encoding="utf-8"))
    assert actions[0]["left_index"] == 0


def test_cue_merge_candidate_analysis_penalizes_fallback_risk(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SUBTITLE_MERGE_ADJACENT", "0")
    bilingual = _write_json(
        tmp_path / "sample.bilingual.json",
        {
            "blocks": [
                {
                    "start": 0.0,
                    "end": 0.7,
                    "ja_text": "あ",
                    "zh_text": "啊",
                    "words": [{"word": "あ", "start": 0.0, "end": 0.7, "source_chunk_index": 10}],
                },
                {
                    "start": 0.9,
                    "end": 1.4,
                    "ja_text": "ん",
                    "zh_text": "嗯",
                    "words": [{"word": "ん", "start": 0.9, "end": 1.4, "source_chunk_index": 11}],
                },
            ]
        },
    )
    diagnostics = _write_jsonl(
        tmp_path / "diagnostics.jsonl",
        [
            {
                "chunk_index": 11,
                "alignment_quality": "vad_coarse",
                "fallback_type": "vad_coarse",
                "fallback_subtype": "vad_coarse_after_sentinel",
                "sentinel_lines": ["sentinel"],
                "failure_reasons": ["alignment_sentinel"],
            }
        ],
    )

    summary = build_summary(
        bilingual_path=bilingual,
        timings_path=None,
        output_dir=tmp_path / "out-risk",
        video_fps=29.97,
        min_score=0.5,
        max_gap_s=0.45,
        max_combined_s=4.8,
        max_text_units=34.0,
        diagnostics_path=diagnostics,
        fallback_risk_policy="penalize",
    )

    assert summary["pair_analysis"]["constraint_counts"]["fallback_risk_pair"] == 1
    assert summary["pair_analysis"]["constraint_counts"]["fallback_risk_boundary"] == 1
    assert summary["after"]["planner_merge_count"] == 0


def test_cue_merge_candidate_analysis_blocks_high_reading_density(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("SUBTITLE_MERGE_ADJACENT", "0")
    bilingual = _write_json(
        tmp_path / "sample.bilingual.json",
        {
            "blocks": [
                {
                    "start": 0.0,
                    "end": 0.7,
                    "ja_text": "あいうえおかきくけこ",
                    "zh_text": "一二三四五六七八九十",
                    "cue_id": 0,
                },
                {
                    "start": 0.9,
                    "end": 1.4,
                    "ja_text": "さしすせそたちつてと",
                    "zh_text": "甲乙丙丁戊己庚辛壬癸",
                    "cue_id": 1,
                },
            ]
        },
    )

    summary = build_summary(
        bilingual_path=bilingual,
        timings_path=None,
        output_dir=tmp_path / "out-reading",
        video_fps=29.97,
        min_score=0.1,
        max_gap_s=0.45,
        max_combined_s=4.8,
        max_text_units=80.0,
        max_reading_units_per_s=12.0,
    )

    assert summary["after"]["planner_merge_count"] == 0
    assert summary["pair_analysis"]["planner_blocker_counts"]["reading_density_too_high"] == 1
    assert summary["planner"]["max_reading_units_per_s"] == 12.0
