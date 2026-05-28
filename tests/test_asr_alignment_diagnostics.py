from __future__ import annotations

import json
from pathlib import Path

from tools.fusionvad_ja.diagnose_asr_alignment import diagnose_case, summarize


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_diagnose_case_marks_alignment_and_asr_drop_candidates(tmp_path):
    aligned_path = tmp_path / "archived" / "sample" / "sample.aligned_segments.json"
    quality_path = tmp_path / "quality_reports" / "sample.quality_report.json"
    _write_json(
        aligned_path,
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 0.5,
                    "text": "ああ",
                    "source_chunk_index": 0,
                    "words": [
                        {
                            "start": 0.0,
                            "end": 0.0,
                            "word": "あ",
                            "source_chunk_index": 0,
                        },
                        {
                            "start": 0.0,
                            "end": 0.0,
                            "word": "あ",
                            "source_chunk_index": 0,
                        },
                    ],
                }
            ],
            "asr_details": {
                "fallback_count": 1,
                "transcript_chunks": [
                    {
                        "index": 0,
                        "start": 0.0,
                        "end": 2.0,
                        "duration": 2.0,
                        "text": "ああ",
                        "raw_text": "ああ",
                    },
                    {
                        "index": 1,
                        "start": 2.0,
                        "end": 5.0,
                        "duration": 3.0,
                        "text": "",
                        "raw_text": "",
                    },
                    {
                        "index": 2,
                        "start": 5.0,
                        "end": 6.0,
                        "duration": 1.0,
                        "text": "~~~♡ｗｗｗ!!!",
                        "raw_text": "~~~♡ｗｗｗ!!!",
                    },
                ],
                "asr_qc": {
                    "items": [
                        {
                            "position": 1,
                            "chunk_index": 1,
                            "severity": "warn",
                            "reasons": ["long_low_value_text"],
                        }
                    ],
                    "dropped_uncertain_items": [
                        {
                            "position": 1,
                            "chunk_index": 1,
                            "reasons": ["long_low_value_text"],
                            "original_text": "んー",
                        }
                    ],
                },
            },
            "asr_log": [
                "chunk 1: Alignment 词数: 2",
                "chunk 1: Alignment 模式: forced_aligner",
                "chunk 1: Alignment 哨兵触发: 时间轴异常，不重新调用 ASR，改用 VAD/比例回退",
                "chunk 1: Alignment 回退: 使用 VAD 约束比例时间戳",
                "chunk 3: Alignment 异常: forced aligner returned empty words",
                "chunk 3: Alignment 模式: even_fallback",
            ],
        },
    )
    _write_json(
        quality_path,
        {
            "alignment_fallback_ratio": 0.5,
            "asr_dropped_uncertain_count": 1,
        },
    )

    rows, case_summary = diagnose_case(aligned_path=aligned_path, workflow_root=tmp_path)
    summary = summarize(rows, [case_summary])

    assert len(rows) == 3
    assert rows[0]["failure_reasons"] == [
        "alignment_fallback",
        "alignment_sentinel",
        "word_timing_zero_heavy",
    ]
    assert rows[0]["alignment_quality"] == "vad_coarse"
    assert rows[0]["fallback_type"] == "vad_coarse"
    assert "asr_dropped_uncertain" in rows[1]["failure_reasons"]
    assert rows[1]["alignment_quality"] == "drop_or_review"
    assert rows[1]["fallback_type"] == "none"
    assert rows[2]["align_text_empty"] is True
    assert rows[2]["alignment_quality"] == "drop_or_review"
    assert rows[2]["fallback_type"] == "proportional"
    assert "alignment_mode_even_fallback" in rows[2]["failure_reasons"]
    assert case_summary["quality_alignment_fallback_ratio"] == 0.5
    assert case_summary["alignment_quality_counts"] == {
        "drop_or_review": 2,
        "vad_coarse": 1,
    }
    assert case_summary["fallback_type_counts"] == {
        "none": 1,
        "proportional": 1,
        "vad_coarse": 1,
    }
    assert summary["failure_candidate_count"] == 3
    assert summary["alignment_quality_counts"] == {
        "drop_or_review": 2,
        "vad_coarse": 1,
    }
