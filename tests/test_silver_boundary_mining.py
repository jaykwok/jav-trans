from __future__ import annotations

import json
from pathlib import Path

from tools.boundary.mine_silver_boundary_labels import mine_case


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_mine_case_filters_forced_silver_labels(tmp_path):
    aligned_path = tmp_path / "archived" / "sample" / "sample.aligned_segments.json"
    _write_json(
        aligned_path,
        {
            "audio_path": str(tmp_path / "audio.wav"),
            "segments": [
                {
                    "start": 1.1,
                    "end": 1.8,
                    "text": "こんにちは",
                    "source_chunk_index": 0,
                    "words": [
                        {
                            "start": 1.1,
                            "end": 1.4,
                            "word": "こん",
                            "source_chunk_index": 0,
                        },
                        {
                            "start": 1.4,
                            "end": 1.8,
                            "word": "にちは",
                            "source_chunk_index": 0,
                        },
                    ],
                },
                {
                    "start": 3.0,
                    "end": 3.5,
                    "text": "ああ",
                    "source_chunk_index": 1,
                    "words": [
                        {
                            "start": 3.0,
                            "end": 3.0,
                            "word": "あ",
                            "source_chunk_index": 1,
                        }
                    ],
                },
            ],
            "asr_details": {
                "transcript_chunks": [
                    {
                        "index": 0,
                        "start": 1.0,
                        "end": 2.0,
                        "duration": 1.0,
                        "text": "こんにちは",
                        "raw_text": "こんにちは",
                        "alignment_fallback_abs_start_s": 1.0,
                        "alignment_fallback_abs_end_s": 1.95,
                        "alignment_fallback_duration_s": 0.95,
                        "alignment_fallback_source": "speech_core",
                    },
                    {
                        "index": 1,
                        "start": 3.0,
                        "end": 4.0,
                        "duration": 1.0,
                        "text": "ああ",
                        "raw_text": "ああ",
                    },
                    {
                        "index": 2,
                        "start": 5.0,
                        "end": 6.0,
                        "duration": 1.0,
                        "text": "~~~",
                        "raw_text": "~~~",
                    },
                ],
            },
            "asr_log": [
                "chunk 1: Alignment 词数: 2",
                "chunk 1: Alignment 模式: forced_aligner",
                "chunk 2: Alignment 词数: 1",
                "chunk 2: Alignment 哨兵触发: 時間軸異常",
                "chunk 2: Alignment 模式: even_fallback",
                "chunk 3: Alignment 模式: even_fallback",
            ],
        },
    )

    silver, hard_cases, summary = mine_case(aligned_path, tmp_path)

    assert len(silver) == 1
    assert silver[0]["chunk_index"] == 0
    assert silver[0]["display_start_label"] == 1.1
    assert silver[0]["display_end_label"] == 1.8
    assert silver[0]["start_error_s"] == -0.1
    assert summary["silver_count"] == 1
    assert summary["hard_case_count"] == len(hard_cases)
    assert any(item["chunk_index"] == 1 for item in hard_cases)
    assert any(item["chunk_index"] == 2 for item in hard_cases)
