from __future__ import annotations

import json
from pathlib import Path

from tools.asr.diagnostics import diagnose_asr_alignment
from tools.asr.diagnostics.diagnose_asr_alignment import diagnose_case, summarize


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_diagnose_case_marks_alignment_candidates(tmp_path):
    aligned_path = tmp_path / "archived" / "sample" / "sample.aligned_segments.json"
    quality_path = tmp_path / "quality_reports" / "sample.quality_report.json"
    _write_json(
        aligned_path,
        {
            "audio_path": str(tmp_path / "audio.wav"),
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
                },
                {
                    "start": 6.0,
                    "end": 7.0,
                    "text": "こんにちは",
                    "source_chunk_index": 3,
                    "words": [
                        {
                            "start": 6.0,
                            "end": 6.4,
                            "word": "こん",
                            "source_chunk_index": 3,
                        }
                    ],
                },
            ],
            "asr_details": {
                "fallback_count": 1,
                "transcript_chunks": [
                    {
                        "index": 0,
                        "start": 0.0,
                        "end": 2.0,
                        "duration": 2.0,
                        "alignment_fallback_start_s": 0.5,
                        "alignment_fallback_end_s": 1.5,
                        "alignment_fallback_duration_s": 1.0,
                        "alignment_fallback_abs_start_s": 0.5,
                        "alignment_fallback_abs_end_s": 1.5,
                        "alignment_fallback_source": "speech_core",
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
                    {
                        "index": 3,
                        "start": 6.0,
                        "end": 7.0,
                        "duration": 1.0,
                        "text": "こんにちは",
                        "raw_text": "こんにちは",
                    },
                ],
            },
            "asr_log": [
                "chunk 1: Alignment 词数: 2",
                "chunk 1: Alignment 模式: forced_aligner",
                "chunk 1: Alignment 哨兵触发: 时间轴异常，不重新调用 ASR，改用 VAD/比例回退",
                "chunk 1: Alignment 回退: 使用 VAD 约束比例时间戳",
                "chunk 1: Alignment 模式: aligner_vad_fallback",
                "chunk 3: Alignment 异常: forced aligner returned empty words",
                "chunk 3: Alignment 模式: even_fallback",
                "chunk 4: Alignment 词数: 1",
                "chunk 4: Alignment 模式: forced_aligner",
            ],
        },
    )
    _write_json(
        quality_path,
        {
            "alignment_fallback_ratio": 0.5,
        },
    )

    rows, case_summary = diagnose_case(aligned_path=aligned_path, workflow_root=tmp_path)
    summary = summarize(rows, [case_summary])

    assert len(rows) == 4
    by_chunk = {row["chunk_index"]: row for row in rows}
    assert by_chunk[0]["failure_reasons"] == [
        "alignment_fallback",
        "alignment_sentinel",
        "word_timing_zero_heavy",
    ]
    assert by_chunk[0]["alignment_quality"] == "vad_coarse"
    assert by_chunk[0]["fallback_type"] == "vad_coarse"
    assert by_chunk[0]["fallback_subtype"] == "vad_coarse_after_sentinel"
    assert "word_timing_zero_heavy" in by_chunk[0]["word_timing_failure_reasons"]
    assert by_chunk[0]["aligned_path"].endswith("sample.aligned_segments.json")
    assert by_chunk[0]["source_audio_path"].endswith("audio.wav")
    assert by_chunk[0]["failure_candidate"] is True
    assert by_chunk[0]["failure_bucket"] == "vad_coarse_alignment"
    assert by_chunk[0]["fallback_window_start"] == 0.5
    assert by_chunk[0]["fallback_window_end"] == 1.5
    assert by_chunk[0]["fallback_duration_s"] == 1.0
    assert by_chunk[0]["fallback_window_source"] == "speech_core"
    assert "empty_text_for_chunk" in by_chunk[1]["failure_reasons"]
    assert by_chunk[1]["alignment_quality"] == "drop_or_review"
    assert by_chunk[1]["fallback_type"] == "none"
    assert by_chunk[1]["fallback_subtype"] == "asr_empty_text"
    assert by_chunk[1]["failure_bucket"] == "empty_text_for_chunk"
    assert by_chunk[2]["align_text_empty"] is True
    assert by_chunk[2]["alignment_quality"] == "drop_or_review"
    assert by_chunk[2]["fallback_type"] == "proportional"
    assert by_chunk[2]["fallback_subtype"] == "align_text_empty"
    assert by_chunk[2]["failure_bucket"] == "align_text_empty"
    assert "alignment_fallback" in by_chunk[2]["failure_reasons"]
    assert by_chunk[3]["alignment_quality"] == "forced"
    assert by_chunk[3]["failure_candidate"] is False
    assert by_chunk[3]["failure_bucket"] == ""
    assert case_summary["quality_alignment_fallback_ratio"] == 0.5
    assert case_summary["alignment_quality_counts"] == {
        "forced": 1,
        "drop_or_review": 2,
        "vad_coarse": 1,
    }
    assert case_summary["fallback_type_counts"] == {
        "none": 2,
        "proportional": 1,
        "vad_coarse": 1,
    }
    assert case_summary["fallback_subtype_counts"] == {
        "align_text_empty": 1,
        "asr_empty_text": 1,
        "none": 1,
        "vad_coarse_after_sentinel": 1,
    }
    assert case_summary["failure_bucket_counts"] == {
        "align_text_empty": 1,
        "empty_text_for_chunk": 1,
        "vad_coarse_alignment": 1,
    }
    assert summary["failure_candidate_count"] == 3
    assert summary["alignment_quality_counts"] == {
        "forced": 1,
        "drop_or_review": 2,
        "vad_coarse": 1,
    }
    assert summary["failure_bucket_counts"] == {
        "align_text_empty": 1,
        "empty_text_for_chunk": 1,
        "vad_coarse_alignment": 1,
    }
    assert summary["fallback_subtype_counts"] == {
        "align_text_empty": 1,
        "asr_empty_text": 1,
        "none": 1,
        "vad_coarse_after_sentinel": 1,
    }


def test_diagnose_case_separates_punctuation_only_nonlexical_text(tmp_path):
    aligned_path = tmp_path / "archived" / "sample" / "sample.aligned_segments.json"
    _write_json(
        aligned_path,
        {
            "audio_path": str(tmp_path / "audio.wav"),
            "segments": [
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": "...",
                    "source_chunk_index": 0,
                    "words": [],
                }
            ],
            "asr_details": {
                "transcript_chunks": [
                    {
                        "index": 0,
                        "start": 1.0,
                        "end": 2.0,
                        "duration": 1.0,
                        "text": "...、...",
                        "raw_text": "…、…",
                    }
                ],
            },
            "asr_log": [
                "chunk 1: Alignment 词数: 0",
                "chunk 1: Alignment VAD 回退语音区间: 1",
                "chunk 1: Alignment 模式: nonlexical",
            ],
        },
    )

    rows, case_summary = diagnose_case(aligned_path=aligned_path, workflow_root=tmp_path)

    assert len(rows) == 1
    assert rows[0]["align_text_empty"] is True
    assert rows[0]["nonlexical_text"] is True
    assert rows[0]["alignment_quality"] == "nonlexical"
    assert rows[0]["fallback_type"] == "none"
    assert rows[0]["fallback_subtype"] == "nonlexical_text"
    assert rows[0]["failure_bucket"] == "nonlexical_text"
    assert rows[0]["failure_reasons"] == ["nonlexical_text"]
    assert case_summary["alignment_quality_counts"] == {"nonlexical": 1}
    assert case_summary["failure_bucket_counts"] == {"nonlexical_text": 1}


def test_diagnose_case_exports_repetition_and_text_density_fields(tmp_path):
    aligned_path = tmp_path / "archived" / "sample" / "sample.aligned_segments.json"
    _write_json(
        aligned_path,
        {
            "audio_path": str(tmp_path / "audio.wav"),
            "segments": [
                {
                    "start": 4.0,
                    "end": 4.5,
                    "text": "んんんん",
                    "source_chunk_index": 1,
                }
            ],
            "asr_details": {
                "transcript_chunks": [
                    {
                        "index": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "text": "あっ、あっ、あっ、あっ、あっ、あっ、",
                        "raw_text": "あっ、あっ、あっ、あっ、あっ、あっ、",
                    },
                    {
                        "index": 1,
                        "start": 4.0,
                        "end": 8.0,
                        "duration": 4.0,
                        "text": "んんんん",
                        "raw_text": "んんんん",
                    },
                ],
            },
            "asr_log": [
                "chunk 1: Alignment 模式: forced_aligner",
                "chunk 2: Alignment 模式: forced_aligner",
            ],
        },
    )

    rows, case_summary = diagnose_case(aligned_path=aligned_path, workflow_root=tmp_path)
    summary = summarize(rows, [case_summary])

    by_chunk = {row["chunk_index"]: row for row in rows}
    assert by_chunk[0]["failure_bucket"] == "repeat_repair_suggested"
    assert by_chunk[0]["repetition_repair"]["changed"] is True
    assert by_chunk[1]["failure_bucket"] == ""
    assert by_chunk[1]["text_density_level"] == "repeated_vocalization_candidate"
    assert case_summary["failure_bucket_counts"] == {
        "repeat_repair_suggested": 1,
    }
    assert case_summary["repeat_repair_suggested_count"] == 1
    assert summary["text_density_counts"] == {
        "repeated_vocalization_candidate": 2,
    }


def test_cli_broadcasts_single_case_label_for_multiple_aligned_jsons(tmp_path, monkeypatch):
    workflow_root = tmp_path / "workflow"
    for stem in ("sample-a", "sample-b"):
        _write_json(
            workflow_root / "archived" / stem / f"{stem}.aligned_segments.json",
            {
                "audio_path": str(tmp_path / f"{stem}.wav"),
                "segments": [
                    {
                        "start": 4.0,
                        "end": 4.5,
                        "text": "んんんん",
                        "source_chunk_index": 1,
                    }
                ],
                "asr_details": {
                    "transcript_chunks": [
                        {
                            "index": 0,
                            "start": 0.0,
                            "end": 1.0,
                            "duration": 1.0,
                            "text": "こんにちは",
                            "raw_text": "こんにちは",
                        }
                    ],
                },
                "asr_log": ["chunk 1: Alignment 模式: forced_aligner"],
            },
        )

    output_dir = tmp_path / "diag"
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose_asr_alignment.py",
            "--workflow-root",
            str(workflow_root),
            "--case-label",
            "shared_label",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert diagnose_asr_alignment.main() == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    labels = {item["case_label"] for item in summary["cases"]}

    assert summary["case_count"] == 2
    assert labels == {"shared_label"}
