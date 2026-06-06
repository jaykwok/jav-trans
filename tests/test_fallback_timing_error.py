from __future__ import annotations

import json
from pathlib import Path

from tools.asr.diagnostics.measure_fallback_timing_error import run_measurement


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_measure_fallback_timing_error_writes_gate_summary(tmp_path):
    boundary_manifest = tmp_path / "boundary_manifest.jsonl"
    _write_jsonl(
        boundary_manifest,
        [
            {
                "audio_id": "clip-a",
                "audio": "audio/clip-a.wav",
                "duration_s": 8.0,
                "actual_speech_segments": [
                    {"start": 1.0, "end": 3.0},
                    {"start": 5.0, "end": 7.0},
                ],
            }
        ],
    )

    aligned_path = tmp_path / "workflow" / "archived" / "clip-a" / "clip-a.aligned_segments.json"
    _write_json(
        aligned_path,
        {
            "audio_path": "audio/clip-a.wav",
            "segments": [
                {
                    "start": 1.05,
                    "end": 2.95,
                    "text": "こんにちは",
                    "source_chunk_index": 0,
                    "words": [{"start": 1.05, "end": 2.95, "word": "こんにちは", "source_chunk_index": 0}],
                },
                {
                    "start": 5.20,
                    "end": 6.90,
                    "text": "こんばんは",
                    "source_chunk_index": 1,
                    "words": [{"start": 5.20, "end": 6.90, "word": "こんばんは", "source_chunk_index": 1}],
                },
            ],
            "asr_details": {
                "transcript_chunks": [
                    {
                        "index": 0,
                        "start": 1.0,
                        "end": 3.0,
                        "duration": 2.0,
                        "text": "こんにちは",
                        "raw_text": "こんにちは",
                    },
                    {
                        "index": 1,
                        "start": 5.0,
                        "end": 7.0,
                        "duration": 2.0,
                        "text": "こんばんは",
                        "raw_text": "こんばんは",
                    },
                ],
                "asr_qc": {},
            },
            "asr_log": [
                "chunk 1: Alignment 词数: 1",
                "chunk 1: Alignment 模式: forced_aligner",
                "chunk 2: Alignment 词数: 1",
                "chunk 2: Alignment 模式: forced_aligner",
                "chunk 2: Alignment 哨兵触发: 时间轴异常，不重新调用 ASR，改用 VAD/比例回退",
                "chunk 2: Alignment 回退: 使用 VAD 约束比例时间戳",
                "chunk 2: Alignment 模式: aligner_vad_fallback",
            ],
        },
    )

    output_dir = tmp_path / "out"
    summary = run_measurement(
        boundary_manifest=boundary_manifest,
        workflow_root=tmp_path / "workflow",
        aligned_jsons=[],
        output_dir=output_dir,
        max_gap_s=0.5,
        gate_unit="cue",
        report_title="Fallback Timing Error Phase 0",
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "measurements.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    markdown = (output_dir / "summary.md").read_text(encoding="utf-8")

    assert summary["case_count"] == 1
    assert summary["gate"]["status"] == "PASS_RECLASSIFICATION_CLEANUP"
    assert any(row["unit"] == "output_segment" and row["quality"] == "forced" for row in rows)
    assert any(row["unit"] == "cue" and row["quality"] == "vad_coarse" for row in rows)
    vad_cue = next(row for row in rows if row["unit"] == "cue" and row["quality"] == "vad_coarse")
    assert vad_cue["abs_start_error_s"] == 0.2
    assert vad_cue["abs_end_error_s"] == 0.1
    assert "Fallback Timing Error Phase 0" in markdown
    assert "`PASS_RECLASSIFICATION_CLEANUP`" in markdown
