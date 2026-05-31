from __future__ import annotations

import json
from pathlib import Path

from tools.fusionvad_ja.measure_fallback_safe_boundaries import main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_fallback_safe_boundary_metrics_flags_long_coarse_fallback(tmp_path):
    cache = tmp_path / "vad-cache.json"
    _write_json(
        cache,
        {
            "processing_spans": [
                {
                    "start": 0.0,
                    "end": 28.0,
                    "split_reason": "overlong",
                    "core_start": 2.0,
                    "core_end": 26.0,
                    "vad_segments": [{"start": 2.0, "end": 26.0, "score": 0.9}],
                },
                {
                    "start": 40.0,
                    "end": 44.0,
                    "split_reason": "gap",
                    "core_start": 40.5,
                    "core_end": 43.5,
                    "vad_segments": [{"start": 40.5, "end": 43.5, "score": 0.9}],
                },
                {
                    "start": 50.0,
                    "end": 54.0,
                    "split_reason": "gap",
                    "core_start": 50.5,
                    "core_end": 53.5,
                    "vad_segments": [{"start": 50.5, "end": 53.5, "score": 0.9}],
                },
            ],
        },
    )
    diagnostics = tmp_path / "diagnostics.jsonl"
    _write_jsonl(
        diagnostics,
        [
            {
                "chunk_index": 0,
                "start": 0.0,
                "end": 28.0,
                "alignment_quality": "vad_coarse",
                "fallback_subtype": "vad_coarse_after_sentinel",
                "failure_bucket": "vad_coarse_alignment",
                "display_text": "long fallback",
            },
            {
                "chunk_index": 1,
                "start": 40.0,
                "end": 44.0,
                "alignment_quality": "vad_coarse",
                "fallback_subtype": "vad_coarse_after_sentinel",
                "failure_bucket": "vad_coarse_alignment",
                "display_text": "short fallback",
            },
            {
                "chunk_index": 2,
                "start": 50.0,
                "end": 54.0,
                "alignment_quality": "forced",
                "fallback_subtype": "none",
                "failure_bucket": "",
            },
        ],
    )
    output_dir = tmp_path / "out"

    assert main(
        [
            "--vad-cache",
            str(cache),
            "--diagnostics",
            str(diagnostics),
            "--output-dir",
            str(output_dir),
            "--target-duration-s",
            "8",
        ]
    ) == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (output_dir / "chunk_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unsafe = [
        json.loads(line)
        for line in (output_dir / "unsafe_fallback_chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["fallback_chunk_count"] == 2
    assert summary["fallback_unsafe_count"] == 1
    assert summary["sentinel_fallback_count"] == 2
    assert summary["fallback_safe_ratio"] == 0.5
    assert rows[0]["fallback_safe"] is False
    assert rows[1]["fallback_safe"] is True
    assert rows[2]["fallback_reason"] == ""
    assert unsafe[0]["chunk_index"] == 0


def test_fallback_safe_boundary_metrics_adds_truth_and_silence_stats(tmp_path):
    import numpy as np
    import soundfile as sf

    audio_path = tmp_path / "chunk.wav"
    audio = np.concatenate(
        [
            np.ones(1600, dtype=np.float32) * 0.1,
            np.zeros(16000, dtype=np.float32),
            np.ones(1600, dtype=np.float32) * 0.1,
        ]
    )
    sf.write(audio_path, audio, 16000)
    cache = tmp_path / "vad-cache.json"
    _write_json(
        cache,
        {
            "signature": {"audio": {"name": "clip.wav"}},
            "processing_spans": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "split_reason": "overlong",
                    "core_start": 0.0,
                    "core_end": 1.2,
                    "vad_segments": [{"start": 0.0, "end": 1.2, "score": 0.9}],
                },
            ],
        },
    )
    diagnostics = tmp_path / "diagnostics.jsonl"
    _write_jsonl(
        diagnostics,
        [
            {
                "chunk_index": 0,
                "video": "clip.wav",
                "start": 0.0,
                "end": 1.2,
                "alignment_quality": "vad_coarse",
                "fallback_subtype": "vad_coarse_after_sentinel",
                "failure_bucket": "vad_coarse_alignment",
                "source_audio_path": str(audio_path),
                "display_text": "fallback",
            },
        ],
    )
    boundary_manifest = tmp_path / "boundary_manifest.jsonl"
    _write_jsonl(
        boundary_manifest,
        [
            {
                "audio_id": "clip",
                "duration_s": 1.2,
                "actual_speech_segments": [{"start": 0.1, "end": 0.2}],
            },
        ],
    )
    output_dir = tmp_path / "out"

    assert main(
        [
            "--vad-cache",
            str(cache),
            "--diagnostics",
            str(diagnostics),
            "--boundary-manifest",
            str(boundary_manifest),
            "--measure-audio-silence",
            "--target-duration-s",
            "0.5",
            "--long-silence-s",
            "0.8",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    row = json.loads((output_dir / "chunk_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert summary["truth_audio_id"] == "clip"
    assert summary["truth_segment_count"] == 1
    assert summary["truth_start_abs_error_s"]["p50"] == 0.1
    assert summary["truth_end_abs_error_s"]["p50"] == 1.0
    assert summary["fallback_long_silence_count"] == 1
    assert row["longest_silence_s"] >= 0.9
    assert "fallback_crosses_long_silence" in row["risk_reasons"]
