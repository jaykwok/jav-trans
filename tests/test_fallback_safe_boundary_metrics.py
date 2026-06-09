from __future__ import annotations

import json
from pathlib import Path

from tools.asr.diagnostics.measure_fallback_safe_boundaries import main
from tools.boundary.ja.analyze_fallback_cut_signal import analyze


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_fallback_safe_boundary_metrics_flags_long_alignment_fallback(tmp_path):
    cache = tmp_path / "boundary-cache.json"
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
                    "speech_segments": [{"start": 2.0, "end": 26.0, "score": 0.9}],
                },
                {
                    "start": 40.0,
                    "end": 44.0,
                    "split_reason": "gap",
                    "core_start": 40.5,
                    "core_end": 43.5,
                    "speech_segments": [{"start": 40.5, "end": 43.5, "score": 0.9}],
                },
                {
                    "start": 50.0,
                    "end": 54.0,
                    "split_reason": "gap",
                    "core_start": 50.5,
                    "core_end": 53.5,
                    "speech_segments": [{"start": 50.5, "end": 53.5, "score": 0.9}],
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
                "fallback_type": "vad_coarse",
                "fallback_subtype": "vad_coarse_after_sentinel",
                "sentinel_lines": ["sentinel"],
                "failure_bucket": "vad_coarse_alignment",
                "display_text": "long fallback",
            },
            {
                "chunk_index": 1,
                "start": 40.0,
                "end": 44.0,
                "alignment_quality": "vad_coarse",
                "fallback_type": "vad_coarse",
                "fallback_subtype": "vad_coarse_after_sentinel",
                "sentinel_lines": ["sentinel"],
                "failure_bucket": "vad_coarse_alignment",
                "display_text": "short fallback",
            },
            {
                "chunk_index": 2,
                "start": 50.0,
                "end": 54.0,
                "alignment_quality": "forced",
                "fallback_type": "none",
                "fallback_subtype": "none",
                "failure_bucket": "",
            },
        ],
    )
    output_dir = tmp_path / "out"

    assert main(
        [
            "--boundary-cache",
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


def test_fallback_safe_boundary_metrics_counts_proportional_sentinel_fallback(tmp_path):
    cache = tmp_path / "boundary-cache.json"
    _write_json(
        cache,
        {
            "processing_spans": [
                {
                    "start": 10.0,
                    "end": 20.0,
                    "split_reason": "boundary_refiner:learned_split",
                    "core_start": 10.0,
                    "core_end": 20.0,
                    "speech_segments": [{"start": 10.0, "end": 20.0, "score": 0.9}],
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
                "start": 10.0,
                "end": 20.0,
                "alignment_quality": "proportional",
                "fallback_type": "proportional",
                "fallback_subtype": "proportional_after_sentinel",
                "sentinel_lines": ["sentinel"],
                "failure_bucket": "proportional_alignment",
                "display_text": "sentinel fallback",
            },
        ],
    )
    output_dir = tmp_path / "out"

    assert main(
        [
            "--boundary-cache",
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
    row = json.loads((output_dir / "chunk_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    unsafe = [
        json.loads(line)
        for line in (output_dir / "unsafe_fallback_chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["fallback_chunk_count"] == 1
    assert summary["fallback_unsafe_count"] == 1
    assert summary["sentinel_fallback_count"] == 1
    assert summary["fallback_reason_counts"] == {"proportional_after_sentinel": 1}
    assert row["fallback_reason"] == "proportional_after_sentinel"
    assert "sentinel_fallback" in row["risk_reasons"]
    assert unsafe[0]["fallback_reason"] == "proportional_after_sentinel"


def test_fallback_safe_boundary_metrics_uses_fallback_window_duration(tmp_path):
    cache = tmp_path / "boundary-cache.json"
    _write_json(
        cache,
        {
            "signature": {"audio": {"name": "clip.wav"}},
            "processing_spans": [
                {
                    "start": 0.0,
                    "end": 12.0,
                    "split_reason": "soft_valley_candidate",
                    "core_start": 2.0,
                    "core_end": 8.0,
                    "speech_segments": [{"start": 2.0, "end": 8.0, "score": 0.9}],
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
                "end": 12.0,
                "duration_s": 12.0,
                "fallback_window_start": 2.0,
                "fallback_window_end": 8.0,
                "fallback_duration_s": 6.0,
                "fallback_window_source": "speech_core",
                "alignment_quality": "proportional",
                "fallback_type": "proportional",
                "fallback_subtype": "proportional_after_sentinel",
                "sentinel_lines": ["sentinel"],
                "failure_bucket": "proportional_alignment",
                "display_text": "core fallback",
            },
        ],
    )
    boundary_manifest = tmp_path / "boundary_manifest.jsonl"
    _write_jsonl(
        boundary_manifest,
        [
            {
                "audio_id": "clip",
                "duration_s": 12.0,
                "actual_speech_segments": [{"start": 2.0, "end": 8.0}],
            },
        ],
    )
    output_dir = tmp_path / "out"

    assert main(
        [
            "--boundary-cache",
            str(cache),
            "--diagnostics",
            str(diagnostics),
            "--boundary-manifest",
            str(boundary_manifest),
            "--output-dir",
            str(output_dir),
            "--target-duration-s",
            "8",
        ]
    ) == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    row = json.loads((output_dir / "chunk_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    unsafe = (output_dir / "unsafe_fallback_chunks.jsonl").read_text(encoding="utf-8")

    assert summary["fallback_chunk_count"] == 1
    assert summary["fallback_unsafe_count"] == 0
    assert summary["fallback_duration_s"]["p50"] == 6.0
    assert summary["fallback_chunk_duration_s"]["p50"] == 12.0
    assert summary["truth_start_abs_error_s"]["p50"] == 0.0
    assert summary["truth_end_abs_error_s"]["p50"] == 0.0
    assert row["duration_s"] == 12.0
    assert row["core_offset_left_s"] == 2.0
    assert row["core_offset_right_s"] == 4.0
    assert row["fallback_duration_s"] == 6.0
    assert row["fallback_window_source"] == "speech_core"
    assert row["fallback_safe"] is True
    assert "long_fallback_chunk" not in row["risk_reasons"]
    assert unsafe == ""


def test_fallback_safe_boundary_metrics_uses_sentinel_lines_not_subtype_name(tmp_path):
    cache = tmp_path / "boundary-cache.json"
    _write_json(
        cache,
        {
            "processing_spans": [
                {
                    "start": 0.0,
                    "end": 12.0,
                    "split_reason": "boundary_refiner:learned_split",
                    "core_start": 0.0,
                    "core_end": 12.0,
                    "speech_segments": [{"start": 0.0, "end": 12.0, "score": 0.9}],
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
                "end": 12.0,
                "alignment_quality": "partial",
                "fallback_type": "none",
                "fallback_subtype": "proportional_after_sentinel",
                "sentinel_lines": ["sentinel"],
                "failure_bucket": "partial_alignment",
            },
        ],
    )
    output_dir = tmp_path / "out"

    assert main(
        [
            "--boundary-cache",
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
    row = json.loads((output_dir / "chunk_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert summary["fallback_chunk_count"] == 0
    assert summary["sentinel_fallback_count"] == 0
    assert row["fallback_reason"] == ""
    assert row["sentinel_fallback"] is False


def test_fallback_safe_boundary_metrics_rejects_legacy_diagnostics_without_fallback_type(tmp_path):
    cache = tmp_path / "boundary-cache.json"
    _write_json(
        cache,
        {
            "processing_spans": [
                {
                    "start": 0.0,
                    "end": 12.0,
                    "split_reason": "boundary_refiner:learned_split",
                    "core_start": 0.0,
                    "core_end": 12.0,
                    "speech_segments": [{"start": 0.0, "end": 12.0, "score": 0.9}],
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
                "end": 12.0,
                "alignment_quality": "vad_coarse",
                "fallback_subtype": "vad_coarse_after_sentinel",
            },
        ],
    )

    try:
        main(
            [
                "--boundary-cache",
                str(cache),
                "--diagnostics",
                str(diagnostics),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    except ValueError as exc:
        assert "fallback_type" in str(exc)
    else:
        raise AssertionError("legacy diagnostics without fallback_type should be rejected")


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
    cache = tmp_path / "boundary-cache.json"
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
                    "speech_segments": [{"start": 0.0, "end": 1.2, "score": 0.9}],
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
                "fallback_type": "vad_coarse",
                "fallback_subtype": "vad_coarse_after_sentinel",
                "sentinel_lines": ["sentinel"],
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
            "--boundary-cache",
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


def test_fallback_cut_signal_analysis_reports_feasible_candidates(tmp_path):
    unsafe = tmp_path / "unsafe.jsonl"
    _write_jsonl(
        unsafe,
        [
            {
                "chunk_index": 3,
                "start": 0.0,
                "end": 24.0,
                "duration_s": 24.0,
                "core_start": 0.0,
                "core_end": 24.0,
                "split_reason": "overlong",
                "fallback_type": "vad_coarse",
                "fallback_subtype": "vad_coarse_after_sentinel",
                "sentinel_lines": ["sentinel"],
                "display_text": "omitted by output",
            },
        ],
    )
    frame_scores = tmp_path / "frame_scores.json"
    _write_json(
        frame_scores,
        {
            "frame_hop_s": 1.0,
            "scores": [0.9] * 24,
            "cut_scores": [0.0] * 8 + [0.99] + [0.0] * 7 + [0.99] + [0.0] * 7,
        },
    )

    summary = analyze(
        unsafe_fallback_chunks=unsafe,
        frame_scores_path=frame_scores,
        output_dir=tmp_path / "out",
        cut_threshold=0.94,
        valley_threshold=0.2,
        target_child_s=9.0,
        min_child_s=1.5,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "cut_signal_details.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["rows_feasible_to_target"] == 1
    assert rows[0]["cut_candidate_count"] == 2
    assert rows[0]["can_reach_target_child"] is True
    assert "display_text" not in rows[0]
    assert rows[0]["text_chars"] == len("omitted by output")


def test_fallback_cut_signal_analysis_reports_missing_candidates(tmp_path):
    unsafe = tmp_path / "unsafe.jsonl"
    _write_jsonl(
        unsafe,
        [
            {
                "chunk_index": 4,
                "start": 0.0,
                "end": 24.0,
                "duration_s": 24.0,
                "core_start": 0.0,
                "core_end": 24.0,
                "split_reason": "overlong",
                "fallback_type": "vad_coarse",
                "fallback_subtype": "vad_coarse_after_sentinel",
                "sentinel_lines": ["sentinel"],
            },
        ],
    )
    frame_scores = tmp_path / "frame_scores.json"
    _write_json(
        frame_scores,
        {
            "frame_hop_s": 1.0,
            "scores": [0.9] * 24,
            "cut_scores": [0.0] * 24,
        },
    )

    summary = analyze(
        unsafe_fallback_chunks=unsafe,
        frame_scores_path=frame_scores,
        output_dir=tmp_path / "out",
        cut_threshold=0.94,
        valley_threshold=0.2,
        target_child_s=9.0,
        min_child_s=1.5,
    )

    assert summary["rows_with_any_candidate"] == 0
    assert summary["rows_feasible_to_target"] == 0


def test_fallback_cut_signal_analysis_filters_chunk_metrics_input(tmp_path):
    metrics = tmp_path / "chunk_metrics.jsonl"
    _write_jsonl(
        metrics,
        [
            {
                "chunk_index": 1,
                "start": 0.0,
                "end": 24.0,
                "duration_s": 24.0,
                "core_start": 0.0,
                "core_end": 24.0,
                "fallback_reason": "vad_coarse_after_sentinel",
                "fallback_safe": False,
            },
            {
                "chunk_index": 2,
                "start": 30.0,
                "end": 34.0,
                "duration_s": 4.0,
                "core_start": 30.0,
                "core_end": 34.0,
                "fallback_reason": "",
                "fallback_safe": True,
            },
            {
                "chunk_index": 3,
                "start": 40.0,
                "end": 44.0,
                "duration_s": 4.0,
                "core_start": 40.0,
                "core_end": 44.0,
                "fallback_reason": "vad_coarse_after_sentinel",
                "fallback_safe": True,
            },
        ],
    )
    frame_scores = tmp_path / "frame_scores.json"
    _write_json(frame_scores, {"frame_hop_s": 1.0, "scores": [0.9] * 48, "cut_scores": [0.0] * 48})

    summary = analyze(
        unsafe_fallback_chunks=metrics,
        frame_scores_path=frame_scores,
        output_dir=tmp_path / "out",
        cut_threshold=0.94,
        valley_threshold=0.2,
        target_child_s=9.0,
        min_child_s=1.5,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "cut_signal_details.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["input_row_count"] == 3
    assert summary["row_count"] == 1
    assert rows[0]["chunk_index"] == 1
