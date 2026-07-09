from __future__ import annotations

import json
from pathlib import Path

from tools.datasets.export_pre_asr_long_chunk_qc import collect_long_chunks, export_long_chunk_qc


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_collect_long_chunks_includes_split_rejection_evidence(tmp_path: Path) -> None:
    reexport = tmp_path / "reexport"
    _write_jsonl(
        reexport / "source_windows.jsonl",
        [
            {
                "window_id": "vid-w00",
                "video_id": "vid",
                "source_video": "E:/video/vid.mp4",
                "source_start_s": 100.0,
                "source_end_s": 175.0,
                "audio_wav": "D:/audio/vid-w00.wav",
            }
        ],
    )
    feature_dir = reexport / "features" / "vid-w00"
    _write_jsonl(
        feature_dir / "pre_asr_candidates.jsonl",
        [
            {
                "candidate_id": "preasr-vid-w00-chunk00000",
                "audio_id": "vid-w00",
                "chunk_index": 0,
                "start": 0.0,
                "end": 2.0,
                "duration_s": 2.0,
            },
            {
                "candidate_id": "preasr-vid-w00-chunk00001",
                "audio_id": "vid-w00",
                "chunk_index": 1,
                "start": 2.0,
                "end": 25.0,
                "duration_s": 23.0,
            },
        ],
    )
    _write_jsonl(
        feature_dir / "semantic_split_features.jsonl",
        [
            {"time_s": 10.0, "label": "cut", "p_cut": 0.72, "accepted": False},
            {"time_s": 22.0, "label": "cut", "p_cut": 0.91, "accepted": True},
            {"time_s": 23.0, "label": "continue", "p_cut": 0.05, "accepted": False},
        ],
    )

    rows, summary = collect_long_chunks(reexport_dir=reexport, min_duration_s=15.0)

    assert summary["candidate_count"] == 2
    assert summary["long_chunk_count"] == 1
    assert summary["threshold_counts"]["gt_20s"] == 1
    assert rows[0]["candidate_id"] == "preasr-vid-w00-chunk00001"
    assert rows[0]["source_video_time_start_s"] == 102.0
    assert rows[0]["source_video_time_end_s"] == 125.0
    assert rows[0]["split_label_cut_count"] == 2
    assert rows[0]["split_accepted_cut_count"] == 1
    assert rows[0]["split_cut_ge_0p7_not_accepted_count"] == 1
    assert rows[0]["split_top_cut_accepted"] is True


def test_export_long_chunk_qc_writes_manifest_and_summary(tmp_path: Path) -> None:
    reexport = tmp_path / "reexport"
    feature_dir = reexport / "features" / "vid-w00"
    _write_jsonl(reexport / "source_windows.jsonl", [{"window_id": "vid-w00", "source_start_s": 10.0}])
    _write_jsonl(
        feature_dir / "pre_asr_candidates.jsonl",
        [
            {
                "candidate_id": "preasr-vid-w00-chunk00000",
                "audio_id": "vid-w00",
                "chunk_index": 0,
                "start": 0.0,
                "end": 16.0,
                "duration_s": 16.0,
            }
        ],
    )

    summary = export_long_chunk_qc(reexport_dir=reexport, output_dir=tmp_path / "qc", min_duration_s=15.0)

    manifest_rows = (tmp_path / "qc" / "long_pre_asr_chunks.jsonl").read_text(encoding="utf-8").splitlines()
    stored_summary = json.loads((tmp_path / "qc" / "summary.json").read_text(encoding="utf-8"))
    assert len(manifest_rows) == 1
    assert summary["long_chunk_count"] == 1
    assert stored_summary["review_required"] is True
    assert stored_summary["manifest"].endswith("long_pre_asr_chunks.jsonl")
