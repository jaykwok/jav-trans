from __future__ import annotations

import json
from pathlib import Path

from tools.subtitles.sweep_speaker_sidecar_cue_planner import build_sweep


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_speaker_sidecar_cue_planner_sweep(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SUBTITLE_MERGE_ADJACENT", "0")
    bilingual = _write_json(
        tmp_path / "sample.bilingual.json",
        {
            "blocks": [
                {"start": 0.0, "end": 0.7, "ja_text": "あ", "zh_text": "啊", "cue_id": 0},
                {"start": 0.9, "end": 1.4, "ja_text": "ん", "zh_text": "嗯", "cue_id": 1},
                {"start": 1.6, "end": 2.2, "ja_text": "だめ", "zh_text": "不行", "cue_id": 2},
            ]
        },
    )
    embeddings = _write_jsonl(
        tmp_path / "speaker_embeddings.jsonl",
        [
            {"segment_id": "0", "start": 0.0, "end": 0.7, "embedding": [1.0, 0.0]},
            {"segment_id": "1", "start": 0.9, "end": 1.4, "embedding": [0.9, 0.1]},
            {"segment_id": "2", "start": 1.6, "end": 2.2, "embedding": [0.0, 1.0]},
        ],
    )

    summary = build_sweep(
        embeddings_path=embeddings,
        bilingual_path=bilingual,
        timings_path=None,
        diagnostics_path=None,
        output_dir=tmp_path / "out",
        thresholds=[0.2, 0.8],
        speaker_policies=["block", "penalize"],
        fallback_risk_policy="penalize",
        video_fps=29.97,
        min_score=0.5,
        max_gap_s=0.45,
        max_combined_s=4.8,
        max_text_units=34.0,
        max_reading_units_per_s=0.0,
    )

    assert len(summary["results"]) == 4
    assert summary["results"][0]["speaker_pairs"]["pair_count"] == 2
    assert summary["planner"]["max_reading_units_per_s"] == 0.0
    assert (tmp_path / "out" / "sweep_summary.json").exists()
    assert (tmp_path / "out" / "sweep_summary.md").exists()
    assert any(row["case"] == "th20-block" for row in summary["results"])
