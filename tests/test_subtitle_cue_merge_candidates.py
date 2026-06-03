from __future__ import annotations

import json
from pathlib import Path

from tools.fusionvad_ja.analyze_subtitle_cue_merge_candidates import build_summary


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_cue_merge_candidate_analysis_keeps_speaker_change(tmp_path: Path, monkeypatch):
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
                    "speaker": "S0",
                },
                {
                    "start": 0.90,
                    "end": 1.48,
                    "ja_text": "ん",
                    "zh_text": "嗯",
                    "speaker": "S0",
                },
                {
                    "start": 1.65,
                    "end": 2.30,
                    "ja_text": "だめ",
                    "zh_text": "不行",
                    "speaker": "S1",
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
    assert summary["after"]["planner_merge_count"] == 1
    assert summary["after"]["block_count"] == 2
    assert "speaker_change" in summary["pair_analysis"]["dense_blocker_counts"]
    actions = json.loads((out_dir / "planner_actions.json").read_text(encoding="utf-8"))
    assert len(actions) == 1
    assert actions[0]["left_index"] == 0
    assert (out_dir / "summary.md").exists()
