from __future__ import annotations

import json
from pathlib import Path

from tools.audits import generate_cueqc_prediction_audit_html as audit


def _prediction(index: int, *, video_id: str, p_drop: float, text: str = "...") -> dict:
    return {
        "schema": "cueqc_prediction_v3_fusion_v1",
        "sample_id": f"cueqc-{video_id}-chunk{index:05d}",
        "audio_id": video_id,
        "video_id": video_id,
        "chunk_index": index,
        "start": float(index),
        "end": float(index) + 0.5,
        "text": text,
        "display_hint": "drop",
        "confidence": p_drop,
        "display_prob_drop": p_drop,
        "display_prob_keep": 1.0 - p_drop,
    }


def test_cueqc_prediction_audit_samples_across_videos_and_bins():
    rows = [
        _prediction(0, video_id="AAA", p_drop=0.851, text="..."),
        _prediction(1, video_id="AAA", p_drop=0.912, text="今日はいい"),
        _prediction(2, video_id="BBB", p_drop=0.872, text="あ"),
        _prediction(3, video_id="BBB", p_drop=0.932, text="長い文章です"),
    ]

    selected = audit.select_audit_rows(rows, max_drop=3, min_drop_confidence=0.85, seed=7)

    assert len(selected) == 3
    assert {row["video_id"] for row in selected} == {"AAA", "BBB"}
    assert all(row["audit_id"] for row in selected)
    assert all(row["audit_bucket"] for row in selected)


def test_cueqc_prediction_audit_builds_html_with_label_schema(tmp_path: Path):
    baseline_root = tmp_path / "baseline"
    archived = baseline_root / "archived" / "AAA"
    audio_dir = baseline_root / "jobs" / "AAA_b5" / "audio"
    archived.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    (archived / "AAA.ja.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n...\n\n", encoding="utf-8")
    (archived / "AAA.aligned_segments.json").write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "..."}]}),
        encoding="utf-8",
    )
    (audio_dir / "AAA.wav").write_bytes(b"RIFF")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps(_prediction(0, video_id="AAA", p_drop=0.9)) + "\n", encoding="utf-8")

    summary = audit.build_audit(
        predictions_jsonl=predictions,
        baseline_root=baseline_root,
        output_dir=tmp_path / "audit",
        title="CueQC false drop",
        dataset_id="test-dataset",
        max_drop=10,
    )

    html = (tmp_path / "audit" / "index.html").read_text(encoding="utf-8")
    assert summary["review_item_count"] == 1
    assert summary["label_schema"] == audit.LABEL_SCHEMA
    assert "误删应保留" in html
    assert "呼吸声" in html
    assert "cueqc_false_drop_audit_labels.jsonl" in html
    assert {"value": "breath", "label": "呼吸声"} in summary["reason_tag_options"]
