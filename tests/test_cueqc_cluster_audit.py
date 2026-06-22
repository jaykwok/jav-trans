from __future__ import annotations

import json
from pathlib import Path

from tools.audits import generate_cueqc_cluster_audit_html as audit


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_cueqc_cluster_audit_uses_explicit_media_root(tmp_path: Path):
    archived_root = tmp_path / "archived"
    media_root = tmp_path / "jobs"
    archived = archived_root / "AAA"
    audio_dir = media_root / "custom_job_name" / "audio"
    archived.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    (archived / "AAA.ja.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n...\n\n", encoding="utf-8")
    (archived / "AAA.aligned_segments.json").write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "..."}]}),
        encoding="utf-8",
    )
    (audio_dir / "AAA.somehash.wav").write_bytes(b"RIFF")
    clusters = tmp_path / "cueqc_clusters.jsonl"
    summaries = tmp_path / "cueqc_cluster_summaries.jsonl"
    _write_jsonl(
        clusters,
        [
            {
                "schema": "cueqc_candidate_v4",
                "sample_id": "cueqc-AAA-chunk00000",
                "audio_id": "AAA.somehash",
                "video_id": "AAA",
                "cluster_id": "cluster_00",
                "chunk_index": 0,
                "start": 0.0,
                "end": 0.5,
                "duration_s": 0.5,
                "text": "...",
                "raw_text": "…",
            }
        ],
    )
    _write_jsonl(summaries, [{"cluster_id": "cluster_00", "count": 1}])

    summary = audit.build_audit(
        clusters_jsonl=clusters,
        summaries_jsonl=summaries,
        archived_root=archived_root,
        media_roots=[media_root],
        output_dir=tmp_path / "audit",
        title="CueQC cluster audit",
        dataset_id="test-dataset",
    )

    assert summary["review_item_count"] == 1
    assert summary["cluster_display_decision_options"] == ["keep", "drop"]
    assert summary["missing_media_videos"] == []
    assert summary["missing_subtitle_videos"] == []
    assert summary["media_by_video"]["AAA"]["audio_exists"] is True
    assert summary["media_roots"] == [audit.project_rel(media_root)]
    assert summary["cluster_review_audio_render_mode"] == "per_chunk_inline_audio"
    assert summary["cluster_seed_action_options"] == ["use_seed", "mixed_skip", "skip"]
    assert summary["cluster_training_label_rule"].startswith("only seed_action=use_seed")
    assert "baseline_root" not in summary
    html = (tmp_path / "audit" / "index.html").read_text(encoding="utf-8")
    assert "playChunkBtn" in html
    assert "playContextBtn" in html
    assert "data-play-sample" in html
    assert "data-inline-audio" in html
    assert "clusterSeedActionButtons" in html
    assert "mixed_skip" in html
    assert "每条独立播放器" in html
    assert 'document.createElement("audio")' not in html
    assert "keep" in html
    assert "drop" in html
