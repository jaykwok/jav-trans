from __future__ import annotations

import json
from pathlib import Path

from tools.audits import generate_cueqc_cluster_broadcast_html as audit


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cluster_broadcast_audit_generates_separate_lightweight_page(tmp_path: Path):
    clusters = tmp_path / "clusters.jsonl"
    summaries = tmp_path / "summaries.jsonl"
    _write_jsonl(
        clusters,
        [
            {
                "sample_id": "sample-keep-0",
                "candidate_id": "cand-keep-0",
                "cluster_id": "cluster_keep",
                "video_id": "AAA",
                "chunk_index": 0,
                "start": 0.0,
                "end": 0.6,
                "duration_s": 0.6,
                "cluster_confidence": 0.9,
                "text": "短い台詞",
            },
            {
                "sample_id": "sample-keep-1",
                "cluster_id": "cluster_keep",
                "video_id": "AAA",
                "chunk_index": 1,
                "start": 1.0,
                "end": 1.7,
                "duration_s": 0.7,
                "cluster_confidence": 0.8,
                "text": "保留候補",
            },
            {
                "sample_id": "sample-mixed-0",
                "cluster_id": "cluster_mixed",
                "video_id": "BBB",
                "chunk_index": 2,
                "start": 2.0,
                "end": 4.4,
                "duration_s": 2.4,
                "cluster_confidence": 0.4,
                "text": "混ざった簇",
            },
        ],
    )
    _write_jsonl(
        summaries,
        [
            {"cluster_id": "cluster_keep", "count": 2, "cluster_label": "keep-like"},
            {"cluster_id": "cluster_mixed", "count": 1, "cluster_label": "mixed"},
        ],
    )

    summary = audit.build_audit(
        clusters_jsonl=clusters,
        summaries_jsonl=summaries,
        output_dir=tmp_path / "audit",
        title="Cluster broadcast audit",
        dataset_id="unit-dataset",
    )

    html = (tmp_path / "audit" / "index.html").read_text(encoding="utf-8")
    saved_summary = json.loads((tmp_path / "audit" / "summary.json").read_text(encoding="utf-8"))

    assert summary["cluster_review_mode"] == "cluster_label_broadcast"
    assert summary["media_enabled"] is False
    assert summary["separate_from_audio_audit"] is True
    assert summary["cluster_label_export"] == "cueqc_cluster_labels.jsonl"
    assert summary["broadcast_label_export"] == "cueqc_cluster_broadcast_labels.jsonl"
    assert saved_summary["review_item_count"] == 3
    assert "保留并广播" in html
    assert "丢弃并广播" in html
    assert "混簇跳过" in html
    assert "下载广播样本标签" in html
    assert "cueqc_cluster_broadcast_labels.jsonl" in html
    assert "<audio" not in html
    assert "playChunkBtn" not in html


def test_cluster_broadcast_page_only_exports_keep_drop_sample_labels(tmp_path: Path):
    clusters = tmp_path / "clusters.jsonl"
    summaries = tmp_path / "summaries.jsonl"
    _write_jsonl(
        clusters,
        [
            {"sample_id": "sample-a", "cluster_id": "cluster_a", "chunk_index": 0, "text": "a"},
            {"sample_id": "sample-b", "cluster_id": "cluster_b", "chunk_index": 1, "text": "b"},
        ],
    )
    _write_jsonl(
        summaries,
        [
            {"cluster_id": "cluster_a", "count": 1},
            {"cluster_id": "cluster_b", "count": 1},
        ],
    )

    audit.build_audit(
        clusters_jsonl=clusters,
        summaries_jsonl=summaries,
        output_dir=tmp_path / "audit",
        title="Cluster broadcast audit",
        dataset_id="unit-dataset",
    )

    html = (tmp_path / "audit" / "index.html").read_text(encoding="utf-8")

    assert 'ann.seed_action === "use_seed" && (decision === "keep" || decision === "drop")' in html
    assert 'if (!(decision === "keep" || decision === "drop")) return [];' in html
    assert 'training_label_included: Boolean(decision)' in html
    assert "skipped_sample_count: decision ? 0 : rows.length" in html
    assert 'schema: "cueqc_cluster_broadcast_label_v1"' in html
    assert 'label_source: "cluster_broadcast"' in html
