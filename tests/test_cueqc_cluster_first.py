from __future__ import annotations

import json
from pathlib import Path

from asr import cueqc
from tools.asr.cueqc.cluster_candidates import cluster_rows, main as cluster_main
from tools.asr.cueqc.compile_training_set import compile_records
from tools.asr.cueqc.enrich_embeddings import enrich_text_embeddings
from tools.asr.cueqc.export_candidates import aligned_payload_to_candidates


def _candidate(
    index: int,
    text: str,
    *,
    severity: str = "ok",
    density: str = "normal_dialogue",
    embedding: list[float] | None = None,
) -> dict:
    duration = 0.6 if len(text) <= 3 else 2.0
    tf = cueqc.text_features(text, text, duration_s=duration)
    row = {
        "schema": "cueqc_candidate_v1",
        "sample_id": f"sample-{index:03d}",
        "cluster_id": "",
        "chunk_index": index,
        "start": float(index),
        "end": float(index) + duration,
        "duration_s": duration,
        "audio": {"path": f"chunk-{index}.wav", "exists": False, "sha1": ""},
        "text": text,
        "raw_text": text,
        "text_preview": text,
        "text_features": {
            key: value
            for key, value in tf.items()
            if key not in {"text", "raw_text", "compact_text", "context_compact_text"}
        },
        "qc": {
            "severity": severity,
            "reasons": ["repeated_nonlexical_vocalization"] if severity == "warn" else [],
            "text_density": {"level": density},
            "vocalization_repetition": {"preserve_candidate": density != "normal_dialogue"},
        },
        "boundary": {"speech_segment_count": 1, "boundary_reason": "test"},
        "adjacency": {
            "prev_gap_s": 0.2,
            "next_gap_s": 0.2,
            "same_text_run_length": 3 if text == "あ" else 1,
        },
        "asr_signals": {},
        "alignment_diagnostics": {},
    }
    if embedding is not None:
        row["embeddings"] = {
            "text": {
                "model": "unit-test-text",
                "dim": len(embedding),
                "normalized": False,
                "vector": list(embedding),
            },
            "audio": {
                "model": "unit-test-audio",
                "dim": len(embedding),
                "normalized": False,
                "vector": list(reversed(embedding)),
            },
        }
    return row


def test_cueqc_candidate_export_from_aligned_payload_preserves_required_fields(tmp_path: Path):
    payload = {
        "audio_path": str(tmp_path / "audio.wav"),
        "asr_details": {
            "transcript_chunks": [
                {
                    "index": 0,
                    "start": 0.0,
                    "end": 0.5,
                    "duration": 0.5,
                    "text": "あ",
                    "raw_text": "あ",
                    "language": "Japanese",
                    "speech_segment_count": 1,
                    "boundary_reason": "unit_test",
                },
                {
                    "index": 1,
                    "start": 1.0,
                    "end": 3.0,
                    "duration": 2.0,
                    "text": "今日はいい天気ですね",
                    "raw_text": "今日はいい天気ですね",
                    "language": "Japanese",
                },
            ],
            "asr_qc": {
                "items": [
                    {
                        "position": 0,
                        "chunk_index": 0,
                        "severity": "warn",
                        "reasons": ["repeated_nonlexical_vocalization"],
                        "metrics": {
                            "text_density": {"level": "short_vocalization_candidate"},
                            "vocalization_repetition": {"preserve_candidate": True},
                            "max_repeat": {"run": 1},
                        },
                    }
                ]
            },
        },
    }

    rows = aligned_payload_to_candidates(payload, video_id="clip")

    assert len(rows) == 2
    assert rows[0]["schema"] == "cueqc_candidate_v1"
    assert rows[0]["audio"]["path"]
    assert rows[0]["text_features"]["char_count"] == 1
    assert rows[0]["boundary"]["speech_segment_count"] == 1
    assert rows[0]["adjacency"]["next_gap_s"] == 0.5
    assert rows[0]["qc"]["severity"] == "warn"
    assert rows[1]["text_features"]["has_stable_vocabulary"] is True


def test_cueqc_finch_cluster_outputs_stable_audit_files(tmp_path: Path):
    rows = [
        _candidate(0, "あ", severity="warn", density="short_vocalization_candidate"),
        _candidate(1, "あ", severity="warn", density="short_vocalization_candidate"),
        _candidate(2, "ん", severity="warn", density="short_vocalization_candidate"),
        _candidate(3, "今日はいい天気ですね"),
        _candidate(4, "明日は学校に行きます"),
        _candidate(5, "これは普通の会話です"),
    ]
    clustered, representatives, summaries, summary = cluster_rows(
        rows,
        method="finch_first_neighbor",
        metric="euclidean",
        min_clusters=2,
        max_clusters=4,
        representatives_per_cluster=2,
    )

    assert len(clustered) == len(rows)
    assert all(row["cluster_id"].startswith("cluster_") for row in clustered)
    assert summary["method"] == "finch_first_neighbor"
    assert 1 <= summary["cluster_count"] <= len(rows)
    assert representatives
    assert summaries

    input_path = tmp_path / "cueqc_candidates.jsonl"
    input_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"
    rc = cluster_main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--method",
            "finch_first_neighbor",
            "--min-clusters",
            "2",
            "--max-clusters",
            "4",
        ]
    )

    assert rc == 0
    assert (output_dir / "cueqc_clusters.jsonl").exists()
    assert (output_dir / "cueqc_cluster_representatives.jsonl").exists()
    html = (output_dir / "cluster_audit.html").read_text(encoding="utf-8")
    assert "CueQC cluster-first" in html
    assert "cueqc_manual_labels.jsonl" in html


def test_cueqc_umap_hdbscan_uses_high_dim_embeddings():
    centers = [
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
    ]
    rows = []
    index = 0
    for cluster_index, center in enumerate(centers):
        for offset in (0.0, 0.1, 0.2, 0.3):
            vector = [
                value + (offset if dim == cluster_index else 0.0)
                for dim, value in enumerate(center)
            ]
            rows.append(_candidate(index, f"テスト{cluster_index}", embedding=vector))
            index += 1

    clustered, representatives, summaries, summary = cluster_rows(
        rows,
        method="umap_hdbscan",
        metric="euclidean",
        feature_space="dense",
        min_clusters=2,
        max_clusters=4,
        min_cluster_size=2,
        min_samples=1,
        selection_method="leaf",
        umap_components=2,
        umap_neighbors=3,
        representatives_per_cluster=1,
        random_state=7,
    )

    assert len(clustered) == len(rows)
    assert summary["method"] == "umap_hdbscan"
    assert summary["feature_space"]["resolved"] == "dense"
    assert summary["feature_space"]["dense"]["sources"] == ["audio", "text"]
    assert summary["cluster_count"] >= 2
    assert representatives
    assert summaries
    assert all(row["cluster_backend"] == "umap_hdbscan" for row in clustered)
    assert all("cluster_confidence" in row for row in clustered)


def test_cueqc_text_embedding_enrichment_writes_dense_vectors(monkeypatch):
    rows = [_candidate(0, "今日はいい天気ですね"), _candidate(1, "あ")]

    def fake_loader(model_name: str, *, device: str):
        def encode(texts: list[str], *, batch_size: int):
            return [[float(index), float(len(text)), 1.0] for index, text in enumerate(texts)]

        return encode, "fake_backend"

    monkeypatch.setattr(
        "tools.asr.cueqc.enrich_embeddings._load_text_embedder",
        fake_loader,
    )

    summary = enrich_text_embeddings(
        rows,
        model_name="fake-text-model",
        device="cpu",
        batch_size=2,
        text_prefix="",
        vector_digits=4,
    )

    assert summary["model"] == "fake-text-model"
    assert summary["dim"] == 3
    assert rows[0]["embeddings"]["text"]["backend"] == "fake_backend"
    assert rows[1]["embeddings"]["text"]["vector"][1] == 1.0


def test_cueqc_training_compile_low_consistency_does_not_hard_reject():
    clusters = [
        {**_candidate(0, "あ"), "cluster_id": "cluster_00"},
        {**_candidate(1, "あ"), "cluster_id": "cluster_00"},
        {**_candidate(2, "今日はいい天気ですね"), "cluster_id": "cluster_01"},
    ]
    labels = [
        {
            "sample_id": "sample-000",
            "cluster_id": "cluster_00",
            "content_label": "non_dialogue",
            "display_decision": "drop",
            "alignment_policy": "skip_align_fallback",
            "qc_decision": "reject",
        },
        {
            "sample_id": "sample-001",
            "cluster_id": "cluster_00",
            "content_label": "dialogue",
            "display_decision": "keep",
            "alignment_policy": "align",
            "qc_decision": "keep",
        },
        {
            "sample_id": "sample-002",
            "cluster_id": "cluster_01",
            "content_label": "dialogue",
            "display_decision": "keep",
            "alignment_policy": "align",
            "qc_decision": "keep",
        },
    ]

    records, skipped, summary = compile_records(
        clusters=clusters,
        manual_labels=labels,
        min_cluster_agreement=0.67,
    )

    assert not skipped
    by_id = {row["sample_id"]: row for row in records}
    assert by_id["sample-000"]["targets"]["qc_decision"] == "review"
    assert by_id["sample-000"]["targets"]["display_hint"] == "review"
    assert by_id["sample-000"]["targets"]["alignment_policy"] == "align"
    assert by_id["sample-000"]["targets"]["hard_reject_target"] is False
    assert by_id["sample-002"]["targets"]["display_hint"] == "keep"
    assert summary["counts"].get("hard_reject_target", 0) == 0


def test_cueqc_shadow_is_conservative_for_stable_dialogue():
    row = _candidate(9, "今日はいい天気ですね")

    decision = cueqc.heuristic_shadow_decision(row)

    assert decision["mode"] == "shadow"
    assert decision["content_type"] == "dialogue"
    assert decision["alignment_policy"] == "align"
    assert decision["display_hint"] == "keep"
