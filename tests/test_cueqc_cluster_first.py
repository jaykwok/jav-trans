from __future__ import annotations

import json
from pathlib import Path

from asr import cueqc
from boundary.sequence_features import FrameSequenceFeatureProvider
from tools.asr.cueqc.cluster_candidates import cluster_rows, main as cluster_main
from tools.asr.cueqc.compile_training_set import _broadcast_cluster_labels, compile_records
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
        "cue_features": {
            "text_density": {"level": density},
            "repeat_profile": tf["repeat_profile"],
            "has_stable_vocabulary": tf["has_stable_vocabulary"],
        },
        "boundary": {"speech_segment_count": 1, "boundary_reason": "test"},
        "adjacency": {
            "prev_gap_s": 0.2,
            "next_gap_s": 0.2,
            "same_text_run_length": 3 if text == "あ" else 1,
        },
        "asr_signals": {},
        "subtitle_timing": {},
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
        },
    }

    rows = aligned_payload_to_candidates(payload, video_id="clip")

    assert len(rows) == 2
    assert rows[0]["schema"] == "cueqc_candidate_v1"
    assert rows[0]["audio"]["path"]
    assert rows[0]["text_features"]["char_count"] == 1
    assert rows[0]["boundary"]["speech_segment_count"] == 1
    assert rows[0]["adjacency"]["next_gap_s"] == 0.5
    assert rows[0]["cue_features"]["text_density"]["level"] == "short_vocalization_candidate"
    assert rows[1]["text_features"]["has_stable_vocabulary"] is True


def test_cueqc_torque_auto_clusters_separated_dense_embeddings():
    """Three well-separated dense clusters -> TORC should recover >=2 clusters."""
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
        metric="euclidean",
        feature_space="dense",
        representatives_per_cluster=1,
    )

    assert len(clustered) == len(rows)
    assert summary["method"] == "torque"
    assert summary["feature_space"]["resolved"] == "dense"
    assert summary["feature_space"]["dense"]["sources"] == ["audio", "text"]
    assert summary["cluster_count"] >= 2
    assert representatives
    assert summaries
    assert all(row["cluster_backend"] == "torque" for row in clustered)
    assert all("cluster_confidence" in row for row in clustered)


def test_cueqc_torque_merge_layer_returns_hierarchy_partition():
    """merge_layer takes a stable hierarchy partition instead of the factor cut."""
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
        metric="euclidean",
        feature_space="dense",
        representatives_per_cluster=1,
        merge_layer=1,
    )

    assert summary["backend"]["mode"] == "merge_layer"
    assert summary["backend"]["merge_layer"] == 1
    # layer 0 is the fine 1-NN layer; layer 1 must be no coarser than layer 0.
    assert summary["backend"]["layer_cluster_counts"][0] >= summary["cluster_count"]
    assert summary["cluster_count"] >= 1
    assert summary["backend"]["noise_count"] == 0  # merge_layer carries no noise flag
    assert len(clustered) == len(rows)


def test_cueqc_torque_outputs_stable_audit_files(tmp_path: Path):
    """TORC over structured features writes the audit artifacts unchanged."""
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
        metric="euclidean",
        feature_space="structured",
        representatives_per_cluster=2,
    )

    assert len(clustered) == len(rows)
    assert summary["method"] == "torque"
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
            "--feature-space",
            "structured",
        ]
    )

    assert rc == 0
    assert (output_dir / "cueqc_clusters.jsonl").exists()
    assert (output_dir / "cueqc_cluster_representatives.jsonl").exists()
    html = (output_dir / "cluster_audit.html").read_text(encoding="utf-8")
    assert "CueQC cluster-first" in html
    assert "cueqc_cluster_labels.jsonl" in html


def test_cueqc_runtime_signature_is_v3_binary_routing(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame")
    monkeypatch.setenv(
        "CUEQC_MODEL_PATH_BY_REPO",
        "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=src/asr/checkpoints/cueqc_mamba_v3_fusion.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt",
    )
    monkeypatch.setenv("CUEQC_DROP_APPLY_ENABLED", "1")

    sig = cueqc.runtime_signature()

    assert sig["policy"] == "cueqc_mamba_v3_fusion"
    assert sig["model_version"] == "cueqc_mamba_v3_fusion"
    assert sig["decision_version"] == "cueqc_display_binary_v1"
    assert sig["drop_apply_enabled"] is True
    assert set(sig) == {
        "schema_version",
        "feature_schema_version",
        "enabled",
        "shadow_only",
        "policy",
        "decision_version",
        "model_version",
        "model_path",
        "checkpoint_sha1",
        "drop_threshold",
        "drop_apply_enabled",
        "shadow_embed_candidates",
    }


def test_cueqc_v3_does_not_extend_boundary_frame_provider_api():
    assert not hasattr(FrameSequenceFeatureProvider, "frames_for_window")


def test_cueqc_training_compile_broadcasts_coarse_cluster_seed_labels():
    clusters = [
        {**_candidate(0, "あ"), "cluster_id": "cluster_00"},
        {**_candidate(1, "あ"), "cluster_id": "cluster_00"},
        {**_candidate(2, "今日はいい天気ですね"), "cluster_id": "cluster_01"},
    ]
    labels = [
        {
            "sample_id": "sample-000",
            "cluster_id": "cluster_00",
            "display_decision": "drop",
        },
        {
            "sample_id": "sample-001",
            "cluster_id": "cluster_00",
            "display_decision": "keep",
        },
        {
            "sample_id": "sample-002",
            "cluster_id": "cluster_01",
            "display_decision": "keep",
        },
    ]

    records, skipped, summary = compile_records(
        clusters=clusters,
        manual_labels=labels,
        min_cluster_agreement=0.67,
    )

    assert not skipped
    by_id = {row["sample_id"]: row for row in records}
    assert by_id["sample-000"]["targets"]["display_decision"] == "drop"
    assert by_id["sample-000"]["targets"]["display_label"] == 0
    assert by_id["sample-001"]["targets"]["display_decision"] == "keep"
    assert by_id["sample-001"]["targets"]["display_label"] == 1
    assert by_id["sample-002"]["targets"]["display_decision"] == "keep"
    assert summary["counts"]["display:keep"] == 2
    assert summary["counts"]["display:drop"] == 1


def test_cueqc_training_compile_cluster_labels_only_keep_drop():
    clusters = [
        {**_candidate(0, "あ"), "cluster_id": "cluster_00"},
        {**_candidate(1, ""), "cluster_id": "cluster_01"},
        {**_candidate(2, ""), "cluster_id": "cluster_02"},
    ]
    labels = [
        {
            "sample_id": "sample-000",
            "cluster_id": "cluster_00",
            "display_decision": "drop",
        },
        {
            "sample_id": "sample-001",
            "cluster_id": "cluster_01",
            "display_decision": "keep",
        },
        {
            "sample_id": "sample-002",
            "cluster_id": "cluster_02",
            "display_decision": "drop",
        },
    ]

    records, skipped, summary = compile_records(
        clusters=clusters,
        manual_labels=labels,
        min_cluster_agreement=0.0,
    )

    assert not skipped
    by_id = {row["sample_id"]: row for row in records}
    assert by_id["sample-000"]["targets"]["display_decision"] == "drop"
    assert by_id["sample-000"]["targets"]["display_label"] == 0
    assert by_id["sample-001"]["targets"]["display_decision"] == "keep"
    assert by_id["sample-001"]["targets"]["display_label"] == 1
    assert by_id["sample-002"]["targets"]["display_decision"] == "drop"
    assert by_id["sample-002"]["targets"]["display_label"] == 0
    assert summary["target_labels"]["display_decision"] == ["drop", "keep"]


def test_cueqc_cluster_broadcast_abstains_mixed_and_skip_labels():
    clusters = [
        {**_candidate(0, "あ"), "cluster_id": "cluster_00"},
        {**_candidate(1, ""), "cluster_id": "cluster_01"},
        {**_candidate(2, ""), "cluster_id": "cluster_02"},
    ]
    cluster_labels = [
        {"cluster_id": "cluster_00", "seed_action": "use_seed", "display_decision": "keep"},
        {"cluster_id": "cluster_01", "seed_action": "mixed_skip", "display_decision": ""},
        {"cluster_id": "cluster_02", "seed_action": "skip", "display_decision": ""},
    ]

    manual_labels = _broadcast_cluster_labels(clusters, cluster_labels)
    records, skipped, summary = compile_records(
        clusters=clusters,
        manual_labels=manual_labels,
        min_cluster_agreement=0.0,
    )

    assert [row["sample_id"] for row in manual_labels] == ["sample-000"]
    assert len(records) == 1
    assert records[0]["targets"]["display_decision"] == "keep"
    assert summary["counts"]["display:keep"] == 1
    assert {row["reason"] for row in skipped} == {"missing_manual_label"}


def test_cueqc_shadow_report_uses_pending_placeholder_before_model_decision():
    row = _candidate(9, "今日はいい天気ですね")

    decision = cueqc.pending_model_decision(row)

    assert decision["mode"] == "pending_cueqc_model"
    assert decision["display_hint"] == "keep"
    assert decision["confidence"] == 1.0
    assert decision["fallback_stage"] == ""
