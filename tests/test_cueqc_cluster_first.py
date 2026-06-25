from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from asr import cueqc
from boundary.sequence_features import FrameSequenceFeatureProvider
from tools.asr.cueqc.cluster_candidates import cluster_rows, main as cluster_main
from tools.asr.cueqc.compile_training_set import _broadcast_cluster_labels, compile_records
from tools.asr.cueqc.export_candidates import aligned_payload_to_candidates
from tools.asr.cueqc.torque import (
    _community_min_distance_matrix,
    _merge_layers_fast,
)


def _candidate(
    index: int,
    text: str,
    *,
    severity: str = "ok",
    embedding: list[float] | None = None,
) -> dict:
    duration = 0.6 if len(text) <= 3 else 2.0
    tf = cueqc.text_features(text, text, duration_s=duration)
    row = {
        "schema": "cueqc_candidate_v4",
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
        },
        "cue_features": {
            "text_observation": {
                "duration_s": duration,
                "char_count": tf["char_count"],
                "raw_char_count": tf["raw_char_count"],
                "unique_chars": tf["unique_chars"],
                "chars_per_sec": tf["chars_per_sec"],
                "kana_only": tf["kana_only"],
                "repeat_run": tf["repeat_profile"]["run"],
                "repeat_unit_len": tf["repeat_profile"]["unit_len"],
                "repeat_ratio": tf["repeat_profile"]["ratio"],
            },
            "repeat_profile": tf["repeat_profile"],
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
    assert rows[0]["schema"] == "cueqc_candidate_v4"
    assert rows[0]["audio"]["path"]
    assert rows[0]["text_features"]["char_count"] == 1
    assert rows[0]["boundary"]["speech_segment_count"] == 1
    assert rows[0]["adjacency"]["next_gap_s"] == 0.5
    assert rows[0]["cue_features"]["text_observation"]["char_count"] == 1
    assert "text_density" not in rows[0]["cue_features"]
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


def test_cueqc_torque_auto_prefers_pre_asr_numeric_features():
    rows = []
    scalar_feature_names = ["duration_s", "scorer_speech_mean", "scorer_split_max"]
    pooled_feature_names = ["chunk_ptm_mean_000", "chunk_ptm_mean_001"]
    feature_names = scalar_feature_names + pooled_feature_names
    for index, values in enumerate(
        [
            [0.25, 0.15, 0.05],
            [0.30, 0.18, 0.08],
            [2.40, 0.94, 0.70],
            [2.55, 0.91, 0.76],
        ]
    ):
        rows.append(
            {
                "schema": "pre_asr_cueqc_v9_audit_candidate",
                "sample_id": f"preasr-sample-{index:03d}",
                "chunk_index": index,
                "start": float(index),
                "end": float(index) + values[0],
                "duration_s": values[0],
                "text": "",
                "features": dict(zip(scalar_feature_names, values)),
                "feature_names": feature_names,
                "ptm_pooling_available": True,
                "pre_asr_ptm_pooled_features": [float(index), float(index + 1)],
            }
        )

    clustered, representatives, summaries, summary = cluster_rows(
        rows,
        metric="euclidean",
        feature_space="auto",
        representatives_per_cluster=1,
    )

    assert len(clustered) == len(rows)
    assert representatives
    assert summaries
    assert summary["feature_space"]["resolved"] == "pre_asr"
    assert (
        summary["feature_space"]["pre_asr"]["source"]
        == "features+pre_asr_ptm_pooled_features"
    )
    assert summary["feature_space"]["pre_asr"]["feature_names"] == feature_names
    assert summary["feature_space"]["pre_asr"]["rows_with_features"] == len(rows)
    assert summary["feature_space"]["pre_asr"]["rows_with_ptm_pooling"] == len(rows)
    assert summary["feature_space"]["pre_asr"]["ptm_pooling_dim"] == len(pooled_feature_names)
    assert summary["feature_space"]["pre_asr"]["total_dim"] == len(feature_names)


def _naive_torc_merge_layers(dm: np.ndarray, *, max_layer: int) -> list[np.ndarray]:
    values = dm.copy()
    np.fill_diagonal(values, np.inf)
    n = values.shape[0]
    nearest = np.argmin(values, axis=1).astype(np.int64)
    graph = sp.csr_matrix(
        (
            np.ones(n * 2, dtype=np.float64),
            (
                np.concatenate([np.arange(n, dtype=np.int64), nearest]),
                np.concatenate([nearest, np.arange(n, dtype=np.int64)]),
            ),
        ),
        shape=(n, n),
    )
    _count, point_labels = connected_components(graph, directed=False)
    layers = [point_labels.astype(np.int64, copy=True)]
    while len(layers) <= max_layer:
        _unique, point_to_comm = np.unique(point_labels, return_inverse=True)
        comm_num = int(point_to_comm.max()) + 1
        sizes = np.bincount(point_to_comm, minlength=comm_num)
        if comm_num <= 1:
            break
        inter = np.full((comm_num, comm_num), np.inf, dtype=np.float64)
        for left in range(comm_num):
            left_points = np.where(point_to_comm == left)[0]
            for right in range(comm_num):
                if left == right:
                    continue
                right_points = np.where(point_to_comm == right)[0]
                inter[left, right] = float(np.min(values[np.ix_(left_points, right_points)]))
        rows = []
        cols = []
        for left in range(comm_num):
            for right in np.argsort(inter[left], kind="mergesort"):
                if left == right:
                    continue
                if sizes[left] <= sizes[int(right)]:
                    rows.extend([left, int(right)])
                    cols.extend([int(right), left])
                    break
        if not rows:
            break
        graph = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(comm_num, comm_num))
        _count, comm_bins = connected_components(graph, directed=False)
        next_labels = comm_bins[point_to_comm].astype(np.int64, copy=False)
        if np.array_equal(next_labels, point_labels):
            break
        point_labels = next_labels
        layers.append(point_labels.copy())
    return layers


def test_torc_fast_community_distance_matches_naive_min_distance():
    dm = np.array(
        [
            [0.0, 1.0, 5.0, 6.0, 9.0],
            [1.0, 0.0, 4.0, 7.0, 8.0],
            [5.0, 4.0, 0.0, 2.0, 3.0],
            [6.0, 7.0, 2.0, 0.0, 1.5],
            [9.0, 8.0, 3.0, 1.5, 0.0],
        ],
        dtype=np.float64,
    )
    point_to_comm = np.array([0, 0, 1, 2, 2], dtype=np.int64)

    fast = _community_min_distance_matrix(dm, point_to_comm, 3, chunk_rows=2)
    naive = np.array(
        [
            [np.inf, 4.0, 6.0],
            [4.0, np.inf, 2.0],
            [6.0, 2.0, np.inf],
        ],
        dtype=np.float64,
    )

    assert np.allclose(fast, naive)


def test_torc_fast_merge_layers_match_naive_reference_and_preserve_input():
    dm = np.array(
        [
            [0.0, 0.2, 3.0, 3.2, 7.0, 7.2],
            [0.2, 0.0, 2.8, 3.4, 7.3, 7.1],
            [3.0, 2.8, 0.0, 0.3, 3.5, 3.8],
            [3.2, 3.4, 0.3, 0.0, 3.2, 3.4],
            [7.0, 7.3, 3.5, 3.2, 0.0, 0.4],
            [7.2, 7.1, 3.8, 3.4, 0.4, 0.0],
        ],
        dtype=np.float64,
    )
    original = dm.copy()

    fast_layers = _merge_layers_fast(dm, max_layer=3)
    naive_layers = _naive_torc_merge_layers(dm, max_layer=3)

    assert np.array_equal(dm, original)
    assert len(fast_layers) == len(naive_layers)
    assert [len(np.unique(layer)) for layer in fast_layers] == [
        len(np.unique(layer)) for layer in naive_layers
    ]
    for fast, naive in zip(fast_layers, naive_layers):
        assert np.array_equal(fast, naive)


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
    assert summary["backend"]["fast_merge_layer"] is True
    assert "auto_config" not in summary["backend"]
    assert "adjustment_factor" not in summary["backend"]
    # layer 0 is the fine 1-NN layer; layer 1 must be no coarser than layer 0.
    assert summary["backend"]["layer_cluster_counts"][0] >= summary["cluster_count"]
    assert summary["cluster_count"] >= 1
    assert summary["backend"]["noise_count"] == 0  # merge_layer carries no noise flag
    assert len(clustered) == len(rows)


def test_cueqc_torque_outputs_stable_audit_files(tmp_path: Path):
    """TORC over structured features writes the audit artifacts unchanged."""
    rows = [
        _candidate(0, "あ", severity="warn"),
        _candidate(1, "あ", severity="warn"),
        _candidate(2, "ん", severity="warn"),
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


def test_cueqc_torc_layer_preview_cli_writes_counts(tmp_path: Path):
    rows = [
        _candidate(0, "あ", severity="warn"),
        _candidate(1, "あ", severity="warn"),
        _candidate(2, "ん", severity="warn"),
        _candidate(3, "今日はいい天気ですね"),
        _candidate(4, "明日は学校に行きます"),
        _candidate(5, "これは普通の会話です"),
    ]
    input_path = tmp_path / "cueqc_candidates.jsonl"
    input_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_dir = tmp_path / "preview"

    rc = cluster_main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--feature-space",
            "structured",
            "--layer-preview-max",
            "2",
            "--layer-preview-only",
        ]
    )

    assert rc == 0
    preview = json.loads((output_dir / "torc_layer_preview.json").read_text(encoding="utf-8"))
    assert preview["schema"] == "cueqc_torc_layer_preview_v1"
    assert preview["method"] == "torque"
    assert preview["max_layer_requested"] == 2
    assert preview["layers"]
    assert preview["layers"][0]["layer"] == 0
    assert preview["layers"][0]["cluster_count"] >= preview["layers"][-1]["cluster_count"]
    assert not (output_dir / "cueqc_clusters.jsonl").exists()


def test_cueqc_runtime_signature_is_v4_binary_shadow(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "cueqc_mamba_v4_binary.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setenv("ASR_BACKEND", "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame")
    monkeypatch.setenv(
        "CUEQC_MODEL_PATH_BY_REPO",
        f"jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame={checkpoint}",
    )
    monkeypatch.setenv("CUEQC_SHADOW_ENABLED", "1")

    sig = cueqc.runtime_signature()

    assert sig["policy"] == "cueqc_mamba_v4_binary_shadow"
    assert sig["model_version"] == "cueqc_mamba_v4_binary"
    assert sig["decision_version"] == "cueqc_display_binary_v1"
    assert sig["drop_apply_enabled"] is False
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


def test_cueqc_v4_does_not_extend_boundary_frame_provider_api():
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
        {**_candidate(3, ""), "cluster_id": "cluster_03"},
        {**_candidate(4, ""), "cluster_id": "cluster_04"},
    ]
    cluster_labels = [
        {"cluster_id": "cluster_00", "seed_action": "use_seed", "display_decision": "keep"},
        {"cluster_id": "cluster_01", "seed_action": "mixed_skip", "display_decision": ""},
        {"cluster_id": "cluster_02", "seed_action": "skip", "display_decision": ""},
        {"cluster_id": "cluster_03", "seed_action": "mixed_skip", "display_decision": "drop"},
        {"cluster_id": "cluster_04", "seed_action": "use_seed", "display_decision": "drop", "training_label_included": False},
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
