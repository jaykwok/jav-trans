from __future__ import annotations

import json
from pathlib import Path

from tools.boundary.export_cueqc_cluster_seed_hardcases import (
    export_cueqc_cluster_seed_hardcases,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _label(
    cluster_id: str,
    *,
    decision: str = "drop",
    seed_action: str = "use_seed",
    included: bool | None = True,
    text_present: int = 0,
    count: int = 2,
) -> dict:
    row = {
        "schema": "cueqc_cluster_label_v1",
        "dataset_id": "test",
        "cluster_id": cluster_id,
        "display_decision": decision,
        "seed_action": seed_action,
        "notes": "",
        "count": count,
        "confidence_avg": 1.0,
        "text_observation_counts": {"empty_text": count - text_present, "text_present": text_present},
        "updated_at": "2026-06-23T00:00:00",
    }
    if included is not None:
        row["training_label_included"] = included
    return row


def _cluster(sample_id: str, cluster_id: str, *, start: float, end: float) -> dict:
    return {
        "schema": "cueqc_candidate_v4",
        "sample_id": sample_id,
        "cluster_id": cluster_id,
        "video_id": "AAA",
        "chunk_index": int(start * 10),
        "start": start,
        "end": end,
        "duration_s": end - start,
        "source_audio_path": "agents/temp/test/audio.wav",
        "audio_id": "AAA.test",
        "text": "...",
        "raw_text": "...",
        "text_features": {"char_count": 0},
        "cluster_confidence": 1.0,
    }


def test_export_cueqc_cluster_seed_hardcases_filters_to_conservative_drop_seed(tmp_path: Path):
    labels = tmp_path / "cueqc_cluster_labels.jsonl"
    clusters = tmp_path / "cueqc_clusters.jsonl"
    _write_jsonl(
        labels,
        [
            _label("cluster_00", decision="drop", seed_action="use_seed", text_present=0, count=2),
            _label("cluster_01", decision="drop", seed_action="mixed_skip", text_present=0, count=1),
            _label("cluster_02", decision="keep", seed_action="use_seed", text_present=0, count=1),
            _label("cluster_03", decision="drop", seed_action="use_seed", included=False, text_present=0, count=1),
            _label("cluster_04", decision="drop", seed_action="use_seed", text_present=1, count=1),
        ],
    )
    _write_jsonl(
        clusters,
        [
            _cluster("sample-000", "cluster_00", start=0.0, end=0.4),
            _cluster("sample-001", "cluster_00", start=1.0, end=1.6),
            _cluster("sample-002", "cluster_01", start=2.0, end=2.5),
            _cluster("sample-003", "cluster_02", start=3.0, end=3.4),
            _cluster("sample-004", "cluster_03", start=4.0, end=4.5),
            _cluster("sample-005", "cluster_04", start=5.0, end=5.5),
        ],
    )

    summary = export_cueqc_cluster_seed_hardcases(
        cluster_labels_path=labels,
        clusters_path=clusters,
        output_dir=tmp_path / "out",
    )

    assert summary["counts"]["seed_drop_clusters"] == 1
    assert summary["counts"]["exported_candidates"] == 2
    assert summary["label_filter_counts"]["excluded_seed_action:mixed_skip"] == 1
    assert summary["label_filter_counts"]["excluded_display:keep"] == 1
    assert summary["label_filter_counts"]["excluded_training_label_false"] == 1
    assert summary["label_filter_counts"]["excluded_text_present_cluster"] == 1

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "cueqc_confirmed_drop_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["sample_id"] for row in rows] == ["sample-000", "sample-001"]
    assert {row["candidate_route"] for row in rows} == {"speech_boundary_frame_negative_candidate"}
    assert rows[0]["source_audio_path"].endswith("agents\\temp\\test\\audio.wav") or rows[0][
        "source_audio_path"
    ].endswith("agents/temp/test/audio.wav")
    assert rows[0]["hard_negative_status"] == "candidate_requires_audio_materialization"


def test_export_cueqc_cluster_seed_hardcases_can_opt_into_text_present_clusters(tmp_path: Path):
    labels = tmp_path / "cueqc_cluster_labels.jsonl"
    clusters = tmp_path / "cueqc_clusters.jsonl"
    _write_jsonl(labels, [_label("cluster_text", decision="drop", seed_action="use_seed", text_present=3, count=3)])
    _write_jsonl(clusters, [_cluster("sample-text", "cluster_text", start=0.0, end=0.5)])

    summary = export_cueqc_cluster_seed_hardcases(
        cluster_labels_path=labels,
        clusters_path=clusters,
        output_dir=tmp_path / "out",
        allow_text_present=True,
    )

    assert summary["allow_text_present"] is True
    assert summary["counts"]["seed_drop_clusters"] == 1
    assert summary["counts"]["exported_candidates"] == 1
