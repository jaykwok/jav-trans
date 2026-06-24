from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr import pre_asr_cueqc
from audio.chunk_packer import PackedChunk
from boundary.base import SpeechSegment
from tools.asr.cueqc.compile_pre_asr_v6_features import compile_features


def _pre_asr_candidate(index: int, *, video_id: str = "AAA", cluster_id: str = "") -> dict:
    candidate = pre_asr_cueqc.candidate_from_span(
        [
            {
                "start": float(index),
                "end": float(index) + 0.5,
                "scorer_speech_mean": 0.8,
                "scorer_split_p90": 0.2,
            }
        ],
        0,
    )
    sample_id = f"preasr-{video_id}-chunk{index:05d}"
    candidate.update(
        {
            "index": index,
            "sample_id": sample_id,
            "candidate_id": sample_id,
            "video_id": video_id,
            "audio_id": video_id,
            "chunk_index": index,
            "duration_s": 0.5,
        }
    )
    if cluster_id:
        candidate["cluster_id"] = cluster_id
    return candidate


def test_pre_asr_cueqc_feature_schema_excludes_asr_text_fields():
    names = " ".join(pre_asr_cueqc.PRE_ASR_CUEQC_FEATURE_NAMES).lower()

    for banned in ("text", "token", "decoder", "asr_confidence", "subtitle_timing"):
        assert banned not in names


def test_pre_asr_cueqc_candidate_uses_numeric_chunk_features_only():
    spans = [
        {
            "start": 0.0,
            "end": 1.2,
            "text": "これは入らない",
            "raw_text": "これも入らない",
            "decoder_stats": {"x": 1},
            "asr_confidence": 0.1,
            "scorer_speech_mean": 0.8,
            "scorer_split_p90": 0.2,
        }
    ]

    candidate = pre_asr_cueqc.candidate_from_span(spans, 0)

    assert candidate["schema"] == pre_asr_cueqc.PRE_ASR_CUEQC_FEATURE_SCHEMA
    assert candidate["feature_names"] == list(pre_asr_cueqc.PRE_ASR_CUEQC_FEATURE_NAMES)
    assert set(candidate["features"]) == set(pre_asr_cueqc.PRE_ASR_CUEQC_FEATURE_NAMES)
    assert "text" not in json.dumps(candidate["features"], ensure_ascii=False).lower()


def test_pre_asr_cueqc_candidate_includes_micro_numeric_features():
    spans = [
        PackedChunk(
            start=0.0,
            end=0.5,
            speech_segments=[SpeechSegment(0.0, 0.5)],
            duration=0.5,
            split_reason="unit",
            subtitle_min_duration_s=20.0 / 24.0,
            below_subtitle_min_duration=True,
            micro_chunk_candidate=True,
            micro_resolve_action="preserve_micro_candidate",
            micro_resolve_reason="balanced_split_evidence",
            left_split_score=0.8,
            right_split_score=0.82,
            left_split_prominence=0.2,
            right_split_prominence=0.21,
            left_split_speech_valley=0.7,
            right_split_speech_valley=0.72,
        )
    ]

    candidate = pre_asr_cueqc.candidate_from_span(spans, 0)

    assert candidate["schema"] == pre_asr_cueqc.PRE_ASR_CUEQC_FEATURE_SCHEMA
    assert candidate["below_subtitle_min_duration"] is True
    assert candidate["micro_chunk_candidate"] is True
    assert candidate["micro_resolve_action"] == "preserve_micro_candidate"
    assert candidate["features"]["below_subtitle_min_duration"] == 1.0
    assert candidate["features"]["micro_action_preserve"] == 1.0
    assert candidate["features"]["micro_action_merge_left"] == 0.0
    assert candidate["features"]["left_split_score"] == 0.8


def test_pre_asr_cueqc_filters_before_wav_export(monkeypatch):
    from asr import pipeline as asr_pipeline

    spans = [
        PackedChunk(
            start=0.0,
            end=1.0,
            speech_segments=[SpeechSegment(0.0, 1.0)],
            duration=1.0,
            split_reason="unit",
        ),
        PackedChunk(
            start=1.2,
            end=2.0,
            speech_segments=[SpeechSegment(1.2, 2.0)],
            duration=0.8,
            split_reason="unit",
        ),
    ]

    class FakeModel:
        def signature(self):
            return {"schema": pre_asr_cueqc.PRE_ASR_CUEQC_SCHEMA, "type": "fake"}

        def decide(self, candidates):
            return [
                {
                    "index": int(candidate["index"]),
                    "route": "drop_before_asr" if int(candidate["index"]) == 0 else "keep_for_asr",
                    "confidence": 0.51 if int(candidate["index"]) == 0 else 0.52,
                    "prob_drop": 0.99 if int(candidate["index"]) == 0 else 0.01,
                    "prob_keep": 0.01 if int(candidate["index"]) == 0 else 0.99,
                }
                for candidate in candidates
            ]

    monkeypatch.setattr(asr_pipeline._pre_asr_cueqc_module, "enabled", lambda: True)
    monkeypatch.setattr(asr_pipeline._pre_asr_cueqc_module, "load_active", lambda **_kwargs: FakeModel())

    log: list[str] = []
    stage_events: list[str] = []
    kept, report = asr_pipeline._apply_pre_asr_cueqc(
        spans,
        log=log,
        on_stage=stage_events.append,
    )

    assert [span.start for span in kept] == [1.2]
    assert report["drop_count"] == 1
    assert report["keep_count"] == 1
    assert report["confidence_min"] == 0.51
    assert report["decisions"][0]["route"] == "drop_before_asr"
    assert {decision["route"] for decision in report["decisions"]} == {
        "drop_before_asr",
        "keep_for_asr",
    }
    combined_log = "\n".join(log)
    assert "keep_for_asr=1" in combined_log
    assert "drop_before_asr=1" in combined_log
    assert "confidence_min=" in combined_log
    assert "fallback" not in combined_log.lower()
    assert "low-confidence" not in combined_log.lower()
    assert stage_events == log


def test_compile_pre_asr_cueqc_features_ignores_text_columns(tmp_path: Path):
    torch = pytest.importorskip("torch")
    del torch
    chunks = tmp_path / "chunks.json"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "features.pt"
    chunks.write_text(
        json.dumps(
            {
                "transcript_chunks": [
                    {
                        "index": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "text": "ignored",
                        "raw_text": "ignored",
                        "scorer_speech_mean": 0.9,
                        "scorer_split_p90": 0.1,
                    },
                    {
                        "index": 1,
                        "start": 1.2,
                        "end": 2.0,
                        "decoder_stats": {"ignored": True},
                        "scorer_speech_mean": 0.2,
                        "scorer_split_p90": 0.0,
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    labels.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "chunks#0", "label": "keep"}),
                json.dumps({"sample_id": "chunks#1", "label": "drop"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = compile_features(
        chunk_paths=[str(chunks)],
        label_paths=[str(labels)],
        output=output,
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame",
    )

    assert summary["count"] == 2
    assert summary["keep"] == 1
    assert summary["drop"] == 1
    assert "text" not in " ".join(summary["feature_names"]).lower()


def test_compile_pre_asr_cueqc_features_reads_jsonl_chunk_candidates(tmp_path: Path):
    torch = pytest.importorskip("torch")
    del torch
    chunks = tmp_path / "chunks.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "features.pt"
    rows = [_pre_asr_candidate(0, video_id="AAA"), _pre_asr_candidate(1, video_id="AAA")]
    chunks.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    labels.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "preasr-AAA-chunk00000", "label": "keep"}),
                json.dumps({"sample_id": "preasr-AAA-chunk00001", "label": "drop"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = compile_features(
        chunk_paths=[str(chunks)],
        label_paths=[str(labels)],
        output=output,
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame",
    )

    assert summary["count"] == 2
    assert summary["keep"] == 1
    assert summary["drop"] == 1


def test_compile_pre_asr_cueqc_sample_labels_do_not_broadcast_by_cluster(tmp_path: Path):
    torch = pytest.importorskip("torch")
    chunks = tmp_path / "chunks.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "features.pt"
    first = _pre_asr_candidate(0, video_id="AAA", cluster_id="cluster_mixed")
    second = _pre_asr_candidate(1, video_id="AAA", cluster_id="cluster_mixed")
    chunks.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in [first, second]) + "\n",
        encoding="utf-8",
    )
    labels.write_text(
        json.dumps(
            {
                "sample_id": "preasr-AAA-chunk00000",
                "cluster_id": "cluster_mixed",
                "label": "drop",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = compile_features(
        chunk_paths=[str(chunks)],
        label_paths=[str(labels)],
        output=output,
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame",
    )
    bundle = torch.load(output, map_location="cpu")

    assert summary["count"] == 1
    assert [row["id"] for row in bundle["rows"]] == ["preasr-AAA-chunk00000"]


def test_compile_pre_asr_cueqc_broadcasts_cluster_examples_to_sample_ids(tmp_path: Path):
    torch = pytest.importorskip("torch")
    chunks = tmp_path / "pre_asr_candidates.json"
    labels = tmp_path / "cluster_labels.jsonl"
    output = tmp_path / "features.pt"
    chunks.write_text(
        json.dumps(
            {
                "video_id": "AAA",
                "pre_asr_candidates": [
                    _pre_asr_candidate(0, video_id="AAA"),
                    _pre_asr_candidate(1, video_id="AAA"),
                    _pre_asr_candidate(2, video_id="AAA"),
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    labels.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema": "cueqc_cluster_label_v1",
                        "cluster_id": "cluster_drop",
                        "display_decision": "drop",
                        "training_label_included": True,
                        "examples": [
                            {"sample_id": "preasr-AAA-chunk00000"},
                            {"sample_id": "preasr-AAA-chunk00001"},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "schema": "cueqc_cluster_label_v1",
                        "cluster_id": "cluster_skip",
                        "display_decision": "keep",
                        "training_label_included": False,
                        "examples": [{"sample_id": "preasr-AAA-chunk00002"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = compile_features(
        chunk_paths=[str(chunks)],
        label_paths=[str(labels)],
        output=output,
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame",
    )
    bundle = torch.load(output, map_location="cpu")

    assert summary["count"] == 2
    assert summary["drop"] == 2
    assert [row["id"] for row in bundle["rows"]] == [
        "preasr-AAA-chunk00000",
        "preasr-AAA-chunk00001",
    ]


def test_compile_pre_asr_cueqc_matches_video_chunk_and_cluster_id_labels(tmp_path: Path):
    torch = pytest.importorskip("torch")
    chunks = tmp_path / "chunks.json"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "features.pt"
    first = _pre_asr_candidate(0, video_id="AAA")
    first.pop("sample_id")
    first.pop("candidate_id")
    second = _pre_asr_candidate(1, video_id="AAA", cluster_id="cluster_keep")
    chunks.write_text(
        json.dumps(
            {
                "video_id": "AAA",
                "pre_asr_candidates": [first, second],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    labels.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema": "cueqc_cluster_label_v1",
                        "cluster_id": "cluster_drop",
                        "display_decision": "drop",
                        "training_label_included": True,
                        "examples": [{"video_id": "AAA", "chunk_index": 0}],
                    }
                ),
                json.dumps(
                    {
                        "schema": "cueqc_cluster_label_v1",
                        "cluster_id": "cluster_keep",
                        "display_decision": "keep",
                        "training_label_included": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = compile_features(
        chunk_paths=[str(chunks)],
        label_paths=[str(labels)],
        output=output,
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame",
    )
    bundle = torch.load(output, map_location="cpu")

    assert summary["count"] == 2
    assert summary["keep"] == 1
    assert summary["drop"] == 1
    assert [row["label"] for row in bundle["rows"]] == ["drop_before_asr", "keep_for_asr"]
