from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np

from asr import pre_asr_cueqc
from audio.chunk_packer import PackedChunk
from boundary.base import SpeechSegment
from boundary.sequence_features import (
    CHUNK_POOLED_PTM_SCHEMA,
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
)
from tools.asr.cueqc.compile_pre_asr_v12_features import compile_features
from tools.asr.cueqc.train_pre_asr_v12_binary import (
    _split_label_masks,
    _window_batch_from_anchors,
)


def _ptm_pool() -> list[float]:
    return [
        float(index) / 1000.0
        for index in range(len(pre_asr_cueqc.PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES))
    ]


def _ptm_pooling_fields() -> dict:
    values = _ptm_pool()
    return {
        "pre_asr_ptm_pooling_schema": CHUNK_POOLED_PTM_SCHEMA,
        "pre_asr_ptm_pooling_bins": pre_asr_cueqc.PRE_ASR_CUEQC_PTM_BINS,
        "pre_asr_ptm_pooling_dim": len(values),
        "pre_asr_ptm_pooled_features": values,
    }


def _pre_asr_candidate(index: int, *, video_id: str = "AAA", cluster_id: str = "") -> dict:
    candidate = pre_asr_cueqc.candidate_from_span(
        [
            {
                "start": float(index),
                "end": float(index) + 0.5,
                "scorer_speech_mean": 0.8,
                "scorer_split_p90": 0.2,
                **_ptm_pooling_fields(),
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


def test_pre_asr_drop_threshold_env_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("PRE_ASR_CUEQC_DROP_THRESHOLD", "0,95")

    with pytest.raises(ValueError, match="PRE_ASR_CUEQC_DROP_THRESHOLD"):
        pre_asr_cueqc._drop_threshold_from_env(0.95)


def test_pre_asr_drop_threshold_env_rejects_out_of_range_value(monkeypatch):
    monkeypatch.setenv("PRE_ASR_CUEQC_DROP_THRESHOLD", "1.5")

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pre_asr_cueqc._drop_threshold_from_env(0.95)


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
    assert set(candidate["features"]) == set(pre_asr_cueqc.PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES)
    assert len(candidate["pre_asr_ptm_pooled_features"]) == len(
        pre_asr_cueqc.PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES
    )
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
            **_ptm_pooling_fields(),
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


def test_pre_asr_cueqc_v9_appends_split_edge_soft_features():
    spans = [
        PackedChunk(
            start=0.0,
            end=0.6,
            speech_segments=[SpeechSegment(0.0, 0.6)],
            duration=0.6,
            split_reason="semantic_split_model",
            primary_cut_candidates=[
                {
                    "kind": "primary",
                    "time_s": 0.0,
                    "proposal_time_s": 0.02,
                    "p_cut": 0.91,
                    "p_continue": 0.06,
                    "p_unsure": 0.03,
                    "role": "noise_to_speech",
                    "p_role": 0.88,
                    "noise_isolation_bracket": True,
                    "bracket_pair_id": "noise-bracket-0.000000-0.600000",
                },
                {
                    "kind": "primary",
                    "time_s": 0.6,
                    "proposal_time_s": 0.58,
                    "p_cut": 0.94,
                    "p_continue": 0.04,
                    "p_unsure": 0.02,
                    "role": "speech_to_noise",
                    "p_role": 0.9,
                    "noise_isolation_bracket": True,
                    "bracket_pair_id": "noise-bracket-0.000000-0.600000",
                },
            ],
            **_ptm_pooling_fields(),
        )
    ]

    candidate = pre_asr_cueqc.candidate_from_span(spans, 0)

    assert pre_asr_cueqc.PRE_ASR_CUEQC_FEATURE_SCHEMA == "pre_asr_cueqc_features_v9"
    assert pre_asr_cueqc.PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES[
        : len(pre_asr_cueqc.PRE_ASR_CUEQC_V8_SCALAR_FEATURE_NAMES)
    ] == pre_asr_cueqc.PRE_ASR_CUEQC_V8_SCALAR_FEATURE_NAMES
    assert candidate["schema"] == "pre_asr_cueqc_features_v9"
    assert candidate["pre_asr_split_edge_pair_id"] == "noise-bracket-0.000000-0.600000"
    assert candidate["pre_asr_split_edges"]["left"]["kind"] == "split_cut"
    assert candidate["pre_asr_split_edges"]["right"]["role"] == "speech_to_noise"
    assert candidate["features"]["left_edge_is_split_cut"] == 1.0
    assert candidate["features"]["right_edge_p_cut"] == pytest.approx(0.94)
    assert candidate["features"]["left_edge_role_noise_to_speech"] == 1.0
    assert candidate["features"]["right_edge_role_speech_to_noise"] == 1.0
    assert candidate["features"]["left_right_same_noise_pair"] == 1.0


def test_pre_asr_cueqc_candidate_uses_chunk_pooled_ptm_embedding():
    spans = [
        PackedChunk(
            start=0.0,
            end=1.0,
            speech_segments=[SpeechSegment(0.0, 1.0)],
            duration=1.0,
            split_reason="unit",
            **_ptm_pooling_fields(),
        )
    ]

    candidate = pre_asr_cueqc.candidate_from_span(spans, 0, require_ptm_pooling=True)
    vector = pre_asr_cueqc.feature_vector(candidate)

    assert candidate["ptm_pooling_available"] is True
    assert candidate["ptm_pooling_schema"] == CHUNK_POOLED_PTM_SCHEMA
    assert vector.shape[0] == len(pre_asr_cueqc.PRE_ASR_CUEQC_FEATURE_NAMES)
    assert vector[-1] == pytest.approx(_ptm_pool()[-1])


def test_frame_sequence_provider_pools_chunk_ptm_features():
    provider = FrameSequenceFeatureProvider(
        duration_s=0.04,
        frame_hop_s=0.01,
        ptm=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        mfcc=[[0.0], [0.0], [0.0], [0.0]],
        config=FrameSequenceFeatureConfig(max_ptm_dims=2),
    )

    names = provider.chunk_pooled_ptm_feature_names(bins=4)
    values = provider.chunk_pooled_ptm_features(start_s=0.0, end_s=0.04, bins=4)

    assert len(names) == 12
    assert values[:2] == pytest.approx([4.0, 5.0])
    assert values[4:] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


def test_pre_asr_cueqc_requires_pooled_ptm_when_requested():
    spans = [
        PackedChunk(
            start=0.0,
            end=1.0,
            speech_segments=[SpeechSegment(0.0, 1.0)],
            duration=1.0,
            split_reason="unit",
        )
    ]

    with pytest.raises(ValueError, match="requires chunk-level pooled PTM"):
        pre_asr_cueqc.candidate_from_span(spans, 0, require_ptm_pooling=True)


def test_pre_asr_cueqc_v12_model_forward_backward_ignores_padding():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")

    model = pre_asr_cueqc.PreAsrCueQCNetwork(
        ptm_dim=4,
        scalar_dim=3,
        hidden_size=128,
        temporal_layers=1,
        dropout=0.0,
    )
    token_count = pre_asr_cueqc.PRE_ASR_CUEQC_MODEL_PTM_TOKENS
    ptm_bins = torch.randn(1, 3, token_count, 4)
    scalar = torch.randn(1, 3, 3)
    chunk_mask = torch.tensor([[1.0, 1.0, 0.0]])
    bin_mask = torch.ones(1, 3, token_count)
    bin_mask[:, 2] = 0.0
    labels = torch.tensor([[1, 0, pre_asr_cueqc.PRE_ASR_CUEQC_IGNORE_LABEL]])

    logits = model(ptm_bins, scalar, chunk_mask=chunk_mask, bin_mask=bin_mask)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 2),
        labels.reshape(-1),
        ignore_index=pre_asr_cueqc.PRE_ASR_CUEQC_IGNORE_LABEL,
    )
    loss.backward()

    assert logits.shape == (1, 3, 2)
    assert torch.isfinite(loss)


def test_pre_asr_cueqc_local_only_mode_and_legacy_default():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")

    assert pre_asr_cueqc.make_model_config({})["temporal_residual_scale"] == 1.0
    model = pre_asr_cueqc.PreAsrCueQCNetwork(
        ptm_dim=4,
        scalar_dim=3,
        hidden_size=128,
        temporal_layers=1,
        temporal_residual_scale=0.0,
        dropout=0.0,
    )
    ptm_bins = torch.randn(1, 2, pre_asr_cueqc.PRE_ASR_CUEQC_MODEL_PTM_TOKENS, 4)
    scalar = torch.randn(1, 2, 3)
    mask = torch.tensor([[1.0, 0.0]])

    logits = model(ptm_bins, scalar, chunk_mask=mask)

    assert logits.shape == (1, 2, 2)
    assert torch.count_nonzero(logits[:, 1]).item() == 0


def _tiny_v12_checkpoint(tmp_path: Path, *, hard_rules: bool = False, split_sha: str = "") -> Path:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")

    feature_names = list(pre_asr_cueqc.PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES[:3])
    model = pre_asr_cueqc.PreAsrCueQCNetwork(
        ptm_dim=pre_asr_cueqc.PRE_ASR_CUEQC_PTM_DIM,
        scalar_dim=len(feature_names),
        hidden_size=128,
        temporal_layers=1,
        temporal_residual_scale=0.0,
        dropout=0.0,
    )
    payload = {
        "schema": pre_asr_cueqc.PRE_ASR_CUEQC_SCHEMA,
        "arch": pre_asr_cueqc.PRE_ASR_CUEQC_MODEL_ARCH,
        "feature_schema": pre_asr_cueqc.PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "runtime_adapter": pre_asr_cueqc.PRE_ASR_CUEQC_RUNTIME_ADAPTER,
        "feature_names": feature_names,
        "model_config": {
            "ptm_dim": pre_asr_cueqc.PRE_ASR_CUEQC_PTM_DIM,
            "scalar_dim": len(feature_names),
            "hidden_size": 128,
            "temporal_layers": 1,
            "temporal_residual_scale": 0.0,
            "dropout": 0.0,
            "num_classes": 2,
        },
        "feature_mean": [0.0] * len(feature_names),
        "feature_std": [1.0] * len(feature_names),
        "decision_config": {
            "drop_threshold": 0.95,
            "hard_keep_veto": hard_rules,
            "hard_drop_rule": False,
            "keep_veto": False,
            "inference_window_size": 128,
        },
        "metadata": {
            "artifact": dict(pre_asr_cueqc.PRE_ASR_CUEQC_ARTIFACT),
            "asr_repo_id": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
            "semantic_split_weights_sha256": split_sha,
        },
        "model_state_dict": model.state_dict(),
    }
    path = tmp_path / "pre_asr_cueqc_v12.pt"
    torch.save(payload, path)
    return path


def test_pre_asr_cueqc_v12_rejects_enabled_hard_rules(tmp_path: Path):
    checkpoint = _tiny_v12_checkpoint(tmp_path, hard_rules=True)

    with pytest.raises(ValueError, match="must disable hard rules"):
        pre_asr_cueqc.load_checkpoint(checkpoint, device="cpu")


def test_pre_asr_cueqc_load_active_validates_split_sha(monkeypatch, tmp_path: Path):
    split = tmp_path / "split.pt"
    split.write_bytes(b"active split")
    checkpoint = _tiny_v12_checkpoint(tmp_path, split_sha="0" * 64)
    monkeypatch.setattr(pre_asr_cueqc, "_checkpoint_path", lambda _repo_id=None: str(checkpoint))
    monkeypatch.setattr(
        pre_asr_cueqc,
        "_semantic_split_checkpoint_path",
        lambda _repo_id=None: str(split),
    )

    with pytest.raises(ValueError, match="split checkpoint sha mismatch"):
        pre_asr_cueqc.load_active(
            expected_asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
        )


def test_pre_asr_cueqc_filters_before_wav_export(monkeypatch):
    from asr import pipeline as asr_pipeline

    spans = [
        PackedChunk(
            start=0.0,
            end=1.0,
            speech_segments=[SpeechSegment(0.0, 1.0)],
            duration=1.0,
            split_reason="unit",
            **_ptm_pooling_fields(),
        ),
        PackedChunk(
            start=1.2,
            end=2.0,
            speech_segments=[SpeechSegment(1.2, 2.0)],
            duration=0.8,
            split_reason="unit",
            **_ptm_pooling_fields(),
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
                        **_ptm_pooling_fields(),
                    },
                    {
                        "index": 1,
                        "start": 1.2,
                        "end": 2.0,
                        "decoder_stats": {"ignored": True},
                        "scorer_speech_mean": 0.2,
                        "scorer_split_p90": 0.0,
                        **_ptm_pooling_fields(),
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
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )

    assert summary["chunk_count"] == 2
    assert summary["keep"] == 1
    assert summary["drop"] == 1
    assert "text" not in " ".join(summary["feature_names"]).lower()


def test_compile_pre_asr_cueqc_features_reads_jsonl_chunk_candidates(tmp_path: Path):
    torch = pytest.importorskip("torch")
    del torch
    from tools.asr.cueqc.train_pre_asr_v12_binary import load_feature_bundle

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
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )

    assert summary["chunk_count"] == 2
    assert summary["keep"] == 1
    assert summary["drop"] == 1
    bundle = load_feature_bundle(output)
    assert bundle["groups"][0]["audio_id"] == "AAA"
    assert (
        bundle["ptm_bin_count"]
        == pre_asr_cueqc.PRE_ASR_CUEQC_MODEL_PTM_TOKENS
    )
    assert tuple(bundle["ptm_bins"].shape[-2:]) == (
        pre_asr_cueqc.PRE_ASR_CUEQC_MODEL_PTM_TOKENS,
        pre_asr_cueqc.PRE_ASR_CUEQC_PTM_DIM,
    )


def test_compile_pre_asr_cueqc_features_expands_source_windows_manifest(tmp_path: Path):
    torch = pytest.importorskip("torch")
    manifest = tmp_path / "source_windows.jsonl"
    chunks = tmp_path / "features" / "w00" / "pre_asr_candidates.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "features.pt"
    chunks.parent.mkdir(parents=True)
    rows = [_pre_asr_candidate(0, video_id="AAA"), _pre_asr_candidate(1, video_id="AAA")]
    chunks.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps({"window_id": "w00", "pre_asr_candidates": str(chunks)}, ensure_ascii=False)
        + "\n",
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
        chunk_paths=[str(manifest)],
        label_paths=[str(labels)],
        output=output,
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )

    assert summary["chunk_count"] == 2
    bundle = torch.load(output, map_location="cpu")
    assert bundle["source_files"][0].replace("/", "\\").endswith(
        "features\\w00\\pre_asr_candidates.jsonl"
    )


def test_compile_pre_asr_cueqc_features_reads_single_object_candidate_file(tmp_path: Path):
    torch = pytest.importorskip("torch")
    chunks = tmp_path / "single_candidate.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "features.pt"
    row = _pre_asr_candidate(0, video_id="AAA")
    chunks.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    labels.write_text(
        json.dumps({"sample_id": "preasr-AAA-chunk00000", "label": "drop"}) + "\n",
        encoding="utf-8",
    )

    summary = compile_features(
        chunk_paths=[str(chunks)],
        label_paths=[str(labels)],
        output=output,
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    bundle = torch.load(output, map_location="cpu")

    assert summary["chunk_count"] == 1
    assert bundle["rows"][0]["id"] == "preasr-AAA-chunk00000"


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
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    bundle = torch.load(output, map_location="cpu")

    assert summary["chunk_count"] == 1
    assert bundle["labels"][0, 0].item() == 0
    assert bundle["labels"][0, 1].item() == pre_asr_cueqc.PRE_ASR_CUEQC_IGNORE_LABEL


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
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    bundle = torch.load(output, map_location="cpu")

    assert summary["chunk_count"] == 2
    assert summary["drop"] == 2
    assert [row["id"] for row in bundle["rows"][:2]] == [
        "preasr-AAA-chunk00000",
        "preasr-AAA-chunk00001",
    ]
    assert bundle["labels"][0, 2].item() == pre_asr_cueqc.PRE_ASR_CUEQC_IGNORE_LABEL


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
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    bundle = torch.load(output, map_location="cpu")

    assert summary["chunk_count"] == 2
    assert summary["keep"] == 1
    assert summary["drop"] == 1
    assert [row["label"] for row in bundle["rows"]] == ["drop_before_asr", "keep_for_asr"]


def test_pre_asr_training_chunk_stratified_split_samples_both_films():
    y = np.asarray(
        [
            [0, 0, 1, 1, pre_asr_cueqc.PRE_ASR_CUEQC_IGNORE_LABEL, 0],
            [1, 1, 0, 0, pre_asr_cueqc.PRE_ASR_CUEQC_IGNORE_LABEL, 1],
        ],
        dtype=np.int64,
    )
    chunk_mask = np.ones_like(y, dtype=np.float32)

    train_mask, val_mask, summary = _split_label_masks(
        y=y,
        chunk_mask=chunk_mask,
        group_rows=[
            {"audio_id": "film-a", "planned_island_id": "sequence"},
            {"audio_id": "film-b", "planned_island_id": "sequence"},
        ],
        split_mode="chunk_stratified",
        val_ratio=0.4,
        rng=np.random.default_rng(17),
    )

    assert summary["train_group_count"] == 2
    assert summary["val_group_count"] == 2
    assert summary["train_counts"]["drop"] > 0
    assert summary["train_counts"]["keep"] > 0
    assert summary["val_counts"]["drop"] > 0
    assert summary["val_counts"]["keep"] > 0
    assert not np.any(train_mask & val_mask)


def test_pre_asr_anchor_batch_supports_pre_windowed_bundle():
    import torch

    labels = torch.tensor([[0, 1, -100], [1, 0, -100]])
    anchors = torch.tensor([[0, 1], [1, 0]])
    ptm = torch.zeros((2, 3, 2, 4))
    scalar = torch.zeros((2, 3, 5))
    chunk_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    bin_mask = torch.ones((2, 3, 2), dtype=torch.bool)

    *_, targets = _window_batch_from_anchors(
        anchor_positions=anchors,
        ptm_bins=ptm,
        scalar=scalar,
        chunk_mask=chunk_mask,
        bin_mask=bin_mask,
        y=labels,
        sequence_window_size=3,
    )

    assert targets.tolist() == [[-100, 1, -100], [1, -100, -100]]
