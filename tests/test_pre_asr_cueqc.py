from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

from asr import pre_asr_cueqc
from audio.chunk_packer import PackedChunk
from boundary.base import SpeechSegment
from boundary.sequence_features import (
    CHUNK_LEARNED_PROJECTED_PTM_SCHEMA,
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
)
from tools.asr.cueqc.pre_asr_feature_compiler import compile_features, read_labels
from tools.asr.cueqc.pre_asr_binary_trainer import (
    _apply_forced_group_splits,
    _boost_anchor_positions,
    _excluded_training_label_count,
    _matching_candidate_positions,
    _prediction_rows,
    _predict_logits_windowed,
    _split_label_masks,
    _window_batch_from_anchors,
    classification_metrics,
)


def _ptm_pool() -> list[float]:
    return [
        float(index) / 1000.0
        for index in range(len(pre_asr_cueqc.PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES))
    ]


def _ptm_pooling_fields() -> dict:
    values = _ptm_pool()
    return {
        "pre_asr_ptm_pooling_schema": CHUNK_LEARNED_PROJECTED_PTM_SCHEMA,
        "pre_asr_ptm_pooling_bins": pre_asr_cueqc.PRE_ASR_CUEQC_PTM_BINS,
        "pre_asr_ptm_pooling_dim": len(values),
        "pre_asr_ptm_pooled_features": values,
        "pre_asr_ptm_projection_digest": "learned-projection-sha",
    }


def _pre_asr_candidate(index: int, *, video_id: str = "AAA", cluster_id: str = "") -> dict:
    candidate = pre_asr_cueqc.candidate_from_span(
        [
            {
                "start": float(index),
                "end": float(index) + 0.5,
                "scorer_speech_mean": 0.8,
                "scorer_split_p90": 0.2,
                "boundary_pipeline_version": 11,
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
            "boundary_pipeline_version": 11,
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
            boundary_pipeline_version=11,
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


def test_pre_asr_cueqc_v10_preserves_split_edge_soft_features():
    spans = [
        PackedChunk(
            start=0.0,
            end=0.6,
            speech_segments=[SpeechSegment(0.0, 0.6)],
            duration=0.6,
            split_reason="semantic_split_model",
            boundary_pipeline_version=11,
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

    assert pre_asr_cueqc.PRE_ASR_CUEQC_FEATURE_SCHEMA == "pre_asr_cueqc_features_v10"
    assert pre_asr_cueqc.PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES[
        : len(pre_asr_cueqc.PRE_ASR_CUEQC_V8_SCALAR_FEATURE_NAMES)
    ] == pre_asr_cueqc.PRE_ASR_CUEQC_V8_SCALAR_FEATURE_NAMES
    assert candidate["schema"] == "pre_asr_cueqc_features_v10"
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
    assert candidate["ptm_pooling_schema"] == CHUNK_LEARNED_PROJECTED_PTM_SCHEMA
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

    with pytest.raises(ValueError, match="requires chunk-level learned-projection PTM"):
        pre_asr_cueqc.candidate_from_span(spans, 0, require_ptm_pooling=True)


def test_pre_asr_cueqc_v13_model_forward_backward_ignores_padding():
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


def _tiny_v13_checkpoint(
    tmp_path: Path,
    *,
    hard_rules: bool = False,
    split_sha: str = "",
    inner_sha: str = "",
) -> Path:
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
        num_classes=2,
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
            "decision_mode": "argmax",
            "hard_keep_veto": hard_rules,
            "hard_drop_rule": False,
            "keep_veto": False,
            "inference_window_size": 128,
        },
        "metadata": {
            "artifact": dict(pre_asr_cueqc.PRE_ASR_CUEQC_ARTIFACT),
            "asr_repo_id": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
            "semantic_split_weights_sha256": split_sha,
            "inner_edge_refiner_weights_sha256": inner_sha,
            "training_labels": ["drop", "keep"],
            "excluded_training_labels": ["unsure"],
            "excluded_training_label_count": 0,
        },
        "model_state_dict": model.state_dict(),
    }
    path = tmp_path / "pre_asr_cueqc_v13.pt"
    torch.save(payload, path)
    return path


def test_pre_asr_cueqc_v13_rejects_enabled_hard_rules(tmp_path: Path):
    checkpoint = _tiny_v13_checkpoint(tmp_path, hard_rules=True)

    with pytest.raises(ValueError, match="must disable hard rules"):
        pre_asr_cueqc.load_checkpoint(checkpoint, device="cpu")


def test_pre_asr_cueqc_load_active_validates_split_sha(monkeypatch, tmp_path: Path):
    split = tmp_path / "split.pt"
    split.write_bytes(b"active split")
    inner = tmp_path / "inner.pt"
    inner.write_bytes(b"active inner")
    checkpoint = _tiny_v13_checkpoint(
        tmp_path,
        split_sha="0" * 64,
        inner_sha=pre_asr_cueqc._file_sha256(inner),
    )
    monkeypatch.setattr(pre_asr_cueqc, "_checkpoint_path", lambda _repo_id=None: str(checkpoint))
    monkeypatch.setattr(
        pre_asr_cueqc,
        "_semantic_split_checkpoint_path",
        lambda _repo_id=None: str(split),
    )
    monkeypatch.setattr(
        pre_asr_cueqc,
        "_inner_edge_refiner_checkpoint_path",
        lambda _repo_id=None: str(inner),
    )

    with pytest.raises(ValueError, match="split checkpoint sha mismatch"):
        pre_asr_cueqc.load_active(
            expected_asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
        )


def test_pre_asr_cueqc_17b_rejects_v12_checkpoint(tmp_path: Path):
    torch = pytest.importorskip("torch")
    checkpoint = _tiny_v13_checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["schema"] = pre_asr_cueqc.PRE_ASR_CUEQC_LEGACY_SCHEMA
    payload["arch"] = pre_asr_cueqc.PRE_ASR_CUEQC_LEGACY_MODEL_ARCH
    payload["feature_schema"] = pre_asr_cueqc.PRE_ASR_CUEQC_LEGACY_FEATURE_SCHEMA
    payload["runtime_adapter"] = pre_asr_cueqc.PRE_ASR_CUEQC_LEGACY_RUNTIME_ADAPTER
    payload["metadata"]["artifact"] = dict(
        pre_asr_cueqc.PRE_ASR_CUEQC_LEGACY_ARTIFACT
    )
    legacy = tmp_path / "pre_asr_cueqc_v12.pt"
    torch.save(payload, legacy)

    with pytest.raises(ValueError, match="unsupported Pre-ASR CueQC schema"):
        pre_asr_cueqc.load_checkpoint(
            legacy,
            device="cpu",
            expected_asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
        )


def test_pre_asr_cueqc_load_active_rejects_explicit_v11_checkpoint(monkeypatch):
    repo_id = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"
    legacy = (
        "src/checkpoints/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf/"
        "pre_asr_cueqc_v11.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf.pt"
    )
    monkeypatch.setenv("PRE_ASR_CUEQC_ENABLED", "1")
    monkeypatch.setenv("PRE_ASR_CUEQC_DEVICE", "cpu")
    monkeypatch.setenv("PRE_ASR_CUEQC_MODEL_PATH_BY_REPO", f"{repo_id}={legacy}")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        pre_asr_cueqc.load_active(expected_asr_repo_id=repo_id)


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


def test_pre_asr_compiler_preserves_raw_teacher_after_manual_override(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "canonical.jsonl"
    labels.write_text(
        json.dumps(
            {
                "subisland_id": "item",
                "teacher_label": "unsure",
                "label": "drop",
                "manual_override_applied": True,
            }
        )
        + "\n",
        "utf-8",
    )

    compiled = read_labels([str(labels)])["item"]

    assert compiled["label_index"] == 0
    assert compiled["teacher_label"] == "unsure"
    assert compiled["training_label_included"] is True
    assert compiled["training_ignore_reason"] == ""


def test_pre_asr_v13_wrappers_run_as_script_paths():
    root = Path(__file__).resolve().parents[1]
    for script in (
        root / "tools" / "asr" / "cueqc" / "compile_pre_asr_v13_features.py",
        root / "tools" / "asr" / "cueqc" / "train_pre_asr_v13.py",
    ):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_compile_pre_asr_cueqc_features_reads_jsonl_chunk_candidates(tmp_path: Path):
    torch = pytest.importorskip("torch")
    del torch
    from tools.asr.cueqc.pre_asr_binary_trainer import load_feature_bundle

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


def test_pre_asr_feature_bundle_rejects_non_binary_training_class(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from tools.asr.cueqc.pre_asr_binary_trainer import load_feature_bundle

    path = tmp_path / "invalid-labels.pt"
    torch.save(
        {
            "schema": "cueqc_pre_asr_semantic_chunk_v13_features",
            "feature_schema": pre_asr_cueqc.PRE_ASR_CUEQC_FEATURE_SCHEMA,
            "runtime_adapter": pre_asr_cueqc.PRE_ASR_CUEQC_RUNTIME_ADAPTER,
            "feature_names": list(pre_asr_cueqc.PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),
            "ptm_bin_count": pre_asr_cueqc.PRE_ASR_CUEQC_MODEL_PTM_TOKENS,
            "ptm_dim": pre_asr_cueqc.PRE_ASR_CUEQC_PTM_DIM,
            "ptm_bins": torch.zeros(
                1,
                1,
                pre_asr_cueqc.PRE_ASR_CUEQC_MODEL_PTM_TOKENS,
                pre_asr_cueqc.PRE_ASR_CUEQC_PTM_DIM,
            ),
            "labels": torch.tensor([[3]]),
        },
        path,
    )

    with pytest.raises(ValueError, match="unsupported training labels: \\[3\\]"):
        load_feature_bundle(path)


def test_compile_pre_asr_cueqc_keeps_per_row_source_identity_in_combined_jsonl(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    chunks = tmp_path / "runtime_v11_provisional.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "features.pt"
    source_rows = []
    for index, (video_id, partition) in enumerate((("AAA", "train"), ("BBB", "test"))):
        candidate = _pre_asr_candidate(index, video_id=video_id)
        source_rows.append(
            {
                "schema": "runtime_v11_provisional_subisland_v1",
                "sample_id": video_id,
                "subisland_id": f"{video_id}__v11s00",
                "source_partition": partition,
                "pre_asr_candidate": candidate,
            }
        )
    chunks.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows),
        "utf-8",
    )
    labels.write_text(
        "".join(
            json.dumps({"subisland_id": row["subisland_id"], "label": "keep"}) + "\n"
            for row in source_rows
        ),
        "utf-8",
    )

    compile_features(
        chunk_paths=[str(chunks)],
        label_paths=[str(labels)],
        output=output,
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    bundle = torch.load(output, map_location="cpu", weights_only=False)

    assert [group["audio_id"] for group in bundle["groups"]] == ["AAA", "BBB"]
    assert [group["dataset_role"] for group in bundle["groups"]] == ["train", "test"]


def test_compile_pre_asr_cueqc_preserves_selected_teacher_unsure_only(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    included = tmp_path / "included.jsonl"
    excluded = tmp_path / "excluded.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "features.pt"
    included.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in [
                _pre_asr_candidate(0, video_id="AAA"),
                _pre_asr_candidate(1, video_id="AAA"),
            ]
        )
        + "\n",
        "utf-8",
    )
    excluded.write_text(
        json.dumps(_pre_asr_candidate(0, video_id="BBB"), ensure_ascii=False) + "\n",
        "utf-8",
    )
    labels.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "preasr-AAA-chunk00000", "label": "keep"}),
                json.dumps({"sample_id": "preasr-AAA-chunk00001", "label": "unsure"}),
                json.dumps({"sample_id": "preasr-BBB-chunk00000", "label": "unsure"}),
            ]
        )
        + "\n",
        "utf-8",
    )

    summary = compile_features(
        chunk_paths=[str(included), str(excluded)],
        label_paths=[str(labels)],
        output=output,
        asr_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    bundle = torch.load(output, map_location="cpu", weights_only=False)

    assert summary["chunk_count"] == 1
    assert summary["teacher_unsure_ignored"] == 1
    assert summary["ambiguous_ignore"] == 0
    unsure_row = next(row for row in bundle["rows"] if row["teacher_label"] == "unsure")
    assert unsure_row["label"] == "teacher_unsure_ignored"
    assert unsure_row["training_ignore_reason"] == "teacher_unsure"
    assert bundle["teacher_unsure_ignored"] == 1


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


def test_pre_asr_training_role_holdout_keeps_val_and_test_out_of_train():
    y = np.asarray([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.int64)
    chunk_mask = np.ones_like(y, dtype=np.float32)

    train_mask, val_mask, summary = _split_label_masks(
        y=y,
        chunk_mask=chunk_mask,
        group_rows=[
            {"audio_id": "train", "dataset_role": "train"},
            {"audio_id": "val", "dataset_role": "val"},
            {"audio_id": "test", "dataset_role": "test"},
        ],
        split_mode="role_holdout",
        val_ratio=0.15,
        rng=np.random.default_rng(17),
    )

    assert np.array_equal(train_mask[0], [True, True, False])
    assert not np.any(train_mask[1:])
    assert not np.any(val_mask[0])
    assert np.array_equal(val_mask[1:], [[True, True, False], [True, True, False]])
    assert summary["train_counts"]["excluded_unsure"] == 0
    assert summary["val_counts"]["excluded_unsure"] == 0
    assert summary["all_counts"]["excluded_unsure"] == 3


def test_pre_asr_excluded_training_count_includes_canonical_and_legacy_unsure():
    y = np.asarray([[0, 1, 2, -100], [2, -100, 1, 0]], dtype=np.int64)

    assert _excluded_training_label_count(
        {"teacher_unsure_ignored": 3},
        y,
    ) == 5


def test_pre_asr_prediction_rows_keep_unsure_out_of_metrics_and_runtime_labels():
    rows = _prediction_rows(
        bundle={
            "rows": [
                {"id": "a", "teacher_label": "keep", "dataset_role": "train"},
                {
                    "id": "b",
                    "teacher_label": "unsure",
                    "canonical_label": "unsure",
                    "exact_core_label": "keep",
                    "training_ignore_reason": "teacher_unsure",
                    "dataset_role": "test",
                },
            ]
        },
        group_rows=[
            {
                "audio_id": "source",
                "dataset_role": "train",
                "row_ids": ["a", "b"],
            }
        ],
        y=np.asarray([[1, -100]], dtype=np.int64),
        chunk_mask=np.ones((1, 2), dtype=np.float32),
        train_mask=np.asarray([[True, False]]),
        val_mask=np.asarray([[False, False]]),
        probabilities=np.asarray([[[0.8, 0.2], [0.9, 0.1]]], dtype=np.float32),
    )

    assert [row["prediction"] for row in rows] == ["drop", "drop"]
    assert rows[0]["false_drop"] is True
    assert rows[0]["included_in_metrics"] is True
    assert rows[1]["truth_label"] == "unsure"
    assert rows[1]["included_in_metrics"] is False
    assert rows[1]["split_membership"] == "excluded"
    assert all(row["prediction"] in {"drop", "keep"} for row in rows)


def test_pre_asr_forced_group_splits_never_restore_teacher_unsure() -> None:
    y = np.asarray([[0, 1, 2, -100], [0, 1, 2, -100]], dtype=np.int64)
    chunk_mask = np.ones_like(y, dtype=np.float32)
    train = np.asarray(
        [[False, False, False, False], [True, True, False, False]],
        dtype=bool,
    )
    val = np.asarray(
        [[True, True, False, False], [False, False, False, False]],
        dtype=bool,
    )

    _apply_forced_group_splits(
        train=train,
        val=val,
        y=y,
        chunk_mask=chunk_mask,
        force_train_groups={0},
        force_val_groups={1},
    )

    assert np.array_equal(train, [[True, True, False, False], [False, False, False, False]])
    assert np.array_equal(val, [[False, False, False, False], [True, True, False, False]])


def test_pre_asr_v13_metrics_exclude_teacher_unsure() -> None:
    probs = np.asarray(
        [
            [0.8, 0.2],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.6, 0.4],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 2, 1], dtype=np.int64)
    durations = np.ones((4,), dtype=np.float32)

    metrics = classification_metrics(probs, labels, durations)

    assert metrics["drop_recall"] == 1.0
    assert metrics["semantic_keep_recall"] == 0.5
    assert metrics["false_drop_count"] == 1.0


def test_pre_asr_v13_windowed_prediction_is_binary() -> None:
    torch = pytest.importorskip("torch")

    class BinaryModel:
        def __call__(self, ptm_bins, scalar, *, chunk_mask, bin_mask):
            del scalar, chunk_mask, bin_mask
            batch, chunks = ptm_bins.shape[:2]
            return torch.ones((batch, chunks, 2), dtype=torch.float32)

    logits = _predict_logits_windowed(
        model=BinaryModel(),
        ptm_bins=torch.zeros((1, 5, 10, 128), dtype=torch.float32),
        scalar=torch.zeros((1, 5, 140), dtype=torch.float32),
        chunk_mask=torch.ones((1, 5), dtype=torch.float32),
        bin_mask=torch.ones((1, 5, 10), dtype=torch.float32),
        sequence_window_size=2,
    )

    assert logits.shape == (1, 5, 2)
    assert torch.all(logits == 1.0)


def test_sequence_tensor_contract_rejects_unassigned_candidates(monkeypatch):
    monkeypatch.setattr(pre_asr_cueqc, "planned_island_sequences", lambda _rows: [])
    with pytest.raises(RuntimeError, match="left candidates unassigned"):
        pre_asr_cueqc.sequence_tensors([{"candidate_id": "missing"}])


def test_pre_asr_training_video_group_split_keeps_windows_together():
    y = np.asarray([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.int64)
    chunk_mask = np.ones_like(y, dtype=np.float32)
    groups = [
        {"audio_id": "video-a-w00", "video_id": "video-a"},
        {"audio_id": "video-a-w01", "video_id": "video-a"},
        {"audio_id": "video-b-w00", "video_id": "video-b"},
        {"audio_id": "video-b-w01", "video_id": "video-b"},
    ]

    train_mask, val_mask, summary = _split_label_masks(
        y=y,
        chunk_mask=chunk_mask,
        group_rows=groups,
        split_mode="video_group",
        val_ratio=0.5,
        rng=np.random.default_rng(17),
    )

    assert summary["train_group_count"] == 2
    assert summary["val_group_count"] == 2
    assert bool(val_mask[0].any()) == bool(val_mask[1].any())
    assert bool(val_mask[2].any()) == bool(val_mask[3].any())
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


def test_pre_asr_candidate_anchor_boost_is_exact():
    import torch

    groups = [
        {"row_ids": ["candidate-a", "candidate-b"]},
        {"row_ids": ["candidate-c"]},
    ]
    positions = {1: torch.tensor([[0, 0], [0, 1], [1, 0]])}
    selected = _matching_candidate_positions(groups, ["candidate-b"])

    boosted = _boost_anchor_positions(
        positions,
        group_indexes=set(),
        candidate_positions=selected,
        boost=3,
    )

    assert selected == {(0, 1)}
    assert boosted[1].tolist().count([0, 1]) == 3
    assert boosted[1].tolist().count([0, 0]) == 1
    assert boosted[1].tolist().count([1, 0]) == 1


def test_pre_asr_candidate_anchor_boost_rejects_missing_id():
    with pytest.raises(ValueError, match="candidate-missing"):
        _matching_candidate_positions(
            [{"row_ids": ["candidate-a"]}],
            ["candidate-missing"],
        )


def test_pre_asr_valid_prefix_temporal_is_padding_invariant():
    import torch

    torch.manual_seed(17)
    model = pre_asr_cueqc.PreAsrCueQCNetwork(
        ptm_dim=128,
        scalar_dim=len(pre_asr_cueqc.PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),
        hidden_size=128,
        valid_prefix_temporal=True,
        dropout=0.0,
    )
    model.eval()
    valid_length = 3
    padded_length = 7
    ptm = torch.randn(2, valid_length, 10, 128)
    scalar = torch.randn(
        2,
        valid_length,
        len(pre_asr_cueqc.PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),
    )
    valid_mask = torch.ones(2, valid_length)
    bin_mask = torch.ones(2, valid_length, 10)
    padded_ptm = torch.zeros(2, padded_length, 10, 128)
    padded_scalar = torch.randn(
        2,
        padded_length,
        len(pre_asr_cueqc.PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),
    )
    padded_chunk_mask = torch.zeros(2, padded_length)
    padded_bin_mask = torch.zeros(2, padded_length, 10)
    padded_ptm[:, :valid_length] = ptm
    padded_scalar[:, :valid_length] = scalar
    padded_chunk_mask[:, :valid_length] = 1
    padded_bin_mask[:, :valid_length] = 1

    with torch.inference_mode():
        trimmed_logits = model(ptm, scalar, valid_mask, bin_mask)
        padded_logits = model(
            padded_ptm,
            padded_scalar,
            padded_chunk_mask,
            padded_bin_mask,
        )[:, :valid_length]

    assert torch.allclose(trimmed_logits, padded_logits, atol=1e-5, rtol=1e-5)


def test_pre_asr_token_attention_auxiliary_is_padding_invariant():
    import torch

    torch.manual_seed(23)
    scalar_dim = len(pre_asr_cueqc.PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES)
    model = pre_asr_cueqc.PreAsrCueQCNetwork(
        ptm_dim=128,
        scalar_dim=scalar_dim,
        hidden_size=128,
        valid_prefix_temporal=True,
        ptm_encoder_mode="token_attention",
        semantic_auxiliary=True,
        late_fusion=True,
        dropout=0.0,
    )
    model.eval()
    ptm = torch.randn(2, 3, 10, 128)
    scalar = torch.randn(2, 3, scalar_dim)
    mask = torch.ones(2, 3)
    bin_mask = torch.ones(2, 3, 10)
    padded_ptm = torch.zeros(2, 6, 10, 128)
    padded_scalar = torch.randn(2, 6, scalar_dim)
    padded_mask = torch.zeros(2, 6)
    padded_bin_mask = torch.zeros(2, 6, 10)
    padded_ptm[:, :3] = ptm
    padded_scalar[:, :3] = scalar
    padded_mask[:, :3] = 1
    padded_bin_mask[:, :3] = 1

    with torch.inference_mode():
        trimmed, auxiliary = model(
            ptm,
            scalar,
            mask,
            bin_mask,
            return_auxiliary=True,
        )
        padded = model(
            padded_ptm,
            padded_scalar,
            padded_mask,
            padded_bin_mask,
        )[:, :3]

    assert auxiliary["semantic_logits"].shape == (2, 3, 2)
    assert torch.allclose(trimmed, padded, atol=1e-5, rtol=1e-5)
