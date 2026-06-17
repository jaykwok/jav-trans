from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.asr.cueqc import extract_features_v3_fusion
from tools.asr.cueqc import merge_features_v3_fusion
from tools.asr.cueqc import predict_v3_fusion
from tools.asr.cueqc import train_mamba_v3_fusion


def _feature_sample(label: int = -1) -> dict:
    return {
        "sample_id": "cueqc-VIDEO-chunk00001",
        "audio_id": "VIDEO",
        "asr_frames": torch.randn(2, 4),
        "token_trace": torch.randn(3, 2),
        "decoder_stats": torch.randn(3),
        "structured": torch.randn(2),
        "__label__": label,
    }


def _feature_bundle(label: int = -1) -> dict:
    return {
        "schema": "cueqc_mamba_v3_fusion_features",
        "version": 3,
        "samples": [_feature_sample(label)],
        "labels": torch.tensor([label], dtype=torch.long),
        "meta": [
            {
                "sample_id": "cueqc-VIDEO-chunk00001",
                "audio_id": "VIDEO",
                "video_id": "VIDEO",
                "chunk_index": 1,
                "start": 0.0,
                "end": 1.0,
                "text": "あ",
            }
        ],
        "feature_config": {
            "feature_source": "asr_internal",
            "uses_bge": False,
            "text_embedding": "none",
            "asr_dim": 4,
            "token_dim": 2,
            "decoder_dim": 3,
            "structured_dim": 2,
            "token_feature_names": ["tok_a", "tok_b"],
            "decoder_feature_names": ["dec_a", "dec_b", "dec_c"],
            "structured_feature_names": ["struct_a", "struct_b"],
        },
        "label_config": {"drop": 0, "keep": 1},
    }


def test_cueqc_v3_feature_extractor_marks_unlabeled_candidate_rows():
    row = {
        "schema": "cueqc_candidate_v1",
        "sample_id": "cueqc-VIDEO-chunk00001",
        "targets": {},
    }

    assert extract_features_v3_fusion._label_from_row(row) == -1


def test_cueqc_v3_feature_merge_preserves_unlabeled_shards(tmp_path: Path):
    shard_a = tmp_path / "a.pt"
    shard_b = tmp_path / "b.pt"
    out = tmp_path / "merged.pt"
    torch.save(_feature_bundle(label=-1), shard_a)
    bundle_b = _feature_bundle(label=1)
    bundle_b["samples"][0]["sample_id"] = "cueqc-VIDEO-chunk00002"
    bundle_b["meta"][0]["sample_id"] = "cueqc-VIDEO-chunk00002"
    torch.save(bundle_b, shard_b)

    summary = merge_features_v3_fusion.merge([shard_a, shard_b], out)
    merged = torch.load(out, map_location="cpu", weights_only=False)

    assert summary["samples"] == 2
    assert summary["labels_unlabeled"] == 1
    assert merged["labels"].tolist() == [-1, 1]
    assert len(merged["samples"]) == 2


def test_cueqc_v3_training_rejects_unlabeled_feature_bundle(tmp_path: Path):
    features_path = tmp_path / "features.pt"
    torch.save(_feature_bundle(label=-1), features_path)

    with pytest.raises(RuntimeError, match="unlabeled"):
        train_mamba_v3_fusion.train(
            train_mamba_v3_fusion.TrainConfig(max_steps=1, device="cpu", test_audio_id=""),
            features_path=features_path,
            output_dir=tmp_path / "out",
        )


def test_cueqc_v3_predict_exports_high_confidence_pseudo_labels(tmp_path: Path, monkeypatch):
    features_path = tmp_path / "features.pt"
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_dir = tmp_path / "pred"
    torch.save(_feature_bundle(label=-1), features_path)
    checkpoint = {
        "schema": "cueqc_mamba_checkpoint_v3_fusion",
        "model_config": {
            "asr_dim": 4,
            "token_dim": 2,
            "decoder_dim": 3,
            "structured_dim": 2,
            "hidden_size": 8,
            "num_layers": 1,
            "state_size": 8,
            "num_heads": 2,
            "head_dim": 8,
            "n_groups": 1,
            "chunk_size": 4,
            "mlp_dim": 4,
            "dropout": 0.0,
        },
        "state_dict": {},
        "normalization": {
            "asr_mean": torch.zeros(4),
            "asr_std": torch.ones(4),
            "token_mean": torch.zeros(2),
            "token_std": torch.ones(2),
            "decoder_mean": torch.zeros(3),
            "decoder_std": torch.ones(3),
            "structured_mean": torch.zeros(2),
            "structured_std": torch.ones(2),
        },
        "decision_config": {"drop_threshold": 0.85},
    }
    torch.save(checkpoint, checkpoint_path)

    class StubModel(torch.nn.Module):
        def to(self, device):
            return self

        def eval(self):
            return self

        def forward(self, **kwargs):
            return torch.tensor([[4.0, 0.0]], dtype=torch.float32)

    def fake_model_from_checkpoint(checkpoint, device):
        return StubModel()

    monkeypatch.setattr(predict_v3_fusion, "_model_from_checkpoint", fake_model_from_checkpoint)

    rc = predict_v3_fusion.run(
        SimpleNamespace(
            features=str(features_path),
            checkpoint=str(checkpoint_path),
            output_dir=str(output_dir),
            device="cpu",
            batch_size=8,
            drop_threshold=None,
            keep_threshold=0.95,
        )
    )

    assert rc == 0
    pseudo_rows = [
        json.loads(line)
        for line in (output_dir / "cueqc_pseudo_labels.high_conf.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(pseudo_rows) == 1
    assert pseudo_rows[0]["targets"]["display_decision"] == "drop"
    assert pseudo_rows[0]["targets"]["display_label"] == 0
