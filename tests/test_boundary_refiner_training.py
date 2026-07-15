from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boundary.cut_refiner import (
    CUT_EDGE_FEATURE_SCHEMA,
    CUT_EDGE_REFINER_RUNTIME_ADAPTER,
    CUT_EDGE_REFINER_SCHEMA,
)
from boundary.outer_refiner import (
    OUTER_EDGE_FEATURE_SCHEMA,
    OUTER_EDGE_REFINER_RUNTIME_ADAPTER,
    OUTER_EDGE_REFINER_SCHEMA,
)
from boundary.outer_refiner_v2 import (
    FullIslandOuterEdgeNetwork,
    OUTER_EDGE_REFINER_V2_FEATURE_SCHEMA,
    OUTER_EDGE_REFINER_V2_MODEL_ARCH,
    OUTER_EDGE_REFINER_V2_RUNTIME_ADAPTER,
    OUTER_EDGE_REFINER_V2_SCHEMA,
    decode_outer_edge_probabilities,
)
from boundary.sequence_features import (
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
)
from boundary.split_model import (
    SEMANTIC_SPLIT_FEATURE_SCHEMA,
    SEMANTIC_SPLIT_LABELS,
    SEMANTIC_SPLIT_V2_RUNTIME_ADAPTER,
    SEMANTIC_SPLIT_V2_SCHEMA,
)
from tools.boundary.ja.train_outer_edge_refiner_v2 import (
    _adaptive_sampling_probabilities,
    _edge_categories_by_audio_id,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"


def _outer_probabilities(*labels: str) -> np.ndarray:
    names = ("discardable", "semantic_target", "unsure")
    rows = []
    for label in labels:
        row = np.full(3, 0.05, dtype=np.float32)
        row[names.index(label)] = 0.9
        rows.append(row)
    return np.stack(rows)


def test_outer_v2_is_full_island_argmax_without_delta_cap() -> None:
    assert OUTER_EDGE_REFINER_V2_SCHEMA == "outer_edge_refiner_v2"
    assert (
        OUTER_EDGE_REFINER_V2_MODEL_ARCH
        == "full_island_learned_ptm_edges_mamba_v2"
    )
    assert OUTER_EDGE_REFINER_V2_RUNTIME_ADAPTER == "paired_outer_edges_v2"
    assert (
        OUTER_EDGE_REFINER_V2_FEATURE_SCHEMA
        == "full_island_raw_ptm_edge_features_v3"
    )

    prediction = decode_outer_edge_probabilities(
        _outer_probabilities(
            "discardable",
            "discardable",
            "semantic_target",
            "semantic_target",
            "discardable",
        ),
        raw_start_s=10.0,
        raw_end_s=20.0,
        frame_hop_s=1.0,
    )
    assert prediction.start_s == 12.0
    assert prediction.end_s == 14.0
    assert prediction.start_delta_s == 2.0
    assert prediction.end_delta_s == -6.0
    assert prediction.start_action == "refined"
    assert prediction.end_action == "refined"


def test_outer_v2_ptm_2048_to_128_projection_is_trainable() -> None:
    model = FullIslandOuterEdgeNetwork()

    assert model.ptm_projector.in_features == 2048
    assert model.ptm_projector.out_features == 128
    assert model.ptm_projector.weight.requires_grad is True
    assert model.ptm_projector.bias.requires_grad is True


def test_outer_v2_runtime_uses_raw_ptm_tail_dims_before_learned_projection() -> None:
    ptm = np.zeros((4, 2048), dtype=np.float32)
    ptm[:, -1] = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    provider = FrameSequenceFeatureProvider(
        duration_s=0.08,
        frame_hop_s=0.02,
        ptm=ptm,
        mfcc=np.zeros((4, 40), dtype=np.float32),
        config=FrameSequenceFeatureConfig(max_ptm_dims=2048),
    )

    features = provider.features_for_outer_island_v2(
        start_s=0.0,
        end_s=0.08,
        raw_ptm_dim=2048,
    )

    assert features.shape == (4, 2089)
    assert features[:, 2047].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert features[:, -1].tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


def test_outer_v2_edge_sampling_is_side_aware_and_adaptively_balances_rare_types(
    tmp_path: Path,
) -> None:
    negative_manifest = tmp_path / "negative.jsonl"
    negative_manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "audio_id": "breath",
                        "background_type": "breathing",
                        "omni_flags": ["breathing"],
                    }
                ),
                json.dumps(
                    {
                        "audio_id": "cry",
                        "background_type": "crying",
                        "omni_flags": ["crying"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    synthetic_details = tmp_path / "synthetic.jsonl"
    synthetic_details.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "audio_id": "common-1",
                        "sources": [
                            {"audio_id": "breath"},
                            {"source_audio_id": "core-1"},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "audio_id": "common-2",
                        "sources": [
                            {"audio_id": "breath"},
                            {"source_audio_id": "core-2"},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "audio_id": "rare",
                        "sources": [
                            {"source_audio_id": "core-3"},
                            {"audio_id": "cry"},
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    categories = _edge_categories_by_audio_id(
        synthetic_details=synthetic_details,
        negative_manifest=negative_manifest,
    )
    probabilities, counts, effective_size = _adaptive_sampling_probabilities(
        [
            {"audio_id": "common-1"},
            {"audio_id": "common-2"},
            {"audio_id": "rare"},
        ],
        categories,
    )

    assert categories["common-1"] == ("leading:breathing",)
    assert categories["rare"] == ("trailing:crying",)
    assert counts == {"leading:breathing": 2, "trailing:crying": 1}
    assert probabilities[2] > probabilities[0]
    assert probabilities.sum() == pytest.approx(1.0)
    assert effective_size < 3.0


def test_outer_v2_abstains_only_on_ambiguous_edge() -> None:
    prediction = decode_outer_edge_probabilities(
        _outer_probabilities(
            "unsure",
            "discardable",
            "semantic_target",
            "semantic_target",
            "discardable",
        ),
        raw_start_s=10.0,
        raw_end_s=15.0,
        frame_hop_s=1.0,
    )
    assert prediction.start_s == 10.0
    assert prediction.start_action == "abstain"
    assert prediction.end_s == 14.0
    assert prediction.end_action == "refined"
    assert prediction.abstain_reason == "unsure_before_target"


@pytest.mark.parametrize(
    ("relative_path", "schema", "feature_schema", "runtime_adapter"),
    [
        (
            "src/checkpoints/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf/outer_edge_refiner_v2."
            "jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt",
            OUTER_EDGE_REFINER_V2_SCHEMA,
            OUTER_EDGE_REFINER_V2_FEATURE_SCHEMA,
            OUTER_EDGE_REFINER_V2_RUNTIME_ADAPTER,
        ),
        (
            "src/checkpoints/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf/semantic_split_model_v2."
            "jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt",
            SEMANTIC_SPLIT_V2_SCHEMA,
            SEMANTIC_SPLIT_FEATURE_SCHEMA,
            SEMANTIC_SPLIT_V2_RUNTIME_ADAPTER,
        ),
        (
            "src/checkpoints/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf/cut_edge_refiner_v1."
            "jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt",
            CUT_EDGE_REFINER_SCHEMA,
            CUT_EDGE_FEATURE_SCHEMA,
            CUT_EDGE_REFINER_RUNTIME_ADAPTER,
        ),
    ],
)
def test_promoted_boundary_checkpoint_contract(
    relative_path: str,
    schema: str,
    feature_schema: str,
    runtime_adapter: str,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = ROOT / relative_path
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert payload["schema"] == schema
    assert payload["feature_config"]["schema"] == feature_schema
    assert payload["metadata"]["runtime_adapter"] == runtime_adapter
    assert payload["metadata"]["ptm_repo_id"] == REPO_ID


def test_semantic_split_checkpoint_has_v2_label_contract() -> None:
    torch = pytest.importorskip("torch")
    checkpoint = (
        ROOT
        / "src/checkpoints/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf/semantic_split_model_v2."
        "jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert payload["schema"] == SEMANTIC_SPLIT_V2_SCHEMA
    assert tuple(payload["metadata"]["labels"]) == SEMANTIC_SPLIT_LABELS
