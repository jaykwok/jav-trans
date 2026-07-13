from __future__ import annotations

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
    OUTER_EDGE_REFINER_V2_FEATURE_SCHEMA,
    OUTER_EDGE_REFINER_V2_MODEL_ARCH,
    OUTER_EDGE_REFINER_V2_RUNTIME_ADAPTER,
    OUTER_EDGE_REFINER_V2_SCHEMA,
    decode_outer_edge_probabilities,
)
from boundary.split_model import (
    SEMANTIC_SPLIT_FEATURE_SCHEMA,
    SEMANTIC_SPLIT_LABELS,
    SEMANTIC_SPLIT_V2_RUNTIME_ADAPTER,
    SEMANTIC_SPLIT_V2_SCHEMA,
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
    assert OUTER_EDGE_REFINER_V2_MODEL_ARCH == "full_island_semantic_edges_mamba_v1"
    assert OUTER_EDGE_REFINER_V2_RUNTIME_ADAPTER == "paired_outer_edges_v2"
    assert OUTER_EDGE_REFINER_V2_FEATURE_SCHEMA == "full_island_semantic_edge_features_v2"

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
            "src/checkpoints/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf/outer_edge_refiner_v1."
            "jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt",
            OUTER_EDGE_REFINER_SCHEMA,
            OUTER_EDGE_FEATURE_SCHEMA,
            OUTER_EDGE_REFINER_RUNTIME_ADAPTER,
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
