from __future__ import annotations

from pathlib import Path

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
from boundary.split_model import (
    SEMANTIC_SPLIT_FEATURE_SCHEMA,
    SEMANTIC_SPLIT_LABELS,
    SEMANTIC_SPLIT_V2_RUNTIME_ADAPTER,
    SEMANTIC_SPLIT_V2_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"


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
