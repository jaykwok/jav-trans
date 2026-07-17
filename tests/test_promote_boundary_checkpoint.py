from __future__ import annotations

from pathlib import Path

import pytest

from boundary.split_model import SEMANTIC_SPLIT_V2_SCHEMA, SEMANTIC_SPLIT_V4_SCHEMA
from boundary.outer_refiner_v2 import OUTER_EDGE_REFINER_V2_SCHEMA
from tools.boundary.ja.promote_boundary_checkpoint import promote_checkpoint


def test_promote_boundary_checkpoint_adds_canonical_artifact_without_weight_change(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "candidate.pt"
    output = tmp_path / "semantic_split_model_v2.repo.pt"
    torch.save(
        {
            "schema": SEMANTIC_SPLIT_V2_SCHEMA,
            "metadata": {"artifact": {"name": "semantic_split_model"}},
            "model_state_dict": {"weight": torch.tensor([[1.0, 2.0]])},
        },
        source,
    )

    summary = promote_checkpoint(
        checkpoint=source,
        output=output,
        source_training_run="agents/temp/train",
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)

    assert summary["weights_unchanged"] is True
    assert payload["metadata"]["artifact"]["production_filename"] == output.name
    assert payload["metadata"]["artifact"]["promoted"] is True
    assert payload["metadata"]["artifact"]["source_training_run"] == "agents/temp/train"


def test_promote_boundary_checkpoint_supports_outer_v2(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "candidate.pt"
    output = tmp_path / "outer_edge_refiner_v2.repo.pt"
    torch.save(
        {
            "schema": OUTER_EDGE_REFINER_V2_SCHEMA,
            "metadata": {"artifact": {"name": "outer_edge_refiner"}},
            "model_state_dict": {"weight": torch.tensor([[3.0, 4.0]])},
        },
        source,
    )

    summary = promote_checkpoint(
        checkpoint=source,
        output=output,
        source_training_run="agents/temp/outer-v2-train",
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)

    assert summary["weights_unchanged"] is True
    assert payload["metadata"]["artifact"]["version"] == "v2"
    assert payload["metadata"]["artifact"]["pipeline_stage"] == 2


def test_promote_boundary_checkpoint_supports_split_v4(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "candidate.pt"
    output = tmp_path / "semantic_split_model_v4.repo.pt"
    torch.save(
        {
            "schema": SEMANTIC_SPLIT_V4_SCHEMA,
            "metadata": {"artifact": {"name": "semantic_split_model"}},
            "model_state_dict": {"weight": torch.tensor([[5.0, 6.0]])},
        },
        source,
    )

    promote_checkpoint(
        checkpoint=source,
        output=output,
        source_training_run="agents/temp/split-v4-train",
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)

    assert payload["metadata"]["artifact"]["version"] == "v4"
    assert payload["metadata"]["artifact"]["pipeline_role"] == (
        "acoustic_binary_boundary_event_planner"
    )
