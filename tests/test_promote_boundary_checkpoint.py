from __future__ import annotations

from pathlib import Path

import pytest

from boundary.split_model import SEMANTIC_SPLIT_V2_SCHEMA
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
