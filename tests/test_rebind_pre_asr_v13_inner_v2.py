from __future__ import annotations

import hashlib

import pytest

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.asr.cueqc.rebind_pre_asr_v13_inner_v2 import rebind


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rebind_updates_both_inner_contract_locations_without_model_change(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    cueqc_path = tmp_path / "cueqc.pt"
    inner_path = tmp_path / "inner.pt"
    output = tmp_path / "out.pt"
    cueqc = {
        "schema": "cueqc_pre_asr_semantic_chunk_v13",
        "model_config": {"num_classes": 2},
        "decision_config": {"decision_mode": "argmax"},
        "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "metadata": {"training_labels": ["drop", "keep"]},
    }
    inner = {
        "schema": "inner_edge_refiner_v2",
        "metadata": {
            "runtime_adapter": "paired_acoustic_inner_edges_binary_v2",
            "training_labels": ["background", "semantic_core"],
            "excluded_training_labels": ["unsure"],
            "boundary_serialization_contract_id": (
                ACOUSTIC_BINARY_V12_CONTRACT.contract_id
            ),
        },
    }
    torch.save(cueqc, cueqc_path)
    torch.save(inner, inner_path)

    summary = rebind(
        cueqc_checkpoint=cueqc_path, inner_checkpoint=inner_path, output=output
    )
    rebound = torch.load(output, map_location="cpu", weights_only=False)
    assert rebound["inner_edge_refiner_weights_sha256"] == _sha(inner_path)
    assert rebound["metadata"]["inner_edge_refiner_weights_sha256"] == _sha(inner_path)
    assert rebound["metadata"]["boundary_serialization_contract_id"] == (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    )
    assert torch.equal(rebound["model_state_dict"]["weight"], cueqc["model_state_dict"]["weight"])
    assert summary["cueqc_checkpoint_sha256"] == _sha(output)


def test_rebind_rejects_old_inner_v1(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    cueqc_path = tmp_path / "cueqc.pt"
    inner_path = tmp_path / "inner.pt"
    torch.save({
        "schema": "cueqc_pre_asr_semantic_chunk_v13",
        "model_config": {"num_classes": 2},
        "decision_config": {"decision_mode": "argmax"},
        "model_state_dict": {}, "metadata": {},
    }, cueqc_path)
    torch.save({"schema": "inner_edge_refiner_v1"}, inner_path)
    with pytest.raises(ValueError, match="Inner Edge Refiner v2"):
        rebind(cueqc_checkpoint=cueqc_path, inner_checkpoint=inner_path, output=tmp_path / "out.pt")
