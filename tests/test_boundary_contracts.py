from __future__ import annotations

import pytest

from boundary.contracts import (
    ACOUSTIC_BINARY_V12_CONTRACT,
    require_boundary_contract_id,
)
from boundary.ja.model import (
    SPEECH_ISLAND_SCORER_V8_MODEL_TYPE,
    SPEECH_ISLAND_SCORER_V8_SCHEMA,
    load_speech_island_scorer_checkpoint,
)
from boundary.ja.proposal import (
    BOUNDARY_PROPOSAL_SCORER_MODEL_TYPE,
    BOUNDARY_PROPOSAL_SCORER_SCHEMA,
    load_boundary_proposal_checkpoint,
)
from boundary.split_model import (
    SEMANTIC_SPLIT_V4_MODEL_ARCH,
    SEMANTIC_SPLIT_V4_SCHEMA,
    load_acoustic_split_v4_planner,
)


def test_current_boundary_contract_centralizes_serialization_and_prediction_schema() -> None:
    assert ACOUSTIC_BINARY_V12_CONTRACT.contract_id == "boundary_acoustic_binary_v12"
    assert (
        ACOUSTIC_BINARY_V12_CONTRACT.inner_prediction_schema
        == "binary_acoustic_inner_edges_v2_prediction"
    )


@pytest.mark.parametrize("contract_id", ["", "boundary_acoustic_binary_v11"])
def test_boundary_contract_rejects_missing_or_retired_ids(contract_id: str) -> None:
    with pytest.raises(ValueError, match="unsupported Boundary serialization contract"):
        require_boundary_contract_id(contract_id)


def test_current_contract_id_is_the_only_runtime_key() -> None:
    assert require_boundary_contract_id(ACOUSTIC_BINARY_V12_CONTRACT.contract_id) == (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    )
    assert ACOUSTIC_BINARY_V12_CONTRACT.matches(
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    )
    assert not hasattr(ACOUSTIC_BINARY_V12_CONTRACT, "pipeline_version")
    assert not hasattr(ACOUSTIC_BINARY_V12_CONTRACT, "cache_version")
    assert not hasattr(ACOUSTIC_BINARY_V12_CONTRACT, "pipeline_generation")
    assert not hasattr(ACOUSTIC_BINARY_V12_CONTRACT, "cache_generation")


def _scorer_payload(contract_id: str | None) -> dict:
    metadata = {}
    if contract_id is not None:
        metadata["boundary_serialization_contract_id"] = contract_id
    return {
        "schema": SPEECH_ISLAND_SCORER_V8_SCHEMA,
        "model_type": SPEECH_ISLAND_SCORER_V8_MODEL_TYPE,
        "model_config": {"ptm_dim": 1, "mfcc_dim": 1, "input_dim": 2},
        "metadata": metadata,
    }


def _proposal_payload(contract_id: str | None) -> dict:
    metadata = {}
    if contract_id is not None:
        metadata["boundary_serialization_contract_id"] = contract_id
    return {
        "schema": BOUNDARY_PROPOSAL_SCORER_SCHEMA,
        "model_type": BOUNDARY_PROPOSAL_SCORER_MODEL_TYPE,
        "model_config": {
            "ptm_dim": 1,
            "mfcc_dim": 1,
            "input_dim": 2,
            "hidden_size": 1,
            "num_layers": 1,
            "output_dim": 1,
        },
        "metadata": metadata,
    }


def _split_payload(contract_id: str | None) -> dict:
    metadata = {}
    if contract_id is not None:
        metadata["boundary_serialization_contract_id"] = contract_id
    return {
        "schema": SEMANTIC_SPLIT_V4_SCHEMA,
        "model_arch": SEMANTIC_SPLIT_V4_MODEL_ARCH,
        "metadata": metadata,
    }


@pytest.mark.parametrize("contract_id", [None, "boundary_acoustic_binary_v11"])
@pytest.mark.parametrize(
    ("loader", "payload_factory"),
    [
        (load_speech_island_scorer_checkpoint, _scorer_payload),
        (load_boundary_proposal_checkpoint, _proposal_payload),
        (load_acoustic_split_v4_planner, _split_payload),
    ],
)
def test_boundary_model_loaders_reject_missing_or_retired_contract_ids(
    monkeypatch,
    tmp_path,
    contract_id,
    loader,
    payload_factory,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: payload_factory(contract_id),
    )

    with pytest.raises(ValueError, match="unsupported Boundary serialization contract"):
        loader(checkpoint, device="cpu")
