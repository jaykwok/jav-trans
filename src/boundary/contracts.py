from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundarySerializationContract:
    contract_id: str
    inner_prediction_schema: str

    def matches(self, contract_id: object) -> bool:
        return str(contract_id or "") == self.contract_id

ACOUSTIC_BINARY_V12_CONTRACT = BoundarySerializationContract(
    contract_id="boundary_acoustic_binary_v12",
    inner_prediction_schema="binary_acoustic_inner_edges_v2_prediction",
)

KNOWN_BOUNDARY_CONTRACT_IDS = frozenset({ACOUSTIC_BINARY_V12_CONTRACT.contract_id})


def require_boundary_contract_id(value: object) -> str:
    contract_id = str(value or "")
    if contract_id not in KNOWN_BOUNDARY_CONTRACT_IDS:
        raise ValueError(f"unsupported Boundary serialization contract: {contract_id!r}")
    return contract_id
