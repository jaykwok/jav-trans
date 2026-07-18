from __future__ import annotations

from typing import Any, NoReturn

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT


# Outer v3 is intentionally a placeholder until its architecture, training
# distribution, and edge-audit contract are reviewed.  Keeping a schema name
# here lets the pipeline reject stale artifacts without retaining an
# executable, unaudited model path.
OUTER_EDGE_REFINER_V3_SCHEMA = "outer_edge_refiner_v3"
OUTER_EDGE_REFINER_V3_STATUS = "pending_outer_v3_audit"
OUTER_EDGE_REFINER_V3_ARTIFACT = {
    "name": "outer_edge_refiner",
    "display_name": "Outer Edge Refiner",
    "version": "v3",
    "pipeline_stage": 2,
    "pipeline_role": "full_island_binary_semantic_outer_edges",
    "status": OUTER_EDGE_REFINER_V3_STATUS,
}


class OuterEdgeRefinerV3:
    """Type-only placeholder for the not-yet-audited Outer v3 contract."""

    status = OUTER_EDGE_REFINER_V3_STATUS

    def signature(self) -> dict[str, Any]:
        return {
            "schema": OUTER_EDGE_REFINER_V3_SCHEMA,
            "status": OUTER_EDGE_REFINER_V3_STATUS,
            "boundary_serialization_contract_id": (
                ACOUSTIC_BINARY_V12_CONTRACT.contract_id
            ),
        }

    def predict_islands(self, **_: Any) -> NoReturn:
        raise RuntimeError(
            "pending_outer_v3_audit: Outer v3 is a placeholder and cannot run"
        )


def load_outer_edge_refiner_v3(*_: Any, **__: Any) -> NoReturn:
    """Reject every artifact until the independent Outer v3 audit is closed."""

    raise RuntimeError(
        "pending_outer_v3_audit: Outer v3 checkpoint loading is disabled"
    )


def build_outer_edge_refiner_v3_checkpoint(*_: Any, **__: Any) -> NoReturn:
    """Prevent accidental training/promotion through an unaudited placeholder."""

    raise RuntimeError(
        "pending_outer_v3_audit: Outer v3 checkpoint construction is disabled"
    )
