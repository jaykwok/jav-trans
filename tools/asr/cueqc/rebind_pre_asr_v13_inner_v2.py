#!/usr/bin/env python3
"""Rebind an audited binary CueQC v13 checkpoint to acoustic Inner v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.pre_asr_cueqc import PRE_ASR_CUEQC_SCHEMA  # noqa: E402
from boundary.inner_refiner_v2 import (  # noqa: E402
    INNER_EDGE_REFINER_V2_RUNTIME_ADAPTER,
    INNER_EDGE_REFINER_V2_SCHEMA,
)
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rebind(*, cueqc_checkpoint: Path, inner_checkpoint: Path, output: Path) -> dict:
    import torch

    cueqc = torch.load(cueqc_checkpoint, map_location="cpu", weights_only=False)
    inner = torch.load(inner_checkpoint, map_location="cpu", weights_only=False)
    if cueqc.get("schema") != PRE_ASR_CUEQC_SCHEMA:
        raise ValueError("rebind requires CueQC v13 schema")
    if int((cueqc.get("model_config") or {}).get("num_classes") or 0) != 2:
        raise ValueError("rebind requires binary CueQC v13")
    if (cueqc.get("decision_config") or {}).get("decision_mode") != "argmax":
        raise ValueError("rebind requires CueQC argmax")
    if inner.get("schema") != INNER_EDGE_REFINER_V2_SCHEMA:
        raise ValueError("rebind requires Inner Edge Refiner v2")
    inner_metadata = dict(inner.get("metadata") or {})
    if inner_metadata.get("runtime_adapter") != INNER_EDGE_REFINER_V2_RUNTIME_ADAPTER:
        raise ValueError("rebind requires acoustic Inner v2 runtime adapter")
    if inner_metadata.get("training_labels") != ["background", "semantic_core"]:
        raise ValueError("rebind requires binary Inner v2 training labels")
    if inner_metadata.get("excluded_training_labels") != ["unsure"]:
        raise ValueError("rebind requires Inner v2 unsure exclusion")
    if (
        inner_metadata.get("boundary_serialization_contract_id")
        != ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError("rebind requires current boundary serialization contract")

    inner_sha = file_sha256(inner_checkpoint)
    try:
        inner_display_path = str(inner_checkpoint.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        inner_display_path = str(inner_checkpoint)
    metadata = dict(cueqc.get("metadata") or {})
    rebind_record = {
        "schema": "pre_asr_cueqc_v13_inner_v2_rebind_v1",
        "rebound_at": datetime.now(timezone.utc).isoformat(),
        "inner_edge_refiner_checkpoint": inner_display_path,
        "inner_edge_refiner_weights_sha256": inner_sha,
        "cueqc_model_state_sha256": hashlib.sha256(
            b"".join(
                tensor.detach().cpu().numpy().tobytes()
                for _key, tensor in sorted((cueqc.get("model_state_dict") or {}).items())
            )
        ).hexdigest(),
    }
    cueqc["inner_edge_refiner_checkpoint"] = inner_display_path
    cueqc["inner_edge_refiner_weights_sha256"] = inner_sha
    metadata["inner_edge_refiner_checkpoint"] = inner_display_path
    metadata["inner_edge_refiner_weights_sha256"] = inner_sha
    metadata["boundary_serialization_contract_id"] = (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    )
    metadata["inner_v2_rebind"] = rebind_record
    cueqc["metadata"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(cueqc, temporary)
    temporary.replace(output)
    return {
        **rebind_record,
        "cueqc_checkpoint": str(output),
        "cueqc_checkpoint_sha256": file_sha256(output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cueqc-checkpoint", required=True)
    parser.add_argument("--inner-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = rebind(
        cueqc_checkpoint=Path(args.cueqc_checkpoint),
        inner_checkpoint=Path(args.inner_checkpoint),
        output=Path(args.output),
    )
    if args.summary:
        Path(args.summary).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
