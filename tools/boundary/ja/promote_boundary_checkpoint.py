#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.ja.proposal import (  # noqa: E402
    BOUNDARY_PROPOSAL_SCORER_ARTIFACT,
    BOUNDARY_PROPOSAL_SCORER_SCHEMA,
)
from boundary.outer_refiner_v2 import (  # noqa: E402
    OUTER_EDGE_REFINER_V2_ARTIFACT,
    OUTER_EDGE_REFINER_V2_SCHEMA,
)
from boundary.split_model import (  # noqa: E402
    SEMANTIC_SPLIT_V2_ARTIFACT,
    SEMANTIC_SPLIT_V2_SCHEMA,
    SEMANTIC_SPLIT_V4_ARTIFACT,
    SEMANTIC_SPLIT_V4_SCHEMA,
)

ARTIFACT_BY_SCHEMA = {
    BOUNDARY_PROPOSAL_SCORER_SCHEMA: BOUNDARY_PROPOSAL_SCORER_ARTIFACT,
    SEMANTIC_SPLIT_V2_SCHEMA: SEMANTIC_SPLIT_V2_ARTIFACT,
    SEMANTIC_SPLIT_V4_SCHEMA: SEMANTIC_SPLIT_V4_ARTIFACT,
    OUTER_EDGE_REFINER_V2_SCHEMA: OUTER_EDGE_REFINER_V2_ARTIFACT,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(payload: Mapping[str, Any]) -> str:
    import torch

    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint missing model_state_dict")
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"model_state_dict.{key} must be a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def promote_checkpoint(
    *,
    checkpoint: Path,
    output: Path,
    source_training_run: str,
) -> dict[str, Any]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    schema = str(payload.get("schema") or "")
    try:
        artifact_template = ARTIFACT_BY_SCHEMA[schema]
    except KeyError as exc:
        raise ValueError(f"unsupported boundary checkpoint schema: {schema!r}") from exc
    weights_before = _state_digest(payload)
    metadata = dict(payload.get("metadata") or {})
    metadata["artifact"] = {
        **artifact_template,
        "checkpoint_format_version": 1,
        "production_filename": output.name,
        "promoted": True,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "self_contained": True,
        "source_training_run": source_training_run,
    }
    payload["metadata"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.promoting")
    torch.save(payload, temp)
    reloaded = torch.load(temp, map_location="cpu", weights_only=False)
    weights_after = _state_digest(reloaded)
    if weights_after != weights_before:
        temp.unlink(missing_ok=True)
        raise ValueError("model weights changed during boundary checkpoint promotion")
    temp.replace(output)
    return {
        "schema": "boundary_checkpoint_promotion_v1",
        "checkpoint_schema": schema,
        "source_checkpoint": str(checkpoint),
        "output_checkpoint": str(output),
        "output_sha256": _file_sha256(output),
        "weights_digest_before": weights_before,
        "weights_digest_after": weights_after,
        "weights_unchanged": True,
        "artifact": dict(reloaded["metadata"]["artifact"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-training-run", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args(argv)
    summary = promote_checkpoint(
        checkpoint=Path(args.checkpoint),
        output=Path(args.output),
        source_training_run=str(args.source_training_run),
    )
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
