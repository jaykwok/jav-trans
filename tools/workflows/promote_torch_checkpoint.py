#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.align.checkpoint_io import (  # noqa: E402
    add_checkpoint_arguments,
    load_checkpoint,
)


def promote_checkpoint(
    *,
    input_path: Path,
    output_path: Path,
    artifact_name: str,
    display_name: str,
    version: str,
    pipeline_stage: int,
    pipeline_role: str,
    source_training_run: str,
    selected_validation: dict[str, Any] | None = None,
    metadata_updates: dict[str, Any] | None = None,
    promotion_reason: str = "",
    promoted_at: str | None = None,
    trusted: bool = False,
) -> dict[str, Any]:
    import torch

    payload = load_checkpoint(input_path, trusted=trusted)
    metadata = dict(payload.get("metadata") or {})
    if metadata_updates is not None:
        metadata.update(metadata_updates)
    timestamp = promoted_at or datetime.now(timezone.utc).isoformat()
    metadata["artifact"] = {
        **dict(metadata.get("artifact") or {}),
        "name": artifact_name,
        "display_name": display_name,
        "version": version,
        "pipeline_stage": int(pipeline_stage),
        "pipeline_role": pipeline_role,
        "checkpoint_format_version": 1,
        "production_filename": output_path.name,
        "promoted": True,
        "promoted_at": timestamp,
        "self_contained": True,
        "source_training_run": source_training_run,
    }
    if selected_validation is not None:
        metadata["selected_validation"] = selected_validation
    if promotion_reason:
        metadata["promotion_reason"] = promotion_reason
    payload["metadata"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.promoting")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return payload


def _read_selected_validation(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("--selected-validation-file must contain a JSON object")
    return value


def _read_metadata_json(value: str) -> dict[str, Any] | None:
    if not value:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--metadata-json must contain a JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a trained PyTorch checkpoint into a production artifact "
            "while preserving model weights and completing the artifact contract."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--pipeline-stage", type=int, required=True)
    parser.add_argument("--pipeline-role", required=True)
    parser.add_argument("--source-training-run", required=True)
    parser.add_argument("--selected-validation-file", type=Path)
    parser.add_argument("--metadata-json", default="")
    parser.add_argument("--promotion-reason", default="")
    add_checkpoint_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = promote_checkpoint(
        input_path=args.input,
        output_path=args.output,
        artifact_name=args.artifact_name,
        display_name=args.display_name,
        version=args.version,
        pipeline_stage=args.pipeline_stage,
        pipeline_role=args.pipeline_role,
        source_training_run=args.source_training_run,
        selected_validation=_read_selected_validation(args.selected_validation_file),
        metadata_updates=_read_metadata_json(args.metadata_json),
        promotion_reason=args.promotion_reason,
        trusted=args.trust_checkpoint,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact": payload["metadata"]["artifact"],
                "decision_config": payload.get("decision_config"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
