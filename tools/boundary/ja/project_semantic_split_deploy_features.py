#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.sequence_features import (  # noqa: E402
    PTM_PROJECTION_SCHEMA,
    ptm_projection_digest,
)
from boundary.sequence_store import (  # noqa: E402
    StreamingFrameWriter,
    load_sequence_arrays,
    save_sequence_dataset,
)

OUTPUT_SCHEMA = "semantic_split_deploy_feature_projection_v1"


def _load_deploy_projection(
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_config = dict(checkpoint.get("feature_config") or {})
    projection = dict(feature_config.get("ptm_projection") or {})
    if str(projection.get("schema") or "") != PTM_PROJECTION_SCHEMA:
        raise ValueError("checkpoint does not contain a deploy PTM projection")

    mean = np.asarray(projection.get("mean"), dtype=np.float32)
    components = np.asarray(projection.get("components"), dtype=np.float32)
    input_dim = int(projection.get("input_dim") or mean.size)
    projected_dim = int(projection.get("dim") or components.shape[0])
    mfcc_dim = int(feature_config.get("mfcc_dim") or 0)
    if mean.shape != (input_dim,):
        raise ValueError(f"projection mean shape {mean.shape} != ({input_dim},)")
    if components.shape != (projected_dim, input_dim):
        raise ValueError(
            f"projection components shape {components.shape} "
            f"!= ({projected_dim}, {input_dim})"
        )
    digest = ptm_projection_digest(mean, components)
    expected_digest = str(projection.get("digest") or "")
    if not expected_digest or digest != expected_digest:
        raise ValueError(
            f"checkpoint PTM projection digest mismatch: {digest} != {expected_digest}"
        )
    if mfcc_dim <= 0:
        raise ValueError("checkpoint feature_config.mfcc_dim must be positive")
    return mean, components, mfcc_dim, digest


def project_dataset(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: str,
    batch_rows: int,
) -> dict[str, Any]:
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if output_path.resolve() == dataset_path.resolve():
        raise ValueError("output must differ from the source dataset")

    mean, components, mfcc_dim, digest = _load_deploy_projection(checkpoint_path)
    arrays = load_sequence_arrays(dataset_path)
    frames = arrays.pop("frame_features")
    input_dim = int(mean.size)
    projected_dim = int(components.shape[0])
    expected_source_dim = input_dim + mfcc_dim
    if int(frames.shape[2]) != expected_source_dim:
        raise ValueError(
            f"source frame dim {frames.shape[2]} != raw PTM+MFCC "
            f"{input_dim}+{mfcc_dim}"
        )

    torch_device = torch.device(device)
    vram_safety_ratio = None
    if torch_device.type == "cuda":
        vram_safety_ratio = apply_vram_safety_cap()
    mean_tensor = torch.from_numpy(mean).to(torch_device)
    components_tensor = torch.from_numpy(components).to(torch_device)

    writer = StreamingFrameWriter(output_path)
    row_count = int(frames.shape[0])
    with torch.inference_mode():
        for start in range(0, row_count, batch_rows):
            end = min(row_count, start + batch_rows)
            block = np.asarray(frames[start:end], dtype=np.float32)
            ptm = torch.from_numpy(
                np.ascontiguousarray(block[..., :input_dim])
            ).to(torch_device)
            projected = torch.matmul(ptm - mean_tensor, components_tensor.T)
            output_block = np.concatenate(
                (
                    projected.cpu().numpy(),
                    block[..., input_dim : input_dim + mfcc_dim],
                ),
                axis=-1,
            )
            writer.append(output_block)
            if end == row_count or end % max(batch_rows, 1024) == 0:
                print(f"projected_rows={end}/{row_count}", flush=True)
    writer.finalize()
    save_sequence_dataset(output_path, frames_finalized=True, **arrays)

    summary = {
        "schema": OUTPUT_SCHEMA,
        "source_dataset": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "row_count": row_count,
        "bin_count": int(frames.shape[1]),
        "source_frame_dim": int(frames.shape[2]),
        "projected_ptm_dim": projected_dim,
        "mfcc_dim": mfcc_dim,
        "output_frame_dim": projected_dim + mfcc_dim,
        "projection_digest": digest,
        "device": str(torch_device),
        "vram_safety_ratio": vram_safety_ratio,
        "shared_vram_budget": False,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project raw Semantic Split frames with a deploy checkpoint."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-rows", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = project_dataset(
        dataset_path=Path(args.dataset),
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        device=str(args.device),
        batch_rows=int(args.batch_rows),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
