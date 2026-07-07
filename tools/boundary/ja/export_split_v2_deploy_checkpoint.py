"""Strip the in-model PTM projector from a projection-trained Semantic Split v2
checkpoint and fold it into the runtime pre-projection path.

A checkpoint trained with ``--ptm-projector-dim`` carries an in-model
``Linear(ptm_input_dim -> ptm_projected_dim)`` and full-dim (e.g. 2088) frame
normalization. The runtime boundary backend never feeds raw full-dim PTM to the
split model: it pre-projects via the affine embedded in ``feature_config`` and
the sequence-feature provider emits ``ptm_projected_dim`` features. A
projection-trained checkpoint cannot deploy on that path as-is (its forward
expects the raw ``ptm_input_dim`` block).

This tool produces a deployment checkpoint that:

1. removes the in-model projector (``ptm_input_dim``/``ptm_projected_dim`` -> 0)
   so the model consumes the pre-projected frame directly — ``frame_proj`` is
   already ``Linear(ptm_projected_dim + mfcc_dim)`` and is left untouched;
2. embeds the folded affine (``mean``, ``components = weight / std``,
   ``digest``) into ``feature_config.ptm_projection`` so the pipeline
   materializes the projection npz and the backend pre-projects;
3. rewrites the frame normalization so the PTM block is identity (the affine
   already normalized it) and the MFCC block keeps its trained stats — this is
   what prevents the double normalization the in-model path would otherwise
   incur once the affine has folded normalization in.

The transform is exact: ``W((x - mu) / sigma) == (x - mu) @ (W / sigma).T``.
Identity is covered by ``tests/test_split_v2_deploy_identity.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from boundary.sequence_features import PTM_PROJECTION_SCHEMA, ptm_projection_digest
from boundary.split_model import (
    SEMANTIC_SPLIT_V2_SCHEMA,
    load_semantic_split_verifier,
)


def _load_affine(projection_npz: Path) -> dict[str, Any]:
    bundle = np.load(projection_npz)
    if str(bundle["schema"]) != PTM_PROJECTION_SCHEMA:
        raise ValueError(
            f"projection npz schema {str(bundle['schema'])!r} != "
            f"{PTM_PROJECTION_SCHEMA!r}: {projection_npz}"
        )
    mean = np.asarray(bundle["mean"], dtype=np.float32)
    components = np.asarray(bundle["components"], dtype=np.float32)
    return {
        "schema": PTM_PROJECTION_SCHEMA,
        "mean": mean,
        "components": components,
        "digest": ptm_projection_digest(mean, components),
    }


def build_deploy_checkpoint(
    checkpoint_path: Path,
    projection_npz: Path,
    output_path: Path,
) -> Path:
    """Transform a projection-trained v2 checkpoint into a projector-free
    deployment checkpoint with the affine embedded. Returns the output path."""

    import torch

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SEMANTIC_SPLIT_V2_SCHEMA:
        raise ValueError(
            f"{checkpoint_path} is not a Semantic Split v2 checkpoint "
            f"(schema={payload.get('schema')!r})"
        )
    model_config = dict(payload.get("model_config") or {})
    ptm_input_dim = int(model_config.get("ptm_input_dim") or 0)
    ptm_projected_dim = int(model_config.get("ptm_projected_dim") or 0)
    if ptm_input_dim <= 0 or ptm_projected_dim <= 0:
        raise ValueError(
            "checkpoint has no in-model PTM projector (ptm_input_dim="
            f"{ptm_input_dim}, ptm_projected_dim={ptm_projected_dim}); nothing "
            "to fold — deploy it directly."
        )
    if model_config.get("ptm_projector_residual"):
        raise ValueError(
            "checkpoint uses a non-foldable residual projector; the affine "
            "cannot reproduce it. Train with a plain linear projector."
        )
    frame_dim = int(model_config["frame_dim"])
    mfcc_dim = frame_dim - ptm_projected_dim
    if mfcc_dim <= 0:
        raise ValueError(
            f"frame_dim {frame_dim} <= ptm_projected_dim {ptm_projected_dim}; "
            "no MFCC block to preserve"
        )

    state_dict = dict(payload["model_state_dict"])
    projector_key = "ptm_projector.weight"
    if projector_key not in state_dict:
        raise ValueError(
            f"model_config declares a projector but {projector_key} is absent "
            "from the state_dict"
        )
    projector_weight = np.asarray(
        state_dict.pop(projector_key).cpu().numpy(), dtype=np.float64
    )
    if projector_weight.shape != (ptm_projected_dim, ptm_input_dim):
        raise ValueError(
            f"ptm_projector.weight shape {projector_weight.shape} != "
            f"({ptm_projected_dim}, {ptm_input_dim})"
        )

    affine = _load_affine(projection_npz)
    if affine["mean"].shape != (ptm_input_dim,):
        raise ValueError(
            f"projection mean dim {affine['mean'].shape[0]} != ptm_input_dim "
            f"{ptm_input_dim}"
        )
    if affine["components"].shape != (ptm_projected_dim, ptm_input_dim):
        raise ValueError(
            f"projection components shape {affine['components'].shape} != "
            f"({ptm_projected_dim}, {ptm_input_dim})"
        )

    normalization = dict(payload.get("normalization") or {})
    frame_mean = np.asarray(normalization["frame_mean"], dtype=np.float64)
    frame_std = np.asarray(normalization["frame_std"], dtype=np.float64)
    if frame_mean.shape[0] != ptm_input_dim + mfcc_dim:
        raise ValueError(
            f"frame_mean dim {frame_mean.shape[0]} != ptm_input_dim+mfcc_dim "
            f"({ptm_input_dim}+{mfcc_dim})"
        )
    # Fold integrity: the exported components must equal projector weight divided
    # by the PTM-block std the model normalized with. Catches a stale/wrong npz.
    ptm_std = np.maximum(frame_std[:ptm_input_dim], 1e-6)
    reconstructed_components = (projector_weight / ptm_std[None, :]).astype(np.float32)
    if not np.allclose(reconstructed_components, affine["components"], atol=1e-5):
        max_drift = float(
            np.abs(reconstructed_components - affine["components"]).max()
        )
        raise ValueError(
            f"projection npz components do not reproduce ptm_projector.weight / "
            f"frame_std (max drift {max_drift:.2e}); the affine was not exported "
            f"from this checkpoint."
        )

    new_model_config = dict(model_config)
    new_model_config["ptm_input_dim"] = 0
    new_model_config["ptm_projected_dim"] = 0

    mfcc_mean = frame_mean[ptm_input_dim : ptm_input_dim + mfcc_dim].astype(np.float32)
    mfcc_std = frame_std[ptm_input_dim : ptm_input_dim + mfcc_dim].astype(np.float32)
    new_normalization = {
        # PTM block identity (the affine already normalized it), MFCC block kept.
        "frame_mean": np.concatenate(
            [np.zeros(ptm_projected_dim, dtype=np.float32), mfcc_mean]
        ),
        "frame_std": np.concatenate(
            [np.ones(ptm_projected_dim, dtype=np.float32), mfcc_std]
        ),
        "scalar_mean": normalization["scalar_mean"],
        "scalar_std": normalization["scalar_std"],
    }

    feature_config = dict(payload.get("feature_config") or {})
    feature_config["ptm_projection"] = {
        "schema": PTM_PROJECTION_SCHEMA,
        "mean": affine["mean"],
        "components": affine["components"],
        "digest": affine["digest"],
        "input_dim": ptm_input_dim,
        "dim": ptm_projected_dim,
    }

    metadata = dict(payload.get("metadata") or {})
    metadata["deploy"] = {
        "source_checkpoint": str(checkpoint_path),
        "projection_npz": str(projection_npz),
        "projection_digest": affine["digest"],
        "stripped_in_model_projector": True,
        "fold_integrity_max_drift": float(
            np.abs(reconstructed_components - affine["components"]).max()
        ),
    }

    # Transform the already-valid v2 payload in place. schema / model_arch /
    # decision_config are unchanged; only the projector, model_config,
    # normalization, feature_config and metadata provenance change.
    deploy_payload = dict(payload)
    deploy_payload["model_config"] = new_model_config
    deploy_payload["model_state_dict"] = state_dict
    deploy_payload["normalization"] = new_normalization
    deploy_payload["feature_config"] = feature_config
    deploy_payload["metadata"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deploy_payload, output_path)

    # Canonical reload self-check: the stripped checkpoint must validate and load
    # as a normal v2 checkpoint with the projector gone.
    load_semantic_split_verifier(output_path, device="cpu")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fold a projection-trained Semantic Split v2 checkpoint's in-model "
            "PTM projector into the runtime pre-projection path."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Projection-trained v2 checkpoint (has ptm_projector).",
    )
    parser.add_argument(
        "--projection-npz",
        required=True,
        help=(
            "Folded affine npz exported by the trainer "
            "(learned_ptm_projection.npz)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output projector-free deployment checkpoint path.",
    )
    args = parser.parse_args()
    out = build_deploy_checkpoint(
        Path(args.checkpoint), Path(args.projection_npz), Path(args.output)
    )
    print(json.dumps({"deploy_checkpoint": str(out)}))


if __name__ == "__main__":
    main()
