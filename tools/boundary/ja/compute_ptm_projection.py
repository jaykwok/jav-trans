#!/usr/bin/env python3
"""Compute a variance-preserving PTM projection from a feature cache.

Replaces the arbitrary ``ptm[:, :128]`` truncation: sampled cache frames give a
mean + top-K PCA basis over the full PTM dim; datasets and runtime then use
``(ptm - mean) @ components.T``. Output npz: ``mean`` (D,), ``components``
(K, D), ``explained_variance_ratio`` (K,), ``schema``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (SRC_ROOT, PROJECT_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

PTM_PROJECTION_SCHEMA = "speech_boundary_ja_ptm_projection_v1"


def run(args: argparse.Namespace) -> None:
    rows = []
    with Path(args.feature_manifest).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("empty feature manifest")
    rng = np.random.default_rng(args.seed)
    picked = rng.permutation(len(rows))[: args.max_windows]
    mean_sum: np.ndarray | None = None
    cov_sum: np.ndarray | None = None
    total = 0
    for count, row_index in enumerate(picked, start=1):
        bundle = np.load(str(rows[int(row_index)]["feature_path"]))
        ptm = np.asarray(bundle["ptm"], dtype=np.float64)
        if ptm.shape[0] > args.frames_per_window:
            frame_indexes = np.sort(
                rng.permutation(ptm.shape[0])[: args.frames_per_window]
            )
            ptm = ptm[frame_indexes]
        if mean_sum is None:
            mean_sum = np.zeros(ptm.shape[1], dtype=np.float64)
            cov_sum = np.zeros((ptm.shape[1], ptm.shape[1]), dtype=np.float64)
        mean_sum += ptm.sum(axis=0)
        cov_sum += ptm.T @ ptm
        total += ptm.shape[0]
        if args.log_every and count % args.log_every == 0:
            print(f"ptm_projection_windows={count}/{picked.size} frames={total}", flush=True)
    assert mean_sum is not None and cov_sum is not None
    mean = mean_sum / total
    covariance = cov_sum / total - np.outer(mean, mean)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][: args.output_dim]
    components = eigenvectors[:, order].T
    explained = eigenvalues[order] / max(1e-12, eigenvalues.sum())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        schema=np.asarray(PTM_PROJECTION_SCHEMA),
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        explained_variance_ratio=explained.astype(np.float32),
    )
    truncation_explained = float(
        np.sort(np.diag(covariance))[::-1][: args.output_dim].sum()
        / max(1e-12, np.trace(covariance))
    )
    summary = {
        "schema": PTM_PROJECTION_SCHEMA,
        "output": str(output),
        "input_dim": int(mean.shape[0]),
        "output_dim": int(args.output_dim),
        "sampled_windows": int(picked.size),
        "sampled_frames": int(total),
        "explained_variance_ratio_sum": round(float(explained.sum()), 6),
        "first128_truncation_variance_ratio_upper_bound": round(
            truncation_explained, 6
        ),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute variance-preserving PTM projection from a feature cache."
    )
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--max-windows", type=int, default=2048)
    parser.add_argument("--frames-per-window", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--log-every", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
