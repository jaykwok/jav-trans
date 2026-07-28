#!/usr/bin/env python3
"""Pack the per-example PTM/MFCC caches into one contiguous float16 memmap.

Training reads the same frames many times; the source caches are 165 GB of
per-example compressed npz spread over two directory layouts, so re-reading them
each epoch would make the experiment I/O-bound rather than GPU-bound.  This walks
them once, in the frame order `materialize_typed_span_frames.py` already fixed,
and writes `features.f16` so a training step is a slice.

PTM is truncated to the leading `--ptm-dim` columns.  For real omni windows that
is the whole stored array; for synthetic composites it selects the same leading
dims out of 2048.  Those are the same coordinates - the check that they are is in
the materializer's docstring - and this tool re-asserts the stored width per
example rather than assuming the manifest was right.

float16 is safe here: PTM activations sit around 1e-2 with |x| < 1, so the
representable resolution is ~1e-6, far below any signal these features carry.
MFCC is appended unscaled at the end of each row.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.gpu_safety import apply_host_memory_cap  # noqa: E402
from boundary.sequence_features import load_ptm_projection  # noqa: E402

OUTPUT_SCHEMA = "typed_span_features_v1"


def _rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def pack(
    *,
    index_path: Path,
    output: Path,
    ptm_dim: int,
    mfcc_dim: int,
    workers: int,
    projection_path: str = "",
) -> dict:
    index = list(_rows(index_path))
    if not index:
        raise SystemExit(f"empty index: {index_path}")

    # A projection replaces truncation: it consumes the full stored PTM and
    # emits its own width, so `--ptm-dim` no longer applies.
    projection = load_ptm_projection(projection_path) if projection_path else None
    if projection is not None:
        mean = projection["mean"].reshape(1, -1)
        components = projection["components"]
        expected_input = int(components.shape[1])
        ptm_dim = int(components.shape[0])
    else:
        mean = None
        components = None
        expected_input = 0

    total_frames = max(int(r["frame_offset"]) + int(r["frame_count"]) for r in index)
    width = ptm_dim + mfcc_dim
    output.mkdir(parents=True, exist_ok=True)
    memmap_path = output / "features.f16"

    features = np.memmap(
        memmap_path, dtype=np.float16, mode="w+", shape=(total_frames, width)
    )

    # Each worker holds a decompressed [T, 2048] float32 array while it slices,
    # so RAM scales with worker count, not with the 8 GB output. Checked per
    # example rather than once at startup for that reason.
    guard = apply_host_memory_cap()

    lock = threading.Lock()
    state = {"done": 0, "failed": 0, "ptm_narrow": 0, "short": 0}

    def load_one(row: dict) -> None:
        offset = int(row["frame_offset"])
        count = int(row["frame_count"])
        try:
            with np.load(Path(row["feature_path"])) as handle:
                ptm = handle["ptm"]
                mfcc = handle["mfcc"]
                stored = int(ptm.shape[1])
                # A projection needs the exact width it was fitted on; a
                # truncation only needs enough columns. Mismatch is recorded,
                # never silently zero-padded.
                if components is not None:
                    narrow = stored != expected_input
                else:
                    narrow = stored < ptm_dim
                if narrow or int(mfcc.shape[1]) < mfcc_dim:
                    with lock:
                        state["ptm_narrow"] += 1
                    return
                if int(ptm.shape[0]) < count or int(mfcc.shape[0]) < count:
                    with lock:
                        state["short"] += 1
                    return
                block = np.empty((count, width), dtype=np.float16)
                if components is not None:
                    projected = (
                        ptm[:count].astype(np.float32) - mean
                    ) @ components.T
                    block[:, :ptm_dim] = projected.astype(np.float16)
                else:
                    block[:, :ptm_dim] = ptm[:count, :ptm_dim].astype(np.float16)
                block[:, ptm_dim:] = mfcc[:count, :mfcc_dim].astype(np.float16)
            features[offset : offset + count] = block
        except Exception:
            with lock:
                state["failed"] += 1
            return
        with lock:
            state["done"] += 1
            if state["done"] % 2000 == 0:
                print(f"  packed {state['done']}/{len(index)}", flush=True)
                guard.check()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(load_one, index))

    features.flush()
    del features
    guard.stop()

    summary = {
        "schema": OUTPUT_SCHEMA,
        "index": str(index_path),
        "features": str(memmap_path),
        "dtype": "float16",
        "total_frames": int(total_frames),
        "width": int(width),
        "ptm_dim": int(ptm_dim),
        "mfcc_dim": int(mfcc_dim),
        "examples": len(index),
        "packed": state["done"],
        "failed": state["failed"],
        "ptm_too_narrow": state["ptm_narrow"],
        "too_few_frames": state["short"],
        "bytes": int(total_frames) * int(width) * 2,
        "host_memory_ratio": guard.ratio,
        "host_peak_gib": round(guard.peak_bytes / 2**30, 3),
        "basis": "learned_projection" if projection is not None else "truncation",
        "projection": str(projection_path) if projection is not None else "",
        "projection_digest": projection["digest"] if projection is not None else "",
        "projection_input_dim": expected_input,
    }
    with (output / "features_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="index.jsonl from the materializer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ptm-dim", type=int, default=128)
    parser.add_argument("--mfcc-dim", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--projection",
        default="",
        help=(
            "Optional learned PTM projection npz. Consumes the full stored PTM "
            "and overrides --ptm-dim with the projection's output width."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    pack(
        index_path=Path(args.index).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        ptm_dim=max(1, int(args.ptm_dim)),
        mfcc_dim=max(1, int(args.mfcc_dim)),
        workers=max(1, int(args.workers)),
        projection_path=args.projection,
    )
