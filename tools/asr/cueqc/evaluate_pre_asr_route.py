#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from asr.pre_asr_cueqc import load_checkpoint  # noqa: E402
from tools.asr.cueqc.compile_pre_asr_v11_features import (  # noqa: E402
    candidate_for_chunk,
    read_chunk_document,
)


def _parse_range(raw: str) -> tuple[float, float]:
    try:
        left, right = (float(value) for value in raw.split(":", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("range must be START:END") from exc
    if right <= left:
        raise argparse.ArgumentTypeError("range END must be greater than START")
    return left, right


def evaluate_route(
    *,
    candidates_path: Path,
    checkpoint_path: Path,
    device: str,
    threshold: float | None,
    inspect_ranges: list[tuple[float, float]],
) -> tuple[list[dict], dict]:
    _audio_id, chunks = read_chunk_document(candidates_path)
    candidates = [candidate_for_chunk(chunks, index) for index in range(len(chunks))]
    model = load_checkpoint(checkpoint_path, device=device)
    if threshold is not None:
        model.drop_threshold = float(threshold)
    decisions = model.decide(candidates)
    rows = [
        {
            "index": int(candidate["index"]),
            "sample_id": str(candidate["sample_id"]),
            "start": float(chunk["start"]),
            "end": float(chunk["end"]),
            "duration_s": float(candidate["duration_s"]),
            "route": str(decision["route"]),
            "prob_drop": float(decision["prob_drop"]),
            "prob_keep": float(decision["prob_keep"]),
        }
        for chunk, candidate, decision in zip(chunks, candidates, decisions)
    ]
    kept = [row for row in rows if row["route"] == "keep_for_asr"]
    durations = np.asarray(
        sorted(row["duration_s"] for row in kept),
        dtype=np.float64,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "threshold": float(model.drop_threshold),
        "candidates": len(rows),
        "keep": len(kept),
        "drop": len(rows) - len(kept),
        "kept_duration_p99_s": (
            float(np.quantile(durations, 0.99)) if durations.size else 0.0
        ),
        "kept_duration_max_s": float(durations[-1]) if durations.size else 0.0,
        "kept_over_20s": [row for row in kept if row["duration_s"] > 20.0],
        "inspected_ranges": [
            {
                "range": [left, right],
                "rows": [
                    row
                    for row in rows
                    if row["end"] > left and row["start"] < right
                ],
            }
            for left, right in inspect_ranges
        ],
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Pre-ASR v11 checkpoint on exported runtime candidates."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--inspect-range", action="append", type=_parse_range, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    rows, summary = evaluate_route(
        candidates_path=args.candidates,
        checkpoint_path=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        inspect_ranges=args.inspect_range,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary["output"] = str(args.output)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
