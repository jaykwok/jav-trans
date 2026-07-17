#!/usr/bin/env python3
"""Evaluate Split v4 predictions on manual cut/continue/ignore overrides."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.split_model import load_acoustic_split_v4_planner  # noqa: E402
from tools.boundary.ja.train_acoustic_split_v4_model import _pad_batch  # noqa: E402
from tools.boundary.ja.train_semantic_split_island_model import load_island_dataset  # noqa: E402


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def evaluate(args: argparse.Namespace) -> dict:
    import torch

    planner = load_acoustic_split_v4_planner(args.checkpoint, device=args.device)
    data = load_island_dataset(Path(args.dataset))
    metadata = _rows(Path(args.metadata))
    overrides = _rows(Path(args.overrides))
    keys = {(str(row["audio_id"]), round(float(row["time_s"]), 6)): row for row in overrides}
    indexes = [
        index for index, row in enumerate(metadata)
        if (str(row["audio_id"]), round(float(row["time_s"]), 6)) in keys
    ]
    groups = sorted({name for name, rows in data["groups"].items() if any(int(i) in indexes for i in rows)})
    norm = {key: np.asarray(value, dtype=np.float32) for key, value in planner.normalization.items()}
    output_rows: list[dict] = []
    with torch.inference_mode():
        frames, scalars, mask, *_ = _pad_batch(
            data, groups,
            frame_mean=norm["frame_mean"], frame_std=norm["frame_std"],
            scalar_mean=norm["scalar_mean"], scalar_std=norm["scalar_std"],
        )
        probs = torch.softmax(planner.model(frames.to(planner.device), scalars.to(planner.device), mask.to(planner.device))["label"], dim=-1).cpu().numpy()
    for group_index, group_name in enumerate(groups):
        for position, index in enumerate(data["groups"][group_name].tolist()):
            meta = metadata[index]
            key = (str(meta["audio_id"]), round(float(meta["time_s"]), 6))
            if key not in keys:
                continue
            override = keys[key]
            predicted = "cut" if int(np.argmax(probs[group_index, position])) == 0 else "continue"
            output_rows.append({
                **override,
                "prediction": predicted,
                "p_cut": float(probs[group_index, position, 0]),
                "p_continue": float(probs[group_index, position, 1]),
                "gate_match": (
                    None if override["training_label"] == "ignore"
                    else predicted == override["training_label"]
                ),
            })
    definite = [row for row in output_rows if row["training_label"] in {"cut", "continue"}]
    payload = {
        "schema": "split_v4_manual_override_evaluation_v1",
        "checkpoint": planner.signature(),
        "row_count": len(output_rows),
        "definite_count": len(definite),
        "definite_correct": sum(row["gate_match"] is True for row in definite),
        "definite_errors": [row for row in definite if row["gate_match"] is False],
        "gate_passed": bool(definite) and all(row["gate_match"] is True for row in definite),
        "rows": output_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--overrides", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    result = evaluate(parse_args())
    print(json.dumps({key: result[key] for key in ("row_count", "definite_count", "definite_correct", "gate_passed")}, ensure_ascii=False))
