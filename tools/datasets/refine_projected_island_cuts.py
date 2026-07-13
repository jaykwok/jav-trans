#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import (  # noqa: E402
    DEFAULT_CUT_EDGE_REFINER_CHECKPOINT_BY_REPO,
    QWEN_ASR_17B_REPO_ID,
)
from boundary.cut_refiner import load_cut_edge_refiner  # noqa: E402


SCHEMA = "semantic_split_v3_island_cut_refined_label_v1"
PROMPT_VERSION = "semantic_split_v3_candidate_projection_cut_refiner_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _feature_lookup_time_s(
    cut: dict[str, Any], *, span_start_s: float, audit_rows: list[dict[str, Any]]
) -> float:
    absolute_time = float(span_start_s) + float(cut["time_s"])
    if str(cut.get("candidate_kind") or "") != "primary":
        return absolute_time
    primary_rows = [
        row
        for row in audit_rows
        if str(row.get("kind") or "") == "primary"
        and row.get("proposal_time_s") is not None
    ]
    if not primary_rows:
        return absolute_time
    primary = min(
        primary_rows, key=lambda row: abs(float(row["time_s"]) - absolute_time)
    )
    if abs(float(primary["time_s"]) - absolute_time) > 0.02:
        return absolute_time
    return float(primary["proposal_time_s"])


def run(
    *,
    reexport_dir: Path,
    projected_labels: Path,
    output_dir: Path,
    checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    import torch

    if torch.cuda.is_available() and str(device).lower() in {"auto", "cuda", "cuda:0"}:
        torch.cuda.set_per_process_memory_fraction(0.95, 0)
    sources = {
        str(row["window_id"]): row
        for row in _read_jsonl(reexport_dir / "source_windows.jsonl")
    }
    refiner = load_cut_edge_refiner(
        checkpoint,
        device=device,
        expected_ptm_repo_id=QWEN_ASR_17B_REPO_ID,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    conflicts = []
    all_deltas = []
    for label in _read_jsonl(projected_labels):
        source = sources[str(label["window_id"])]
        audit_rows = _read_jsonl(Path(source["boundary_audit"]))
        bundle = np.load(Path(source["semantic_split_features"]), allow_pickle=False)
        proposal_times = np.asarray(bundle["proposal_times_s"], dtype=np.float32)
        frame_features = np.asarray(bundle["frame_features"], dtype=np.float32)
        scalar_features = np.asarray(bundle["scalar_features"], dtype=np.float32)
        core_starts = np.asarray(bundle["core_starts_s"], dtype=np.float32)
        core_ends = np.asarray(bundle["core_ends_s"], dtype=np.float32)
        selected_indexes = []
        input_times = []
        source_cuts = []
        for cut in label.get("cuts") or []:
            absolute_time = float(label["span_start_s"]) + float(cut["time_s"])
            feature_time = _feature_lookup_time_s(
                cut,
                span_start_s=float(label["span_start_s"]),
                audit_rows=audit_rows,
            )
            index = int(np.argmin(np.abs(proposal_times - feature_time)))
            lookup_delta = float(proposal_times[index]) - feature_time
            if abs(lookup_delta) > 0.02:
                conflicts.append(
                    {
                        "island_id": label["island_id"],
                        "time_s": cut["time_s"],
                        "reason": "feature_row_not_found",
                        "lookup_delta_s": lookup_delta,
                    }
                )
                continue
            selected_indexes.append(index)
            input_times.append(float(proposal_times[index]))
            source_cuts.append(cut)
        if selected_indexes:
            indexes = np.asarray(selected_indexes, dtype=np.int64)
            refined = refiner.refine(
                proposal_times_s=np.asarray(input_times, dtype=np.float32),
                frame_features=frame_features[indexes],
                scalar_features=scalar_features[indexes],
                core_start_s=core_starts[indexes],
                core_end_s=core_ends[indexes],
            )
        else:
            refined = np.asarray([], dtype=np.float32)
        cuts = []
        previous = 0.0
        for source_cut, input_time, refined_absolute in zip(
            source_cuts, input_times, refined
        ):
            refined_relative = float(refined_absolute) - float(label["span_start_s"])
            if not previous < refined_relative < float(label["duration_s"]):
                conflicts.append(
                    {
                        "island_id": label["island_id"],
                        "time_s": refined_relative,
                        "reason": "refined_order_or_bounds",
                    }
                )
                continue
            delta = float(refined_absolute) - float(input_time)
            all_deltas.append(abs(delta))
            cuts.append(
                {
                    **source_cut,
                    "time_s": refined_relative,
                    "projected_candidate_time_s": float(source_cut["time_s"]),
                    "cut_refiner_delta_s": delta,
                    "cut_refiner_sha256": refiner.sha256,
                }
            )
            previous = refined_relative
        output_rows.append(
            {
                **label,
                "schema": SCHEMA,
                "prompt_version": PROMPT_VERSION,
                "cuts": cuts,
                "complete_search": len(cuts) == len(label.get("cuts") or []),
                "cut_refiner_path": refiner.path,
                "cut_refiner_sha256": refiner.sha256,
                "reason": "Projected candidates refined to shared absolute timestamps by active Cut Refiner.",
            }
        )
    output = output_dir / "refined_island_labels.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    summary = {
        "schema": "semantic_split_v3_island_cut_refinement_summary_v1",
        "island_count": len(output_rows),
        "input_cut_count": sum(len(row.get("cuts") or []) for row in _read_jsonl(projected_labels)),
        "refined_cut_count": sum(len(row["cuts"]) for row in output_rows),
        "conflict_count": len(conflicts),
        "mean_abs_refiner_delta_s": sum(all_deltas) / len(all_deltas) if all_deltas else 0.0,
        "max_abs_refiner_delta_s": max(all_deltas, default=0.0),
        "cut_refiner_sha256": refiner.sha256,
        "vram_safety_ratio": 0.95,
        "output": str(output),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "conflicts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in conflicts),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reexport-dir", required=True)
    parser.add_argument("--projected-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CUT_EDGE_REFINER_CHECKPOINT_BY_REPO[QWEN_ASR_17B_REPO_ID],
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                reexport_dir=Path(args.reexport_dir),
                projected_labels=Path(args.projected_labels),
                output_dir=Path(args.output_dir),
                checkpoint=Path(args.checkpoint),
                device=args.device,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
