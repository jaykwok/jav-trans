#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SRC_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from tools.boundary.ja.evaluate_semantic_split_island import (  # noqa: E402
    _dataset_names,
    _load_model,
)
from tools.boundary.ja.train_semantic_split_island_model import (  # noqa: E402
    CORE_DURATION_SCALAR_INDEX,
    evaluate_island_model,
    load_island_dataset,
)

SCHEMA = "joint_split_cueqc_calibration_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _thresholds(raw: str) -> list[float]:
    values = sorted({round(float(value.strip()), 6) for value in raw.split(",") if value.strip()})
    if not values or any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("threshold grids must contain values in (0, 1)")
    return values


def _partition(video_id: str, val_percent: int) -> str:
    bucket = int(hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "val" if bucket < val_percent else "train"


def _selected_rows(data: Mapping[str, Any], names: Sequence[str]) -> list[int]:
    return [
        int(index)
        for name in names
        for index in data["groups"][name].tolist()
    ]


def split_metrics(
    data: Mapping[str, Any],
    names: Sequence[str],
    gate_by_row: Mapping[int, float],
    *,
    normal_threshold: float,
    short_threshold: float,
    short_core_max_s: float,
    baseline_accept: Mapping[int, bool] | None = None,
) -> dict[str, Any]:
    tp = fp = fn = continue_total = 0
    pair_hits: dict[int, list[bool]] = {}
    accepted: dict[int, bool] = {}
    rows = _selected_rows(data, names)
    for name in names:
        indexes = data["groups"][name]
        core_duration = float(data["scalars"][indexes[0], CORE_DURATION_SCALAR_INDEX])
        threshold = short_threshold if core_duration <= short_core_max_s else normal_threshold
        for raw_index in indexes.tolist():
            index = int(raw_index)
            is_cut = float(gate_by_row[index]) >= threshold
            accepted[index] = is_cut
            label = int(data["labels"][index])
            if label == 0:
                if is_cut:
                    tp += 1
                else:
                    fn += 1
                pair_id = int(data["pair_ids"][index])
                if pair_id >= 0:
                    pair_hits.setdefault(pair_id, []).append(is_cut)
            elif label == 1:
                continue_total += 1
                if is_cut:
                    fp += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    complete_pairs = [hits for hits in pair_hits.values() if len(hits) >= 2]
    changed = (
        sum(accepted[index] != bool(baseline_accept[index]) for index in rows)
        if baseline_accept is not None
        else 0
    )
    return {
        "normal_threshold": normal_threshold,
        "short_threshold": short_threshold,
        "candidate_count": len(rows),
        "accepted_count": sum(accepted.values()),
        "cut_precision": precision,
        "cut_recall": recall,
        "cut_f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        "continue_false_cut": fp / max(1, continue_total),
        "pair_isolation_rate": (
            sum(all(hits[:2]) for hits in complete_pairs) / max(1, len(complete_pairs))
        ),
        "complete_pair_count": len(complete_pairs),
        "decision_change_count": changed,
        "decision_change_ratio": changed / max(1, len(rows)),
        "_accepted": accepted,
    }


def cueqc_metrics(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    drop_total = keep_total = tp = fp = fn = 0
    drop_duration = keep_duration = false_drop_duration = false_keep_duration = 0.0
    for row in rows:
        truth = str(row["truth"])
        duration = float(row.get("duration_s") or 0.0)
        predicted_drop = float(row["v12_prob_drop"]) >= threshold
        if truth == "drop":
            drop_total += 1
            drop_duration += duration
            if predicted_drop:
                tp += 1
            else:
                fn += 1
                false_keep_duration += duration
        elif truth == "keep":
            keep_total += 1
            keep_duration += duration
            if predicted_drop:
                fp += 1
                false_drop_duration += duration
        else:
            raise ValueError(f"unexpected CueQC truth: {truth!r}")
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, drop_total)
    keep_recall = 1.0 - fp / max(1, keep_total)
    return {
        "threshold": threshold,
        "candidate_count": len(rows),
        "drop_count": drop_total,
        "keep_count": keep_total,
        "drop_precision": precision,
        "drop_recall": recall,
        "drop_f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        "semantic_keep_recall": keep_recall,
        "false_drop_count": fp,
        "false_keep_count": fn,
        "false_drop_duration_s": false_drop_duration,
        "false_keep_duration_s": false_keep_duration,
        "speech_loss_duration_ratio": false_drop_duration / max(1e-12, keep_duration),
        "noise_residual_duration_ratio": false_keep_duration / max(1e-12, drop_duration),
    }


def select_operating_point(
    split_grid: Sequence[Mapping[str, Any]],
    cueqc_grid: Sequence[Mapping[str, Any]],
    *,
    baseline_split: Mapping[str, Any],
    baseline_cueqc: Mapping[str, Any],
    min_drop_recall: float,
    allow_lower_cueqc_threshold: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_eligible = [
        row
        for row in split_grid
        if float(row["cut_recall"]) + 1e-12 >= float(baseline_split["cut_recall"])
        and float(row["continue_false_cut"]) <= float(baseline_split["continue_false_cut"]) + 1e-12
    ]
    cueqc_eligible = [
        row
        for row in cueqc_grid
        if float(row["drop_recall"]) + 1e-12 >= min_drop_recall
        and float(row["semantic_keep_recall"]) + 1e-12
        >= float(baseline_cueqc["semantic_keep_recall"])
        and (
            allow_lower_cueqc_threshold
            or float(row["threshold"]) + 1e-12 >= float(baseline_cueqc["threshold"])
        )
    ]
    if not split_eligible:
        raise ValueError("no Split operating point preserves baseline recall and false-cut rate")
    if not cueqc_eligible:
        raise ValueError("no CueQC operating point satisfies drop and keep recall gates")

    joint_rows: list[dict[str, Any]] = []
    baseline_normal = float(baseline_split["normal_threshold"])
    baseline_short = float(baseline_split["short_threshold"])
    baseline_drop = float(baseline_cueqc["threshold"])
    for split_row, cueqc_row in product(split_eligible, cueqc_eligible):
        joint_rows.append(
            {
                "normal_threshold": float(split_row["normal_threshold"]),
                "short_threshold": float(split_row["short_threshold"]),
                "cueqc_drop_threshold": float(cueqc_row["threshold"]),
                "split_cut_precision": float(split_row["cut_precision"]),
                "split_cut_recall": float(split_row["cut_recall"]),
                "split_cut_f1": float(split_row["cut_f1"]),
                "split_continue_false_cut": float(split_row["continue_false_cut"]),
                "split_decision_change_count": int(split_row["decision_change_count"]),
                "split_decision_change_ratio": float(split_row["decision_change_ratio"]),
                "cueqc_drop_recall": float(cueqc_row["drop_recall"]),
                "cueqc_keep_recall": float(cueqc_row["semantic_keep_recall"]),
                "speech_loss_duration_ratio": float(cueqc_row["speech_loss_duration_ratio"]),
                "noise_residual_duration_ratio": float(cueqc_row["noise_residual_duration_ratio"]),
                "false_drop_count": int(cueqc_row["false_drop_count"]),
                "false_keep_count": int(cueqc_row["false_keep_count"]),
                "distance_from_baseline": (
                    abs(float(split_row["normal_threshold"]) - baseline_normal)
                    + abs(float(split_row["short_threshold"]) - baseline_short)
                    + abs(float(cueqc_row["threshold"]) - baseline_drop)
                ),
            }
        )
    selected = min(
        joint_rows,
        key=lambda row: (
            row["speech_loss_duration_ratio"],
            row["noise_residual_duration_ratio"],
            -row["split_cut_f1"],
            row["split_decision_change_ratio"],
            row["distance_from_baseline"],
        ),
    )
    return selected, joint_rows


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_cueqc_gate(
    summary_path: Path, paired_decisions_path: Path
) -> tuple[dict[str, Any], float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("schema") or "") != "pre_asr_cueqc_v12_gate_summary_v1":
        raise ValueError("CueQC model gate summary schema mismatch")
    expected_paired = _resolve(str(dict(summary.get("outputs") or {}).get("paired_decisions") or ""))
    if expected_paired.resolve() != paired_decisions_path.resolve():
        raise ValueError(
            f"paired decisions do not belong to the model gate: "
            f"{paired_decisions_path} != {expected_paired}"
        )
    threshold = float(summary["v12_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("CueQC gate v12_threshold must be in (0, 1)")
    return summary, threshold


def chain_tail_consistency(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    source_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    counts = Counter()
    role_counts = Counter()
    for row in rows:
        if float(row["v12_prob_drop"]) < threshold:
            continue
        counts["cueqc_drop_count"] += 1
        source = _resolve(str(row["source"]))
        if source not in source_cache:
            source_cache[source] = {
                str(candidate["candidate_id"]): candidate for candidate in _read_jsonl(source)
            }
        candidate = source_cache[source].get(str(row["id"]))
        if candidate is None:
            counts["missing_candidate_metadata"] += 1
            continue
        edges = dict(candidate.get("pre_asr_split_edges") or {})
        edge_rows = [dict(edges.get(side) or {}) for side in ("left", "right")]
        split_edges = [edge for edge in edge_rows if edge.get("kind") == "split_cut"]
        bracketed = [edge for edge in split_edges if edge.get("noise_isolation_bracket")]
        counts["drop_with_any_split_edge"] += int(bool(split_edges))
        counts["drop_without_split_edge"] += int(not split_edges)
        counts["drop_with_noise_isolation_bracket"] += int(bool(bracketed))
        for edge in split_edges:
            role_counts[str(edge.get("role") or "unknown")] += 1
    return {
        "schema": "split_cueqc_chain_tail_consistency_v1",
        "policy": "report_only",
        "counts": dict(counts),
        "split_edge_roles": dict(role_counts),
    }


def _v3_val_windows(source_windows_path: Path, val_percent: int) -> set[str]:
    return {
        str(row["window_id"])
        for row in _read_jsonl(source_windows_path)
        if _partition(str(row["video_id"]), val_percent) == "val"
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vram_safety_ratio = apply_vram_safety_cap()
    model, normalization, split_decision = _load_model(
        Path(args.split_checkpoint), args.device
    )
    data = load_island_dataset(Path(args.split_dataset))
    names = _dataset_names(data, args.split_partition)
    baseline_normal = float(split_decision["normal_cut_threshold"])
    baseline_short = float(split_decision["short_core_cut_threshold"])
    short_core_max_s = float(split_decision["short_core_max_s"])
    inference = evaluate_island_model(
        model,
        data,
        names,
        normalization=normalization,
        device=args.device,
        batch_islands=args.batch_islands,
        max_batch_candidates=args.max_batch_candidates,
        normal_cut_threshold=baseline_normal,
        short_core_cut_threshold=baseline_short,
        short_core_max_s=short_core_max_s,
    )
    gate_by_row = inference["gate_by_row"]
    baseline_split = split_metrics(
        data,
        names,
        gate_by_row,
        normal_threshold=baseline_normal,
        short_threshold=baseline_short,
        short_core_max_s=short_core_max_s,
    )
    baseline_accept = baseline_split.pop("_accepted")
    split_grid = []
    for normal, short in product(
        _thresholds(args.normal_thresholds), _thresholds(args.short_thresholds)
    ):
        metrics = split_metrics(
            data,
            names,
            gate_by_row,
            normal_threshold=normal,
            short_threshold=short,
            short_core_max_s=short_core_max_s,
            baseline_accept=baseline_accept,
        )
        metrics.pop("_accepted")
        split_grid.append(metrics)

    cueqc_gate, baseline_drop = load_cueqc_gate(
        Path(args.cueqc_model_gate_summary), Path(args.cueqc_paired_decisions)
    )
    val_windows = _v3_val_windows(Path(args.source_windows), args.val_percent)
    cueqc_rows = [
        row
        for row in _read_jsonl(Path(args.cueqc_paired_decisions))
        if str(row["audio_id"]) in val_windows
    ]
    if not cueqc_rows:
        raise ValueError("no CueQC paired decisions matched the v3 validation windows")
    baseline_cueqc = cueqc_metrics(cueqc_rows, baseline_drop)
    cueqc_grid = [
        cueqc_metrics(cueqc_rows, threshold)
        for threshold in _thresholds(args.cueqc_thresholds)
    ]
    selected, joint_grid = select_operating_point(
        split_grid,
        cueqc_grid,
        baseline_split=baseline_split,
        baseline_cueqc=baseline_cueqc,
        min_drop_recall=args.min_drop_recall,
        allow_lower_cueqc_threshold=args.allow_lower_cueqc_threshold,
    )
    requires_de_prime = (
        float(selected["split_decision_change_ratio"]) > args.max_split_churn_ratio
    )
    consistency = chain_tail_consistency(
        cueqc_rows, float(selected["cueqc_drop_threshold"])
    )
    requires_cueqc_reaudit = (
        float(selected["cueqc_drop_threshold"]) + 1e-12 < baseline_drop
    )
    recommendation = {
        "status": (
            "requires_de_prime"
            if requires_de_prime
            else "requires_cueqc_reaudit"
            if requires_cueqc_reaudit
            else "ready_to_apply"
        ),
        "requires_de_prime": requires_de_prime,
        "requires_cueqc_reaudit": requires_cueqc_reaudit,
        "split_decision_config": {
            "normal_cut_threshold": selected["normal_threshold"],
            "short_core_cut_threshold": selected["short_threshold"],
        },
        "cueqc_decision_config": {
            "drop_threshold": selected["cueqc_drop_threshold"]
        },
        "weights_must_remain_unchanged": True,
    }
    summary = {
        "schema": SCHEMA,
        "split_checkpoint": args.split_checkpoint,
        "split_dataset": args.split_dataset,
        "split_partition": args.split_partition,
        "cueqc_checkpoint": cueqc_gate["v12_checkpoint"],
        "cueqc_model_gate_summary": args.cueqc_model_gate_summary,
        "cueqc_paired_decisions": args.cueqc_paired_decisions,
        "source_windows": args.source_windows,
        "v3_val_window_count": len(val_windows),
        "cueqc_val_candidate_count": len(cueqc_rows),
        "baseline": {"split": baseline_split, "cueqc": baseline_cueqc},
        "selected": selected,
        "gates": {
            "min_drop_recall": args.min_drop_recall,
            "min_keep_recall": baseline_cueqc["semantic_keep_recall"],
            "audited_min_cueqc_threshold": baseline_drop,
            "allow_lower_cueqc_threshold": args.allow_lower_cueqc_threshold,
            "min_split_cut_recall": baseline_split["cut_recall"],
            "max_split_continue_false_cut": baseline_split["continue_false_cut"],
            "max_split_churn_ratio_without_de_prime": args.max_split_churn_ratio,
        },
        "selection_policy": [
            "preserve_split_cut_recall_and_continue_false_cut",
            "preserve_cueqc_drop_gate_and_baseline_keep_recall",
            "do_not_expand_the_audited_drop_set_without_explicit_reaudit",
            "minimize_speech_loss_duration_ratio",
            "minimize_noise_residual_duration_ratio",
            "maximize_split_cut_f1",
            "minimize_split_churn_then_distance_from_baseline",
        ],
        "recommendation": recommendation,
        "chain_tail_consistency": consistency,
        "vram_safety_ratio": vram_safety_ratio,
        "shared_vram_budget": False,
        "outputs": {
            "split_grid": str(output_dir / "split_grid.jsonl"),
            "cueqc_grid": str(output_dir / "cueqc_grid.jsonl"),
            "joint_grid": str(output_dir / "joint_grid.jsonl"),
            "recommendation": str(output_dir / "recommended_decision_config.json"),
        },
    }
    _write_jsonl(output_dir / "split_grid.jsonl", split_grid)
    _write_jsonl(output_dir / "cueqc_grid.jsonl", cueqc_grid)
    _write_jsonl(output_dir / "joint_grid.jsonl", joint_grid)
    _write_json(output_dir / "recommended_decision_config.json", recommendation)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly calibrate Semantic Split and Pre-ASR CueQC thresholds."
    )
    parser.add_argument("--split-checkpoint", required=True)
    parser.add_argument("--split-dataset", required=True)
    parser.add_argument("--split-partition", choices=("val", "train", "all"), default="val")
    parser.add_argument("--cueqc-model-gate-summary", required=True)
    parser.add_argument("--cueqc-paired-decisions", required=True)
    parser.add_argument("--source-windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--normal-thresholds", default="0.70,0.725,0.75,0.775,0.80")
    parser.add_argument("--short-thresholds", default="0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--cueqc-thresholds", default="0.45,0.475,0.50,0.525,0.55,0.575,0.60")
    parser.add_argument("--min-drop-recall", type=float, default=0.98)
    parser.add_argument(
        "--allow-lower-cueqc-threshold",
        action="store_true",
        help="Permit expanding the audited drop set; recommendation then requires a new audit.",
    )
    parser.add_argument("--max-split-churn-ratio", type=float, default=0.005)
    parser.add_argument("--val-percent", type=int, default=20)
    parser.add_argument("--batch-islands", type=int, default=4)
    parser.add_argument("--max-batch-candidates", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if not 0.0 < args.min_drop_recall <= 1.0:
        parser.error("--min-drop-recall must be in (0, 1]")
    if not 0.0 <= args.max_split_churn_ratio <= 1.0:
        parser.error("--max-split-churn-ratio must be in [0, 1]")
    if not 1 <= args.val_percent <= 50:
        parser.error("--val-percent must be in [1, 50]")
    return args


if __name__ == "__main__":
    run(parse_args())
