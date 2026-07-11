#!/usr/bin/env python3
"""Diagnose padding, temporal-context, and local-representation errors in v12."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.pre_asr_cueqc import PreAsrCueQCNetwork, make_model_config  # noqa: E402
from tools.asr.cueqc.pre_asr_feature_compiler import project_path  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(dict(row))
    return rows


def classify_error_cause(
    *,
    padded_prob_drop: float,
    trimmed_prob_drop: float,
    local_prob_drop: float,
    threshold: float,
) -> str:
    if padded_prob_drop >= threshold and trimmed_prob_drop < threshold:
        return "padding_sensitive_flip"
    if trimmed_prob_drop >= threshold and local_prob_drop < threshold:
        return "temporal_context_flip"
    if local_prob_drop >= threshold:
        return "local_representation_error"
    return "not_reproduced_as_trimmed_error"


def _load_model_inputs(bundle: Mapping[str, Any], checkpoint: Mapping[str, Any], device: str):
    import torch

    feature_names = tuple(str(item) for item in checkpoint.get("feature_names") or ())
    source_names = tuple(str(item) for item in bundle.get("feature_names") or ())
    index_by_name = {name: index for index, name in enumerate(source_names)}
    indexes = [index_by_name[name] for name in feature_names]
    scalar = bundle["scalar_features"][:, :, indexes].float()
    mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32)
    std = torch.as_tensor(checkpoint["feature_std"], dtype=torch.float32).clamp_min(1e-6)
    scalar = torch.nan_to_num(
        (scalar - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    dev = torch.device(device)
    model = PreAsrCueQCNetwork(**make_model_config(checkpoint.get("model_config")))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(dev).eval()
    return (
        model,
        bundle["ptm_bins"].float().to(dev),
        scalar.to(dev),
        bundle["chunk_mask"].float().to(dev),
        bundle["bin_mask"].float().to(dev),
    )


def _trimmed_probs(model, ptm, scalar, chunk_mask, bin_mask) -> np.ndarray:
    import torch

    output = torch.full(
        (*chunk_mask.shape, 2),
        float("nan"),
        dtype=torch.float32,
        device=ptm.device,
    )
    groups_by_length: dict[int, list[int]] = defaultdict(list)
    for group_index in range(int(chunk_mask.shape[0])):
        groups_by_length[int(chunk_mask[group_index].sum().detach().cpu().item())].append(group_index)
    with torch.inference_mode():
        for length, group_indexes in sorted(groups_by_length.items()):
            if length <= 0:
                continue
            indexes = torch.as_tensor(group_indexes, dtype=torch.long, device=ptm.device)
            logits = model(
                ptm[indexes, :length],
                scalar[indexes, :length],
                chunk_mask=chunk_mask[indexes, :length],
                bin_mask=bin_mask[indexes, :length],
            )
            output[indexes, :length] = torch.softmax(logits, dim=-1)
    return output.float().cpu().numpy()


def _single_candidate_probs(model, ptm, scalar, chunk_mask, bin_mask, positions):
    import torch

    ptm_rows = torch.stack([ptm[group, index] for group, index in positions], dim=0).unsqueeze(1)
    scalar_rows = torch.stack([scalar[group, index] for group, index in positions], dim=0).unsqueeze(1)
    chunk_rows = torch.ones((len(positions), 1), dtype=chunk_mask.dtype, device=ptm.device)
    bin_rows = torch.stack([bin_mask[group, index] for group, index in positions], dim=0).unsqueeze(1)
    scale = float(model.temporal_residual_scale)
    with torch.inference_mode():
        model.temporal_residual_scale = 0.0
        local = torch.softmax(model(ptm_rows, scalar_rows, chunk_rows, bin_rows), dim=-1)[:, 0, 0]
        ptm_only = torch.softmax(
            model(ptm_rows, torch.zeros_like(scalar_rows), chunk_rows, bin_rows), dim=-1
        )[:, 0, 0]
        scalar_only = torch.softmax(
            model(torch.zeros_like(ptm_rows), scalar_rows, chunk_rows, bin_rows), dim=-1
        )[:, 0, 0]
        model.temporal_residual_scale = scale
        single_temporal = torch.softmax(
            model(ptm_rows, scalar_rows, chunk_rows, bin_rows), dim=-1
        )[:, 0, 0]
    return {
        "local": local.float().cpu().numpy(),
        "ptm_only_local": ptm_only.float().cpu().numpy(),
        "scalar_only_local": scalar_only.float().cpu().numpy(),
        "single_temporal": single_temporal.float().cpu().numpy(),
    }


def _ptm_summary(ptm_bins: np.ndarray) -> np.ndarray:
    local = ptm_bins[:, 2:]
    return np.concatenate(
        [ptm_bins[:, 0], ptm_bins[:, 1], local.mean(axis=1), local.max(axis=1)],
        axis=1,
    )


def _linear_probes(
    *,
    bundle: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    holdout_audio_ids: set[str],
    holdout_candidate_ids: list[str],
    position_by_id: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = bundle["labels"].detach().cpu().numpy()
    chunk_mask = bundle["chunk_mask"].detach().cpu().numpy() > 0
    scalar = bundle["scalar_features"].detach().cpu().numpy()
    ptm = bundle["ptm_bins"].detach().cpu().numpy()
    group_audio_ids = [str(group.get("audio_id") or "") for group in bundle.get("groups") or ()]
    train_positions = [
        (group, index)
        for group in range(labels.shape[0])
        if group_audio_ids[group] not in holdout_audio_ids
        for index in range(labels.shape[1])
        if chunk_mask[group, index] and labels[group, index] in (0, 1)
    ]
    holdout_positions = [position_by_id[candidate_id] for candidate_id in holdout_candidate_ids]
    y_train = np.asarray([labels[group, index] for group, index in train_positions], dtype=np.int64)
    y_holdout = np.asarray([labels[group, index] for group, index in holdout_positions], dtype=np.int64)
    scalar_train = np.stack([scalar[group, index] for group, index in train_positions])
    scalar_holdout = np.stack([scalar[group, index] for group, index in holdout_positions])
    ptm_train = _ptm_summary(np.stack([ptm[group, index] for group, index in train_positions]))
    ptm_holdout = _ptm_summary(np.stack([ptm[group, index] for group, index in holdout_positions]))

    outputs: dict[str, Any] = {}
    for name, train_x, holdout_x in (
        ("scalar", scalar_train, scalar_holdout),
        ("ptm_summary", ptm_train, ptm_holdout),
        ("combined", np.concatenate([scalar_train, ptm_train], axis=1), np.concatenate([scalar_holdout, ptm_holdout], axis=1)),
    ):
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=17),
        )
        probe.fit(train_x, y_train)
        probabilities = probe.predict_proba(holdout_x)
        drop_column = list(probe[-1].classes_).index(0)
        p_drop = probabilities[:, drop_column]
        prediction = np.where(p_drop >= 0.5, 0, 1)
        outputs[name] = {
            "holdout_accuracy": float(np.mean(prediction == y_holdout)),
            "holdout_keep_false_drop_count": int(np.sum((y_holdout == 1) & (prediction == 0))),
            "holdout_drop_false_keep_count": int(np.sum((y_holdout == 0) & (prediction == 1))),
            "candidate_prob_drop": {
                candidate_id: float(probability)
                for candidate_id, probability in zip(holdout_candidate_ids, p_drop, strict=True)
            },
        }
    return outputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    bundle = torch.load(project_path(args.features), map_location="cpu", weights_only=False)
    checkpoint = torch.load(project_path(args.checkpoint), map_location="cpu", weights_only=False)
    paired_rows = _read_jsonl(project_path(args.paired_decisions))
    manual_rows = _read_jsonl(project_path(args.manual_labels))
    holdout_candidate_ids = [
        line.strip()
        for line in project_path(args.holdout_candidate_ids).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    holdout_audio_ids = {
        line.strip()
        for line in project_path(args.holdout_audio_ids).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    position_by_id = {
        str(row_id): (group_index, chunk_index)
        for group_index, group in enumerate(bundle.get("groups") or ())
        for chunk_index, row_id in enumerate(group.get("row_ids") or ())
    }
    paired_by_id = {str(row["id"]): row for row in paired_rows}
    manual_by_id = {str(row["candidate_id"]): row for row in manual_rows}

    model, ptm, scalar, chunk_mask, bin_mask = _load_model_inputs(
        bundle, checkpoint, args.device
    )
    trimmed = _trimmed_probs(model, ptm, scalar, chunk_mask, bin_mask)
    positions = [position_by_id[candidate_id] for candidate_id in holdout_candidate_ids]
    single = _single_candidate_probs(model, ptm, scalar, chunk_mask, bin_mask, positions)

    padding_deltas: list[float] = []
    padding_flips: list[str] = []
    for candidate_id, paired in paired_by_id.items():
        group, index = position_by_id[candidate_id]
        padded = float(paired["v12_prob_drop"])
        unpadded = float(trimmed[group, index, 0])
        padding_deltas.append(abs(padded - unpadded))
        if (padded >= args.threshold) != (unpadded >= args.threshold):
            padding_flips.append(candidate_id)

    holdout_rows: list[dict[str, Any]] = []
    for offset, candidate_id in enumerate(holdout_candidate_ids):
        group, index = position_by_id[candidate_id]
        padded = float(paired_by_id[candidate_id]["v12_prob_drop"])
        unpadded = float(trimmed[group, index, 0])
        local = float(single["local"][offset])
        truth = "keep" if manual_by_id[candidate_id]["label"] == "definite_keep" else "drop"
        row = {
            "candidate_id": candidate_id,
            "truth": truth,
            "duration_s": float(paired_by_id[candidate_id]["duration_s"]),
            "padded_prob_drop": padded,
            "trimmed_prob_drop": unpadded,
            "local_prob_drop": local,
            "single_temporal_prob_drop": float(single["single_temporal"][offset]),
            "ptm_only_local_prob_drop": float(single["ptm_only_local"][offset]),
            "scalar_only_local_prob_drop": float(single["scalar_only_local"][offset]),
        }
        if truth == "keep" and padded >= args.threshold:
            row["error_cause"] = classify_error_cause(
                padded_prob_drop=padded,
                trimmed_prob_drop=unpadded,
                local_prob_drop=local,
                threshold=float(args.threshold),
            )
        holdout_rows.append(row)

    delta_array = np.asarray(padding_deltas, dtype=np.float64)
    summary = {
        "schema": "pre_asr_v12_representation_diagnostic_v1",
        "features": str(project_path(args.features)),
        "checkpoint": str(project_path(args.checkpoint)),
        "threshold": float(args.threshold),
        "device": str(args.device),
        "padding_stability": {
            "candidate_count": len(padding_deltas),
            "absolute_delta_mean": float(delta_array.mean()),
            "absolute_delta_p95": float(np.quantile(delta_array, 0.95)),
            "absolute_delta_p99": float(np.quantile(delta_array, 0.99)),
            "absolute_delta_max": float(delta_array.max()),
            "threshold_flip_count": len(padding_flips),
            "threshold_flip_candidate_ids": padding_flips,
        },
        "holdout_error_causes": dict(
            Counter(
                str(row["error_cause"])
                for row in holdout_rows
                if row.get("error_cause")
            )
        ),
        "holdout": holdout_rows,
        "linear_probes": _linear_probes(
            bundle=bundle,
            checkpoint=checkpoint,
            holdout_audio_ids=holdout_audio_ids,
            holdout_candidate_ids=holdout_candidate_ids,
            position_by_id=position_by_id,
        ),
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "padding_stability": {
                    key: value
                    for key, value in summary["padding_stability"].items()
                    if key != "threshold_flip_candidate_ids"
                },
                "holdout_error_causes": summary["holdout_error_causes"],
                "linear_probe_metrics": {
                    name: {
                        key: value
                        for key, value in probe.items()
                        if key != "candidate_prob_drop"
                    }
                    for name, probe in summary["linear_probes"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Pre-ASR v12 representation failures.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--paired-decisions", required=True)
    parser.add_argument("--manual-labels", required=True)
    parser.add_argument("--holdout-candidate-ids", required=True)
    parser.add_argument("--holdout-audio-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.50)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
