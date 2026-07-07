#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from asr.pre_asr_cueqc import (  # noqa: E402
    PRE_ASR_CUEQC_IGNORE_LABEL,
    PreAsrCueQCNetwork,
    make_model_config,
)


SUMMARY_SCHEMA = "pre_asr_cueqc_v12_gate_summary_v1"
PAIRED_SCHEMA = "pre_asr_cueqc_v12_gate_paired_decision_v1"
LONG_FALSE_DROP_SCHEMA = "pre_asr_cueqc_v12_gate_long_false_drop_v1"


def _project_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _read_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint must be a mapping: {path}")
    return dict(payload)


def _read_bundle(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"feature bundle must be a mapping: {path}")
    return dict(payload)


def _selected_scalar(bundle: Mapping[str, Any], feature_names: tuple[str, ...]) -> Any:
    source_names = tuple(str(item) for item in bundle.get("feature_names") or ())
    index_by_name = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in feature_names if name not in index_by_name]
    if missing:
        raise ValueError(
            "feature bundle is missing checkpoint scalar fields: "
            + ", ".join(missing[:8])
        )
    indexes = [index_by_name[name] for name in feature_names]
    return bundle["scalar_features"][:, :, indexes].float()


def _predict_probs(
    *,
    bundle: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    device: str,
) -> np.ndarray:
    import torch

    feature_names = tuple(str(item) for item in checkpoint.get("feature_names") or ())
    if not feature_names:
        raise ValueError("checkpoint feature_names must not be empty")
    config = make_model_config(checkpoint.get("model_config"))
    if int(config["scalar_dim"]) != len(feature_names):
        raise ValueError("checkpoint model_config.scalar_dim does not match feature_names")
    model = PreAsrCueQCNetwork(**config)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint missing model_state_dict")
    model.load_state_dict(state)
    normalized_device = (device or "auto").strip().lower()
    if normalized_device == "auto":
        normalized_device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(normalized_device)
    model.to(dev)
    model.eval()

    scalar = _selected_scalar(bundle, feature_names)
    mean = torch.as_tensor(
        checkpoint.get("feature_mean", [0.0] * len(feature_names)),
        dtype=torch.float32,
    )
    std = torch.as_tensor(
        checkpoint.get("feature_std", [1.0] * len(feature_names)),
        dtype=torch.float32,
    ).clamp_min(1e-6)
    if tuple(mean.shape) != (len(feature_names),) or tuple(std.shape) != (len(feature_names),):
        raise ValueError("checkpoint feature normalization shape mismatch")
    scalar = (scalar - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    scalar = torch.nan_to_num(scalar, nan=0.0, posinf=0.0, neginf=0.0)

    ptm_bins = bundle["ptm_bins"].float().to(dev)
    scalar = scalar.to(dev)
    chunk_mask = bundle["chunk_mask"].float().to(dev)
    bin_mask = bundle["bin_mask"].float().to(dev)
    decision = dict(checkpoint.get("decision_config") or {})
    window_size = max(0, int(decision.get("inference_window_size") or 512))
    with torch.inference_mode():
        if window_size > 0 and int(ptm_bins.shape[1]) > window_size:
            group_count, max_chunks = tuple(chunk_mask.shape)
            logits_all = torch.zeros(
                (group_count, max_chunks, 2),
                dtype=torch.float32,
                device=dev,
            )
            counts = torch.zeros(
                (group_count, max_chunks, 1),
                dtype=torch.float32,
                device=dev,
            )
            for group_index in range(int(group_count)):
                length = int(chunk_mask[group_index].sum().detach().cpu().item())
                for start in range(0, length, window_size):
                    end = min(length, start + window_size)
                    logits = model(
                        ptm_bins[group_index : group_index + 1, start:end],
                        scalar[group_index : group_index + 1, start:end],
                        chunk_mask=chunk_mask[group_index : group_index + 1, start:end],
                        bin_mask=bin_mask[group_index : group_index + 1, start:end],
                    )
                    logits_all[group_index, start:end] += logits[0].float()
                    counts[group_index, start:end] += 1.0
            logits = logits_all / counts.clamp_min(1.0)
        else:
            logits = model(ptm_bins, scalar, chunk_mask=chunk_mask, bin_mask=bin_mask)
        return torch.softmax(logits, dim=-1).float().cpu().numpy()


def _duration(features: Mapping[str, float]) -> float:
    return float(features.get("refined_duration_s") or features.get("duration_s") or 0.0)


def _legacy_hard_keep(features: Mapping[str, float], config: Mapping[str, Any]) -> str:
    duration = _duration(features)
    if duration >= float(config.get("hard_keep_min_duration_s", 0.80)):
        return "duration_at_or_above_hard_keep_min"
    speech_p90 = float(features.get("scorer_speech_p90") or 0.0)
    active05 = float(features.get("scorer_speech_active_ratio_05") or 0.0)
    if (
        speech_p90 >= float(config.get("high_speech_p90", 0.85))
        and active05 >= float(config.get("high_active_ratio", 0.50))
    ):
        return "high_stable_speech"
    if bool(features.get("micro_chunk_candidate")):
        if float(features.get("left_split_score") or 0.0) >= 0.75 and float(
            features.get("right_split_score") or 0.0
        ) >= 0.75:
            return "strong_micro_split_evidence"
    if (
        float(features.get("prev_scorer_speech_p90") or 0.0) >= 0.70
        and float(features.get("next_scorer_speech_p90") or 0.0) >= 0.70
    ):
        return "between_strong_speech_neighbors"
    return ""


def _legacy_hard_drop(features: Mapping[str, float], config: Mapping[str, Any]) -> str:
    if (
        _duration(features) < 0.12
        and float(features.get("scorer_speech_p90") or 0.0)
        < float(config.get("very_low_speech_p90", 0.05))
        and float(features.get("scorer_speech_active_ratio_05") or 0.0)
        < float(config.get("very_low_active_ratio", 0.05))
        and not bool(features.get("micro_chunk_candidate"))
    ):
        return "very_short_very_low_speech"
    return ""


def _legacy_keep_veto(features: Mapping[str, float], config: Mapping[str, Any]) -> str:
    speech_p90 = float(features.get("scorer_speech_p90") or 0.0)
    active05 = float(features.get("scorer_speech_active_ratio_05") or 0.0)
    if (
        speech_p90 >= float(config.get("high_speech_p90", 0.85))
        and active05 >= float(config.get("high_active_ratio", 0.50))
    ):
        return "high_stable_speech"
    if (
        float(features.get("subtitle_min_duration_s") or 0.0) > 0.0
        and _duration(features) >= float(features.get("subtitle_min_duration_s") or 0.0)
        and float(features.get("scorer_speech_mean") or 0.0) >= 0.20
    ):
        return "display_duration_with_speech"
    return ""


def _legacy_decision(
    *,
    p_drop: float,
    threshold: float,
    features: Mapping[str, float],
    config: Mapping[str, Any],
    force_rules: bool,
) -> tuple[bool, str, str, str]:
    hard_keep_enabled = bool(config.get("hard_keep_veto", False)) or force_rules
    hard_drop_enabled = bool(config.get("hard_drop_rule", False)) or force_rules
    keep_veto_enabled = bool(config.get("keep_veto", False)) or force_rules
    hard_keep = _legacy_hard_keep(features, config) if hard_keep_enabled else ""
    hard_drop = (
        _legacy_hard_drop(features, config)
        if hard_drop_enabled and not hard_keep
        else ""
    )
    keep_veto = (
        _legacy_keep_veto(features, config)
        if keep_veto_enabled and not hard_keep and not hard_drop
        else ""
    )
    if hard_keep:
        return False, "hard_keep_veto", keep_veto or hard_keep, hard_drop
    if hard_drop:
        return True, "hard_drop_rule", keep_veto, hard_drop
    if p_drop >= threshold and not keep_veto:
        return True, "model_drop_threshold", "", ""
    return False, "keep_veto" if keep_veto else "model_keep_default", keep_veto, ""


def _metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    tp = sum(row["truth"] == "drop" and row[prediction_key] == "drop" for row in rows)
    fp = sum(row["truth"] == "keep" and row[prediction_key] == "drop" for row in rows)
    fn = sum(row["truth"] == "drop" and row[prediction_key] == "keep" for row in rows)
    tn = sum(row["truth"] == "keep" and row[prediction_key] == "keep" for row in rows)
    return {
        "count": len(rows),
        "accuracy": (tp + tn) / max(1, len(rows)),
        "drop_precision": tp / max(1, tp + fp),
        "drop_recall": tp / max(1, tp + fn),
        "drop_f1": (2 * tp) / max(1, 2 * tp + fp + fn),
        "semantic_keep_recall": tn / max(1, tn + fp),
        "false_drop_count": fp,
        "false_keep_count": fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle_path = _project_path(args.features)
    v12_path = _project_path(args.v12_checkpoint)
    legacy_path = _project_path(args.legacy_v11_checkpoint)
    output_dir = _project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = _read_bundle(bundle_path)
    v12_checkpoint = _read_checkpoint(v12_path)
    legacy_checkpoint = _read_checkpoint(legacy_path)
    bundle_feature_names = tuple(str(item) for item in bundle.get("feature_names") or ())
    labels = bundle["labels"].detach().cpu().numpy()
    mask = bundle["chunk_mask"].detach().cpu().numpy() > 0
    scalar = bundle["scalar_features"].detach().cpu().numpy()
    rows_meta = [dict(item) for item in (bundle.get("rows") or [])]
    if int(mask.sum()) != len(rows_meta):
        raise ValueError("feature bundle rows do not align with chunk_mask")

    v12_probs = _predict_probs(bundle=bundle, checkpoint=v12_checkpoint, device=args.device)
    legacy_probs = _predict_probs(
        bundle=bundle,
        checkpoint=legacy_checkpoint,
        device=args.device,
    )
    v12_config = dict(v12_checkpoint.get("decision_config") or {})
    legacy_config = dict(legacy_checkpoint.get("decision_config") or {})
    v12_threshold = (
        float(args.v12_threshold)
        if args.v12_threshold is not None
        else float(v12_config.get("drop_threshold", 0.95))
    )
    legacy_threshold = (
        float(args.legacy_threshold)
        if args.legacy_threshold is not None
        else float(legacy_config.get("drop_threshold", 0.95))
    )

    paired_rows: list[dict[str, Any]] = []
    long_false_drops: list[dict[str, Any]] = []
    row_offset = 0
    for group_index in range(labels.shape[0]):
        for chunk_index in range(labels.shape[1]):
            if not mask[group_index, chunk_index]:
                continue
            meta = rows_meta[row_offset] if row_offset < len(rows_meta) else {}
            row_offset += 1
            label_index = int(labels[group_index, chunk_index])
            if label_index not in (0, 1):
                continue
            features = {
                name: float(scalar[group_index, chunk_index, feature_index])
                for feature_index, name in enumerate(bundle_feature_names)
            }
            truth = "drop" if label_index == 0 else "keep"
            v12_p_drop = float(v12_probs[group_index, chunk_index, 0])
            legacy_p_drop = float(legacy_probs[group_index, chunk_index, 0])
            v12_drop = v12_p_drop >= v12_threshold
            legacy_drop, legacy_reason, legacy_veto, legacy_hard = _legacy_decision(
                p_drop=legacy_p_drop,
                threshold=legacy_threshold,
                features=features,
                config=legacy_config,
                force_rules=bool(args.force_legacy_rules),
            )
            item = {
                "schema": PAIRED_SCHEMA,
                "id": str(meta.get("id") or ""),
                "audio_id": str(meta.get("audio_id") or ""),
                "source": str(meta.get("source") or ""),
                "group_index": int(group_index),
                "chunk_index": int(chunk_index),
                "start": float(meta.get("start") or 0.0),
                "end": float(meta.get("end") or 0.0),
                "duration_s": float(features.get("refined_duration_s") or meta.get("end", 0.0) - meta.get("start", 0.0)),
                "truth": truth,
                "v12_prediction": "drop" if v12_drop else "keep",
                "v12_prob_drop": v12_p_drop,
                "v12_reason": "model_drop_threshold" if v12_drop else "model_keep_default",
                "legacy_prediction": "drop" if legacy_drop else "keep",
                "legacy_prob_drop": legacy_p_drop,
                "legacy_reason": legacy_reason,
                "legacy_veto_reason": legacy_veto,
                "legacy_hard_rule_reason": legacy_hard,
            }
            paired_rows.append(item)
            if (
                truth == "keep"
                and item["v12_prediction"] == "drop"
                and float(item["duration_s"]) >= float(args.long_false_drop_min_s)
            ):
                long_false_drops.append(
                    {
                        "schema": LONG_FALSE_DROP_SCHEMA,
                        **item,
                    }
                )

    v12_metrics = _metrics(paired_rows, "v12_prediction")
    legacy_metrics = _metrics(paired_rows, "legacy_prediction")
    keep_margin = (
        float(v12_metrics["semantic_keep_recall"])
        - float(legacy_metrics["semantic_keep_recall"])
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "features": str(bundle_path.relative_to(ROOT) if bundle_path.is_relative_to(ROOT) else bundle_path),
        "v12_checkpoint": str(v12_path.relative_to(ROOT) if v12_path.is_relative_to(ROOT) else v12_path),
        "legacy_v11_checkpoint": str(
            legacy_path.relative_to(ROOT) if legacy_path.is_relative_to(ROOT) else legacy_path
        ),
        "device": args.device,
        "v12_threshold": v12_threshold,
        "legacy_threshold": legacy_threshold,
        "force_legacy_rules": bool(args.force_legacy_rules),
        "class_counts": {
            "drop": sum(row["truth"] == "drop" for row in paired_rows),
            "keep": sum(row["truth"] == "keep" for row in paired_rows),
        },
        "v12": v12_metrics,
        "legacy_v11": legacy_metrics,
        "gate": {
            "min_drop_recall": float(args.min_drop_recall),
            "keep_recall_margin_vs_legacy": keep_margin,
            "keep_recall_pass": keep_margin >= -1e-12,
            "drop_recall_pass": float(v12_metrics["drop_recall"]) >= float(args.min_drop_recall),
            "long_false_drop_min_s": float(args.long_false_drop_min_s),
            "long_false_drop_count": len(long_false_drops),
            "long_false_drop_audit_required": len(long_false_drops) > 0,
        },
    }
    summary["gate"]["promote_candidate"] = (
        bool(summary["gate"]["keep_recall_pass"])
        and bool(summary["gate"]["drop_recall_pass"])
        and not bool(summary["gate"]["long_false_drop_audit_required"])
    )

    paired_path = output_dir / "paired_decisions.jsonl"
    paired_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in paired_rows),
        encoding="utf-8",
    )
    long_path = output_dir / "v12_false_drops_ge_0p8s.jsonl"
    long_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in long_false_drops),
        encoding="utf-8",
    )
    summary["outputs"] = {
        "paired_decisions": str(paired_path.relative_to(ROOT)),
        "v12_false_drops_ge_0p8s": str(long_path.relative_to(ROOT)),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate a Pre-ASR CueQC v12 candidate against explicit v11 legacy replay."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--v12-checkpoint", required=True)
    parser.add_argument("--legacy-v11-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--v12-threshold", type=float)
    parser.add_argument("--legacy-threshold", type=float)
    parser.add_argument("--min-drop-recall", type=float, default=0.98)
    parser.add_argument("--long-false-drop-min-s", type=float, default=0.8)
    parser.add_argument(
        "--force-legacy-rules",
        action="store_true",
        help="Diagnostic only: force old hard rules on for the v11 replay arm.",
    )
    args = parser.parse_args(argv)
    for name in ("v12_threshold", "legacy_threshold"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= float(value) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not 0.0 <= float(args.min_drop_recall) <= 1.0:
        parser.error("--min-drop-recall must be in [0, 1]")
    if float(args.long_false_drop_min_s) < 0.0:
        parser.error("--long-false-drop-min-s must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
