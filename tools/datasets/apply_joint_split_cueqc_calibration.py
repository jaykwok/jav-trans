#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SRC_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from asr.backends.qwen import (  # noqa: E402
    DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO,
    DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO,
    QWEN_ASR_17B_REPO_ID,
)

SCHEMA = "joint_split_cueqc_calibration_apply_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(payload: Mapping[str, Any]) -> str:
    import torch

    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint missing model_state_dict")
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"model_state_dict.{key} must be a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _pressure_config(summary: Mapping[str, Any]) -> dict[str, Any]:
    pressure = dict(summary.get("semantic_split_duration_pressure") or {})
    if not bool(pressure.get("enabled")):
        raise ValueError("operating-point summary must enable duration pressure")
    return {
        "duration_pressure_enabled": True,
        "duration_pressure_log_median": float(pressure["log_median"]),
        "duration_pressure_log_mad": float(pressure["log_mad"]),
        "duration_pressure_z": float(pressure["z"]),
        "duration_pressure_floor": float(pressure["floor"]),
    }


def apply_calibration(
    *,
    calibration_summary_path: Path,
    operating_point_summary_path: Path,
    split_checkpoint_path: Path,
    cueqc_checkpoint_path: Path,
    output_summary_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    import torch

    calibration = _read_json(calibration_summary_path)
    if str(calibration.get("schema") or "") != "joint_split_cueqc_calibration_v1":
        raise ValueError("calibration summary schema mismatch")
    recommendation = dict(calibration.get("recommendation") or {})
    if str(recommendation.get("status") or "") != "ready_to_apply":
        raise ValueError("calibration recommendation is not ready_to_apply")
    if bool(recommendation.get("requires_de_prime")) or bool(
        recommendation.get("requires_cueqc_reaudit")
    ):
        raise ValueError("calibration still requires reconvergence or re-audit")
    if not bool(recommendation.get("weights_must_remain_unchanged")):
        raise ValueError("calibration must preserve model weights")

    split_updates = dict(recommendation["split_decision_config"])
    cueqc_updates = dict(recommendation["cueqc_decision_config"])
    pressure_updates = _pressure_config(_read_json(operating_point_summary_path))
    split_payload = torch.load(
        split_checkpoint_path, map_location="cpu", weights_only=False
    )
    cueqc_payload = torch.load(
        cueqc_checkpoint_path, map_location="cpu", weights_only=False
    )
    if str(split_payload.get("schema") or "") != "semantic_split_verifier_v2":
        raise ValueError("active Split checkpoint is not v2")
    if str(cueqc_payload.get("schema") or "") != "cueqc_pre_asr_semantic_chunk_v12_binary":
        raise ValueError("active CueQC checkpoint is not v12")

    split_weights_before = _state_digest(split_payload)
    cueqc_weights_before = _state_digest(cueqc_payload)
    split_file_before = _file_sha256(split_checkpoint_path)
    cueqc_file_before = _file_sha256(cueqc_checkpoint_path)
    applied_at = datetime.now(timezone.utc).isoformat()
    calibration_record = {
        "schema": "joint_split_cueqc_selected_calibration_v1",
        "source": str(calibration_summary_path),
        "operating_point_source": str(operating_point_summary_path),
        "applied_at": applied_at,
        "selected": dict(calibration["selected"]),
    }

    split_payload["decision_config"] = {
        **dict(split_payload.get("decision_config") or {}),
        "normal_cut_threshold": float(split_updates["normal_cut_threshold"]),
        "short_core_cut_threshold": float(split_updates["short_core_cut_threshold"]),
        **pressure_updates,
    }
    split_metadata = dict(split_payload.get("metadata") or {})
    split_metadata["selected_calibration"] = calibration_record
    split_payload["metadata"] = split_metadata

    split_temp = split_checkpoint_path.with_name(
        f".{split_checkpoint_path.name}.stage-e-calibrating"
    )
    cueqc_temp = cueqc_checkpoint_path.with_name(
        f".{cueqc_checkpoint_path.name}.stage-e-calibrating"
    )
    torch.save(split_payload, split_temp)
    calibrated_split_sha = _file_sha256(split_temp)

    cueqc_payload["decision_config"] = {
        **dict(cueqc_payload.get("decision_config") or {}),
        "drop_threshold": float(cueqc_updates["drop_threshold"]),
    }
    cueqc_payload["semantic_split_weights_sha256"] = calibrated_split_sha
    cueqc_metadata = dict(cueqc_payload.get("metadata") or {})
    cueqc_metadata["semantic_split_weights_sha256"] = calibrated_split_sha
    cueqc_metadata["selected_calibration"] = calibration_record
    cueqc_payload["metadata"] = cueqc_metadata
    torch.save(cueqc_payload, cueqc_temp)

    split_reloaded = torch.load(split_temp, map_location="cpu", weights_only=False)
    cueqc_reloaded = torch.load(cueqc_temp, map_location="cpu", weights_only=False)
    split_weights_after = _state_digest(split_reloaded)
    cueqc_weights_after = _state_digest(cueqc_reloaded)
    if split_weights_after != split_weights_before:
        raise ValueError("Split model weights changed during calibration")
    if cueqc_weights_after != cueqc_weights_before:
        raise ValueError("CueQC model weights changed during calibration")

    summary = {
        "schema": SCHEMA,
        "dry_run": bool(dry_run),
        "calibration_summary": str(calibration_summary_path),
        "operating_point_summary": str(operating_point_summary_path),
        "split_checkpoint": str(split_checkpoint_path),
        "cueqc_checkpoint": str(cueqc_checkpoint_path),
        "split_file_sha256_before": split_file_before,
        "split_file_sha256_after": calibrated_split_sha,
        "cueqc_file_sha256_before": cueqc_file_before,
        "cueqc_file_sha256_after": _file_sha256(cueqc_temp),
        "split_weights_digest_before": split_weights_before,
        "split_weights_digest_after": split_weights_after,
        "cueqc_weights_digest_before": cueqc_weights_before,
        "cueqc_weights_digest_after": cueqc_weights_after,
        "weights_unchanged": True,
        "split_decision_config": dict(split_reloaded["decision_config"]),
        "cueqc_decision_config": dict(cueqc_reloaded["decision_config"]),
        "cueqc_bound_split_sha256": calibrated_split_sha,
        "applied_at": applied_at,
    }
    if dry_run:
        split_temp.unlink()
        cueqc_temp.unlink()
    else:
        split_temp.replace(split_checkpoint_path)
        cueqc_temp.replace(cueqc_checkpoint_path)
        if _file_sha256(split_checkpoint_path) != calibrated_split_sha:
            raise ValueError("calibrated Split checkpoint hash changed after replace")
    _write_json(output_summary_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atomically apply a ready Split x CueQC calibration."
    )
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--operating-point-summary", required=True)
    parser.add_argument(
        "--split-checkpoint",
        default=DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO[QWEN_ASR_17B_REPO_ID],
    )
    parser.add_argument(
        "--cueqc-checkpoint",
        default=DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO[QWEN_ASR_17B_REPO_ID],
    )
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = apply_calibration(
        calibration_summary_path=Path(args.calibration_summary),
        operating_point_summary_path=Path(args.operating_point_summary),
        split_checkpoint_path=Path(args.split_checkpoint),
        cueqc_checkpoint_path=Path(args.cueqc_checkpoint),
        output_summary_path=Path(args.output_summary),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
