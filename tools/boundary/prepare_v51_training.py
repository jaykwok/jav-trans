#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.boundary.build_refiner_frame_sequence_dataset import DATASET_SCHEMA  # noqa: E402
from tools.boundary.boundary_preference import read_jsonl  # noqa: E402


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "agents"
    / "temp"
    / "speech-boundary-ja"
    / "true-v5-boundary-preference-pilot"
    / "compiled_preference_v5.jsonl"
)
DEFAULT_INIT_CHECKPOINT = PROJECT_ROOT / "src" / "boundary" / "checkpoints" / "boundary_refiner.pt"
DEFAULT_MIN_FORMAL_PREFERENCES = 300
DEFAULT_LR = 0.0001
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_STEPS = 100


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("/", "\\")
    except ValueError:
        return str(path)


def local_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def file_sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def summarize_dataset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("v5.1 training prep requires at least one dataset row")

    schema_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    axis_counts: Counter[str] = Counter()
    winner_counts: Counter[str] = Counter()
    feature_schema_counts: Counter[str] = Counter()
    feature_hash_counts: Counter[str] = Counter()
    feature_dims: Counter[int] = Counter()
    sequence_items = 0
    start_supervised = 0
    end_supervised = 0
    invalid_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=1):
        schema = str(row.get("schema") or "")
        schema_counts[schema] += 1
        if schema != DATASET_SCHEMA:
            invalid_rows.append({"row": row_index, "error": f"unsupported schema {schema!r}"})
            continue

        features = row.get("sequence_features")
        targets = row.get("sequence_boundary_delta_targets")
        weights = row.get("sequence_boundary_delta_weights")
        feature_names = row.get("feature_names")
        if not isinstance(features, list) or not features:
            invalid_rows.append({"row": row_index, "error": "missing sequence_features"})
            continue
        if not isinstance(targets, list) or len(targets) != len(features):
            invalid_rows.append({"row": row_index, "error": "target length mismatch"})
            continue
        if not isinstance(weights, list) or len(weights) != len(features):
            invalid_rows.append({"row": row_index, "error": "weight length mismatch"})
            continue
        if not isinstance(feature_names, list) or not feature_names:
            invalid_rows.append({"row": row_index, "error": "missing feature_names"})
            continue
        for item_index, feature in enumerate(features):
            if not isinstance(feature, list) or len(feature) != len(feature_names):
                invalid_rows.append(
                    {
                        "row": row_index,
                        "item": item_index,
                        "error": "feature dim mismatch",
                    }
                )
                break
        for item_index, weight in enumerate(weights):
            if not isinstance(weight, list) or len(weight) != 2:
                invalid_rows.append(
                    {
                        "row": row_index,
                        "item": item_index,
                        "error": "weight must be [start_weight, end_weight]",
                    }
                )
                break
            start_supervised += 1 if float(weight[0]) > 0.0 else 0
            end_supervised += 1 if float(weight[1]) > 0.0 else 0

        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        source_counts[str(row.get("source") or "")] += 1
        video_counts[str(metadata.get("video_id") or row.get("audio_id") or "")] += 1
        axis_counts[str(metadata.get("axis") or "")] += 1
        winner_counts[str(metadata.get("winner") or "")] += 1
        feature_schema_counts[str(row.get("feature_schema") or "")] += 1
        feature_hash_counts[str(row.get("feature_schema_hash") or "")] += 1
        feature_dims[len(feature_names)] += 1
        sequence_items += len(features)

    if invalid_rows:
        preview = invalid_rows[:5]
        raise ValueError(f"invalid v5.1 dataset rows: {preview}")

    if schema_counts != Counter({DATASET_SCHEMA: len(rows)}):
        raise ValueError(f"dataset contains unsupported schema values: {dict(schema_counts)}")

    return {
        "schema": DATASET_SCHEMA,
        "row_count": len(rows),
        "sequence_items": sequence_items,
        "start_supervised_items": start_supervised,
        "end_supervised_items": end_supervised,
        "source_counts": dict(source_counts),
        "video_counts": dict(video_counts),
        "axis_counts": dict(axis_counts),
        "winner_counts": dict(winner_counts),
        "feature_schema_counts": dict(feature_schema_counts),
        "feature_hash_counts": dict(feature_hash_counts),
        "feature_dim_counts": {str(key): value for key, value in sorted(feature_dims.items())},
    }


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import failure depends on host env.
        raise RuntimeError("torch is required to inspect the Boundary checkpoint") from exc

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Boundary checkpoint must be a mapping")
    schema = str(payload.get("schema") or "")
    model_config = dict(payload.get("model_config") or {})
    output_dim = int(model_config.get("output_dim", -1))
    if schema != "boundary_refiner_v5":
        raise ValueError(f"init checkpoint schema must be boundary_refiner_v5, got {schema!r}")
    if output_dim != 2:
        raise ValueError(f"init checkpoint output_dim must be 2, got {output_dim}")
    feature_names = list(payload.get("feature_names") or ())
    if not feature_names:
        raise ValueError("init checkpoint is missing feature_names")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    return {
        "path": repo_display_path(path),
        "sha1": file_sha1(path),
        "schema": schema,
        "feature_dim": len(feature_names),
        "feature_schema": metadata.get("feature_schema") or payload.get("feature_schema"),
        "feature_schema_hash": metadata.get("feature_schema_hash") or payload.get("feature_schema_hash"),
        "runtime_adapter": metadata.get("runtime_adapter"),
        "model_config": {
            "hidden_size": int(model_config.get("hidden_size", 0)),
            "num_layers": int(model_config.get("num_layers", 0)),
            "state_size": int(model_config.get("state_size", 0)),
            "num_heads": int(model_config.get("num_heads", 0)),
            "n_groups": int(model_config.get("n_groups", 0)),
            "chunk_size": int(model_config.get("chunk_size", 0)),
            "bidirectional": bool(model_config.get("bidirectional", True)),
            "input_dim": int(model_config.get("input_dim", len(feature_names))),
            "output_dim": output_dim,
        },
    }


def _ps_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _ps_expandable(value: str | Path) -> str:
    return '"' + str(value).replace("`", "``").replace('"', '`"') + '"'


def _train_command_lines(
    *,
    dataset: str,
    output_dir: str,
    init_checkpoint: str,
    tensor_cache_path: str,
    model_config: Mapping[str, Any],
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    log_interval_steps: int,
    expand_output_dir: bool = False,
) -> list[str]:
    output_arg = _ps_expandable(output_dir) if expand_output_dir else _ps_literal(output_dir)
    lines = [
        "$env:PYTHONIOENCODING='utf-8'",
        "uv run python -m tools.boundary.train_refiner `",
        f"  --dataset {_ps_literal(dataset)} `",
        f"  --output-dir {output_arg} `",
        f"  --max-steps {int(max_steps)} `",
        f"  --batch-size {int(batch_size)} `",
        f"  --learning-rate {float(learning_rate):.8g} `",
        f"  --device {_ps_literal(device)} `",
        f"  --hidden-size {int(model_config['hidden_size'])} `",
        f"  --num-layers {int(model_config['num_layers'])} `",
        f"  --state-size {int(model_config['state_size'])} `",
        f"  --num-heads {int(model_config['num_heads'])} `",
        f"  --n-groups {int(model_config['n_groups'])} `",
        f"  --chunk-size {int(model_config['chunk_size'])} `",
        f"  --init-checkpoint {_ps_literal(init_checkpoint)} `",
        "  --preserve-init-normalization `",
        "  --freeze-backbone `",
        f"  --tensor-cache-path {_ps_literal(tensor_cache_path)} `",
        f"  --log-interval-steps {int(log_interval_steps)}",
    ]
    if not bool(model_config.get("bidirectional", True)):
        lines.insert(-1, "  --unidirectional `")
    return lines


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare_training_prep(
    *,
    dataset_path: Path,
    init_checkpoint: Path,
    output_dir: Path,
    min_formal_preferences: int = DEFAULT_MIN_FORMAL_PREFERENCES,
    formal_max_steps: int = DEFAULT_MAX_STEPS,
    formal_batch_size: int = DEFAULT_BATCH_SIZE,
    formal_learning_rate: float = DEFAULT_LR,
    formal_device: str = "auto",
    dry_run_device: str = "cpu",
) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    init_checkpoint = init_checkpoint.resolve()
    output_dir = output_dir.resolve()
    dataset_summary = summarize_dataset(read_jsonl(dataset_path))
    checkpoint_summary = inspect_checkpoint(init_checkpoint)
    model_config = checkpoint_summary["model_config"]

    dataset_rel = repo_display_path(dataset_path)
    init_rel = repo_display_path(init_checkpoint)
    output_rel = repo_display_path(output_dir)
    tensor_cache_rel = repo_display_path(output_dir / "compiled_preference_v5.tensor-cache.pt")
    dry_run_output_rel = repo_display_path(output_dir / "dry-run-1step-matched-config")

    formal_output_template = "agents\\temp\\$($stamp)_boundary-v5.1-preference-ft-head"
    formal_prefix = [
        "$env:PYTHONIOENCODING='utf-8'",
        "$stamp = Get-Date -Format yyyyMMdd_HHmmss",
    ]
    formal_lines = formal_prefix + _train_command_lines(
        dataset=dataset_rel,
        output_dir=formal_output_template,
        init_checkpoint=init_rel,
        tensor_cache_path=tensor_cache_rel,
        model_config=model_config,
        max_steps=formal_max_steps,
        batch_size=formal_batch_size,
        learning_rate=formal_learning_rate,
        device=formal_device,
        log_interval_steps=10,
        expand_output_dir=True,
    )[1:]
    dry_run_lines = _train_command_lines(
        dataset=dataset_rel,
        output_dir=dry_run_output_rel,
        init_checkpoint=init_rel,
        tensor_cache_path=tensor_cache_rel,
        model_config=model_config,
        max_steps=1,
        batch_size=8,
        learning_rate=formal_learning_rate,
        device=dry_run_device,
        log_interval_steps=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dry_run_script = output_dir / "dry_run_1step_cpu.ps1"
    formal_script = output_dir / "train_preference_head.ps1"
    write_text(dry_run_script, "\n".join(dry_run_lines) + "\n")
    write_text(formal_script, "\n".join(formal_lines) + "\n")

    row_count = int(dataset_summary["row_count"])
    formal_ready = row_count >= int(min_formal_preferences)
    summary = {
        "schema": "boundary_v51_training_prep_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": output_rel,
        "dataset_path": dataset_rel,
        "init_checkpoint": checkpoint_summary,
        "dataset": dataset_summary,
        "recommendations": {
            "formal_min_preference_rows": int(min_formal_preferences),
            "formal_training_recommended": formal_ready,
            "pilot_training_possible": row_count > 0,
            "replace_default_checkpoint": False,
            "reason": (
                "enough preference rows for a head-only pilot"
                if formal_ready
                else "preference rows below formal gate; use only dry-run or pilot, not default replacement"
            ),
        },
        "hard_negative_plan": [
            "CueQC display=drop/drop_ok is not a direct v5.1 delta label.",
            "Pure non-speech, short noise, and invalid speech islands should become SpeechBoundary-JA frame-level negatives.",
            "Over-fragmented chunks or chunks that should attach to neighbors should become Boundary preference/hard-case rows.",
            "Mixed or uncertain drops stay in an audit pool until a precise boundary target exists.",
        ],
        "commands": {
            "dry_run_script": repo_display_path(dry_run_script),
            "formal_head_finetune_script": repo_display_path(formal_script),
            "dry_run": dry_run_lines,
            "formal_head_finetune": formal_lines,
        },
    }
    summary_json = output_dir / "training_prep_summary.json"
    summary_md = output_dir / "summary.md"
    write_text(summary_json, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_text(summary_md, render_markdown_summary(summary))
    return summary


def render_markdown_summary(summary: Mapping[str, Any]) -> str:
    dataset = summary["dataset"]
    checkpoint = summary["init_checkpoint"]
    rec = summary["recommendations"]
    lines = [
        "# Boundary v5.1 Training Prep Summary",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{summary['dataset_path']}`",
        f"- Rows/items: `{dataset['row_count']}` / `{dataset['sequence_items']}`",
        f"- Start/end supervised items: `{dataset['start_supervised_items']}` / `{dataset['end_supervised_items']}`",
        f"- Axis counts: `{dataset['axis_counts']}`",
        f"- Winner counts: `{dataset['winner_counts']}`",
        f"- Feature hashes: `{dataset['feature_hash_counts']}`",
        "",
        "## Init Checkpoint",
        "",
        f"- Path: `{checkpoint['path']}`",
        f"- SHA1: `{checkpoint['sha1']}`",
        f"- Schema/runtime: `{checkpoint['schema']}` / `{checkpoint.get('runtime_adapter')}`",
        f"- Model config: `{checkpoint['model_config']}`",
        "",
        "## Gate",
        "",
        f"- Formal min preference rows: `{rec['formal_min_preference_rows']}`",
        f"- Formal training recommended now: `{rec['formal_training_recommended']}`",
        f"- Pilot training possible: `{rec['pilot_training_possible']}`",
        f"- Replace default checkpoint: `{rec['replace_default_checkpoint']}`",
        f"- Reason: {rec['reason']}",
        "",
        "## Scripts",
        "",
        f"- Dry run: `{summary['commands']['dry_run_script']}`",
        f"- Head finetune: `{summary['commands']['formal_head_finetune_script']}`",
        "",
        "## Hard Negative Plan",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["hard_negative_plan"])
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a reproducible Boundary Refiner v5.1 preference-finetune package."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT_CHECKPOINT))
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to agents/temp/YYYYMMDD_HHMMSS_boundary-v5.1-training-prep.",
    )
    parser.add_argument("--min-formal-preferences", type=int, default=DEFAULT_MIN_FORMAL_PREFERENCES)
    parser.add_argument("--formal-max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--formal-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--formal-learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--formal-device", default="auto")
    parser.add_argument("--dry-run-device", default="cpu")
    parser.add_argument(
        "--strict-formal-gate",
        action="store_true",
        help="Exit with code 2 when row count is below --min-formal-preferences.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_formal_preferences <= 0:
        raise ValueError("--min-formal-preferences must be positive")
    if args.formal_max_steps <= 0:
        raise ValueError("--formal-max-steps must be positive")
    if args.formal_batch_size <= 0:
        raise ValueError("--formal-batch-size must be positive")
    if args.formal_learning_rate <= 0.0:
        raise ValueError("--formal-learning-rate must be positive")

    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "agents" / "temp" / f"{local_timestamp()}_boundary-v5.1-training-prep"
    )
    summary = prepare_training_prep(
        dataset_path=project_path(args.dataset),
        init_checkpoint=project_path(args.init_checkpoint),
        output_dir=output_dir,
        min_formal_preferences=args.min_formal_preferences,
        formal_max_steps=args.formal_max_steps,
        formal_batch_size=args.formal_batch_size,
        formal_learning_rate=args.formal_learning_rate,
        formal_device=args.formal_device,
        dry_run_device=args.dry_run_device,
    )
    print(f"output_dir={summary['output_dir']}")
    print(f"summary={summary['output_dir']}\\training_prep_summary.json")
    print(f"dry_run_script={summary['commands']['dry_run_script']}")
    print(f"formal_head_finetune_script={summary['commands']['formal_head_finetune_script']}")
    print(
        "rows={rows} items={items} formal_ready={ready}".format(
            rows=summary["dataset"]["row_count"],
            items=summary["dataset"]["sequence_items"],
            ready=summary["recommendations"]["formal_training_recommended"],
        )
    )
    if args.strict_formal_gate and not summary["recommendations"]["formal_training_recommended"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
