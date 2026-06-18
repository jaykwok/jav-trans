#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import QWEN_ASR_REPO_ID, qwen_asr_repo_tag  # noqa: E402
from boundary.ja import load_label_records  # noqa: E402


SUMMARY_SCHEMA = "speech_boundary_ja_frame_boundary_scorer_v3_prep"
DEFAULT_PTM_MODEL_PATH = "models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame"


def local_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_display_path(path: str | Path | None) -> str:
    if not path:
        return ""
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(PROJECT_ROOT)).replace("/", "\\")
    except ValueError:
        return str(raw)


def _ps_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def label_summary(labels: Path) -> dict[str, Any]:
    records = load_label_records(labels)
    source_counts = Counter(record.source for record in records)
    quality_counts = Counter(record.label_quality for record in records)
    cut_point_count = 0
    cut_drop_zone_count = 0
    metadata_records = 0
    for record in records:
        metadata = dict(record.boundary_metadata or {})
        if metadata:
            metadata_records += 1
        cut_point_count += len(list(metadata.get("cut_point_segments") or []))
        cut_drop_zone_count += len(list(metadata.get("cut_drop_zones") or []))
    return {
        "labels": repo_display_path(labels),
        "records": len(records),
        "metadata_records": metadata_records,
        "cut_point_segments": cut_point_count,
        "cut_drop_zones": cut_drop_zone_count,
        "source_counts": dict(sorted(source_counts.items())),
        "label_quality_counts": dict(sorted(quality_counts.items())),
    }


def feature_cache_script(
    *,
    labels: Path,
    manifest: Path,
    output_dir: Path,
    ptm: str,
    model_path: str,
    device: str,
    batch_size: int,
    prepare_workers: int,
) -> str:
    lines = [
        "$env:PYTHONIOENCODING='utf-8'",
        "uv run python -m tools.boundary.ja.build_feature_cache `",
        f"  --labels {_ps_literal(repo_display_path(labels))} `",
        f"  --manifest {_ps_literal(repo_display_path(manifest))} `",
        f"  --output-dir {_ps_literal(repo_display_path(output_dir))} `",
        f"  --ptm {_ps_literal(ptm)} `",
    ]
    if model_path:
        lines.append(f"  --model-path {_ps_literal(repo_display_path(model_path))} `")
    lines.extend(
        [
            f"  --device {_ps_literal(device)} `",
            "  --dtype 'bfloat16' `",
            f"  --batch-size {int(batch_size)} `",
            f"  --prepare-workers {int(prepare_workers)} `",
            "  --no-compress `",
            "  --log-every 25",
            "",
        ]
    )
    return "\n".join(lines)


def train_script(
    *,
    labels: Path,
    feature_manifest: Path,
    output_dir: Path,
    checkpoint_name: str,
    device: str,
    max_steps: int,
    batch_hidden_size: int,
    positive_weight: float,
    negative_weight: float,
    cut_positive_weight: float,
    cut_negative_weight: float,
    cut_loss_weight: float,
    focal_gamma: float,
) -> str:
    return "\n".join(
        [
            "$env:PYTHONIOENCODING='utf-8'",
            "uv run python -m tools.boundary.ja.train_feature_scorer `",
            f"  --labels {_ps_literal(repo_display_path(labels))} `",
            f"  --feature-manifest {_ps_literal(repo_display_path(feature_manifest))} `",
            f"  --output-dir {_ps_literal(repo_display_path(output_dir))} `",
            f"  --checkpoint-name {_ps_literal(checkpoint_name)} `",
            f"  --device {_ps_literal(device)} `",
            f"  --max-steps {int(max_steps)} `",
            f"  --hidden-size {int(batch_hidden_size)} `",
            "  --num-layers 2 `",
            "  --state-size 32 `",
            "  --num-heads 4 `",
            "  --n-groups 2 `",
            "  --chunk-size 8 `",
            f"  --positive-weight {float(positive_weight)} `",
            f"  --negative-weight {float(negative_weight)} `",
            f"  --cut-positive-weight {float(cut_positive_weight)} `",
            f"  --cut-negative-weight {float(cut_negative_weight)} `",
            f"  --cut-loss-weight {float(cut_loss_weight)} `",
            "  --cut-min-gap-s 0.5 `",
            "  --cut-boundary-radius-frames 1 `",
            f"  --focal-gamma {float(focal_gamma)} `",
            "  --eval-ratio 0.1 `",
            "  --threshold 0.5 `",
            "  --cut-threshold 0.5",
            "",
        ]
    )


def eval_script(
    *,
    checkpoint: Path,
    labels: Path,
    feature_manifest: Path,
    output_dir: Path,
    eval_device: str,
    batch_size: int,
    runtime_profiles: list[str],
) -> str:
    lines = [
        "$env:PYTHONIOENCODING='utf-8'",
        "uv run python -m tools.boundary.ja.evaluate_feature_scorer_thresholds `",
        f"  --checkpoint {_ps_literal(repo_display_path(checkpoint))} `",
        f"  --labels {_ps_literal(repo_display_path(labels))} `",
        f"  --feature-manifest {_ps_literal(repo_display_path(feature_manifest))} `",
        f"  --output-dir {_ps_literal(repo_display_path(output_dir))} `",
        f"  --device {_ps_literal(eval_device)} `",
        f"  --batch-size {int(batch_size)}",
    ]
    for profile in runtime_profiles:
        lines[-1] += " `"
        lines.append(f"  --runtime-profile {_ps_literal(profile)}")
    lines.append("")
    return "\n".join(lines)


def prepare_frame_boundary_scorer_v3(
    *,
    labels: Path,
    manifest: Path,
    output_dir: Path,
    ptm: str = QWEN_ASR_REPO_ID,
    model_path: str = DEFAULT_PTM_MODEL_PATH,
    device: str = "cuda",
    batch_size: int = 64,
    prepare_workers: int = 2,
    max_steps: int = 1000,
    hidden_size: int = 128,
    positive_weight: float = 1.0,
    negative_weight: float = 15.0,
    cut_positive_weight: float = 4.0,
    cut_negative_weight: float = 1.0,
    cut_loss_weight: float = 1.0,
    focal_gamma: float = 2.0,
    eval_device: str = "cpu",
    eval_batch_size: int = 1,
    runtime_profiles: list[str] | None = None,
    allow_no_cut_targets: bool = False,
) -> dict[str, Any]:
    labels = labels.resolve()
    manifest = manifest.resolve()
    output_dir = output_dir.resolve()
    if not labels.exists():
        raise FileNotFoundError(labels)
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    summary = label_summary(labels)
    cut_targets = int(summary["cut_point_segments"]) + int(summary["cut_drop_zones"])
    if cut_targets <= 0 and not allow_no_cut_targets:
        raise ValueError("v3 first-scorer labels must contain cut_point_segments or cut_drop_zones")
    feature_dir = output_dir / "feature-cache"
    scorer_dir = output_dir / "frame-boundary-scorer-v3"
    eval_dir = output_dir / "threshold-eval"
    feature_manifest = feature_dir / "feature_manifest.jsonl"
    ptm_repo_tag = qwen_asr_repo_tag(ptm)
    checkpoint_name = f"speech_boundary_ja_feature_scorer.{ptm_repo_tag}.pt"
    checkpoint = scorer_dir / checkpoint_name

    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        output_dir / "build_feature_cache.ps1",
        feature_cache_script(
            labels=labels,
            manifest=manifest,
            output_dir=feature_dir,
            ptm=ptm,
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            prepare_workers=prepare_workers,
        ),
    )
    write_text(
        output_dir / "train_frame_boundary_scorer_v3.ps1",
        train_script(
            labels=labels,
            feature_manifest=feature_manifest,
            output_dir=scorer_dir,
            checkpoint_name=checkpoint_name,
            device=device,
            max_steps=max_steps,
            batch_hidden_size=hidden_size,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
            cut_positive_weight=cut_positive_weight,
            cut_negative_weight=cut_negative_weight,
            cut_loss_weight=cut_loss_weight,
            focal_gamma=focal_gamma,
        ),
    )
    write_text(
        output_dir / "evaluate_frame_boundary_scorer_v3.ps1",
        eval_script(
            checkpoint=checkpoint,
            labels=labels,
            feature_manifest=feature_manifest,
            output_dir=eval_dir,
            eval_device=eval_device,
            batch_size=eval_batch_size,
            runtime_profiles=list(runtime_profiles or []),
        ),
    )
    payload = {
        "schema": SUMMARY_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": repo_display_path(output_dir),
        "label_summary": summary,
        "runtime_caveat": {
            "default_replaced": False,
            "opt_in_env": "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO",
            "required_schema": "speech_boundary_ja_mamba2_frame_boundary_scorer_v3",
        },
        "commands": {
            "build_feature_cache": repo_display_path(output_dir / "build_feature_cache.ps1"),
            "train_frame_boundary_scorer_v3": repo_display_path(output_dir / "train_frame_boundary_scorer_v3.ps1"),
            "evaluate_frame_boundary_scorer_v3": repo_display_path(output_dir / "evaluate_frame_boundary_scorer_v3.ps1"),
        },
        "outputs": {
            "feature_manifest": repo_display_path(feature_manifest),
            "checkpoint": repo_display_path(checkpoint),
            "threshold_eval": repo_display_path(eval_dir),
        },
        "training_config": {
            "ptm": str(ptm),
            "ptm_repo_tag": ptm_repo_tag,
            "model_path": repo_display_path(model_path),
            "checkpoint_name": checkpoint_name,
            "max_steps": int(max_steps),
            "hidden_size": int(hidden_size),
            "positive_weight": float(positive_weight),
            "negative_weight": float(negative_weight),
            "cut_positive_weight": float(cut_positive_weight),
            "cut_negative_weight": float(cut_negative_weight),
            "cut_loss_weight": float(cut_loss_weight),
            "focal_gamma": float(focal_gamma),
        },
        "eval_config": {
            "device": str(eval_device),
            "batch_size": int(eval_batch_size),
            "runtime_profiles": list(runtime_profiles or []),
        },
    }
    write_json(output_dir / "summary.json", payload)
    write_text(output_dir / "summary.md", render_markdown(payload))
    return payload


def render_markdown(summary: Mapping[str, Any]) -> str:
    labels = summary["label_summary"]
    lines = [
        "# SpeechBoundary-JA Frame Boundary Scorer v3 Prep",
        "",
        f"- Output: `{summary['output_dir']}`",
        f"- Labels: `{labels['labels']}`",
        f"- Records: `{labels['records']}`",
        f"- Metadata records: `{labels['metadata_records']}`",
        f"- Cut point segments: `{labels['cut_point_segments']}`",
        f"- Cut drop zones: `{labels['cut_drop_zones']}`",
        f"- PTM: `{summary['training_config']['ptm']}`",
        f"- PTM repo tag: `{summary['training_config']['ptm_repo_tag']}`",
        f"- PTM model path: `{summary['training_config']['model_path']}`",
        f"- Checkpoint: `{summary['outputs']['checkpoint']}`",
        f"- Training weights: `speech +{summary['training_config']['positive_weight']} / -{summary['training_config']['negative_weight']}; cut +{summary['training_config']['cut_positive_weight']} / -{summary['training_config']['cut_negative_weight']}`",
        f"- Eval device: `{summary['eval_config']['device']}`",
        f"- Eval batch size: `{summary['eval_config']['batch_size']}`",
        f"- Eval runtime profiles: `{len(summary['eval_config']['runtime_profiles'])}`",
        "",
        "## Scripts",
        "",
    ]
    for key, value in summary["commands"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "This prep only writes scripts. It does not build feature cache, train, evaluate, or replace runtime defaults.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare scripts for SpeechBoundary-JA Mamba2 speech+cut frame boundary scorer v3."
    )
    parser.add_argument("--labels", required=True, help="Synthetic true-structure SpeechBoundary-JA labels JSONL.")
    parser.add_argument("--manifest", required=True, help="Audio manifest matching labels.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--ptm", default=QWEN_ASR_REPO_ID)
    parser.add_argument("--model-path", default=DEFAULT_PTM_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prepare-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=15.0)
    parser.add_argument("--cut-positive-weight", type=float, default=4.0)
    parser.add_argument("--cut-negative-weight", type=float, default=1.0)
    parser.add_argument("--cut-loss-weight", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--eval-device",
        default="cpu",
        help="Threshold-eval device. Defaults to CPU so offline sweeps do not consume GPU/shared VRAM.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--runtime-profile",
        action="append",
        default=[],
        help="Add a threshold eval runtime profile as speech_on,speech_off,cut.",
    )
    parser.add_argument("--allow-no-cut-targets", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.prepare_workers < 0:
        parser.error("--prepare-workers must be non-negative")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    if args.positive_weight <= 0.0:
        parser.error("--positive-weight must be positive")
    if args.negative_weight <= 0.0:
        parser.error("--negative-weight must be positive")
    if args.cut_positive_weight <= 0.0:
        parser.error("--cut-positive-weight must be positive")
    if args.cut_negative_weight <= 0.0:
        parser.error("--cut-negative-weight must be positive")
    if args.cut_loss_weight <= 0.0:
        parser.error("--cut-loss-weight must be positive")
    if args.focal_gamma < 0.0:
        parser.error("--focal-gamma must be non-negative")
    if args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be positive")
    for profile in args.runtime_profile:
        parts = [part.strip() for part in str(profile).split(",")]
        if len(parts) != 3:
            parser.error("--runtime-profile must be speech_on,speech_off,cut")
        try:
            on_threshold, off_threshold, cut_threshold = (float(part) for part in parts)
        except ValueError:
            parser.error("--runtime-profile values must be numbers")
        if not (0.0 <= on_threshold <= 1.0 and 0.0 <= off_threshold <= 1.0 and 0.0 <= cut_threshold <= 1.0):
            parser.error("--runtime-profile values must be in [0, 1]")
        if on_threshold < off_threshold:
            parser.error("--runtime-profile speech_on must be >= speech_off")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT
        / "agents"
        / "temp"
        / f"{local_timestamp()}_speech-boundary-frame-boundary-scorer-v3-{qwen_asr_repo_tag(args.ptm)}-prep"
    )
    summary = prepare_frame_boundary_scorer_v3(
        labels=project_path(args.labels),
        manifest=project_path(args.manifest),
        output_dir=output_dir,
        ptm=args.ptm,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        prepare_workers=args.prepare_workers,
        max_steps=args.max_steps,
        hidden_size=args.hidden_size,
        positive_weight=args.positive_weight,
        negative_weight=args.negative_weight,
        cut_positive_weight=args.cut_positive_weight,
        cut_negative_weight=args.cut_negative_weight,
        cut_loss_weight=args.cut_loss_weight,
        focal_gamma=args.focal_gamma,
        eval_device=args.eval_device,
        eval_batch_size=args.eval_batch_size,
        runtime_profiles=args.runtime_profile,
        allow_no_cut_targets=args.allow_no_cut_targets,
    )
    print(f"output_dir={summary['output_dir']}")
    print(f"summary={summary['output_dir']}\\summary.json")
    print(f"build_feature_cache={summary['commands']['build_feature_cache']}")
    print(f"train_frame_boundary_scorer_v3={summary['commands']['train_frame_boundary_scorer_v3']}")
    print(f"evaluate_frame_boundary_scorer_v3={summary['commands']['evaluate_frame_boundary_scorer_v3']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
