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

from boundary.ja import load_label_records  # noqa: E402


SUMMARY_SCHEMA = "speech_boundary_ja_frame_boundary_scorer_v3_prep"


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
    device: str,
    batch_size: int,
    prepare_workers: int,
) -> str:
    return "\n".join(
        [
            "$env:PYTHONIOENCODING='utf-8'",
            "uv run python -m tools.boundary.ja.build_feature_cache `",
            f"  --labels {_ps_literal(repo_display_path(labels))} `",
            f"  --manifest {_ps_literal(repo_display_path(manifest))} `",
            f"  --output-dir {_ps_literal(repo_display_path(output_dir))} `",
            f"  --device {_ps_literal(device)} `",
            "  --dtype 'bfloat16' `",
            f"  --batch-size {int(batch_size)} `",
            f"  --prepare-workers {int(prepare_workers)} `",
            "  --no-compress `",
            "  --log-every 25",
            "",
        ]
    )


def train_script(
    *,
    labels: Path,
    feature_manifest: Path,
    output_dir: Path,
    device: str,
    max_steps: int,
    batch_hidden_size: int,
    cut_positive_weight: float,
    cut_loss_weight: float,
) -> str:
    return "\n".join(
        [
            "$env:PYTHONIOENCODING='utf-8'",
            "uv run python -m tools.boundary.ja.train_feature_scorer `",
            f"  --labels {_ps_literal(repo_display_path(labels))} `",
            f"  --feature-manifest {_ps_literal(repo_display_path(feature_manifest))} `",
            f"  --output-dir {_ps_literal(repo_display_path(output_dir))} `",
            f"  --device {_ps_literal(device)} `",
            f"  --max-steps {int(max_steps)} `",
            f"  --hidden-size {int(batch_hidden_size)} `",
            "  --num-layers 2 `",
            "  --state-size 32 `",
            "  --num-heads 4 `",
            "  --n-groups 2 `",
            "  --chunk-size 8 `",
            "  --positive-weight 1.0 `",
            "  --negative-weight 15.0 `",
            f"  --cut-positive-weight {float(cut_positive_weight)} `",
            "  --cut-negative-weight 1.0 `",
            f"  --cut-loss-weight {float(cut_loss_weight)} `",
            "  --cut-min-gap-s 0.5 `",
            "  --cut-boundary-radius-frames 1 `",
            "  --focal-gamma 2.0 `",
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
    device: str,
) -> str:
    return "\n".join(
        [
            "$env:PYTHONIOENCODING='utf-8'",
            "uv run python -m tools.boundary.ja.evaluate_feature_scorer_thresholds `",
            f"  --checkpoint {_ps_literal(repo_display_path(checkpoint))} `",
            f"  --labels {_ps_literal(repo_display_path(labels))} `",
            f"  --feature-manifest {_ps_literal(repo_display_path(feature_manifest))} `",
            f"  --output-dir {_ps_literal(repo_display_path(output_dir))} `",
            f"  --device {_ps_literal(device)}",
            "",
        ]
    )


def prepare_frame_boundary_scorer_v3(
    *,
    labels: Path,
    manifest: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 64,
    prepare_workers: int = 2,
    max_steps: int = 1000,
    hidden_size: int = 128,
    cut_positive_weight: float = 4.0,
    cut_loss_weight: float = 1.0,
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
    checkpoint = scorer_dir / "speech_boundary_ja_feature_scorer.pt"

    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        output_dir / "build_feature_cache.ps1",
        feature_cache_script(
            labels=labels,
            manifest=manifest,
            output_dir=feature_dir,
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
            device=device,
            max_steps=max_steps,
            batch_hidden_size=hidden_size,
            cut_positive_weight=cut_positive_weight,
            cut_loss_weight=cut_loss_weight,
        ),
    )
    write_text(
        output_dir / "evaluate_frame_boundary_scorer_v3.ps1",
        eval_script(
            checkpoint=checkpoint,
            labels=labels,
            feature_manifest=feature_manifest,
            output_dir=eval_dir,
            device=device,
        ),
    )
    payload = {
        "schema": SUMMARY_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": repo_display_path(output_dir),
        "label_summary": summary,
        "runtime_caveat": {
            "default_replaced": False,
            "opt_in_env": "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT",
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prepare-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--cut-positive-weight", type=float, default=4.0)
    parser.add_argument("--cut-loss-weight", type=float, default=1.0)
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
    if args.cut_positive_weight <= 0.0:
        parser.error("--cut-positive-weight must be positive")
    if args.cut_loss_weight <= 0.0:
        parser.error("--cut-loss-weight must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "agents" / "temp" / f"{local_timestamp()}_speech-boundary-frame-boundary-scorer-v3-prep"
    )
    summary = prepare_frame_boundary_scorer_v3(
        labels=project_path(args.labels),
        manifest=project_path(args.manifest),
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
        prepare_workers=args.prepare_workers,
        max_steps=args.max_steps,
        hidden_size=args.hidden_size,
        cut_positive_weight=args.cut_positive_weight,
        cut_loss_weight=args.cut_loss_weight,
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
