#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.ja import build_training_examples, load_label_records, load_manifest_audio_map, write_jsonl  # noqa: E402


SUMMARY_SCHEMA = "speech_boundary_hard_negative_finetune_prep_v1"
DEFAULT_MIN_NEGATIVES = 300
DEFAULT_MIN_POSITIVE_ANCHORS = 1000
DEFAULT_MAX_NEGATIVE_SHARE = 0.35


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


def local_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(dict(row))
    return rows


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"JSON manifest must be a list: {path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"JSON manifest row must be an object: {path}:{index}")
        rows.append(dict(row))
    return rows


def discover_latest_sources() -> tuple[Path, Path]:
    paths = sorted(
        (PROJECT_ROOT / "agents" / "temp").glob(
            "*_boundary-v5.1-sources-from-cueqc/speech_boundary_negative_labels.jsonl"
        )
    )
    if not paths:
        raise FileNotFoundError(
            "no speech_boundary_negative_labels.jsonl found under agents/temp/*_boundary-v5.1-sources-from-cueqc"
        )
    labels = paths[-1]
    manifest = labels.with_name("speech_boundary_negative_manifest.json")
    return labels, manifest


def label_summary(labels_path: Path, manifest_path: Path | None) -> dict[str, Any]:
    records = load_label_records(labels_path)
    audio_map = load_manifest_audio_map(manifest_path)
    examples, skipped = build_training_examples(
        records,
        manifest_audio_map=audio_map,
        audio_root=None,
        trainable_only=True,
    )
    quality_counts = Counter(record.label_quality for record in records)
    source_counts = Counter(record.source for record in records)
    frame_count = sum(len(record.speech_frames) for record in records)
    speech_frames = sum(sum(int(value) for value in record.speech_frames) for record in records)
    duration_s = sum(float(record.duration_s) for record in records)
    return {
        "labels_path": repo_display_path(labels_path),
        "manifest_path": repo_display_path(manifest_path) if manifest_path else "",
        "records": len(records),
        "trainable_examples": len(examples),
        "skipped": len(skipped),
        "skipped_reasons": dict(Counter(str(row.get("reason") or "") for row in skipped)),
        "label_quality_counts": dict(quality_counts),
        "source_counts": dict(source_counts),
        "frame_count": frame_count,
        "speech_frame_count": speech_frames,
        "speech_frame_ratio": speech_frames / frame_count if frame_count else 0.0,
        "duration_s": round(duration_s, 6),
    }


def _ps_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def merge_manifests(
    *,
    negative_manifest: Path | None,
    anchor_manifests: Sequence[Path],
    output_path: Path,
) -> int:
    rows: list[dict[str, Any]] = []
    rows.extend(read_json(negative_manifest))
    for path in anchor_manifests:
        rows.extend(read_json(path))
    write_json(output_path, rows)
    return len(rows)


def merge_labels(
    *,
    negative_labels: Path,
    anchor_labels: Sequence[Path],
    output_path: Path,
) -> int:
    records = load_label_records(negative_labels)
    for path in anchor_labels:
        records.extend(load_label_records(path))
    write_jsonl(output_path, records)
    return len(records)


def feature_cache_script(
    *,
    labels: Path,
    manifest: Path | None,
    output_dir: Path,
    device: str,
    batch_size: int,
    limit: int | None = None,
) -> str:
    lines = [
        "$env:PYTHONIOENCODING='utf-8'",
        "uv run python -m tools.boundary.ja.build_feature_cache `",
        f"  --labels {_ps_literal(repo_display_path(labels))} `",
    ]
    if manifest:
        lines.append(f"  --manifest {_ps_literal(repo_display_path(manifest))} `")
    lines.extend(
        [
            f"  --output-dir {_ps_literal(repo_display_path(output_dir))} `",
            f"  --device {_ps_literal(device)} `",
            "  --dtype 'bfloat16' `",
            f"  --batch-size {int(batch_size)} `",
            "  --prepare-workers 2 `",
            "  --no-compress `",
            "  --log-every 25",
        ]
    )
    if limit is not None:
        lines[-1] += " `"
        lines.append(f"  --limit {int(limit)}")
    return "\n".join(lines) + "\n"


def tiny_smoke_script(
    *,
    labels: Path,
    manifest: Path | None,
    output_dir: Path,
) -> str:
    lines = [
        "$env:PYTHONIOENCODING='utf-8'",
        "uv run python -m tools.boundary.ja.train_tiny `",
        f"  --labels {_ps_literal(repo_display_path(labels))} `",
    ]
    if manifest:
        lines.append(f"  --manifest {_ps_literal(repo_display_path(manifest))} `")
    lines.extend(
        [
            f"  --output-dir {_ps_literal(repo_display_path(output_dir))} `",
            "  --device 'cpu' `",
            "  --max-steps 1 `",
            "  --window-s 1.0",
        ]
    )
    return "\n".join(lines) + "\n"


def formal_tiny_train_script(
    *,
    labels: Path,
    manifest: Path | None,
    output_dir: Path,
    device: str,
    max_steps: int,
) -> str:
    lines = [
        "$env:PYTHONIOENCODING='utf-8'",
        "# Research/plumbing model only. Do not promote as SpeechBoundary-JA runtime.",
        "uv run python -m tools.boundary.ja.train_tiny `",
        f"  --labels {_ps_literal(repo_display_path(labels))} `",
    ]
    if manifest:
        lines.append(f"  --manifest {_ps_literal(repo_display_path(manifest))} `")
    lines.extend(
        [
            f"  --output-dir {_ps_literal(repo_display_path(output_dir))} `",
            f"  --device {_ps_literal(device)} `",
            f"  --max-steps {int(max_steps)} `",
            "  --window-s 1.0",
        ]
    )
    return "\n".join(lines) + "\n"


def feature_scorer_train_script(
    *,
    labels: Path,
    feature_manifest: Path,
    output_dir: Path,
    device: str,
    max_steps: int,
) -> str:
    lines = [
        "$env:PYTHONIOENCODING='utf-8'",
        "# Candidate scorer only. Runtime uses it only when SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT is set.",
        "uv run python -m tools.boundary.ja.train_feature_scorer `",
        f"  --labels {_ps_literal(repo_display_path(labels))} `",
        f"  --feature-manifest {_ps_literal(repo_display_path(feature_manifest))} `",
        f"  --output-dir {_ps_literal(repo_display_path(output_dir))} `",
        f"  --device {_ps_literal(device)} `",
        f"  --max-steps {int(max_steps)} `",
        "  --hidden-size 128 `",
        "  --dropout 0.05 `",
        "  --eval-ratio 0.1 `",
        "  --threshold 0.5",
    ]
    return "\n".join(lines) + "\n"


def prepare_hard_negative_finetune(
    *,
    negative_labels: Path,
    negative_manifest: Path | None,
    output_dir: Path,
    anchor_labels: Sequence[Path] = (),
    anchor_manifests: Sequence[Path] = (),
    min_negatives: int = DEFAULT_MIN_NEGATIVES,
    min_positive_anchors: int = DEFAULT_MIN_POSITIVE_ANCHORS,
    max_negative_share: float = DEFAULT_MAX_NEGATIVE_SHARE,
    device: str = "cuda",
    batch_size: int = 64,
    tiny_max_steps: int = 200,
) -> dict[str, Any]:
    if anchor_manifests and len(anchor_manifests) != len(anchor_labels):
        raise ValueError("--anchor-labels and --anchor-manifests must have the same count when manifests are provided")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    negative = label_summary(negative_labels, negative_manifest)
    anchors: list[dict[str, Any]] = []
    for index, labels in enumerate(anchor_labels):
        manifest = anchor_manifests[index] if anchor_manifests else None
        anchors.append(label_summary(labels, manifest))
    anchor_records = sum(int(item["records"]) for item in anchors)
    anchor_trainable = sum(int(item["trainable_examples"]) for item in anchors)
    anchor_positive_frames = sum(int(item["speech_frame_count"]) for item in anchors)
    negative_trainable = int(negative["trainable_examples"])
    total_trainable = negative_trainable + anchor_trainable
    negative_share = negative_trainable / total_trainable if total_trainable else 1.0

    gate_checks = {
        "negative_examples_ge_min": negative_trainable >= int(min_negatives),
        "negative_examples_all_resolved": int(negative["skipped"]) == 0,
        "positive_anchor_examples_ge_min": anchor_trainable >= int(min_positive_anchors),
        "positive_anchor_has_speech_frames": anchor_positive_frames > 0,
        "negative_share_le_max": negative_share <= float(max_negative_share),
    }
    formal_ready = all(gate_checks.values())

    feature_dir = output_dir / "feature-cache-negative-only"
    smoke_dir = output_dir / "tiny-smoke-negative-only"
    mixed_labels_path = output_dir / "speech_boundary_mixed_hard_negative_anchor_labels.jsonl"
    mixed_manifest_path = output_dir / "speech_boundary_mixed_hard_negative_anchor_manifest.json"
    mixed_training_manifest_path = output_dir / "speech_boundary_mixed_training_manifest.jsonl"
    mixed_training_skipped_path = output_dir / "speech_boundary_mixed_training_manifest_skipped.json"
    mixed_feature_dir = output_dir / "feature-cache-mixed-hard-negative-anchor"
    tiny_mixed_dir = output_dir / "tiny-mixed-hard-negative-anchor"
    feature_scorer_dir = output_dir / "feature-scorer-hard-negative-anchor"
    mixed_feature_manifest_path = mixed_feature_dir / "feature_manifest.jsonl"
    mixed_record_count = 0
    mixed_manifest_count = 0
    mixed_examples: list[Any] = []
    mixed_skipped: list[dict[str, Any]] = []
    if formal_ready:
        mixed_record_count = merge_labels(
            negative_labels=negative_labels,
            anchor_labels=anchor_labels,
            output_path=mixed_labels_path,
        )
        mixed_manifest_count = merge_manifests(
            negative_manifest=negative_manifest,
            anchor_manifests=anchor_manifests,
            output_path=mixed_manifest_path,
        )
        mixed_records = load_label_records(mixed_labels_path)
        mixed_examples, mixed_skipped = build_training_examples(
            mixed_records,
            manifest_audio_map=load_manifest_audio_map(mixed_manifest_path),
            audio_root=None,
            trainable_only=True,
        )
        with (mixed_training_manifest_path).open("w", encoding="utf-8") as handle:
            for example in mixed_examples:
                handle.write(json.dumps(example.__dict__, ensure_ascii=False, sort_keys=True) + "\n")
        write_json(mixed_training_skipped_path, mixed_skipped)

    write_text(
        output_dir / "build_negative_feature_cache.ps1",
        feature_cache_script(
            labels=negative_labels,
            manifest=negative_manifest,
            output_dir=feature_dir,
            device=device,
            batch_size=batch_size,
        ),
    )
    write_text(
        output_dir / "tiny_negative_plumbing_smoke.ps1",
        tiny_smoke_script(labels=negative_labels, manifest=negative_manifest, output_dir=smoke_dir),
    )
    if formal_ready:
        write_text(
            output_dir / "build_mixed_feature_cache.ps1",
            feature_cache_script(
                labels=mixed_labels_path,
                manifest=mixed_manifest_path,
                output_dir=mixed_feature_dir,
                device=device,
                batch_size=batch_size,
            ),
        )
        write_text(
            output_dir / "tiny_mixed_plumbing_train.ps1",
            formal_tiny_train_script(
                labels=mixed_labels_path,
                manifest=mixed_manifest_path,
                output_dir=tiny_mixed_dir,
                device=device,
                max_steps=tiny_max_steps,
            ),
        )
        write_text(
            output_dir / "train_mixed_feature_scorer.ps1",
            feature_scorer_train_script(
                labels=mixed_labels_path,
                feature_manifest=mixed_feature_manifest_path,
                output_dir=feature_scorer_dir,
                device=device,
                max_steps=tiny_max_steps,
            ),
        )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": repo_display_path(output_dir),
        "negative_source": negative,
        "anchor_sources": anchors,
        "mixed_source": {
            "emitted": formal_ready,
            "labels_path": repo_display_path(mixed_labels_path) if formal_ready else "",
            "manifest_path": repo_display_path(mixed_manifest_path) if formal_ready else "",
            "training_manifest_path": repo_display_path(mixed_training_manifest_path) if formal_ready else "",
            "training_manifest_skipped_path": repo_display_path(mixed_training_skipped_path) if formal_ready else "",
            "records": mixed_record_count,
            "manifest_rows": mixed_manifest_count,
            "trainable_examples": len(mixed_examples),
            "skipped": len(mixed_skipped),
        },
        "gate": {
            "formal_training_ready": formal_ready,
            "checks": gate_checks,
            "policy": {
                "min_negatives": int(min_negatives),
                "min_positive_anchors": int(min_positive_anchors),
                "max_negative_share": float(max_negative_share),
            },
            "negative_share": round(negative_share, 6),
            "reason": (
                "ready to build a mixed hard-negative finetune dataset"
                if formal_ready
                else "not ready: hard negatives must be mixed with enough positive/synthetic anchor examples before finetune"
            ),
        },
        "runtime_caveat": {
            "speech_boundary_runtime": "qwen-feature-energy-bootstrap-v1",
            "direct_replacement_checkpoint_supported": False,
            "opt_in_scorer_env": "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT",
            "reason": "feature scorer checkpoints are runtime-loadable only when explicitly enabled and still need workflow smoke plus human audit before promotion; train_tiny remains plumbing-only.",
        },
        "commands": {
            "negative_feature_cache_script": repo_display_path(output_dir / "build_negative_feature_cache.ps1"),
            "tiny_negative_plumbing_smoke": repo_display_path(output_dir / "tiny_negative_plumbing_smoke.ps1"),
            "mixed_feature_cache_script": (
                repo_display_path(output_dir / "build_mixed_feature_cache.ps1") if formal_ready else ""
            ),
            "tiny_mixed_plumbing_train": (
                repo_display_path(output_dir / "tiny_mixed_plumbing_train.ps1") if formal_ready else ""
            ),
            "mixed_feature_scorer_train": (
                repo_display_path(output_dir / "train_mixed_feature_scorer.ps1") if formal_ready else ""
            ),
        },
        "next_steps": [
            "add anchor positive/synthetic label sources before any formal SpeechBoundary-JA hard-negative finetune",
            "build a mixed sampling manifest with capped negative share",
            "run mixed feature-cache generation only after the anchor gate passes",
            "train a runtime-loadable feature scorer from the mixed feature cache",
            "gate any trained scorer with full workflow smoke and human audit before runtime promotion",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    write_text(output_dir / "summary.md", render_markdown(summary))
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    gate = summary["gate"]
    neg = summary["negative_source"]
    mixed = summary.get("mixed_source") or {}
    lines = [
        "# SpeechBoundary Hard-Negative Finetune Prep",
        "",
        f"- Output: `{summary['output_dir']}`",
        f"- Negative labels: `{neg['labels_path']}`",
        f"- Negative trainable examples: `{neg['trainable_examples']}`",
        f"- Anchor sources: `{len(summary['anchor_sources'])}`",
        f"- Mixed source emitted: `{mixed.get('emitted', False)}`",
        "",
        "## Gate",
        "",
        f"- Formal training ready: `{gate['formal_training_ready']}`",
        f"- Negative share: `{gate['negative_share']}`",
        f"- Checks: `{gate['checks']}`",
        f"- Reason: {gate['reason']}",
        "",
        "## Scripts",
        "",
    ]
    for key, value in summary["commands"].items():
        if value:
            lines.append(f"- {key}: `{value}`")
    if mixed.get("emitted"):
        lines.extend(
            [
                "",
                "## Mixed Source",
                "",
                f"- Labels: `{mixed.get('labels_path')}`",
                f"- Manifest: `{mixed.get('manifest_path')}`",
                f"- Trainable examples: `{mixed.get('trainable_examples')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- Current SpeechBoundary-JA runtime is a Qwen/MFCC bootstrap scorer, not a trained replaceable TinyFrameClassifier.",
            "- Feature scorer checkpoints are opt-in via `SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT`; generated tiny scripts only validate label/audio plumbing.",
            "- Do not train on the 522 negatives alone; mix with positive/synthetic anchors first.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and gate SpeechBoundary-JA hard-negative finetune sources."
    )
    parser.add_argument("--negative-labels", default="")
    parser.add_argument("--negative-manifest", default="")
    parser.add_argument("--anchor-labels", action="append", default=[])
    parser.add_argument("--anchor-manifests", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to agents/temp/YYYYMMDD_HHMMSS_speech-boundary-hard-negative-finetune-prep.",
    )
    parser.add_argument("--min-negatives", type=int, default=DEFAULT_MIN_NEGATIVES)
    parser.add_argument("--min-positive-anchors", type=int, default=DEFAULT_MIN_POSITIVE_ANCHORS)
    parser.add_argument("--max-negative-share", type=float, default=DEFAULT_MAX_NEGATIVE_SHARE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tiny-max-steps", type=int, default=200)
    args = parser.parse_args(argv)
    if args.min_negatives <= 0:
        parser.error("--min-negatives must be positive")
    if args.min_positive_anchors < 0:
        parser.error("--min-positive-anchors must be non-negative")
    if not 0.0 < args.max_negative_share <= 1.0:
        parser.error("--max-negative-share must be in (0, 1]")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.tiny_max_steps <= 0:
        parser.error("--tiny-max-steps must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.negative_labels:
        negative_labels = project_path(args.negative_labels)
        negative_manifest = project_path(args.negative_manifest) if args.negative_manifest else None
    else:
        negative_labels, discovered_manifest = discover_latest_sources()
        negative_manifest = project_path(args.negative_manifest) if args.negative_manifest else discovered_manifest
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT
        / "agents"
        / "temp"
        / f"{local_timestamp()}_speech-boundary-hard-negative-finetune-prep"
    )
    summary = prepare_hard_negative_finetune(
        negative_labels=negative_labels,
        negative_manifest=negative_manifest,
        output_dir=output_dir,
        anchor_labels=[project_path(path) for path in args.anchor_labels],
        anchor_manifests=[project_path(path) for path in args.anchor_manifests],
        min_negatives=args.min_negatives,
        min_positive_anchors=args.min_positive_anchors,
        max_negative_share=args.max_negative_share,
        device=args.device,
        batch_size=args.batch_size,
        tiny_max_steps=args.tiny_max_steps,
    )
    print(f"output_dir={summary['output_dir']}")
    print(f"summary={summary['output_dir']}\\summary.json")
    print(f"negative_examples={summary['negative_source']['trainable_examples']}")
    print(f"formal_training_ready={summary['gate']['formal_training_ready']}")
    print(f"gate_checks={json.dumps(summary['gate']['checks'], ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
