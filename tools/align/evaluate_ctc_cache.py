#!/usr/bin/env python3
"""Evaluate an alignment checkpoint on a frozen feature cache.

Unlike training loss, this reports text and blank-only rows separately.  When a
teacher manifest is supplied, held-out lexical crops also report absolute edge
error against the teacher island after applying the production speech-extent
walk.  It is still teacher agreement, not human truth, but it reveals whether a
mixed head learned boundaries or merely raised blank everywhere.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment import (  # noqa: E402
    ALIGNMENT_MODEL_SCHEMA,
    BLANK_INDEX,
    AlignmentVocab,
    align_text,
    build_head,
    speech_extent,
)
from tools.align.train_ctc_aligner import FeatureCache, _collate  # noqa: E402
from tools.align.frame_teacher_supervision import (  # noqa: E402
    IGNORE_LABEL,
    compile_sparse_frame_targets,
    load_accepted_frame_teachers,
    summarize_sparse_frame_probabilities,
)
from utils.gpu_safety import apply_vram_safety_cap  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _describe_ms(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64) * 1000.0
    return {
        "count": len(values),
        "median_ms": round(float(np.median(array)), 2),
        "p90_ms": round(float(np.percentile(array, 90)), 2),
        "p99_ms": round(float(np.percentile(array, 99)), 2),
        "mean_ms": round(float(np.mean(array)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", default="val")
    parser.add_argument("--teacher-manifest", default="")
    parser.add_argument("--frame-teacher-results", default="")
    parser.add_argument("--frame-teacher-manifest", default="")
    parser.add_argument("--frame-positive-merge-gap-s", type=float, default=0.15)
    parser.add_argument("--frame-boundary-ignore-s", type=float, default=0.10)
    parser.add_argument("--frame-negative-min-s", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if bool(args.frame_teacher_results) != bool(args.frame_teacher_manifest):
        raise SystemExit(
            "--frame-teacher-results and --frame-teacher-manifest must be given together"
        )

    import torch
    from torch import nn

    apply_vram_safety_cap(0.95)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if str(payload.get("schema") or "") != ALIGNMENT_MODEL_SCHEMA:
        raise SystemExit(f"not an alignment checkpoint: {payload.get('schema')!r}")
    vocab = AlignmentVocab.from_payload(payload["vocab"])
    upsample = int(payload["upsample"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = build_head(
        vocab_size=vocab.size,
        input_dim=int(payload.get("input_dim", 2048)),
        hidden_dim=int(payload["hidden_dim"]),
        upsample=upsample,
        blocks=int(payload["blocks"]),
        dropout=0.0,
    )
    head.load_state_dict(payload["state_dict"])
    head.to(device).eval()

    cache = FeatureCache([Path(args.cache_dir)], domains=["evaluation"])
    rows = [row for row in cache.rows if row.get("partition") == args.partition]
    if not rows:
        raise SystemExit(f"cache has no {args.partition!r} rows")
    teacher = {}
    if args.teacher_manifest:
        teacher = {
            str(row["audio_id"]): row
            for row in _read_jsonl(Path(args.teacher_manifest))
        }
    frame_teachers = {}
    if args.frame_teacher_results:
        frame_teachers, _ = load_accepted_frame_teachers(
            Path(args.frame_teacher_results), Path(args.frame_teacher_manifest)
        )

    criterion = nn.CTCLoss(blank=BLANK_INDEX, reduction="none", zero_infinity=True)
    measures: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    edge_errors: dict[str, list[float]] = defaultdict(list)
    frame_blank_probabilities: list[np.ndarray] = []
    frame_teacher_labels: list[np.ndarray] = []
    align_failures = 0
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        encoded = [vocab.encode(str(row.get("text") or "")) for row in chunk]
        items = [(cache.features(row), target) for row, target in zip(chunk, encoded)]
        features, targets, frame_lengths, target_lengths = _collate(items, torch)
        with torch.inference_mode():
            log_probs = head(features.to(device), frame_lengths)
            losses = criterion(
                log_probs.transpose(0, 1),
                targets.to(device),
                (frame_lengths * upsample).to(device),
                target_lengths.to(device),
            ).float().cpu().numpy()
            probabilities = log_probs.exp().float().cpu()
        for index, (row, loss) in enumerate(zip(chunk, losses)):
            kind = "text" if str(row.get("text") or "") else "blank"
            frames = int(frame_lengths[index]) * upsample
            item = probabilities[index, :frames]
            measures[kind]["ctc_loss"].append(float(loss))
            measures[kind]["blank_argmax_rate"].append(
                float((item.argmax(dim=-1) == BLANK_INDEX).float().mean().item())
            )
            measures[kind]["mean_blank_probability"].append(
                float(item[:, BLANK_INDEX].mean().item())
            )
            source_id = str(row.get("source_id") or row["audio_id"])
            if source_id in frame_teachers:
                labels = compile_sparse_frame_targets(
                    frame_teachers[source_id],
                    output_frames=frames,
                    upsample=upsample,
                    positive_merge_gap_s=args.frame_positive_merge_gap_s,
                    boundary_ignore_s=args.frame_boundary_ignore_s,
                    negative_minimum_s=args.frame_negative_min_s,
                )
                valid = labels != IGNORE_LABEL
                if np.any(valid):
                    frame_blank_probabilities.append(
                        item[:, BLANK_INDEX].numpy()[valid]
                    )
                    frame_teacher_labels.append(labels[valid])
            if kind != "text" or str(row["audio_id"]) not in teacher:
                continue
            truth = teacher[str(row["audio_id"])]
            try:
                spans = align_text(
                    log_probs[index, :frames].float().cpu(),
                    str(row["text"]),
                    vocab,
                    upsample=upsample,
                )
            except (ValueError, RuntimeError):
                align_failures += 1
                continue
            extent = speech_extent(
                log_probs[index, :frames].float().cpu(), spans, upsample=upsample
            )
            if extent is None:
                align_failures += 1
                continue
            crop_start = float(truth["source_start_s"])
            teacher_start = float(truth["teacher_start_s"]) - crop_start
            teacher_end = float(truth["teacher_end_s"]) - crop_start
            edge_errors["onset_absolute"].append(abs(extent[0] - teacher_start))
            edge_errors["end_absolute"].append(abs(extent[1] - teacher_end))
            edge_errors["onset_signed"].append(extent[0] - teacher_start)
            edge_errors["end_signed"].append(extent[1] - teacher_end)

    result = {
        "schema": "asr_ctc_cache_evaluation_v1",
        "checkpoint": str(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "partition": args.partition,
        "rows": len(rows),
        "rows_by_target_kind": {
            kind: len(values["ctc_loss"]) for kind, values in sorted(measures.items())
        },
        "metrics_by_target_kind": {
            kind: {
                metric: round(statistics.fmean(values), 6)
                for metric, values in sorted(metrics.items())
            }
            for kind, metrics in sorted(measures.items())
        },
        "teacher_edge_error": {
            metric: _describe_ms(values) for metric, values in sorted(edge_errors.items())
        },
        "sparse_frame_teacher": (
            summarize_sparse_frame_probabilities(
                np.concatenate(frame_blank_probabilities),
                np.concatenate(frame_teacher_labels),
            )
            if frame_blank_probabilities
            else {}
        ),
        "alignment_failures": align_failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
