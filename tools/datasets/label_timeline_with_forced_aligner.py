#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402


SCHEMA = "timeline_forced_aligner_label_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _existing_ids(path: Path) -> set[str]:
    return {str(row["item_id"]) for row in _read_jsonl(path)} if path.exists() else set()


def _batches(rows: list[dict[str, Any]], batch_size: int, max_batch_s: float):
    current: list[dict[str, Any]] = []
    duration = 0.0
    for row in sorted(rows, key=lambda item: float(item["duration_s"])):
        item_duration = float(row["duration_s"])
        if current and (len(current) >= batch_size or duration + item_duration > max_batch_s):
            yield current
            current = []
            duration = 0.0
        current.append(row)
        duration += item_duration
    if current:
        yield current


def _prepare_batch_inputs(processor, *, audio, transcripts):
    return processor.prepare_forced_aligner_inputs(
        audio=audio,
        transcript=transcripts,
        language=["Japanese"] * len(transcripts),
    )


def _build_anchors(*, item, words, word_list, point_confidence):
    if len(words) != len(word_list):
        raise RuntimeError(
            f"forced-aligner decoded {len(words)} units for {len(word_list)} expected units"
        )
    expected_points = len(word_list) * 2
    if len(point_confidence) != expected_points:
        raise RuntimeError(
            f"forced-aligner returned {len(point_confidence)} timestamp confidences; "
            f"expected {expected_points}"
        )
    anchors = []
    for word_index, (word, expected_text) in enumerate(zip(words, word_list)):
        start_conf = float(point_confidence[word_index * 2])
        end_conf = float(point_confidence[word_index * 2 + 1])
        start_s = float(word["start_time"])
        end_s = float(word["end_time"])
        anchors.append(
            {
                "unit_id": f"u{word_index:04d}",
                "text": str(expected_text),
                "start_s": start_s,
                "end_s": end_s,
                "absolute_start_s": float(item["absolute_start_s"]) + start_s,
                "absolute_end_s": float(item["absolute_start_s"]) + end_s,
                "start_score": start_conf,
                "end_score": end_conf,
                "alignment_score": min(start_conf, end_conf),
            }
        )
    return anchors


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoProcessor

    rows = _read_jsonl(Path(args.items))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "forced_aligner_labels.jsonl"
    existing = _existing_ids(labels_path)
    pending = [row for row in rows if str(row["item_id"]) not in existing]
    if args.limit > 0:
        pending = pending[: args.limit]
    if not pending:
        return {"schema": "timeline_forced_aligner_summary_v1", "pending": 0}

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for forced-aligner teacher labeling")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    applied_ratio = apply_vram_safety_cap(args.vram_safety_ratio)
    print(f"forced-aligner processor load start model={args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    print("forced-aligner processor load done; model load start", flush=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=args.attention,
        low_cpu_mem_usage=True,
    )
    print("forced-aligner model weights loaded; cuda transfer start", flush=True)
    model = model.to(device)
    print("forced-aligner cuda transfer done", flush=True)
    model.eval()

    processed = 0
    for batch_index, batch in enumerate(
        _batches(pending, max(1, args.batch_size), max(1.0, args.max_batch_s)),
        start=1,
    ):
        audio = [load_audio_16k_mono(str(row["audio_path"]))[0] for row in batch]
        transcripts = [str(row["transcript"]) for row in batch]
        inputs, word_lists = _prepare_batch_inputs(
            processor,
            audio=audio,
            transcripts=transcripts,
        )
        moved = inputs.to(device, dtype)
        with torch.inference_mode():
            outputs = model(**moved)
        decoded = processor.decode_forced_alignment(
            outputs.logits,
            moved["input_ids"],
            word_lists,
            timestamp_token_id=int(model.config.timestamp_token_id),
        )
        if len(decoded) != len(batch) or len(word_lists) != len(batch):
            raise RuntimeError(
                "forced-aligner batch cardinality mismatch: "
                f"batch={len(batch)} decoded={len(decoded)} word_lists={len(word_lists)}"
            )
        timestamp_mask = moved["input_ids"] == int(model.config.timestamp_token_id)
        log_norm = torch.logsumexp(outputs.logits.float(), dim=-1)
        max_logit = outputs.logits.float().amax(dim=-1)
        confidence = torch.exp(max_logit - log_norm)
        for row_index, (item, words, word_list) in enumerate(
            zip(batch, decoded, word_lists)
        ):
            point_confidence = confidence[row_index][timestamp_mask[row_index]].detach().cpu().tolist()
            anchors = _build_anchors(
                item=item,
                words=words,
                word_list=word_list,
                point_confidence=point_confidence,
            )
            payload = {
                "schema": SCHEMA,
                "item_id": item["item_id"],
                "source_id": item["source_id"],
                "source_chunk_index": item["source_chunk_index"],
                "duration_s": item["duration_s"],
                "transcript": item["transcript"],
                "audio_path": item["audio_path"],
                "model": args.model,
                "word_units": anchors,
                "word_unit_count": len(anchors),
                "mean_alignment_score": (
                    sum(anchor["alignment_score"] for anchor in anchors) / len(anchors)
                    if anchors
                    else 0.0
                ),
                "vram_safety_ratio": applied_ratio,
            }
            with labels_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            processed += 1
        print(
            f"forced-aligner batch={batch_index} processed={processed}/{len(pending)}",
            flush=True,
        )

    summary = {
        "schema": "timeline_forced_aligner_summary_v1",
        "model": args.model,
        "processed": processed,
        "total_labels": len(_read_jsonl(labels_path)),
        "vram_safety_ratio": applied_ratio,
        "labels": str(labels_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-ForcedAligner-0.6B-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-s", type=float, default=40.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--vram-safety-ratio", type=float, default=0.95)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
