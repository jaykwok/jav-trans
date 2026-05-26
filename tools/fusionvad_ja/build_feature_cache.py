#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.fusionvad_ja import (
    FeatureConfig,
    align_feature_frames,
    build_ptm_feature_extractor,
    build_training_examples,
    extract_mfcc,
    is_low_frame_rate_ptm,
    load_label_records,
    load_manifest_audio_map,
    write_feature_cache,
)
from audio.loading import load_audio_16k_mono


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_label_records(Path(args.labels))
    audio_map = load_manifest_audio_map(Path(args.manifest) if args.manifest else None)
    examples, skipped = build_training_examples(
        records,
        manifest_audio_map=audio_map,
        audio_root=Path(args.audio_root) if args.audio_root else None,
        extension_hints=args.extension,
        trainable_only=not args.include_non_trainable,
    )
    config = FeatureConfig(
        ptm=args.ptm,
        frame_hop_s=args.frame_hop_s,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        device=args.device,
        dtype=args.dtype,
        revision=args.revision,
        model_path=args.model_path,
        download=not args.no_download,
        attention=args.attention,
        language=args.language,
    )
    rows = []
    errors = []
    selected_examples = examples[: args.limit] if args.limit is not None else examples
    print(
        f"feature_cache_start selected={len(selected_examples)} examples={len(examples)} "
        f"device={args.device} dtype={args.dtype} ptm={args.ptm}",
        flush=True,
    )
    ptm_extractor = build_ptm_feature_extractor(config)
    try:
        first_parameter = next(ptm_extractor.model.parameters())
        actual_device = str(first_parameter.device)
        model_path = getattr(ptm_extractor, "model_path", "")
        print(
            f"ptm_extractor={type(ptm_extractor).__name__} model_path={model_path} "
            f"extractor_device={ptm_extractor.device} "
            f"param_device={actual_device} param_dtype={first_parameter.dtype}",
            flush=True,
        )
        if args.device != "cpu":
            if not actual_device.startswith("cuda"):
                raise RuntimeError(f"requested CUDA feature extraction but model is on {actual_device}")
            try:
                import torch

                print(
                    f"cuda_memory allocated={torch.cuda.memory_allocated()} "
                    f"reserved={torch.cuda.memory_reserved()}",
                    flush=True,
                )
            except Exception as exc:
                print(f"cuda_memory_check_error={exc}", flush=True)
        batch_size = max(1, int(args.batch_size))
        for batch_start in range(0, len(selected_examples), batch_size):
            batch_examples = selected_examples[batch_start : batch_start + batch_size]
            batch_time = time.perf_counter()
            prepared = []
            for offset, example in enumerate(batch_examples):
                item_index = batch_start + offset
                try:
                    audio_path = Path(example.audio_path)
                    audio, sample_rate = load_audio_16k_mono(str(audio_path))
                    mfcc = extract_mfcc(audio, sample_rate=sample_rate, config=config)
                    duration_s = len(audio) / sample_rate if sample_rate else 0.0
                    prepared.append(
                        {
                            "index": item_index,
                            "example": example,
                            "audio_path": audio_path,
                            "audio": audio,
                            "sample_rate": sample_rate,
                            "mfcc": mfcc,
                            "duration_s": duration_s,
                        }
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "audio_id": example.audio_id,
                            "audio_path": example.audio_path,
                            "label_index": example.label_index,
                            "error": str(exc),
                        }
                    )
                    print(
                        f"error {item_index + 1}/{len(selected_examples)} audio_id={example.audio_id} "
                        f"error={exc}",
                        flush=True,
                    )
            if not prepared:
                continue
            try:
                sample_rates = {int(item["sample_rate"]) for item in prepared}
                if len(sample_rates) != 1:
                    raise ValueError(f"mixed sample rates in batch: {sorted(sample_rates)}")
                ptm_features = ptm_extractor.extract_batch(
                    [item["audio"] for item in prepared],
                    sample_rate=int(prepared[0]["sample_rate"]),
                )
                for item, ptm in zip(prepared, ptm_features, strict=True):
                    example = item["example"]
                    aligned_ptm, aligned_mfcc = align_feature_frames(
                        ptm,
                        item["mfcc"],
                        resize_ptm=is_low_frame_rate_ptm(config.ptm),
                    )
                    bundle = {
                        "whisper": aligned_ptm,
                        "mfcc": aligned_mfcc,
                        "duration_s": float(item["duration_s"]),
                        "sample_rate": int(item["sample_rate"]),
                    }
                    cached = write_feature_cache(
                        output_dir=feature_dir,
                        audio_id=example.audio_id,
                        source=example.source,
                        audio_path=item["audio_path"],
                        config=config,
                        bundle=bundle,
                    )
                    rows.append(
                        {
                            **asdict(cached),
                            "label_index": example.label_index,
                            "label_quality": example.label_quality,
                            "speech_frame_count": example.speech_frame_count,
                        }
                    )
                    print(
                        f"cached {item['index'] + 1}/{len(selected_examples)} audio_id={example.audio_id} "
                        f"source={example.source} frames={cached.frame_count}",
                        flush=True,
                    )
                print(
                    f"cached_batch {batch_start + 1}-{batch_start + len(batch_examples)}/"
                    f"{len(selected_examples)} batch_size={len(prepared)} "
                    f"elapsed_s={time.perf_counter() - batch_time:.2f}",
                    flush=True,
                )
            except Exception as exc:
                for item in prepared:
                    example = item["example"]
                    errors.append(
                        {
                            "audio_id": example.audio_id,
                            "audio_path": example.audio_path,
                            "label_index": example.label_index,
                            "error": str(exc),
                        }
                    )
                print(
                    f"error_batch {batch_start + 1}-{batch_start + len(batch_examples)}/"
                    f"{len(selected_examples)} error={exc}",
                    flush=True,
                )
    finally:
        ptm_extractor.close()

    manifest_path = output_dir / "feature_manifest.json"
    skipped_path = output_dir / "feature_skipped.json"
    errors_path = output_dir / "feature_errors.json"
    summary_path = output_dir / "feature_summary.json"
    manifest_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    errors_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "labels": args.labels,
        "source_manifest": args.manifest,
        "feature_manifest": str(manifest_path),
        "feature_dir": str(feature_dir),
        "records": len(records),
        "examples": len(examples),
        "cached": len(rows),
        "skipped": len(skipped),
        "errors": len(errors),
        "label_quality_counts": dict(sorted(Counter(row["label_quality"] for row in rows).items())),
        "ptm": args.ptm,
        "config": asdict(config),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"feature_manifest={manifest_path}")
    print(f"skipped={skipped_path}")
    print(f"errors={errors_path}")
    print(f"summary={summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached PTM+MFCC features for FusionVAD-JA training.")
    parser.add_argument("--labels", required=True, help="FusionVAD-JA label JSONL.")
    parser.add_argument("--manifest", help="JSON manifest containing audio paths.")
    parser.add_argument("--audio-root", help="Optional directory for resolving audio_id + extension.")
    parser.add_argument("--extension", action="append", default=[".wav", ".flac", ".ogg", ".mp3", ".m4a"])
    parser.add_argument("--include-non-trainable", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ptm", default="whisper-ja-1.5b")
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--revision")
    parser.add_argument("--model-path", default="", help="Optional local PTM model path for Qwen/Whisper feature extraction.")
    parser.add_argument("--no-download", action="store_true", help="Require the PTM model to already exist locally.")
    parser.add_argument("--attention", default="sdpa", help="Attention implementation for Qwen3-ASR feature extraction.")
    parser.add_argument("--language", default="Japanese", help="Qwen3-ASR prompt language used when building audio features.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "feature-cache"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
