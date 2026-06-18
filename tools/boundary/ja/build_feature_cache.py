#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import QWEN_ASR_REPO_ID
from boundary.ja import (
    DEFAULT_TRAINABLE_LABEL_QUALITIES,
    FeatureConfig,
    TrainingExample,
    align_feature_frames,
    extract_mfcc,
    build_ptm_feature_extractor,
    is_low_frame_rate_ptm,
    load_manifest_audio_map,
    write_feature_cache,
)
from audio.loading import load_audio_16k_mono


def _write_jsonl_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _write_jsonl_file(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            _write_jsonl_row(handle, row)
            count += 1
    return count


def _load_existing_manifest(path: Path) -> tuple[set[int], Counter[str], int]:
    label_indexes: set[int] = set()
    label_quality_counts: Counter[str] = Counter()
    count = 0
    if not path.exists():
        return label_indexes, label_quality_counts, count
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            label_index = row.get("label_index")
            if label_index is not None:
                label_indexes.add(int(label_index))
            label_quality_counts[str(row.get("label_quality") or "")] += 1
            count += 1
    return label_indexes, label_quality_counts, count


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _stream_training_examples(
    *,
    labels_path: Path,
    manifest_audio_map: Mapping[str, str],
    audio_root: Path | None,
    extension_hints: Iterable[str],
    trainable_only: bool,
) -> tuple[int, list[TrainingExample], list[dict[str, Any]]]:
    records_count = 0
    examples: list[TrainingExample] = []
    skipped: list[dict[str, Any]] = []
    with labels_path.open("r", encoding="utf-8") as handle:
        for label_index, line in enumerate(handle):
            if not line.strip():
                continue
            records_count += 1
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                skipped.append({"label_index": label_index, "reason": "bad_label_row"})
                continue
            label_quality = str(payload.get("label_quality") or "")
            if trainable_only and label_quality not in DEFAULT_TRAINABLE_LABEL_QUALITIES:
                continue
            audio_id = str(payload.get("audio_id") or "")
            source = str(payload.get("source") or "")
            duration_s = float(payload.get("duration_s") or 0.0)
            frame_hop_s = float(payload.get("frame_hop_s") or 0.02)
            speech_frames = list(payload.get("speech_frames") or [])
            expected_frames = _frame_count(duration_s, frame_hop_s)
            if len(speech_frames) != expected_frames:
                skipped.append(
                    {
                        "audio_id": audio_id,
                        "source": source,
                        "label_quality": label_quality,
                        "reason": "frame_count_mismatch",
                        "expected_frames": expected_frames,
                        "actual_frames": len(speech_frames),
                    }
                )
                continue
            audio_path = _resolve_audio_path(
                audio_id=audio_id,
                source=source,
                manifest_audio_map=manifest_audio_map,
                audio_root=audio_root,
                extension_hints=extension_hints,
            )
            if audio_path is None:
                skipped.append(
                    {
                        "audio_id": audio_id,
                        "source": source,
                        "label_quality": label_quality,
                        "reason": "missing_audio_path",
                    }
                )
                continue
            examples.append(
                TrainingExample(
                    audio_id=audio_id,
                    source=source,
                    label_quality=label_quality,
                    duration_s=duration_s,
                    frame_hop_s=frame_hop_s,
                    audio_path=str(audio_path),
                    label_index=label_index,
                    speech_frame_count=sum(int(value) for value in speech_frames),
                    frame_count=len(speech_frames),
                )
            )
    return records_count, examples, skipped


def _frame_count(duration_s: float, frame_hop_s: float) -> int:
    if duration_s <= 0.0:
        return 0
    return int(math.ceil((duration_s / frame_hop_s) - 1e-9))


def _resolve_audio_path(
    *,
    audio_id: str,
    source: str,
    manifest_audio_map: Mapping[str, str],
    audio_root: Path | None,
    extension_hints: Iterable[str],
) -> Path | None:
    for key in (audio_id, f"{source}:{audio_id}"):
        value = manifest_audio_map.get(key)
        if value:
            return Path(value)
    candidate = Path(audio_id)
    if candidate.exists():
        return candidate
    if audio_root is None:
        return None
    for suffix in extension_hints:
        suffix = suffix if str(suffix).startswith(".") else f".{suffix}"
        candidate = audio_root / f"{audio_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _prepare_batch(
    *,
    batch_start: int,
    batch_examples: list,
    selected_count: int,
    config: FeatureConfig,
) -> tuple[list[dict], list[dict], float]:
    prepared = []
    errors = []
    prepare_start = time.perf_counter()
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
                    "index": item_index,
                    "selected_count": selected_count,
                }
            )
    return prepared, errors, time.perf_counter() - prepare_start


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_map = load_manifest_audio_map(Path(args.manifest) if args.manifest else None)
    records_count, examples, skipped = _stream_training_examples(
        labels_path=Path(args.labels),
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
        feature_dim=args.feature_dim,
        device=args.device,
        dtype=args.dtype,
        revision=args.revision,
        model_path=args.model_path,
        download=not args.no_download,
        attention=args.attention,
        language=args.language,
    )
    selected_examples = examples[: args.limit] if args.limit is not None else examples
    print(
        f"feature_cache_start selected={len(selected_examples)} examples={len(examples)} "
        f"device={args.device} dtype={args.dtype} ptm={args.ptm} "
        f"batch_size={args.batch_size} prepare_workers={args.prepare_workers}",
        flush=True,
    )
    ptm_extractor = build_ptm_feature_extractor(config)
    manifest_path = output_dir / "feature_manifest.jsonl"
    skipped_path = output_dir / "feature_skipped.jsonl"
    errors_path = output_dir / "feature_errors.jsonl"
    summary_path = output_dir / "feature_summary.json"
    existing_indexes, label_quality_counts, cached_count = _load_existing_manifest(manifest_path) if args.resume else (
        set(),
        Counter(),
        0,
    )
    if existing_indexes:
        selected_examples = [example for example in selected_examples if example.label_index not in existing_indexes]
        print(
            f"feature_cache_resume existing={cached_count} remaining={len(selected_examples)}",
            flush=True,
        )
    skipped_count = _write_jsonl_file(skipped_path, skipped) if not args.resume else _write_jsonl_file(skipped_path, skipped)
    error_count = _count_jsonl_rows(errors_path) if args.resume else 0
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
        prepare_workers = max(0, int(args.prepare_workers))
        batch_starts = list(range(0, len(selected_examples), batch_size))

        executor: ThreadPoolExecutor | None = None
        future: Future | None = None

        def submit_prepare(start: int) -> Future:
            batch_examples = selected_examples[start : start + batch_size]
            if executor is None:
                inline_future: Future = Future()
                inline_future.set_result(
                    _prepare_batch(
                        batch_start=start,
                        batch_examples=batch_examples,
                        selected_count=len(selected_examples),
                        config=config,
                    )
                )
                return inline_future
            return executor.submit(
                _prepare_batch,
                batch_start=start,
                batch_examples=batch_examples,
                selected_count=len(selected_examples),
                config=config,
            )

        if prepare_workers > 0:
            executor = ThreadPoolExecutor(max_workers=prepare_workers)
        try:
            manifest_mode = "a" if args.resume else "w"
            errors_mode = "a" if args.resume else "w"
            with manifest_path.open(manifest_mode, encoding="utf-8") as manifest_handle, errors_path.open(
                errors_mode, encoding="utf-8"
            ) as errors_handle:
                if batch_starts:
                    future = submit_prepare(batch_starts[0])
                for batch_index, batch_start in enumerate(batch_starts):
                    if future is None:
                        break
                    batch_examples = selected_examples[batch_start : batch_start + batch_size]
                    prepared, prepare_errors, prepare_elapsed_s = future.result()
                    error_count += len(prepare_errors)
                    for error in prepare_errors:
                        _write_jsonl_row(errors_handle, error)
                    errors_handle.flush()
                    next_index = batch_index + 1
                    next_future: Future | None = None
                    if next_index < len(batch_starts):
                        next_future = submit_prepare(batch_starts[next_index])
                    print(
                        f"prepared_batch {batch_start + 1}-{batch_start + len(batch_examples)}/"
                        f"{len(selected_examples)} prepared={len(prepared)} errors={len(prepare_errors)} "
                        f"elapsed_s={prepare_elapsed_s:.2f}",
                        flush=True,
                    )
                    for error in prepare_errors:
                        print(
                            f"error {int(error['index']) + 1}/{int(error['selected_count'])} "
                            f"audio_id={error['audio_id']} error={error['error']}",
                            flush=True,
                        )
                    batch_time = time.perf_counter()
                    if not prepared:
                        future = next_future
                        continue
                    try:
                        sample_rates = {int(item["sample_rate"]) for item in prepared}
                        if len(sample_rates) != 1:
                            raise ValueError(f"mixed sample rates in batch: {sorted(sample_rates)}")
                        ptm_start = time.perf_counter()
                        ptm_features = ptm_extractor.extract_batch(
                            [item["audio"] for item in prepared],
                            sample_rate=int(prepared[0]["sample_rate"]),
                        )
                        ptm_elapsed_s = time.perf_counter() - ptm_start
                        write_elapsed_s = 0.0
                        for item, ptm in zip(prepared, ptm_features, strict=True):
                            example = item["example"]
                            aligned_ptm, aligned_mfcc = align_feature_frames(
                                ptm,
                                item["mfcc"],
                                resize_ptm=is_low_frame_rate_ptm(config.ptm),
                            )
                            bundle = {
                                "ptm": aligned_ptm,
                                "mfcc": aligned_mfcc,
                                "duration_s": float(item["duration_s"]),
                                "sample_rate": int(item["sample_rate"]),
                            }
                            write_start = time.perf_counter()
                            cached = write_feature_cache(
                                output_dir=feature_dir,
                                audio_id=example.audio_id,
                                source=example.source,
                                audio_path=item["audio_path"],
                                config=config,
                                bundle=bundle,
                                compressed=not args.no_compress,
                            )
                            write_elapsed_s += time.perf_counter() - write_start
                            manifest_row = {
                                **asdict(cached),
                                "label_index": example.label_index,
                                "label_quality": example.label_quality,
                                "speech_frame_count": example.speech_frame_count,
                            }
                            _write_jsonl_row(manifest_handle, manifest_row)
                            manifest_handle.flush()
                            cached_count += 1
                            label_quality_counts[str(example.label_quality)] += 1
                            log_every = max(1, int(args.log_every))
                            is_last = item["index"] + 1 >= len(selected_examples)
                            if log_every == 1 or (item["index"] + 1) % log_every == 0 or is_last:
                                print(
                                    f"cached {item['index'] + 1}/{len(selected_examples)} audio_id={example.audio_id} "
                                    f"source={example.source} frames={cached.frame_count}",
                                    flush=True,
                                )
                        print(
                            f"cached_batch {batch_start + 1}-{batch_start + len(batch_examples)}/"
                            f"{len(selected_examples)} batch_size={len(prepared)} "
                            f"elapsed_s={time.perf_counter() - batch_time:.2f} "
                            f"ptm_elapsed_s={ptm_elapsed_s:.2f} write_elapsed_s={write_elapsed_s:.2f} "
                            f"compressed={not args.no_compress}",
                            flush=True,
                        )
                    except Exception as exc:
                        batch_errors = []
                        for item in prepared:
                            example = item["example"]
                            batch_errors.append(
                                {
                                    "audio_id": example.audio_id,
                                    "audio_path": example.audio_path,
                                    "label_index": example.label_index,
                                    "error": str(exc),
                                }
                            )
                        error_count += len(batch_errors)
                        for error in batch_errors:
                            _write_jsonl_row(errors_handle, error)
                        errors_handle.flush()
                        print(
                            f"error_batch {batch_start + 1}-{batch_start + len(batch_examples)}/"
                            f"{len(selected_examples)} error={exc}",
                            flush=True,
                        )
                    future = next_future
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
    finally:
        ptm_extractor.close()

    summary = {
        "labels": args.labels,
        "source_manifest": args.manifest,
        "feature_manifest": str(manifest_path),
        "feature_manifest_format": "jsonl",
        "feature_dir": str(feature_dir),
        "records": records_count,
        "examples": len(examples),
        "cached": cached_count,
        "skipped": skipped_count,
        "errors": error_count,
        "label_quality_counts": dict(sorted(label_quality_counts.items())),
        "ptm": args.ptm,
        "compressed": not args.no_compress,
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
    parser = argparse.ArgumentParser(description="Build cached PTM+MFCC features for SpeechBoundary-JA training.")
    parser.add_argument("--labels", required=True, help="SpeechBoundary-JA label JSONL.")
    parser.add_argument("--manifest", help="JSON manifest containing audio paths.")
    parser.add_argument("--audio-root", help="Optional directory for resolving audio_id + extension.")
    parser.add_argument("--extension", action="append", default=[".wav", ".flac", ".ogg", ".mp3", ".m4a"])
    parser.add_argument("--include-non-trainable", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ptm", default=QWEN_ASR_REPO_ID)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument(
        "--feature-dim",
        type=int,
        help="Optional PTM feature dimension to keep in the cache. Omit to keep the full PTM.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--prepare-workers",
        type=int,
        default=0,
        help="Background workers for audio loading + MFCC preparation. 0 keeps inline preparation.",
    )
    parser.add_argument("--no-compress", action="store_true", help="Use np.savez instead of np.savez_compressed.")
    parser.add_argument("--revision")
    parser.add_argument("--model-path", default="", help="Optional local PTM model path for Qwen feature extraction.")
    parser.add_argument("--no-download", action="store_true", help="Require the PTM model to already exist locally.")
    parser.add_argument("--attention", default="sdpa", help="Attention implementation for Qwen3-ASR feature extraction.")
    parser.add_argument("--language", default="Japanese", help="Qwen3-ASR prompt language used when building audio features.")
    parser.add_argument("--log-every", type=int, default=1, help="Print one cached row every N examples; 1 logs every row.")
    parser.add_argument("--resume", action="store_true", help="Append to existing feature_manifest.jsonl and skip cached label_index rows.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / "feature-cache"))
    args = parser.parse_args(argv)
    if args.feature_dim is not None and args.feature_dim <= 0:
        parser.error("--feature-dim must be positive when set")
    return args


if __name__ == "__main__":
    run(parse_args())
