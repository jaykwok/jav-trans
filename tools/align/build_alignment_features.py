#!/usr/bin/env python3
"""Cache SFT'd-encoder features for CTC alignment training.

The encoder is frozen, so its output for a given clip never changes and there is
no reason to recompute it every epoch. Phase 0 measured encoder forward at
RTF 0.00069, which puts the whole extraction in the minutes range and makes the
cache almost free to rebuild - the point of caching here is training speed, not
extraction cost.

Features are stored fp16. The head normalises its input first thing
(`LayerNorm`), so fp16's ~3 decimal digits are far below the precision the head
can use, and it halves both disk and the host-to-device copy that dominates a
frozen-feature training loop.

**Held-out clips are excluded here, not at training time.** The alignment kill
gate is measured on the composite corpus, whose cores are drawn from these same
galgame clips - so a clip that appears in both would let the head memorise the
answer and report an alignment accuracy it has not earned. Excluding at
extraction makes the exclusion a property of the cache rather than a flag
someone can forget to pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment import (  # noqa: E402
    ENCODER_FPS,
    minimum_ctc_frames,
    normalize_text,
)
from audio.loading import load_audio_16k_mono  # noqa: E402
from utils.gpu_safety import apply_vram_safety_cap  # noqa: E402
from asr.encoder_features import EncoderFeatureConfig, Qwen3AsrEncoder  # noqa: E402

SAMPLE_RATE = 16000
FEATURE_DIM = 2048
SHARD_SCHEMA = "asr_alignment_feature_cache_v1"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _held_out_ids(composite_manifest: Path | None) -> set[str]:
    """Galgame clips that appear as cores in the evaluation composites."""
    if composite_manifest is None:
        return set()
    ids: set[str] = set()
    for row in _read_jsonl(composite_manifest):
        for core in row.get("core_spans") or ():
            core_id = str(core.get("core_id") or "").strip()
            if core_id:
                ids.add(core_id)
            stem = Path(str(core.get("source_audio") or "")).stem
            if stem:
                ids.add(stem)
    return ids


def _group_key(row: dict, position: int, *, field: str, block: int) -> str:
    """Which held-out group a clip belongs to.

    Validation was a per-clip random draw, which on this corpus is barely
    validation at all: the clips come from visual novels, so the same voice
    actor reading the same script style lands on both sides of the split and the
    val loss reports memorisation of a speaker as generalisation. Any
    architecture A/B decided on that number is decided on a contaminated metric.

    `field` is the honest key and is used whenever the manifest carries one
    (game, voice actor, source work). The galgame manifest does not - it keeps
    only `audio_id`, text, duration and the upstream row order - so the fallback
    groups consecutive manifest rows into blocks. Source order groups clips by
    game, which this extractor already relied on when it refused to subsample by
    truncation, so blocks approximate speaker disjointness without pretending to
    be exact. It is a proxy, and the summary says which one was used.
    """
    if field:
        value = row.get(field)
        if value is None or str(value) == "":
            raise SystemExit(
                f"--group-field {field!r} is missing from the manifest; "
                "grouping cannot silently fall back to a leaky split"
            )
        return f"{field}={value}"
    if block > 0:
        index = row.get("index")
        sequence = int(index) if isinstance(index, (int, float)) else position
        return f"block={sequence // block}"
    return f"clip={position}"


def _assign_partitions(rows: list[dict], *, val_fraction: float, rng) -> dict:
    """Put whole groups in val until it is big enough, then stop.

    Group-level rather than clip-level, so val is a set of speakers the head
    never trained on. The count lands near `val_fraction` rather than on it -
    the last group taken usually overshoots - which is the price of the split
    meaning something.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    order = sorted(groups)
    rng.shuffle(order)
    target = val_fraction * len(rows)
    val_rows, val_groups = 0, 0
    for name in order:
        if val_rows >= target:
            break
        for row in groups[name]:
            row["partition"] = "val"
        val_rows += len(groups[name])
        val_groups += 1
    for row in rows:
        row.setdefault("partition", "train")
    return {
        "groups_total": len(groups),
        "groups_val": val_groups,
        "val_rows": val_rows,
        "val_fraction_actual": round(val_rows / len(rows), 5) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="galgame clip manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--holdout-composites",
        default="",
        help="composite manifest whose cores must be excluded from this cache",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-rows", type=int, default=2000)
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=25.0)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument(
        "--group-field",
        default="",
        help="manifest field naming the game/voice actor a clip belongs to. "
        "When the manifest has one, this is the honest split key.",
    )
    parser.add_argument(
        "--group-block",
        type=int,
        default=200,
        help="fallback when no --group-field exists: treat this many "
        "consecutive manifest rows as one group. Source order groups clips by "
        "game, so contiguous blocks approximate speaker disjointness; 0 "
        "restores the old per-clip random split.",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    held_out = _held_out_ids(
        Path(args.holdout_composites) if args.holdout_composites else None
    )

    rows = _read_jsonl(Path(args.manifest))
    eligible: list[dict] = []
    skipped = Counter()
    for position, row in enumerate(rows):
        audio_id = str(row.get("audio_id") or "")
        text = normalize_text(str(row.get("text") or ""))
        duration = float(row.get("duration_s") or 0.0)
        if audio_id in held_out:
            skipped["held_out_for_evaluation"] += 1
            continue
        if not text:
            skipped["empty_text"] += 1
            continue
        if not args.min_seconds <= duration <= args.max_seconds:
            skipped["duration_out_of_range"] += 1
            continue
        # CTC cannot emit more characters than it has frames. Upsampling gives
        # headroom, but a clip below even the 1x bound is unusable at any factor.
        # `minimum_ctc_frames`, not `len(text)`: repeated characters each need a
        # blank between them, and this is the same judgment inference applies.
        if minimum_ctc_frames(text) > duration * ENCODER_FPS:
            skipped["text_denser_than_frame_rate"] += 1
            continue
        eligible.append({"audio_id": audio_id, "audio": row["audio"], "text": text,
                         "duration_s": duration,
                         "group": _group_key(
                             row,
                             position,
                             field=args.group_field,
                             block=args.group_block,
                         )})

    if not eligible:
        raise SystemExit("no eligible clips after filtering")

    rng = np.random.default_rng(args.seed)
    # Subsample at random rather than truncating. The manifest arrives in source
    # dataset order, which groups clips by game and therefore by voice actor, so
    # taking a prefix would build a cache of a few speakers and report their
    # alignment accuracy as the corpus's.
    if args.limit and args.limit < len(eligible):
        chosen = rng.choice(len(eligible), size=args.limit, replace=False)
        eligible = [eligible[i] for i in sorted(chosen)]
        skipped["not_sampled"] = len(rows) - len(eligible)

    split = _assign_partitions(eligible, val_fraction=args.val_fraction, rng=rng)

    # Sorting by duration keeps each batch nearly uniform in length, so the
    # padding the processor adds is spent on a few frames rather than on the
    # gap between a 1 s clip and a 25 s one.
    order = sorted(range(len(eligible)), key=lambda i: eligible[i]["duration_s"])

    apply_vram_safety_cap(0.95)
    extractor = Qwen3AsrEncoder(
        EncoderFeatureConfig(model_path=args.model_path or "", device="cuda")
    )

    index_path = output_dir / "index.jsonl"
    shard_rows: list[np.ndarray] = []
    shard_index = 0
    shard_offset = 0
    written = 0
    failures = 0
    started = time.perf_counter()
    audio_seconds = 0.0

    def flush() -> None:
        nonlocal shard_rows, shard_index, shard_offset
        if not shard_rows:
            return
        stacked = np.concatenate(shard_rows, axis=0).astype(np.float16)
        np.save(output_dir / f"features_{shard_index:04d}.npy", stacked)
        shard_rows = []
        shard_index += 1
        shard_offset = 0

    with index_path.open("w", encoding="utf-8") as index_handle:
        for start in range(0, len(order), args.batch_size):
            batch_rows = [eligible[i] for i in order[start : start + args.batch_size]]
            audios: list[np.ndarray] = []
            kept: list[dict] = []
            for row in batch_rows:
                path = Path(row["audio"])
                if not path.is_absolute():
                    path = PROJECT_ROOT / path
                try:
                    audio, rate = load_audio_16k_mono(str(path))
                except Exception:
                    failures += 1
                    continue
                if rate != SAMPLE_RATE:
                    failures += 1
                    continue
                audios.append(np.asarray(audio, dtype=np.float32))
                kept.append(row)
            if not audios:
                continue
            features = extractor.encode_batch(audios, sample_rate=SAMPLE_RATE)
            for row, audio, feature in zip(kept, audios, features):
                frames = int(feature.shape[0])
                if frames < minimum_ctc_frames(row["text"]):
                    skipped["frames_below_text_length"] += 1
                    continue
                shard_rows.append(feature.astype(np.float16))
                index_handle.write(
                    json.dumps(
                        {
                            "schema": SHARD_SCHEMA,
                            "audio_id": row["audio_id"],
                            "text": row["text"],
                            "partition": row["partition"],
                            # Carried into the cache so a later run can regroup
                            # or audit the split without the manifest, which is
                            # exactly what was impossible for the first cache.
                            "group": row["group"],
                            "duration_s": round(row["duration_s"], 4),
                            "frames": frames,
                            "shard": f"features_{shard_index:04d}.npy",
                            "offset": shard_offset,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                shard_offset += frames
                audio_seconds += len(audio) / SAMPLE_RATE
                written += 1
                if shard_offset >= args.shard_rows * int(ENCODER_FPS) * 4:
                    flush()
            if written and start % (args.batch_size * 50) == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  {written} clips / {audio_seconds/3600:.2f} h "
                    f"in {elapsed:.0f}s",
                    flush=True,
                )
        flush()

    counts: Counter[str] = Counter()
    partitions: Counter[str] = Counter()
    for row in _read_jsonl(index_path):
        counts.update(row["text"])
        partitions[row["partition"]] += 1

    summary = {
        "schema": SHARD_SCHEMA,
        "manifest": str(args.manifest),
        "clips_written": written,
        "clips_failed_to_load": failures,
        "skipped": dict(skipped),
        "held_out_ids": len(held_out),
        "partitions": dict(partitions),
        # How val was drawn. A run that reports a val loss without saying which
        # of these produced it is not comparable with one that used the other:
        # a per-clip split shares voice actors across the boundary and reads
        # lower for that reason alone.
        "split": {
            "mode": (
                f"field:{args.group_field}"
                if args.group_field
                else (f"block:{args.group_block}" if args.group_block > 0 else "clip")
            ),
            **split,
        },
        "audio_hours": round(audio_seconds / 3600.0, 3),
        "distinct_characters": len(counts),
        "total_characters": int(sum(counts.values())),
        "chars_per_second": round(sum(counts.values()) / audio_seconds, 3)
        if audio_seconds
        else 0.0,
        "shards": shard_index,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "char_counts.json").write_text(
        json.dumps(dict(counts.most_common()), ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
