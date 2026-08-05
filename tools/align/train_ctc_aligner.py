#!/usr/bin/env python3
"""Train the CTC alignment head on cached encoder features.

Only the head learns. The encoder stays frozen and, in this loop, is not even
loaded - its output was cached by `build_alignment_features.py`. That is the
whole reason this route is affordable where the abandoned force-aligner was not:
the domain adaptation was already paid for by the ASR's SFT, so what remains is
a ~10M parameter head over features that never change.

CTC loss is the only objective. There is no alignment supervision anywhere in
this loop - no frame-level labels exist, and inventing them would mean inventing
the thing being measured. Alignment emerges from CTC's marginalisation over
paths, and is evaluated separately against geometry the training never saw.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
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
    ALIGNMENT_MODEL_SCHEMA,
    BLANK_INDEX,
    AlignmentVocab,
    build_head,
    is_acoustic_char,
    minimum_ctc_frames,
)
from utils.gpu_safety import apply_vram_safety_cap  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class FeatureCache:
    """Random access into the fp16 shards, memory-mapped rather than loaded.

    Several caches can be given at once, which is how the galgame and real-JAV
    domains are mixed. Rows carry the cache they came from, so a shard name that
    appears in both (`features_0000.npy` does) cannot resolve to the wrong file
    - a bug that would silently pair one domain's features with the other's
    text and look like a training-stability problem.
    """

    def __init__(
        self,
        cache_dirs: list[Path],
        repeats: list[int] | None = None,
        domains: list[str] | None = None,
    ) -> None:
        self.cache_dirs = list(cache_dirs)
        repeats = repeats or [1] * len(self.cache_dirs)
        if len(repeats) != len(self.cache_dirs):
            raise SystemExit("--cache-repeat must be given once per --cache-dir")
        # The directory name is a decent default but often says nothing useful
        # (`cache`), and the label is what the run summary reports the domain mix
        # under - a summary that says `cache: 24000` is unreadable a week later.
        domains = domains or [path.name for path in self.cache_dirs]
        if len(domains) != len(self.cache_dirs):
            raise SystemExit("--cache-domain must be given once per --cache-dir")
        self._shards: dict[tuple[int, str], np.ndarray] = {}
        self.rows: list[dict] = []
        self.domain_rows: Counter[str] = Counter()
        self.domain_seconds: Counter[str] = Counter()
        for index, cache_dir in enumerate(self.cache_dirs):
            domain = domains[index]
            for row in _read_jsonl(cache_dir / "index.jsonl"):
                row = {**row, "cache_index": index, "domain": domain}
                # Oversampling applies to train only. Repeating val rows would
                # not make the metric wrong, but it would weight the domains
                # differently in train and val and make the two incomparable.
                copies = repeats[index] if row.get("partition") == "train" else 1
                for _ in range(max(1, copies)):
                    self.rows.append(row)
                self.domain_rows[domain] += 1
                self.domain_seconds[domain] += float(row.get("duration_s") or 0.0)

    def _shard(self, cache_index: int, name: str) -> np.ndarray:
        key = (cache_index, name)
        cached = self._shards.get(key)
        if cached is None:
            cached = np.load(self.cache_dirs[cache_index] / name, mmap_mode="r")
            self._shards[key] = cached
        return cached

    def features(self, row: dict) -> np.ndarray:
        shard = self._shard(int(row.get("cache_index") or 0), row["shard"])
        start = int(row["offset"])
        return np.asarray(shard[start : start + int(row["frames"])], dtype=np.float32)


def _collate(batch: list[tuple[np.ndarray, list[int]]], torch):
    frames = [item[0].shape[0] for item in batch]
    width = max(frames)
    features = np.zeros((len(batch), width, batch[0][0].shape[1]), dtype=np.float32)
    for index, (feature, _) in enumerate(batch):
        features[index, : feature.shape[0]] = feature
    targets = [item[1] for item in batch]
    flat = [token for target in targets for token in target]
    return (
        torch.from_numpy(features),
        torch.tensor(flat, dtype=torch.long),
        torch.tensor(frames, dtype=torch.long),
        torch.tensor([len(t) for t in targets], dtype=torch.long),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        required=True,
        action="append",
        help="repeatable; give once per domain being mixed",
    )
    parser.add_argument(
        "--cache-repeat",
        action="append",
        type=int,
        default=None,
        help="repeatable, parallel to --cache-dir. Oversamples a smaller "
        "domain's train rows so it is not drowned out by a larger one.",
    )
    parser.add_argument(
        "--cache-domain",
        action="append",
        default=None,
        help="repeatable, parallel to --cache-dir. Label used in the run "
        "summary; defaults to the directory name.",
    )
    parser.add_argument(
        "--select-domain",
        default="",
        help="pick the best checkpoint on this domain's val loss instead of the "
        "pooled one; the pooled number is dominated by whichever cache is larger",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--upsample", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vocab-max-size", type=int, default=3000)
    parser.add_argument("--vocab-min-count", type=int, default=2)
    parser.add_argument(
        "--acoustic-targets",
        action="store_true",
        help="align only pronounceable characters (vocab v2). Punctuation was "
        "16.9%% of targets and 527 clips were nothing but `...`, all of it "
        "asking the head to emit a class where there is no sound.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    import torch
    from torch import nn

    apply_vram_safety_cap(0.95)
    torch.manual_seed(args.seed)

    cache = FeatureCache(
        [Path(path) for path in args.cache_dir],
        args.cache_repeat,
        args.cache_domain,
    )
    train_rows = [r for r in cache.rows if r["partition"] == "train"]
    val_rows = [r for r in cache.rows if r["partition"] == "val"]
    if not train_rows:
        raise SystemExit("feature cache has no training rows")
    mix = Counter(row["domain"] for row in train_rows)
    print(f"train rows by domain (after repeat): {dict(mix)}")

    counts: Counter[str] = Counter()
    for row in train_rows:
        counts.update(row["text"])
    vocab = AlignmentVocab.from_counts(
        counts,
        max_size=args.vocab_max_size,
        min_count=args.vocab_min_count,
        acoustic_only=args.acoustic_targets,
    )
    covered = sum(counts[ch] for ch in vocab.chars)
    total = sum(counts.values())
    # Coverage is over ALL counted characters, so on the acoustic arm it reads
    # as "share of the corpus this head is asked to explain" - by construction
    # below 1, and the gap is the punctuation that no longer has a class.
    acoustic_total = sum(
        count for char, count in counts.items() if is_acoustic_char(char)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = build_head(
        vocab_size=vocab.size,
        hidden_dim=args.hidden_dim,
        upsample=args.upsample,
        blocks=args.blocks,
        dropout=args.dropout,
    ).to(device)
    parameters = sum(p.numel() for p in head.parameters())

    # `zero_infinity` matters: a clip whose text cannot fit its frames yields an
    # infinite loss that would poison the gradient for the whole batch. The
    # extractor filters those, but a vocab change can create new ones.
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = max(1, math.ceil(len(train_rows) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    # Keyed by cache as well as id: the two corpora name their clips
    # independently, and a collision would hand one clip's features the other's
    # text without any error - the same failure mode the shard key guards.
    encoded = {
        (int(row.get("cache_index") or 0), row["audio_id"]): vocab.encode(row["text"])
        for row in cache.rows
    }

    def _key(row: dict) -> tuple[int, str]:
        return int(row.get("cache_index") or 0), row["audio_id"]

    # What `zero_infinity` is actually hiding. A row whose frames cannot hold its
    # targets contributes a zero loss, not an error: it trains on nothing while
    # counting as a sample, so it dilutes the reported loss by exactly its share
    # and there is no other symptom. The extractor filters these, but it filters
    # on characters at 1x while training runs on encoded targets at `--upsample`,
    # so the two judgments only agree if both use `minimum_ctc_frames` - and a
    # vocab change can create new ones under either. Counted once, up front,
    # because a rate here invalidates every loss number below it.
    infeasible = [
        row
        for row in cache.rows
        if encoded[_key(row)]
        and int(row["frames"]) * args.upsample < minimum_ctc_frames(encoded[_key(row)])
    ]
    infeasible_by_partition = Counter(
        str(row.get("partition") or "train") for row in infeasible
    )
    if infeasible:
        print(
            f"  WARNING: {len(infeasible)} of {len(cache.rows)} rows "
            f"({len(infeasible) / len(cache.rows):.2%}) cannot fit their targets "
            f"at --upsample {args.upsample}; zero_infinity will zero their loss "
            f"silently. by partition: {dict(infeasible_by_partition)}",
            flush=True,
        )

    def batches(rows: list[dict], *, shuffle: bool):
        """Length-bucketed batches, shuffled at the batch level.

        Grouping by length is a throughput win, and it used to be doing a second
        job badly: padded positions are zeros, but the head's input `LayerNorm`
        maps an all-zero vector to its own bias, and the conv stack reaches
        ~2.3 s - so a short clip's tail was convolved against a context that
        does not exist at single-clip inference. Bucketing shrank that padding
        to a few frames without removing it. The head now masks padded positions
        outright, so this is back to being only about throughput. Shuffling the
        batch order rather than the rows keeps the gradient noise that shuffling
        is there to provide.
        """
        order = sorted(range(len(rows)), key=lambda i: rows[i]["frames"])
        groups = [
            order[start : start + args.batch_size]
            for start in range(0, len(order), args.batch_size)
        ]
        if shuffle:
            np.random.default_rng(args.seed + epoch).shuffle(groups)
        for group in groups:
            chunk = [rows[i] for i in group]
            items = [(cache.features(r), encoded[_key(r)]) for r in chunk]
            items = [item for item in items if item[1]]
            if items:
                yield _collate(items, torch)

    def run_epoch(rows: list[dict], *, train: bool) -> tuple[float, int]:
        head.train(train)
        total_loss, seen = 0.0, 0
        for features, targets, frame_lengths, target_lengths in batches(
            rows, shuffle=train
        ):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device)
            input_lengths = frame_lengths * args.upsample
            with torch.set_grad_enabled(train):
                # Padded positions are masked inside the head. Without this the
                # padding's LayerNorm bias is convolved into the real frames near
                # a clip's tail, so a clip's own output depends on what it was
                # batched with - and none of that happens at inference, where
                # clips arrive one at a time.
                log_probs = head(features, frame_lengths)
                # CTCLoss wants (T, B, V).
                loss = criterion(
                    log_probs.transpose(0, 1),
                    targets,
                    input_lengths.to(device),
                    target_lengths.to(device),
                )
            if train:
                nonlocal step
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                optimizer.step()
                step += 1
            total_loss += float(loss.item()) * features.shape[0]
            seen += features.shape[0]
        return total_loss / max(1, seen), seen

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "ctc_aligner.pt"
    step = 0
    best_val = float("inf")
    history: list[dict] = []
    started = time.perf_counter()

    # Validation is reported per domain, not just pooled. The two corpora are
    # very different sizes, so a pooled number is mostly the larger one's - and
    # the whole reason this run mixes domains is a failure that only shows up in
    # the smaller one. `--select-domain` decides which number picks the
    # checkpoint; pooled by default, because a small val split is noisy.
    val_domains = sorted({row["domain"] for row in val_rows})
    if args.select_domain and args.select_domain not in val_domains:
        raise SystemExit(
            f"--select-domain {args.select_domain!r} has no val rows; "
            f"available: {val_domains}"
        )
    val_by_domain = {
        domain: [row for row in val_rows if row["domain"] == domain]
        for domain in val_domains
    }

    for epoch in range(args.epochs):
        train_loss, _ = run_epoch(train_rows, train=True)
        per_domain: dict[str, float] = {}
        pooled_total, pooled_seen = 0.0, 0
        for domain, rows in val_by_domain.items():
            loss, seen = run_epoch(rows, train=False)
            per_domain[domain] = round(loss, 4)
            pooled_total += loss * seen
            pooled_seen += seen
        val_loss = pooled_total / pooled_seen if pooled_seen else float("nan")
        selected = per_domain.get(args.select_domain, val_loss) if args.select_domain else val_loss
        elapsed = time.perf_counter() - started
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_loss_by_domain": dict(per_domain),
                "elapsed_s": round(elapsed, 1),
            }
        )
        detail = "  ".join(f"{name} {value:.4f}" for name, value in per_domain.items())
        print(
            f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"[{detail}]  {elapsed:.0f}s",
            flush=True,
        )
        if not math.isnan(selected) and selected < best_val:
            best_val = selected
            torch.save(
                {
                    "schema": ALIGNMENT_MODEL_SCHEMA,
                    "state_dict": head.state_dict(),
                    "vocab": vocab.to_payload(),
                    "upsample": args.upsample,
                    "hidden_dim": args.hidden_dim,
                    "blocks": args.blocks,
                    "input_dim": 2048,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_loss_by_domain": dict(per_domain),
                    "selection_domain": args.select_domain or "",
                },
                checkpoint_path,
            )

    summary = {
        "schema": "asr_ctc_aligner_training_v1",
        "cache_dirs": [str(path) for path in args.cache_dir],
        "cache_repeat": list(args.cache_repeat or []),
        "cache_domain": list(args.cache_domain or []),
        "select_domain": args.select_domain or "",
        "train_rows_by_domain": dict(Counter(row["domain"] for row in train_rows)),
        "cached_seconds_by_domain": {
            domain: round(seconds, 1)
            for domain, seconds in sorted(cache.domain_seconds.items())
        },
        "train_clips": len(train_rows),
        "val_clips": len(val_rows),
        "vocab_size": vocab.size,
        "vocab_coverage": round(covered / total, 5) if total else 0.0,
        # Two runs with different target sets do NOT have comparable losses -
        # fewer targets is a shorter sequence and a lower CTC loss for that
        # reason alone. Recorded so nobody reads the acoustic arm's val number
        # as an improvement over the punctuated one; the comparison that means
        # something is geometry on the composite set.
        "acoustic_targets": bool(args.acoustic_targets),
        "acoustic_character_share": round(acoustic_total / total, 5) if total else 0.0,
        "head_parameters": parameters,
        # Rows whose loss `zero_infinity` zeroed. Non-zero means every loss in
        # `history` is diluted by that fraction and the run is not comparable
        # with one where it is zero.
        "infeasible_rows": len(infeasible),
        "infeasible_rows_by_partition": dict(infeasible_by_partition),
        "upsample": args.upsample,
        "epochs": args.epochs,
        "best_val_loss": round(best_val, 4) if math.isfinite(best_val) else None,
        "checkpoint": str(checkpoint_path),
        "history": history,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
