#!/usr/bin/env python3
"""Train the CTC alignment head on cached encoder features.

Only the head learns. The encoder stays frozen and, in this loop, is not even
loaded - its output was cached by `build_alignment_features.py`. That is the
whole reason this route is affordable where the abandoned force-aligner was not:
the domain adaptation was already paid for by the ASR's SFT, so what remains is
a ~10M parameter head over features that never change.

Full-clip CTC remains the canonical text objective.  An optional sparse frame
teacher may add a blank-vs-speech auxiliary loss from audited word timestamps:
word islands are positive, only long distant gaps are negative, and uncertain
frames are ignored.  The auxiliary target never replaces dataset text.

Frame teachers are declared per **cache**, not per domain, and repeatably. Not
every source of training text has word timings - blank-only rows have no words
to time, and clips recovered by a local decode have text but no clock - so those
caches simply carry no frame supervision, which the run summary states outright.
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
from tools.align.frame_teacher_supervision import (  # noqa: E402
    IGNORE_LABEL,
    balanced_sparse_frame_loss,
    compile_sparse_frame_targets,
    load_accepted_frame_teachers,
)


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


def _cap_empty_train_rows(
    rows: list[dict], *, maximum_fraction: float, seed: int
) -> tuple[list[dict], dict[str, dict[str, int | float]]]:
    """Cap empty targets within each domain, deterministically.

    A global cap is ineffective when 29k galgame positives hide an overly
    blank real-domain arm.  Per-domain capping keeps the supervision mixture
    honest even when that smaller cache is oversampled with ``--cache-repeat``.
    """

    if not 0.0 <= maximum_fraction < 1.0:
        raise ValueError("maximum empty target fraction must be in [0, 1)")
    by_domain: dict[str, list[dict]] = {}
    for row in rows:
        by_domain.setdefault(str(row.get("domain") or ""), []).append(row)
    kept: list[dict] = []
    report: dict[str, dict[str, int | float]] = {}
    rng = np.random.default_rng(seed)
    for domain in sorted(by_domain):
        domain_rows = by_domain[domain]
        nonempty = [row for row in domain_rows if str(row.get("text") or "")]
        empty = [row for row in domain_rows if not str(row.get("text") or "")]
        maximum_empty = (
            int(math.floor(maximum_fraction * len(nonempty) / (1.0 - maximum_fraction)))
            if nonempty
            else 0
        )
        keep_count = min(len(empty), maximum_empty)
        if keep_count < len(empty):
            chosen = sorted(
                int(index)
                for index in rng.choice(len(empty), size=keep_count, replace=False)
            )
            kept_empty = [empty[index] for index in chosen]
        else:
            kept_empty = empty
        selected = nonempty + kept_empty
        kept.extend(selected)
        report[domain] = {
            "nonempty": len(nonempty),
            "empty_available": len(empty),
            "empty_kept": len(kept_empty),
            "empty_dropped": len(empty) - len(kept_empty),
            "empty_fraction": round(len(kept_empty) / len(selected), 6)
            if selected
            else 0.0,
        }
    return kept, report


def _row_key(row: dict) -> tuple[int, str]:
    """Keyed by cache as well as id: the corpora name their clips
    independently, and a collision would hand one clip's features another's
    text without any error - the same failure the shard key guards against."""
    return int(row.get("cache_index") or 0), row["audio_id"]


def compile_frame_labels_by_cache(
    train_rows: list[dict],
    cache_names: list[str],
    *,
    results: list[str],
    manifests: list[str],
    caches: list[str],
    upsample: int,
    positive_merge_gap_s: float,
    boundary_ignore_s: float,
    negative_minimum_s: float,
) -> tuple[dict[tuple[int, str], np.ndarray], dict[str, object]]:
    """Frame supervision for the caches that declared a teacher archive.

    The guard that matters is per archive: **every** train row of the cache it
    names must have an accepted teacher, or the run stops. That is what catches a
    stale archive or a mistyped path, which would otherwise show up only as a
    quietly weaker auxiliary loss.

    Rows in caches with no declared archive get no frame labels, by design -
    blank-only rows have no words to time, and clips recovered by a local decode
    have text but no clock. The count of those rows is reported rather than left
    implicit, so a mixture that lost supervision cannot pass for one that never
    asked for it.
    """
    cache_index_by_name = {name: index for index, name in enumerate(cache_names)}
    frame_labels: dict[tuple[int, str], np.ndarray] = {}
    archives: list[dict[str, object]] = []
    blank_frames = speech_frames = ignored_frames = supervised_rows = 0

    for result_path, manifest_path, cache_name in zip(results, manifests, caches):
        if cache_name not in cache_index_by_name:
            raise SystemExit(
                f"--frame-teacher-cache {cache_name!r} is not one of the "
                f"--cache-dir names; available: {sorted(cache_index_by_name)}"
            )
        cache_index = cache_index_by_name[cache_name]
        teachers, load_summary = load_accepted_frame_teachers(
            Path(result_path), Path(manifest_path)
        )
        eligible: dict[tuple[int, str], dict] = {}
        for row in train_rows:
            if int(row.get("cache_index") or 0) == cache_index:
                eligible.setdefault(_row_key(row), row)
        missing = sorted(
            str(row.get("source_id") or row["audio_id"])
            for row in eligible.values()
            if str(row.get("source_id") or row["audio_id"]) not in teachers
        )
        if missing:
            raise SystemExit(
                f"{len(missing)} training rows in cache {cache_name!r} lack an "
                f"accepted frame teacher; first={missing[0]}"
            )
        archive_blank = archive_speech = archive_ignored = archive_rows = 0
        for key, row in eligible.items():
            labels = compile_sparse_frame_targets(
                teachers[str(row.get("source_id") or row["audio_id"])],
                output_frames=int(row["frames"]) * upsample,
                upsample=upsample,
                positive_merge_gap_s=positive_merge_gap_s,
                boundary_ignore_s=boundary_ignore_s,
                negative_minimum_s=negative_minimum_s,
                # Crop rows and full-clip rows of the same source sit in one
                # cache; without this the crops' labels are shifted by their
                # own start time and nothing says so.
                start_offset_s=float(row.get("source_start_s") or 0.0),
            )
            frame_labels[key] = labels
            archive_blank += int(np.sum(labels == 0))
            archive_speech += int(np.sum(labels == 1))
            archive_ignored += int(np.sum(labels == IGNORE_LABEL))
            archive_rows += int(np.any(labels != IGNORE_LABEL))
        archives.append(
            {
                **load_summary,
                "cache": cache_name,
                "domain": next(
                    (str(row.get("domain") or "") for row in eligible.values()), ""
                ),
                "eligible_unique_train_rows": len(eligible),
                "supervised_unique_train_rows": archive_rows,
                "blank_frames": archive_blank,
                "speech_frames": archive_speech,
                "ignored_frames": archive_ignored,
            }
        )
        blank_frames += archive_blank
        speech_frames += archive_speech
        ignored_frames += archive_ignored
        supervised_rows += archive_rows

    if not supervised_rows or not speech_frames or not blank_frames:
        raise SystemExit(
            "frame teacher produced no usable balanced supervision "
            f"(rows={supervised_rows}, speech={speech_frames}, blank={blank_frames})"
        )
    unsupervised = sum(
        1 for row in train_rows if _row_key(row) not in frame_labels
    )
    summary = {
        "archives": archives,
        "supervised_unique_train_rows": supervised_rows,
        "blank_frames": blank_frames,
        "speech_frames": speech_frames,
        "ignored_frames": ignored_frames,
        "labelled_frame_share": round(
            (blank_frames + speech_frames)
            / max(1, blank_frames + speech_frames + ignored_frames),
            6,
        ),
        "train_rows_without_frame_supervision": unsupervised,
    }
    return frame_labels, summary


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
    parser.add_argument(
        "--include-empty-targets",
        action="store_true",
        help="train explicit blank-only cache rows instead of dropping them",
    )
    parser.add_argument(
        "--max-empty-train-fraction",
        type=float,
        default=0.30,
        help="per-domain cap applied after cache repeats (default: 0.30)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--upsample", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vocab-max-size", type=int, default=3000)
    parser.add_argument("--vocab-min-count", type=int, default=2)
    parser.add_argument(
        "--vocab-checkpoint",
        default="",
        help="reuse the exact vocabulary from an existing checkpoint so a "
        "data A/B does not also change the output classes",
    )
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
    parser.add_argument(
        "--frame-teacher-results",
        action="append",
        default=None,
        help="repeatable; Grok result JSONL containing word timestamps. Requires "
        "the strict accepted manifest; rejected results are never used.",
    )
    parser.add_argument(
        "--frame-teacher-manifest",
        action="append",
        default=None,
        help="repeatable, parallel to --frame-teacher-results; the strict "
        "accepted teacher manifest that quality-gates its result rows",
    )
    parser.add_argument(
        "--frame-teacher-cache",
        action="append",
        default=None,
        help="repeatable, parallel to --frame-teacher-results. Names the "
        "--cache-dir whose rows this archive covers, by directory name. Every "
        "train row in that cache must have an accepted teacher or the run stops.",
    )
    parser.add_argument("--frame-loss-weight", type=float, default=0.0)
    parser.add_argument("--frame-positive-merge-gap-s", type=float, default=0.15)
    parser.add_argument("--frame-boundary-ignore-s", type=float, default=0.10)
    parser.add_argument("--frame-negative-min-s", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    # Scoped by cache rather than by domain. A domain is a mixture label used to
    # balance the loss, and it can legitimately gather several caches - galgame
    # text with timings, galgame text recovered without them, and blank-only
    # rows that have no words to time at all. A teacher archive covers exactly
    # one of those caches, so requiring every row of a *domain* to carry
    # supervision forced the domains to be split for a reason that has nothing to
    # do with the mixture, and the per-domain blank cap then dropped most of the
    # blank rows. Scoping to the cache keeps the guard - a mismatched archive
    # still stops the run - without deforming the mixture.
    teacher_results = list(args.frame_teacher_results or [])
    teacher_manifests = list(args.frame_teacher_manifest or [])
    teacher_caches = list(args.frame_teacher_cache or [])
    if len({len(teacher_results), len(teacher_manifests), len(teacher_caches)}) != 1:
        raise SystemExit(
            "--frame-teacher-results, --frame-teacher-manifest and "
            "--frame-teacher-cache must be given the same number of times"
        )
    if args.frame_loss_weight < 0.0:
        raise SystemExit("--frame-loss-weight must be non-negative")
    if bool(teacher_results) != (args.frame_loss_weight > 0.0):
        raise SystemExit(
            "a positive --frame-loss-weight and at least one frame teacher "
            "triple are required together"
        )
    if min(
        args.frame_positive_merge_gap_s,
        args.frame_boundary_ignore_s,
        args.frame_negative_min_s,
    ) < 0.0:
        raise SystemExit("frame supervision durations must be non-negative")

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
    empty_excluded = Counter()
    empty_cap_report: dict[str, dict[str, int | float]] = {}
    if args.include_empty_targets:
        train_rows, empty_cap_report = _cap_empty_train_rows(
            train_rows,
            maximum_fraction=float(args.max_empty_train_fraction),
            seed=args.seed,
        )
    else:
        for row in train_rows + val_rows:
            if not str(row.get("text") or ""):
                empty_excluded[str(row.get("partition") or "train")] += 1
        train_rows = [row for row in train_rows if str(row.get("text") or "")]
        val_rows = [row for row in val_rows if str(row.get("text") or "")]
    if not train_rows:
        raise SystemExit("feature cache has no training rows")
    mix = Counter(row["domain"] for row in train_rows)
    print(f"train rows by domain (after repeat): {dict(mix)}")

    counts: Counter[str] = Counter()
    for row in train_rows:
        counts.update(row["text"])
    if args.vocab_checkpoint:
        vocab_payload = torch.load(
            args.vocab_checkpoint, map_location="cpu", weights_only=False
        )
        vocab = AlignmentVocab.from_payload(vocab_payload["vocab"])
        if bool(vocab.acoustic_only) != bool(args.acoustic_targets):
            raise SystemExit(
                "--acoustic-targets must match the vocabulary checkpoint"
            )
    else:
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

    encoded = {_row_key(row): vocab.encode(row["text"]) for row in cache.rows}
    _key = _row_key

    frame_labels: dict[tuple[int, str], np.ndarray] = {}
    frame_teacher_summary: dict[str, object] = {}
    if teacher_results:
        frame_labels, frame_teacher_summary = compile_frame_labels_by_cache(
            train_rows,
            [path.name for path in cache.cache_dirs],
            results=teacher_results,
            manifests=teacher_manifests,
            caches=teacher_caches,
            upsample=args.upsample,
            positive_merge_gap_s=args.frame_positive_merge_gap_s,
            boundary_ignore_s=args.frame_boundary_ignore_s,
            negative_minimum_s=args.frame_negative_min_s,
        )
        print(f"frame teacher: {frame_teacher_summary}", flush=True)

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
            if items:
                yield chunk, _collate(items, torch)

    def run_epoch(
        rows: list[dict], *, train: bool
    ) -> tuple[float, int, float, float | None]:
        head.train(train)
        total_loss, total_ctc_loss, seen = 0.0, 0.0, 0
        total_frame_loss, frame_seen = 0.0, 0
        for chunk, packed in batches(
            rows, shuffle=train
        ):
            features, targets, frame_lengths, target_lengths = packed
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
                ctc_loss = criterion(
                    log_probs.transpose(0, 1),
                    targets,
                    input_lengths.to(device),
                    target_lengths.to(device),
                )
                frame_loss = None
                if train and frame_labels:
                    label_array = np.full(
                        tuple(log_probs.shape[:2]), IGNORE_LABEL, dtype=np.int8
                    )
                    for index, row in enumerate(chunk):
                        labels = frame_labels.get(_key(row))
                        if labels is not None:
                            label_array[index, : len(labels)] = labels
                    label_tensor = torch.from_numpy(label_array).to(device)
                    frame_loss, frame_counts = balanced_sparse_frame_loss(
                        log_probs, label_tensor, torch
                    )
                    if frame_counts["blank_frames"] + frame_counts["speech_frames"]:
                        frame_seen += features.shape[0]
                        total_frame_loss += (
                            float(frame_loss.detach().item()) * features.shape[0]
                        )
                loss = ctc_loss + (
                    args.frame_loss_weight * frame_loss
                    if frame_loss is not None
                    else 0.0
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
            total_ctc_loss += float(ctc_loss.item()) * features.shape[0]
            seen += features.shape[0]
        return (
            total_loss / max(1, seen),
            seen,
            total_ctc_loss / max(1, seen),
            total_frame_loss / frame_seen if frame_seen else None,
        )

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
    val_by_domain_and_kind = {
        domain: {
            kind: [
                row
                for row in rows
                if ("text" if str(row.get("text") or "") else "blank") == kind
            ]
            for kind in ("text", "blank")
            if any(
                ("text" if str(row.get("text") or "") else "blank") == kind
                for row in rows
            )
        }
        for domain, rows in val_by_domain.items()
    }

    for epoch in range(args.epochs):
        train_loss, _, train_ctc_loss, train_frame_loss = run_epoch(
            train_rows, train=True
        )
        per_domain: dict[str, float] = {}
        pooled_total, pooled_seen = 0.0, 0
        for domain, rows in val_by_domain.items():
            loss, seen, _, _ = run_epoch(rows, train=False)
            per_domain[domain] = round(loss, 4)
            pooled_total += loss * seen
            pooled_seen += seen
        per_domain_and_kind: dict[str, dict[str, float]] = {}
        for domain, by_kind in val_by_domain_and_kind.items():
            per_domain_and_kind[domain] = {}
            for kind, rows in by_kind.items():
                kind_loss, _, _, _ = run_epoch(rows, train=False)
                per_domain_and_kind[domain][kind] = round(kind_loss, 4)
        val_loss = pooled_total / pooled_seen if pooled_seen else float("nan")
        selected = per_domain.get(args.select_domain, val_loss) if args.select_domain else val_loss
        elapsed = time.perf_counter() - started
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_ctc_loss": round(train_ctc_loss, 4),
                "train_frame_loss": (
                    round(train_frame_loss, 4)
                    if train_frame_loss is not None
                    else None
                ),
                "val_loss": round(val_loss, 4),
                "val_loss_by_domain": dict(per_domain),
                "val_loss_by_domain_and_target_kind": per_domain_and_kind,
                "elapsed_s": round(elapsed, 1),
            }
        )
        detail = "  ".join(f"{name} {value:.4f}" for name, value in per_domain.items())
        frame_detail = (
            f" ctc {train_ctc_loss:.4f} frame {train_frame_loss:.4f}"
            if train_frame_loss is not None
            else ""
        )
        print(
            f"epoch {epoch:3d}  train {train_loss:.4f}{frame_detail}  "
            f"val {val_loss:.4f}  [{detail}]  {elapsed:.0f}s",
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
                    "frame_teacher": frame_teacher_summary,
                    "frame_loss_weight": float(args.frame_loss_weight),
                },
                checkpoint_path,
            )

    summary = {
        "schema": "asr_ctc_aligner_training_v1",
        "cache_dirs": [str(path) for path in args.cache_dir],
        "cache_repeat": list(args.cache_repeat or []),
        "cache_domain": list(args.cache_domain or []),
        "select_domain": args.select_domain or "",
        "include_empty_targets": bool(args.include_empty_targets),
        "max_empty_train_fraction": float(args.max_empty_train_fraction),
        "empty_target_cap_by_domain": empty_cap_report,
        "empty_targets_excluded": dict(empty_excluded),
        "vocab_checkpoint": str(args.vocab_checkpoint or ""),
        "frame_teacher_results": list(teacher_results),
        "frame_teacher_manifest": list(teacher_manifests),
        "frame_teacher_cache": list(teacher_caches),
        "frame_loss_weight": float(args.frame_loss_weight),
        "frame_positive_merge_gap_s": float(args.frame_positive_merge_gap_s),
        "frame_boundary_ignore_s": float(args.frame_boundary_ignore_s),
        "frame_negative_min_s": float(args.frame_negative_min_s),
        "frame_teacher_summary": frame_teacher_summary,
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
