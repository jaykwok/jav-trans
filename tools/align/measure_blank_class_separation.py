#!/usr/bin/env python3
"""Split the head's blank frames into "voiced but not a word" and "silent".

**The question this answers.** The gate reading of the head (`alignment.blank_runs`,
consumed by `chunking.cut_at_pauses`) places chunk cuts on runs of argmax blank.
The distinction that matters there is not speech/non-speech but **word / voiced
but wordless**: a breath can be loud and still be a legal place to cut, while a
whisper is a word and is not. Grok's word timestamps cannot answer this on their
own, because Grok is equally silent over a moan and over silence - its labels
collapse the two classes this tool exists to separate.

They can be separated without human labels on *clean* single-speaker galgame:
inside an accepted clip, a region with no Grok word but with acoustic energy is
voiced-and-wordless, and a region with neither is silence. That yields the
`margin_vs_non_semantic_pp` quantity the 2026-08-05 pause-frame audit was built
to obtain, over the whole held-out partition instead of 60 hand-labelled windows.

**What the energy split is and is not.** It is an energy decision, so it fires on
breath, room tone, clicks, and any SFX in the clip - it is not a moan detector,
and no single frame's class should be read as truth. What makes it usable is that
the bias belongs to the *audio*, not to the head being measured: it is identical
across arms of an A/B, and the reported quantity is a margin between classes
rather than an absolute rate. The threshold is swept for the same reason - a
conclusion that survives only at one `--relative-db` is a conclusion about the
threshold.

**Held-out only by default.** Train rows had their >=0.5 s gaps used as blank
frame supervision, so measuring blank behaviour there is circular.
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
    BLANK_INDEX,
    ENCODER_FRAME_S,
    SUPPORTED_ALIGNMENT_MODEL_SCHEMAS,
    AlignmentVocab,
    build_head,
)
from audio.loading import load_audio_16k_mono  # noqa: E402
from tools.align.frame_teacher_supervision import (  # noqa: E402
    load_accepted_frame_teachers,
    merge_intervals,
)

SCHEMA = "asr_blank_class_separation_v1"
SAMPLE_RATE = 16000
ENERGY_WINDOW_S = 0.010

WORD = "word"
VOICED_WORDLESS = "voiced_wordless"
SILENT = "silent"
CLASSES = (WORD, VOICED_WORDLESS, SILENT)


def resolve_repo_path(path_text: str) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def frame_energy(
    audio: np.ndarray,
    *,
    frame_count: int,
    frame_s: float,
    sample_rate: int = SAMPLE_RATE,
    window_s: float = ENERGY_WINDOW_S,
) -> np.ndarray | None:
    """Peak short-window RMS inside each head output frame.

    The peak rather than the mean over sub-windows: a frame holding one short
    voiced burst is voiced, and averaging it against surrounding silence would
    report it as a pause - the exact error this tool exists to detect.
    """
    if audio.size == 0 or frame_count < 1:
        return None
    width = max(1, int(round(window_s * sample_rate)))
    usable = (audio.size // width) * width
    if usable < width:
        return None
    windows = audio[:usable].reshape(-1, width).astype(np.float64)
    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    energy = np.zeros(frame_count, dtype=np.float64)
    for frame in range(frame_count):
        start = int(np.floor(frame * frame_s / window_s))
        end = int(np.ceil((frame + 1) * frame_s / window_s))
        start = min(max(0, start), rms.size)
        end = min(max(start + 1, end), rms.size)
        if start >= rms.size:
            break
        energy[frame] = float(rms[start:end].max())
    return energy


def energy_threshold(
    energy: np.ndarray, *, relative_db: float, floor_dbfs: float
) -> float | None:
    """Clip-relative voicing threshold, floored in absolute terms.

    Relative because the corpus is not loudness-normalised; floored because a
    clip that is entirely silence would otherwise report its own noise as the
    loudest thing in it and call that voiced.
    """
    peak = float(energy.max()) if energy.size else 0.0
    absolute_floor = 10.0 ** (floor_dbfs / 20.0)
    if peak <= 0.0 or peak < absolute_floor:
        return None
    return max(peak * (10.0 ** (relative_db / 20.0)), absolute_floor)


def classify_frames(
    *,
    frame_count: int,
    frame_s: float,
    duration_s: float,
    islands: list[tuple[float, float]],
    energy: np.ndarray,
    threshold: float,
    boundary_ignore_s: float,
    long_gap_min_s: float,
) -> dict[str, np.ndarray]:
    """Return boolean masks for the three classes plus the long-gap subset.

    Frames within ``boundary_ignore_s`` of a word edge belong to neither class:
    onset and offset are exactly where the teacher's own timestamps are least
    trustworthy, and letting them fall into either class would move the margin
    without any of the underlying acoustics changing.
    """
    centers = (np.arange(frame_count, dtype=np.float64) + 0.5) * frame_s
    inside_audio = centers < duration_s
    word = np.zeros(frame_count, dtype=bool)
    protected = np.zeros(frame_count, dtype=bool)
    for start, end in islands:
        word |= (centers >= start) & (centers < end)
        protected |= (centers >= start - boundary_ignore_s) & (
            centers < end + boundary_ignore_s
        )
    gap = inside_audio & ~protected

    # Gap runs, so the caller can restrict to the regime the blank labels were
    # actually trained on (>= 0.5 s) instead of extrapolating to the syllable
    # spacing inside continuous speech.
    long_gap = np.zeros(frame_count, dtype=bool)
    cursor = 0.0
    bounds: list[tuple[float, float]] = []
    for start, end in islands:
        bounds.append((cursor, max(cursor, start - boundary_ignore_s)))
        cursor = max(cursor, end + boundary_ignore_s)
    bounds.append((cursor, duration_s))
    for start, end in bounds:
        if end - start >= long_gap_min_s:
            long_gap |= (centers >= start) & (centers < end)
    long_gap &= gap

    voiced = gap & (energy >= threshold)
    silent = gap & (energy < threshold)
    return {
        WORD: word & inside_audio,
        VOICED_WORDLESS: voiced,
        SILENT: silent,
        "long_gap": long_gap,
    }


def _rate(mask: np.ndarray, values: np.ndarray) -> float | None:
    if not bool(mask.any()):
        return None
    return float(values[mask].mean())


def _summarize(
    pooled: dict[str, dict[str, float]],
    per_clip: dict[str, list[float]],
    frame_s: float,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in CLASSES:
        counts = pooled[name]
        frames = counts["frames"]
        out[name] = {
            "frames": int(frames),
            "seconds": round(frames * frame_s, 2),
            "argmax_blank_rate": (
                round(counts["blank"] / frames, 6) if frames else None
            ),
            "mean_blank_probability": (
                round(counts["probability"] / frames, 6) if frames else None
            ),
        }
    rates = {
        name: (
            pooled[name]["blank"] / pooled[name]["frames"]
            if pooled[name]["frames"]
            else None
        )
        for name in CLASSES
    }

    def margin(left: str, right: str) -> float | None:
        if rates[left] is None or rates[right] is None:
            return None
        return round((rates[left] - rates[right]) * 100.0, 4)

    out["margins_pp"] = {
        # The 2026-08-05 definition: blank rate on non-semantic vocalisation
        # minus blank rate on words. Wide => the gate can separate them.
        "margin_vs_non_semantic_pp": margin(VOICED_WORDLESS, WORD),
        "margin_vs_silence_pp": margin(SILENT, WORD),
        # How much of the blank behaviour is only silence detection. Large =>
        # the head answers "is it quiet", not "is it a word".
        "silence_over_voiced_pp": margin(SILENT, VOICED_WORDLESS),
    }
    out["per_clip_median_margins_pp"] = {
        key: (round(statistics.median(values) * 100.0, 4) if values else None)
        for key, values in per_clip.items()
    }
    out["per_clip_counts"] = {key: len(values) for key, values in per_clip.items()}
    return out


def measure(
    *,
    checkpoint: Path,
    cache_dir: Path,
    teacher_results: Path,
    teacher_manifest: Path,
    partition: str,
    relative_db_sweep: list[float],
    floor_dbfs: float,
    boundary_ignore_s: float,
    positive_merge_gap_s: float,
    long_gap_min_s: float,
    limit: int,
) -> dict[str, object]:
    import torch

    from tools.align.train_ctc_aligner import FeatureCache

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if str(payload.get("schema") or "") not in SUPPORTED_ALIGNMENT_MODEL_SCHEMAS:
        raise SystemExit(f"not an alignment checkpoint: {payload.get('schema')!r}")
    vocab = AlignmentVocab.from_payload(payload["vocab"])
    upsample = int(payload["upsample"])
    frame_s = ENCODER_FRAME_S / float(upsample)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # This measurement is entirely about the CTC blank column and does not read
    # the frame classes at all - but a v2 checkpoint carries the extra layer, and
    # `load_state_dict` is strict, so the head has to be built with it or the
    # load fails on a tensor this tool will never look at.
    head = build_head(
        vocab_size=vocab.size,
        input_dim=int(payload.get("input_dim", 2048)),
        hidden_dim=int(payload["hidden_dim"]),
        upsample=upsample,
        blocks=int(payload["blocks"]),
        dropout=0.0,
        frame_classes=len(payload.get("frame_classes") or []),
    )
    head.load_state_dict(payload["state_dict"])
    head.to(device).eval()

    cache = FeatureCache([cache_dir], domains=["measurement"])
    rows = [row for row in cache.rows if row.get("partition") == partition]
    if not rows:
        raise SystemExit(f"cache has no {partition!r} rows")
    if limit > 0:
        rows = rows[:limit]

    teachers, teacher_summary = load_accepted_frame_teachers(
        teacher_results, teacher_manifest
    )
    audio_by_source: dict[str, str] = {}
    with teacher_results.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            audio_by_source[str(record.get("source_id") or "")] = str(
                record.get("audio") or ""
            )

    pooled = {
        db: {name: defaultdict(float) for name in CLASSES}
        for db in relative_db_sweep
    }
    pooled_long = {
        db: {name: defaultdict(float) for name in CLASSES}
        for db in relative_db_sweep
    }
    per_clip = {
        db: defaultdict(list) for db in relative_db_sweep
    }
    skipped: dict[str, int] = defaultdict(int)
    measured_clips = 0

    for row in rows:
        source_id = str(row.get("source_id") or "")
        teacher = teachers.get(source_id)
        if teacher is None:
            skipped["no_accepted_teacher"] += 1
            continue
        audio_path = audio_by_source.get(source_id) or ""
        if not audio_path:
            skipped["no_audio_path"] += 1
            continue
        resolved = resolve_repo_path(audio_path)
        if not resolved.exists():
            skipped["audio_missing"] += 1
            continue

        features = cache.features(row)
        with torch.no_grad():
            tensor = torch.from_numpy(features[None, ...]).to(device)
            lengths = torch.tensor([features.shape[0]], dtype=torch.long)
            log_probs = head(tensor, lengths)[0].float().cpu().numpy()
        frame_count = log_probs.shape[0]
        is_blank = (log_probs.argmax(axis=-1) == BLANK_INDEX).astype(np.float64)
        blank_probability = np.exp(log_probs[:, BLANK_INDEX]).astype(np.float64)

        audio, audio_sample_rate = load_audio_16k_mono(str(resolved))
        energy = frame_energy(
            audio,
            frame_count=frame_count,
            frame_s=frame_s,
            sample_rate=int(audio_sample_rate),
        )
        if energy is None:
            skipped["audio_too_short"] += 1
            continue
        duration_s = min(float(teacher["duration_s"]), frame_count * frame_s)
        islands = merge_intervals(
            teacher["lexical_intervals"], maximum_gap_s=positive_merge_gap_s
        )
        islands = [
            (max(0.0, start), min(duration_s, end))
            for start, end in islands
            if min(duration_s, end) > max(0.0, start)
        ]
        if not islands:
            skipped["no_islands_in_window"] += 1
            continue

        used = False
        for relative_db in relative_db_sweep:
            threshold = energy_threshold(
                energy, relative_db=relative_db, floor_dbfs=floor_dbfs
            )
            if threshold is None:
                skipped[f"below_absolute_floor@{relative_db}"] += 1
                continue
            masks = classify_frames(
                frame_count=frame_count,
                frame_s=frame_s,
                duration_s=duration_s,
                islands=islands,
                energy=energy,
                threshold=threshold,
                boundary_ignore_s=boundary_ignore_s,
                long_gap_min_s=long_gap_min_s,
            )
            for name in CLASSES:
                mask = masks[name]
                pooled[relative_db][name]["frames"] += float(mask.sum())
                pooled[relative_db][name]["blank"] += float(is_blank[mask].sum())
                pooled[relative_db][name]["probability"] += float(
                    blank_probability[mask].sum()
                )
                long_mask = mask & masks["long_gap"] if name != WORD else mask
                pooled_long[relative_db][name]["frames"] += float(long_mask.sum())
                pooled_long[relative_db][name]["blank"] += float(
                    is_blank[long_mask].sum()
                )
                pooled_long[relative_db][name]["probability"] += float(
                    blank_probability[long_mask].sum()
                )
            word_rate = _rate(masks[WORD], is_blank)
            voiced_rate = _rate(masks[VOICED_WORDLESS], is_blank)
            silent_rate = _rate(masks[SILENT], is_blank)
            if word_rate is not None and voiced_rate is not None:
                per_clip[relative_db]["margin_vs_non_semantic_pp"].append(
                    voiced_rate - word_rate
                )
            if word_rate is not None and silent_rate is not None:
                per_clip[relative_db]["margin_vs_silence_pp"].append(
                    silent_rate - word_rate
                )
            if voiced_rate is not None and silent_rate is not None:
                per_clip[relative_db]["silence_over_voiced_pp"].append(
                    silent_rate - voiced_rate
                )
            used = True
        if used:
            measured_clips += 1

    return {
        "schema": SCHEMA,
        "checkpoint": str(checkpoint),
        "cache_dir": str(cache_dir),
        "partition": partition,
        "frame_s": round(frame_s, 6),
        "upsample": upsample,
        "clips_considered": len(rows),
        "clips_measured": measured_clips,
        "skipped": dict(sorted(skipped.items())),
        "teacher": teacher_summary,
        "parameters": {
            "relative_db_sweep": relative_db_sweep,
            "floor_dbfs": floor_dbfs,
            "boundary_ignore_s": boundary_ignore_s,
            "positive_merge_gap_s": positive_merge_gap_s,
            "long_gap_min_s": long_gap_min_s,
            "energy_window_s": ENERGY_WINDOW_S,
        },
        "all_gap_frames": {
            str(db): _summarize(pooled[db], per_clip[db], frame_s)
            for db in relative_db_sweep
        },
        "long_gap_frames_only": {
            str(db): _summarize(pooled_long[db], defaultdict(list), frame_s)
            for db in relative_db_sweep
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/ctc_aligner.pt")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--teacher-results", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--partition", default="val")
    parser.add_argument(
        "--relative-db",
        type=float,
        nargs="+",
        default=[-30.0, -35.0, -40.0],
        help="voicing threshold below each clip's own peak; swept, not tuned",
    )
    parser.add_argument("--floor-dbfs", type=float, default=-55.0)
    parser.add_argument("--boundary-ignore-s", type=float, default=0.10)
    parser.add_argument("--positive-merge-gap-s", type=float, default=0.15)
    parser.add_argument("--long-gap-min-s", type=float, default=0.50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = measure(
        checkpoint=resolve_repo_path(args.checkpoint),
        cache_dir=resolve_repo_path(args.cache_dir),
        teacher_results=resolve_repo_path(args.teacher_results),
        teacher_manifest=resolve_repo_path(args.teacher_manifest),
        partition=args.partition,
        relative_db_sweep=[float(value) for value in args.relative_db],
        floor_dbfs=float(args.floor_dbfs),
        boundary_ignore_s=float(args.boundary_ignore_s),
        positive_merge_gap_s=float(args.positive_merge_gap_s),
        long_gap_min_s=float(args.long_gap_min_s),
        limit=int(args.limit),
    )
    output = resolve_repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["all_gap_frames"], ensure_ascii=False, indent=2))
    print(f"clips measured: {report['clips_measured']} -> {output}")


if __name__ == "__main__":
    main()
