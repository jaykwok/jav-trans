#!/usr/bin/env python3
"""Measure how much silence each composite core carries before its first sound.

**Why this exists.** `evaluate_alignment_geometry.py` states, correctly, that
absolute onset error is not identifiable from the composite manifest alone: the
manifest records where a clip was *placed*, not where speech starts inside it, so
a core whose source has 300 ms of leading silence puts its first character 300 ms
after `start_s` with no error having occurred. That left `context_shift` as the
only free metric, and `context_shift` is blind by construction to the inset that
the geometry pass measured at 230.8 ms - it cancels when the same core is aligned
twice.

The term does not have to stay unknown. The cores are *clean* galgame clips, so
the leading silence can be measured directly on the source audio and subtracted:

    onset_error = predicted_start - (core.start_s + leading_silence)

**What this measurement is and is not.** It is an energy onset, so it is early-
biased in a known direction and late-biased in another, and neither is small
enough to ignore:

  * Fricatives and breathy onsets (`/s/`, `/h/`, whispered vowels) ramp slowly,
    so a threshold crossing lands *after* the sound began -> overstates silence.
  * Room tone, encoder noise and clicks cross the threshold before any voice ->
    understates silence.

So a single number here is not ground truth for one clip. What makes it useful is
that the bias is a property of the *clip*, not of the head being evaluated: it is
identical across arms of an A/B, so differences remain readable even though the
absolute value carries an unknown offset. This tool therefore reports the
distribution and writes per-core values for joining, and deliberately does not
present the result as an error measurement on its own.

Thresholds are relative to each clip's own peak rather than absolute, because the
corpus is not loudness-normalised; `--floor-dbfs` additionally refuses to call
anything speech below an absolute floor, which is what stops a clip of pure
silence from reporting its noise as an onset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from audio.loading import load_audio_16k_mono  # noqa: E402

SAMPLE_RATE = 16000
# v2 reports both edges. The tail was added on 2026-08-10 because
# `CODA_EXTEND_MAX_S` was sized the same contaminated way the onset cap was: its
# 371.7 ms "tail inset" is `core_end_s - last_character_end`, and `core_end_s` is
# where the clip stops, not where its speech does.
SCHEMA = "asr_core_edge_silence_v2"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve(path_text: str) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def edge_silence_s(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    window_s: float = 0.010,
    relative_db: float = -35.0,
    floor_dbfs: float = -55.0,
) -> tuple[float, float] | None:
    """(silence before the first sound, silence after the last), in seconds.

    Returns None when no window clears the absolute floor, which means the clip
    has nothing to measure rather than that its speech fills it - the caller must
    not read a missing value as 0.0.

    Both crossings round OUTWARD, toward "the speech was longer than this": the
    leading edge is reported at the START of the first loud window and the
    trailing edge at the END of the last one. A window is the resolution of the
    measurement, and rounding this way keeps the metric from manufacturing the
    very error it exists to detect - inward rounding would make the head look
    late at the head and early at the tail for free.

    **Both edges are biased the same way, and knowing which way is what makes
    them usable.** At the onset, fricatives and breathy starts (`/s/`, `/h/`)
    ramp slowly, so the crossing lands after the sound began. At the tail the
    same problem is worse in Japanese: vowel decay and word-final devoicing
    (`です` / `ます`, `〜した`) fall under the threshold while still acoustically
    present. So `L_det >= L_true` and `R_det >= R_true`: the detected speech
    interval is always a SUBSET of the real one.

    Read as an error on a predicted boundary, that gives one direction for free:

        onset_error = predicted_start - (core_start + L_det)   <= true error
        end_error   = predicted_end   - (core_end   - R_det)   >= true error

    i.e. the prediction always looks like it reaches FURTHER out than it does -
    earlier at the head, later at the tail. So "starts early" and "ends late" are
    both upper bounds, while "starts late" and "ends early" - the two readings
    that mean speech was cut off - can only be understated. **This detector
    cannot fool you into believing speech was clipped when it was not**, which is
    exactly the direction a cap has to be safe in.
    """
    if audio.size == 0:
        return None
    width = max(1, int(window_s * sample_rate))
    usable = (audio.size // width) * width
    if usable < width:
        return None
    frames = audio[:usable].reshape(-1, width)
    rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
    peak = float(rms.max())
    if peak <= 0.0:
        return None
    absolute_floor = 10.0 ** (floor_dbfs / 20.0)
    relative = peak * (10.0 ** (relative_db / 20.0))
    threshold = max(relative, absolute_floor)
    if peak < absolute_floor:
        return None
    loud = np.flatnonzero(rms >= threshold)
    if loud.size == 0:
        return None
    leading = float(loud[0] * width) / float(sample_rate)
    # Measured against the FULL clip, not the truncated window, so the tail of a
    # clip whose length is not a whole number of windows is not silently lost.
    trailing = (audio.size - float((loud[-1] + 1) * width)) / float(sample_rate)
    return leading, max(0.0, trailing)


def leading_silence_s(audio: np.ndarray, **kwargs) -> float | None:
    """Leading edge only, kept for callers that predate the tail measurement."""
    measured = edge_silence_s(audio, **kwargs)
    return None if measured is None else measured[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composites", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--window-s", type=float, default=0.010)
    parser.add_argument("--relative-db", type=float, default=-35.0)
    parser.add_argument("--floor-dbfs", type=float, default=-55.0)
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.composites))
    rng = np.random.default_rng(args.seed)
    # The same selection `evaluate_alignment_geometry.py` makes from the same
    # seed, so the two outputs join row for row without carrying an index.
    if args.limit and args.limit < len(rows):
        picked = rng.choice(len(rows), size=args.limit, replace=False)
        rows = [rows[i] for i in sorted(picked)]

    measured: list[dict] = []
    values: list[float] = []
    tail_values: list[float] = []
    missing = 0
    unreadable = 0
    for row in rows:
        cores = sorted(row.get("core_spans") or (), key=lambda c: float(c["start_s"]))
        for index, core in enumerate(cores):
            source = _resolve(str(core.get("source_audio") or ""))
            if not source.exists():
                unreadable += 1
                continue
            try:
                clip, rate = load_audio_16k_mono(str(source))
            except Exception:  # noqa: BLE001
                unreadable += 1
                continue
            if rate != SAMPLE_RATE:
                unreadable += 1
                continue
            edges = edge_silence_s(
                np.asarray(clip, dtype=np.float32),
                window_s=args.window_s,
                relative_db=args.relative_db,
                floor_dbfs=args.floor_dbfs,
            )
            if edges is None:
                missing += 1
                silence = trailing = None
            else:
                silence, trailing = edges
                values.append(silence)
                tail_values.append(trailing)
            measured.append(
                {
                    "sample_id": row.get("sample_id"),
                    "core_index": index,
                    "core_id": core.get("core_id"),
                    "core_start_s": round(float(core["start_s"]), 4),
                    "core_end_s": round(float(core["end_s"]), 4),
                    "leading_silence_ms": (
                        None if silence is None else round(silence * 1000.0, 1)
                    ),
                    "trailing_silence_ms": (
                        None if trailing is None else round(trailing * 1000.0, 1)
                    ),
                }
            )

    def _describe(edge: list[float]) -> dict:
        if not edge:
            return {"count": 0}
        array = np.asarray(edge, dtype=np.float64) * 1000.0

        def pct(fraction: float) -> float:
            return round(float(np.percentile(array, fraction * 100.0)), 1)

        return {
            "count": len(edge),
            "median": round(statistics.median(edge) * 1000.0, 1),
            "mean": round(statistics.fmean(edge) * 1000.0, 1),
            "p05": pct(0.05),
            "p90": pct(0.90),
            "p99": pct(0.99),
            "max": round(max(edge) * 1000.0, 1),
            "zero_share": round(float((array <= 0.0).mean()), 4),
        }

    summary = {
        "schema": SCHEMA,
        "composites": str(args.composites),
        "cores_measured": len(measured),
        "cores_without_signal": missing,
        "cores_unreadable": unreadable,
        "detector": {
            "window_s": args.window_s,
            "relative_db": args.relative_db,
            "floor_dbfs": args.floor_dbfs,
        },
        "leading_silence_ms": _describe(values),
        "trailing_silence_ms": _describe(tail_values),
        # The figures the two caps were originally sized against, so a reader can
        # see at a glance how much of each "inset" was never alignment error.
        "recorded_inset_2026_07_31_ms": {"onset": 230.8, "coda": 371.7},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with output.with_suffix(".details.jsonl").open("w", encoding="utf-8") as handle:
        for record in measured:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
