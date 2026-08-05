#!/usr/bin/env python3
"""Sample real-audio windows for the safe-cut frame audit.

The pre-ASR corpus already sorts real JAV chunks into `definite_keep`,
`definite_drop` and `ambiguous_ignore`. Those classes are coarse - they answer
"does this chunk contain a word anywhere" - so they cannot be the audit's ground
truth. They are used here only as a *sampling frame*, which is a different job
and one they are good at: sampling uniformly from the pool would fill the audit
with easy silence, and the case that broke the pre-gate lives in the chunks
where breathing and words share the same seconds.

  * `definite_keep`     - chunks known to contain words. Where `word` frames and
                          the breathing between them sit next to each other.
  * `definite_drop`     - chunks known to contain none. Establishes what the
                          head's blank reading does when there is nothing to
                          lose, and supplies most of the `non_semantic_vocal`.
  * `ambiguous_ignore`  - the chunks a teacher could not decide. The pre-ASR
                          chain dropped these on the floor; for this question
                          they are the most informative pool there is.

**The class is not written into the page or the manifest the page reads.** It is
kept in the selection record only, so it can stratify the report afterwards
without ever reaching the ear that produces the labels.

Windows are cut to a fixed length for the same reason the onset audit fixed its
clip length: any per-stratum difference an auditor can notice stops the pass
being blind.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import wave

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import numpy as np  # noqa: E402

from tools.audits.pause_frame_audit import (  # noqa: E402
    FRAME_HOP_S,
    MANIFEST_SCHEMA,
    SELECTION_SCHEMA,
    write_jsonl,
)

SOURCE_CLASSES = ("definite_keep", "definite_drop", "ambiguous_ignore")
DEFAULT_COUNTS = {"definite_keep": 24, "definite_drop": 16, "ambiguous_ignore": 20}


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate() or 1
        return frames / float(rate)


def _cut(source: Path, target: Path, start_s: float, seconds: float) -> float:
    """Copy `seconds` of audio starting at `start_s`. Returns what was written.

    Rewritten rather than referenced with an offset because the page plays whole
    files: a manifest that pointed at the original with a start time would let a
    misconfigured player expose neighbouring audio the labeller was not asked
    about, and the labels would then describe a window nobody agreed on.
    """
    with wave.open(str(source), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        total = handle.getnframes()
        begin = max(0, min(total, int(start_s * rate)))
        wanted = int(seconds * rate)
        handle.setpos(begin)
        payload = handle.readframes(min(wanted, total - begin))
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(payload)
    return len(payload) / float(width * channels * rate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-root",
        default="datasets/train/omni-joint-boundary-preasr-v2/pre_asr/audio_wav",
        help="directory holding the three pre-ASR class folders",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=8.0,
        help="fixed for every window; varying it would leak the stratum",
    )
    parser.add_argument("--min-source-seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260805)
    for name in SOURCE_CLASSES:
        parser.add_argument(
            f"--{name.replace('_', '-')}-n", type=int, default=DEFAULT_COUNTS[name]
        )
    args = parser.parse_args()

    audio_root = Path(args.audio_root)
    if not audio_root.is_absolute():
        audio_root = PROJECT_ROOT / audio_root
    output_dir = Path(args.output_dir)
    media_dir = output_dir / "media"
    if media_dir.exists():
        shutil.rmtree(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    selection: list[dict] = []
    manifest: list[dict] = []
    skipped = {"too_short": 0, "unreadable": 0}
    index = 0

    for source_class in SOURCE_CLASSES:
        wanted = int(getattr(args, f"{source_class}_n"))
        if wanted <= 0:
            continue
        folder = audio_root / source_class
        candidates = sorted(folder.glob("*.wav"))
        if not candidates:
            raise SystemExit(f"no audio under {folder}")
        order = rng.permutation(len(candidates))
        taken = 0
        for position in order:
            if taken >= wanted:
                break
            path = candidates[int(position)]
            try:
                duration = _wav_duration(path)
            except Exception:  # noqa: BLE001
                skipped["unreadable"] += 1
                continue
            if duration < args.min_source_seconds:
                skipped["too_short"] += 1
                continue
            # Start anywhere that leaves a full window, so the audit is not a
            # study of how chunks happen to begin.
            span = max(0.0, duration - args.window_seconds)
            start_s = float(rng.random() * span) if span > 0 else 0.0
            row_id = f"pause-{index:04d}"
            target = media_dir / f"{row_id}.wav"
            written = _cut(path, target, start_s, args.window_seconds)
            if written <= 0.0:
                skipped["unreadable"] += 1
                continue
            frame_count = max(1, int(written / FRAME_HOP_S))
            manifest.append(
                {
                    "schema": MANIFEST_SCHEMA,
                    "row_id": row_id,
                    "audio": f"media/{row_id}.wav",
                    "duration_s": round(written, 4),
                    "frame_count": frame_count,
                    "frame_hop_s": FRAME_HOP_S,
                }
            )
            selection.append(
                {
                    "schema": SELECTION_SCHEMA,
                    "row_id": row_id,
                    # Kept out of the manifest on purpose: the page must not be
                    # able to show it, and the report needs it.
                    "source_class": source_class,
                    "source_audio": path.as_posix(),
                    "source_start_s": round(start_s, 4),
                    "duration_s": round(written, 4),
                }
            )
            index += 1
            taken += 1
        if taken < wanted:
            print(
                f"  WARNING: only {taken}/{wanted} usable windows in {source_class}",
                flush=True,
            )

    if not manifest:
        raise SystemExit("no windows selected")

    write_jsonl(output_dir / "manifest.jsonl", manifest)
    write_jsonl(output_dir / "selection.jsonl", selection)
    summary = {
        "schema": SELECTION_SCHEMA,
        "windows": len(manifest),
        "window_seconds": args.window_seconds,
        "frame_hop_s": FRAME_HOP_S,
        "by_source_class": {
            name: sum(1 for row in selection if row["source_class"] == name)
            for name in SOURCE_CLASSES
        },
        "labelled_frames_total": sum(row["frame_count"] for row in manifest),
        "skipped": skipped,
        "seed": args.seed,
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
