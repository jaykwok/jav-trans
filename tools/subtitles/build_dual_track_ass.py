#!/usr/bin/env python3
"""Overlay two subtitle versions on one video, one along the top, one along the bottom.

Two runs of this pipeline do not share cue boundaries: a change to the alignment
head moves the starts and ends, and a change to the cue plan merges, splits and
drops cues outright. So the two tracks cannot be zipped into one file with a
shared timeline - any pairing rule would be inventing correspondences and then
displaying them as fact.

Instead both tracks are written as independent event streams in one ASS file,
the old one anchored to the top and the new one to the bottom. Nothing is
matched up; each line simply appears when its own version says it should. What
you are watching for is precisely the disagreement: a line that shows up early
on one row, a row that stays empty through a passage of moaning, two rows whose
text drifts apart.

The summary printed at the end counts only what needs no pairing to be true
(cue counts, total on-screen seconds, character totals). Per-cue timing deltas
are deliberately not reported here - `tools/align/compare_head_to_teacher.py`
does that properly, against speech islands rather than cue boundaries.
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TIME = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)
TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    """(start, end, text) per cue. Tolerant of missing indices and blank runs."""
    raw = path.read_text(encoding="utf-8-sig")
    cues: list[tuple[float, float, str]] = []
    for chunk in re.split(r"\r?\n\r?\n+", raw.strip()):
        lines = [line for line in chunk.splitlines() if line.strip()]
        if not lines:
            continue
        match = None
        text_from = 0
        for index, line in enumerate(lines[:2]):
            match = TIME.search(line)
            if match:
                text_from = index + 1
                break
        if not match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
        start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1.ljust(3, "0")) / 1000
        end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2.ljust(3, "0")) / 1000
        text = "\n".join(lines[text_from:]).strip()
        if text:
            cues.append((start, end, text))
    return cues


def ass_time(value: float) -> str:
    value = max(0.0, value)
    hours, rest = divmod(value, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{int(hours)}:{int(minutes):02d}:{seconds:05.2f}"


def ass_text(text: str) -> str:
    text = TAG.sub("", html.unescape(text))
    text = text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
    return "\\N".join(line.strip() for line in text.splitlines() if line.strip())


def build(
    old: list[tuple[float, float, str]],
    new: list[tuple[float, float, str]],
    *,
    old_label: str,
    new_label: str,
    play_res: tuple[int, int] = (1920, 1080),
    font_size: int = 52,
) -> str:
    width, height = play_res
    # &HAABBGGRR. Top row is deliberately the duller colour: the bottom row is
    # the candidate being judged and should read as the primary track.
    head = f"""[Script Info]
; Two subtitle versions overlaid for comparison - NOT a shipping subtitle.
;   top    = {old_label}
;   bottom = {new_label}
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: {width}
PlayResY: {height}
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Old,Arial,{font_size},&H00B4B4B4,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,1,8,40,40,32,1
Style: New,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00202020,&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    rows = []
    for style, cues, label in (("Old", old, old_label), ("New", new, new_label)):
        for start, end, text in cues:
            if end <= start:
                end = start + 0.05
            rows.append(
                (
                    start,
                    f"Dialogue: 0,{ass_time(start)},{ass_time(end)},{style},"
                    f"{label},0,0,0,,{ass_text(text)}",
                )
            )
    rows.sort(key=lambda row: row[0])
    return head + "\n".join(row for _, row in rows) + "\n"


def on_screen_seconds(cues: list[tuple[float, float, str]]) -> float:
    """Union of the cue intervals, so overlaps are not double counted."""
    total = 0.0
    finish = -1.0
    for start, end, _ in sorted(cues):
        start = max(start, finish)
        if end > start:
            total += end - start
            finish = end
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True, help="SRT shown along the top")
    parser.add_argument("--new", required=True, help="SRT shown along the bottom")
    parser.add_argument("--old-label", default="old")
    parser.add_argument("--new-label", default="new")
    parser.add_argument("--font-size", type=int, default=52)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    old = parse_srt(Path(args.old))
    new = parse_srt(Path(args.new))
    if not old or not new:
        raise SystemExit(f"empty track: old={len(old)} new={len(new)}")

    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build(
            old,
            new,
            old_label=args.old_label,
            new_label=args.new_label,
            font_size=args.font_size,
        ),
        encoding="utf-8-sig",
    )

    print(f"{'':>14} {'cues':>7} {'on-screen s':>12} {'chars':>8} {'median s':>9}")
    for label, cues in ((args.old_label, old), (args.new_label, new)):
        spans = sorted(end - start for start, end, _ in cues)
        median = spans[len(spans) // 2] if spans else 0.0
        chars = sum(len(text.replace("\n", "")) for _, _, text in cues)
        print(
            f"{label:>14} {len(cues):>7} {on_screen_seconds(cues):>12.1f} "
            f"{chars:>8} {median:>9.2f}"
        )
    print(f"\nwrote {out}")
    print("play the video with this file as the subtitle track (mpv/VLC/PotPlayer):")
    print(f"  mpv <video> --sub-file={out}")


if __name__ == "__main__":
    main()
