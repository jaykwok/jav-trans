#!/usr/bin/env python3
"""Adapter for the real-domain safe-cut frame audit: labels, contract, metrics.

**The question this audit exists to settle.** The chunker reads a run of CTC
blank as "no word here" and cuts there. That reading was falsified on real audio
once already: frame-level blank rate is 92.2% inside human-labelled speech
against 99.66% in `non_vocal`, a margin of 7.4pp, and a discarding gate built on
it lost about 26% of real words. The open question since then is *why*, and it
has two candidate answers that no measurement in this repo can currently tell
apart:

  * the head is fine and the reading is wrong - one posterior cannot serve both
    "which character" and "is anyone speaking", so a separate pause branch is
    needed;
  * the head is mistrained for this domain - it only ever saw clean galgame, and
    JAV words live inside breathing, so a pause branch trained on the same clean
    corpus would inherit the same mapping and change nothing.

Both stories predict the falsification that was observed. Separating them needs
frame-resolution truth on *real* audio, which is what these pages collect.

**Why the existing labels do not answer it.** `drop_span_words_v1` asks "is there
a word in this span" over spans whose median length is 7.47 s and p90 is 17.1 s,
while the gate decides at 38.5 ms. A 17-second span marked `speech` says nothing
about where inside it the silence is, so it cannot score a cut point.

**The three labels, and why they are these three.** The distinction that broke
the pre-gate is not speech vs silence, it is *word* vs *voiced-but-wordless*:

  * `word`                - a lexical word is being said. Cutting here truncates
                            content, which is the irreversible failure.
  * `non_semantic_vocal`  - breathing, moaning, laughter, crying. Audible, often
                            loud, carries no word. This is the class the head
                            reads as blank and the class that makes the margin
                            7.4pp instead of something usable.
  * `silence`             - no voice at all. Room tone counts as silence.
  * `unsure`              - required by the core's contract, and load-bearing
                            here: a breath with a word buried in it is exactly
                            the case that must not be forced into either side.

**Two pages, deliberately.** The labelling page shows audio and nothing else -
no blank runs, no cut points, no model output of any kind. Showing the head's
reading next to the question would anchor the ear on the thing being evaluated
and produce labels that agree with it by construction. The review page shows the
comparison, and it is generated from labels that are already frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

LABEL_WORD = "word"
LABEL_NON_SEMANTIC = "non_semantic_vocal"
LABEL_SILENCE = "silence"
LABEL_UNSURE = "unsure"

# Order is the display order on both pages, and the order the review page reads
# when it collapses a partition. `unsure` sits last because it is an escape
# hatch rather than a peer of the other three.
PARTITION_LABELS = (LABEL_WORD, LABEL_NON_SEMANTIC, LABEL_SILENCE, LABEL_UNSURE)

LABEL_TEXT = {
    LABEL_WORD: "A 有词",
    LABEL_NON_SEMANTIC: "B 非语义发声",
    LABEL_SILENCE: "C 静音",
    LABEL_UNSURE: "? 不确定",
}
LABEL_COLOR = {
    LABEL_WORD: "#2f7d32",
    LABEL_NON_SEMANTIC: "#b26a00",
    LABEL_SILENCE: "#41597a",
    LABEL_UNSURE: "#7a7a7a",
}

SELECTION_SCHEMA = "pause_frame_audit_selection_v1"
MANIFEST_SCHEMA = "pause_frame_audit_manifest_v1"
PAGE_SUMMARY_SCHEMA = "pause_frame_audit_page_summary_v1"
MANUAL_LABEL_SCHEMA = "pause_frame_manual_partition_v1"
RESULT_SCHEMA = "pause_frame_audit_result_v1"

# The audit labels at the head's own output resolution, so a labelled boundary
# and a blank run are comparable without either side being resampled. 13 fps
# encoder frames upsampled by 2 - see `asr.alignment.ENCODER_FPS`.
FRAME_HOP_S = 1.0 / 13.0 / 2.0

MANUAL_LABEL_FILENAME = "manual_pause_labels.jsonl"


@dataclass(frozen=True)
class PauseWindow:
    """One window of real audio to be partitioned by ear."""

    row_id: str
    audio: str
    source_class: str
    duration_s: float

    @property
    def frame_count(self) -> int:
        """Frames the partition must cover exactly.

        Floor rather than round: a partial trailing frame has no audio behind
        its second half, and asking someone to label it would be asking about
        silence the file does not contain.
        """
        return max(1, int(self.duration_s / FRAME_HOP_S))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def frames_to_seconds(frame: int) -> float:
    return round(float(frame) * FRAME_HOP_S, 6)


def seconds_to_frame(seconds: float) -> int:
    return int(round(float(seconds) / FRAME_HOP_S))


def partition_label_at(segments: list[dict[str, Any]], frame: int) -> str | None:
    """The label covering `frame`, or None where the partition does not reach."""
    for segment in segments:
        if int(segment["start_frame"]) <= frame < int(segment["end_frame"]):
            return str(segment["label"])
    return None


def expand_partition(segments: list[dict[str, Any]], frame_count: int) -> list[str]:
    """Per-frame labels, so the comparison is not done on span arithmetic.

    Frames the partition does not cover come back as `unsure` rather than as a
    guess: a hole is an unanswered question, and scoring it either way would let
    an incomplete labelling pass masquerade as a result.
    """
    labels = [LABEL_UNSURE] * max(0, int(frame_count))
    for segment in segments:
        begin = max(0, int(segment["start_frame"]))
        finish = min(len(labels), int(segment["end_frame"]))
        for frame in range(begin, finish):
            labels[frame] = str(segment["label"])
    return labels


def blank_frames_from_runs(
    runs: list[tuple[float, float]], frame_count: int
) -> list[bool]:
    """Turn `(start_s, end_s)` blank runs into a per-frame mask.

    A frame counts as blank when its own start falls inside a run, matching how
    `blank_runs` derived the run from the argmax in the first place - so this
    round-trips rather than re-deciding anything at the boundary.
    """
    mask = [False] * max(0, int(frame_count))
    for start_s, end_s in runs:
        begin = max(0, seconds_to_frame(start_s))
        finish = min(len(mask), seconds_to_frame(end_s))
        for frame in range(begin, finish):
            mask[frame] = True
    return mask


def confusion(labels: list[str], blank: list[bool]) -> dict[str, dict[str, int]]:
    """Frames per (human label, head reading).

    The head's reading is binary - blank or not - and the human's is not, which
    is the entire point: the interesting cell is `non_semantic_vocal` x blank,
    because that is audible voice the gate is free to cut through, and the cell
    `word` x blank is the one that loses content.
    """
    table = {
        label: {"blank": 0, "non_blank": 0}
        for label in PARTITION_LABELS
    }
    for label, is_blank in zip(labels, blank):
        bucket = table.setdefault(label, {"blank": 0, "non_blank": 0})
        bucket["blank" if is_blank else "non_blank"] += 1
    return table


def separation_report(table: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Blank rate per label, and the margin the gate would have to live on.

    `margin_pp` is the gap between the blank rate on wordless voice and on
    words. It is the number that decides the open question: a shared blank that
    cannot separate those two is not a pause detector regardless of how it was
    trained, and a pause branch is worth building only if some signal can.
    """

    def rate(label: str) -> float | None:
        row = table.get(label) or {}
        total = int(row.get("blank", 0)) + int(row.get("non_blank", 0))
        if total <= 0:
            return None
        return round(int(row["blank"]) / total, 5)

    word = rate(LABEL_WORD)
    non_semantic = rate(LABEL_NON_SEMANTIC)
    silence = rate(LABEL_SILENCE)
    margins: dict[str, Any] = {
        "blank_rate_word": word,
        "blank_rate_non_semantic_vocal": non_semantic,
        "blank_rate_silence": silence,
        "blank_rate_unsure": rate(LABEL_UNSURE),
    }
    if word is not None and non_semantic is not None:
        margins["margin_vs_non_semantic_pp"] = round((non_semantic - word) * 100.0, 2)
    if word is not None and silence is not None:
        margins["margin_vs_silence_pp"] = round((silence - word) * 100.0, 2)
    return margins


def labelled_frame_totals(labels: list[str]) -> dict[str, int]:
    totals = {label: 0 for label in PARTITION_LABELS}
    for label in labels:
        totals[label] = totals.get(label, 0) + 1
    return totals


PARTITION_EDITOR_CSS = """
.pause-card{background:#fff;border-radius:9px;padding:12px 14px;margin:14px 0;box-shadow:0 1px 3px rgba(16,32,48,.14)}
.pause-card h3{margin:0 0 6px}
.pause-strip{position:relative;height:54px;background:#e7ebef;border-radius:5px;overflow:hidden;cursor:crosshair;margin:8px 0}
.pause-seg{position:absolute;top:0;height:100%;border:0;border-right:1px solid rgba(255,255,255,.75);cursor:pointer;font-size:10px;color:#fff;overflow:hidden;white-space:nowrap;padding:0 3px}
.pause-seg.selected{outline:3px solid #111;outline-offset:-3px}
.pause-tools{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}
.pause-tools button{padding:5px 9px;border:0;border-radius:4px;cursor:pointer}
.pause-legend{display:flex;gap:10px;flex-wrap:wrap;margin:4px 0}
.pause-legend span{display:inline-flex;align-items:center;gap:4px;font-size:12px}
.pause-swatch{width:12px;height:12px;border-radius:3px;display:inline-block}
.pause-error{color:#a11;font-size:12px;min-height:16px}
.pause-compare{display:grid;grid-template-columns:150px 1fr;gap:8px;align-items:center;margin:6px 0}
.pause-disagree{background:#ffe9e9}
"""


def label_legend_html() -> str:
    parts = []
    for label in PARTITION_LABELS:
        parts.append(
            f'<span><i class="pause-swatch" style="background:{LABEL_COLOR[label]}"></i>'
            f"{LABEL_TEXT[label]}</span>"
        )
    return f'<div class="pause-legend">{"".join(parts)}</div>'


def adapter_constants_js() -> str:
    """Constants both pages share, emitted once so they cannot drift apart."""
    return (
        f"const PAUSE_LABELS={json.dumps(list(PARTITION_LABELS))};"
        f"const PAUSE_LABEL_TEXT={json.dumps(LABEL_TEXT, ensure_ascii=False)};"
        f"const PAUSE_LABEL_COLOR={json.dumps(LABEL_COLOR)};"
        f"const PAUSE_FRAME_HOP_S={FRAME_HOP_S!r};"
    )
