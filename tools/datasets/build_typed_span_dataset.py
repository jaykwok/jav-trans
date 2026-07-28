#!/usr/bin/env python3
"""Normalize every existing boundary label family into one typed-span dataset.

Three label families describe the same underlying task at different angles:

  omni-joint pre_asr   real JAV windows, {start,end} chunks with keep/drop and
                       free-form Omni flags.  Chunks cover only part of a window
                       (median ~44s of 75s), so uncovered time is UNKNOWN, not
                       negative - the distinction that makes or breaks the target.
  omni-joint split     real JAV windows, boundary points labelled cut/continue/
                       unsure at candidate times.  Point supervision, kept
                       alongside the spans rather than merged into them.
  hardmix composites   synthetic timelines with exact speech segments and
                       complete coverage.

Output is `typed_span_dataset_v1`: one row per example, carrying spans typed as
speech / non_semantic_vocal / non_vocal / unsure plus an explicit `coverage` flag
and a `group_id` for leakage-safe, provenance-stratified splits.

Two labels are emitted per span on purpose:
  `speech`  - the binary decision, known whenever the teacher said keep or drop
  `type`    - the 3-way subtype, `unsure` when the flags cannot resolve it
A drop whose subtype is unresolvable still supervises "not speech" without
inventing a subtype it never carried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator, Mapping
import wave

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_SCHEMA = "typed_span_dataset_v1"
FRAME_HOP_S = 0.02
# Chunk ends may sit a hair past the decoded window because the Omni teacher
# rounded; anything beyond this is a real inconsistency and gets dropped.
EDGE_TOLERANCE_S = 0.1

TYPE_SPEECH = "speech"
TYPE_NON_SEMANTIC_VOCAL = "non_semantic_vocal"
TYPE_NON_VOCAL = "non_vocal"
TYPE_UNSURE = "unsure"

OMNI_DATASETS = (
    "omni-joint-boundary-preasr-v1",
    "omni-joint-boundary-preasr-v2",
    "omni-joint-boundary-preasr-v3",
)
HARDMIX_DATASETS = (
    "hardmix-real-drop-4096",
    "hardmix-realism-16384",
)
# The 2026-07-27 rebuild. Kept separate from the originals rather than replacing
# them, because the two differ in exactly the defect under test: the originals
# cropped 86% of speech units to a flat 4.000 s, so both of that unit's ends are
# arbitrary mid-utterance cuts labelled as speech onset/offset.
HARDMIX_FIXED_DATASETS = ("hardmix-fixedboundary-4096",)

# The Omni flag vocabulary has 286 surface forms for a handful of concepts
# ("groan"/"groaning", "kiss"/"kiss_sound"/"kissing_sound", "breath"/
# "short_breath"/"only_breathing").  Match on stems, longest first, so
# "non_speech" is not shadowed by "speech".
# Flags that assert only the ABSENCE of lexical content, never what the sound
# actually is. The teacher used `non_speech` as a synonym for "drop" - a moan,
# a breath and a laugh are all non_speech - so reading it as "not a human
# sound" typed 1866 of the 2945 non_vocal spans (63.4%) from a flag that never
# made that claim. They carry no type evidence and fall through to `unsure`.
NON_LEXICAL_STEMS = (
    "non_speech",
    "no_speech",
    "not_speech",
    "speechless",
)
NON_VOCAL_STEMS = (
    "silence",
    "silent",
    "pause",
    "noise",
    "music",
    "bgm",
    "mechanical",
    "impact",
    "water",
    "cloth",
    "ambien",
    "background_only",
    "instrument",
)
NON_SEMANTIC_VOCAL_STEMS = (
    "breath",
    "moan",
    "groan",
    "gasp",
    "pant",
    "sigh",
    "cry",
    "sob",
    "laugh",
    "giggle",
    "kiss",
    "lick",
    "saliva",
    "swallow",
    "scream",
    "shout",
    "grunt",
    "whimper",
    "non_verbal",
    "nonverbal",
    "vocalization",
    "vocalisation",
    "interjection",
)
SPEECH_STEMS = (
    "speech",
    "dialogue",
    "dialog",
    "talk",
    "voice_content",
    "utterance",
    "word",
    "line",
    "singing",
    "sing",
    "whisper",
    "narration",
)


def _classify_flags(flags: Iterable[str]) -> tuple[bool, bool, bool]:
    """Return (has_speech, has_non_semantic_vocal, has_non_vocal) for the flags."""
    has_speech = False
    has_vocal = False
    has_non_vocal = False
    for raw in flags:
        flag = str(raw).strip().lower()
        if not flag:
            continue
        # Order matters. Concrete acoustic descriptions win first, so a flag
        # like "non_speech_noise" is still read as noise rather than being
        # discarded for containing "non_speech". Only a flag with no concrete
        # evidence at all falls through to NON_LEXICAL_STEMS, which is also
        # what keeps "non_speech" away from the "speech" stem it contains.
        if any(stem in flag for stem in NON_VOCAL_STEMS):
            has_non_vocal = True
            continue
        if any(stem in flag for stem in NON_SEMANTIC_VOCAL_STEMS):
            has_vocal = True
            continue
        if any(stem in flag for stem in NON_LEXICAL_STEMS):
            continue
        if any(stem in flag for stem in SPEECH_STEMS):
            has_speech = True
    return has_speech, has_vocal, has_non_vocal


def resolve_drop_subtype(flags: Iterable[str]) -> str:
    """Subtype for a dropped chunk, or `unsure` when the flags do not decide."""
    _, has_vocal, has_non_vocal = _classify_flags(flags)
    if has_vocal and not has_non_vocal:
        return TYPE_NON_SEMANTIC_VOCAL
    if has_non_vocal and not has_vocal:
        return TYPE_NON_VOCAL
    if has_vocal and has_non_vocal:
        # Mixed evidence: a human vocal cue is present, so the safer subtype for
        # a high-recall segmenter is the vocal one.
        return TYPE_NON_SEMANTIC_VOCAL
    return TYPE_UNSURE


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        if rate <= 0:
            raise ValueError(f"invalid sample rate in {path}")
        return handle.getnframes() / float(rate)


def _video_id_of(window_id: str) -> str:
    """Strip the trailing window suffix so splits can group by source video."""
    head, sep, tail = window_id.rpartition("-w")
    if sep and tail.isdigit():
        return head
    return window_id


def _sorted_disjoint(spans: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Sort spans and drop any that overlap an already-accepted one.

    Used for the real chunks, where neighbouring spans carry different types and
    cannot be unioned.  Same-type spans must go through `_merge_same_type`
    instead: dropping one there would turn its tail into a false negative.
    """
    ordered = sorted(spans, key=lambda s: (float(s["start_s"]), float(s["end_s"])))
    accepted: list[dict[str, Any]] = []
    dropped = 0
    for span in ordered:
        if accepted and float(span["start_s"]) < float(accepted[-1]["end_s"]) - 1e-6:
            dropped += 1
            continue
        accepted.append(span)
    return accepted, dropped


def _merge_same_type(spans: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Union overlapping or touching spans that share one type.

    Synthetic composites legitimately place overlapping speech units (the layout
    allows concurrent utterances).  For a speech/non-speech target the union is
    the correct truth; dropping the later span would label its uncovered tail as
    non-speech.
    """
    ordered = sorted(spans, key=lambda s: (float(s["start_s"]), float(s["end_s"])))
    merged: list[dict[str, Any]] = []
    joins = 0
    for span in ordered:
        if merged and float(span["start_s"]) <= float(merged[-1]["end_s"]) + 1e-6:
            if float(span["end_s"]) > float(merged[-1]["end_s"]):
                merged[-1] = dict(merged[-1], end_s=round(float(span["end_s"]), 6))
            joins += 1
            continue
        merged.append(dict(span))
    return merged, joins


def build_omni_examples(
    base: Path, *, stats: dict[str, int]
) -> Iterator[dict[str, Any]]:
    """One example per real JAV window, with sparse typed spans."""
    pre_asr = base / "pre_asr" / "labels.jsonl"
    if not pre_asr.is_file():
        return

    by_window: dict[str, list[dict[str, Any]]] = {}
    for row in _rows(pre_asr):
        by_window.setdefault(str(row.get("window_id")), []).append(row)

    points_by_window: dict[str, list[dict[str, Any]]] = {}
    split_labels = base / "semantic_split" / "labels.jsonl"
    if split_labels.is_file():
        for row in _rows(split_labels):
            label = str(row.get("label") or "")
            if label not in {"cut", "continue", "unsure"}:
                continue
            points_by_window.setdefault(str(row.get("window_id")), []).append(
                {"time_s": float(row.get("time_s", 0.0)), "label": label}
            )

    for window_id in sorted(by_window):
        audio = base / "audio_wav" / f"{window_id}.wav"
        if not audio.is_file():
            stats["window_missing_audio"] += 1
            continue
        duration = _wav_duration_s(audio)

        spans: list[dict[str, Any]] = []
        for row in by_window[window_id]:
            start = float(row.get("start", 0.0))
            end = float(row.get("end", 0.0))
            if end <= start:
                stats["chunk_empty"] += 1
                continue
            if end > duration + EDGE_TOLERANCE_S or start < -EDGE_TOLERANCE_S:
                stats["chunk_out_of_range"] += 1
                continue
            start = max(0.0, start)
            end = min(duration, end)
            if end - start < FRAME_HOP_S:
                stats["chunk_shorter_than_frame"] += 1
                continue

            omni_label = str(row.get("omni_label") or "")
            training_label = str(row.get("label") or "")
            flags = [str(f) for f in (row.get("omni_flags") or [])]

            if training_label == "ambiguous_ignore":
                span_type = TYPE_UNSURE
                speech: bool | None = None
                stats["span_unsure"] += 1
            elif omni_label == "keep":
                span_type = TYPE_SPEECH
                speech = True
                stats["span_speech"] += 1
            elif omni_label == "drop":
                span_type = resolve_drop_subtype(flags)
                speech = False
                stats[f"span_drop_{span_type}"] += 1
            else:
                stats["chunk_unknown_label"] += 1
                continue

            spans.append(
                {
                    "start_s": round(start, 6),
                    "end_s": round(end, 6),
                    "type": span_type,
                    "speech": speech,
                    "source_label": training_label or omni_label,
                    "flags": flags,
                }
            )

        if not spans:
            stats["window_no_spans"] += 1
            continue
        spans, overlap_dropped = _sorted_disjoint(spans)
        stats["span_overlap_dropped"] += overlap_dropped

        covered = sum(float(s["end_s"]) - float(s["start_s"]) for s in spans)
        yield {
            "schema": OUTPUT_SCHEMA,
            "example_id": f"{base.name}:{window_id}",
            "provenance": "real_omni_joint",
            "dataset": base.name,
            "audio": str(audio),
            "audio_sha256": _sha256(audio),
            "duration_s": round(duration, 6),
            "frame_hop_s": FRAME_HOP_S,
            "frame_count": int(round(duration / FRAME_HOP_S)),
            "video_id": _video_id_of(window_id),
            "window_id": window_id,
            "group_id": _video_id_of(window_id),
            # Leakage units: everything this example shares with other examples.
            # Splitting must keep all of them on one side.
            "group_members": [f"video:{_video_id_of(window_id)}"],
            # Uncovered time is unlabelled, never an implicit negative.
            "coverage": "sparse",
            "covered_s": round(covered, 6),
            "spans": spans,
            "boundary_points": sorted(
                points_by_window.get(window_id, []), key=lambda p: p["time_s"]
            ),
        }


def build_hardmix_examples(
    base: Path, *, stats: dict[str, int], limit: int = 0
) -> Iterator[dict[str, Any]]:
    """One example per synthetic composite, with complete coverage.

    `actual_speech_segments` is the undilated truth; `teacher_segments` is the
    dilated training view.  The undilated form is authoritative here so the
    dilation stays a downstream choice rather than being baked in.
    """
    labels = base / "labels.jsonl"
    if not labels.is_file():
        return
    audio_dir = base / "audio"

    for index, row in enumerate(_rows(labels)):
        if limit and index >= limit:
            return
        audio_id = str(row.get("audio_id") or "")
        duration = float(row.get("duration_s") or 0.0)
        if duration <= 0.0:
            stats["synthetic_bad_duration"] += 1
            continue
        audio = audio_dir / f"{audio_id}.wav"
        if not audio.is_file():
            stats["synthetic_missing_audio"] += 1
            continue

        meta = row.get("boundary_metadata") or {}
        segments = meta.get("actual_speech_segments") or []
        speech_spans: list[dict[str, Any]] = []
        for segment in segments:
            start = max(0.0, float(segment.get("start", 0.0)))
            end = min(duration, float(segment.get("end", 0.0)))
            if end - start < FRAME_HOP_S:
                stats["synthetic_segment_too_short"] += 1
                continue
            speech_spans.append(
                {
                    "start_s": round(start, 6),
                    "end_s": round(end, 6),
                    "type": TYPE_SPEECH,
                    "speech": True,
                    "source_label": "synthetic_speech_unit",
                    "flags": [],
                }
            )
        speech_spans, overlap_joined = _merge_same_type(speech_spans)
        stats["synthetic_overlap_merged"] += overlap_joined

        # Fill the complement.  The 0-units are real Omni definite-drop audio, but
        # this manifest does not say which subtype each gap used, so the gaps
        # supervise "not speech" and leave the subtype unsure.
        spans: list[dict[str, Any]] = []
        cursor = 0.0
        for span in speech_spans:
            if float(span["start_s"]) - cursor >= FRAME_HOP_S:
                spans.append(
                    {
                        "start_s": round(cursor, 6),
                        "end_s": round(float(span["start_s"]), 6),
                        "type": TYPE_UNSURE,
                        "speech": False,
                        "source_label": "synthetic_non_speech_unit",
                        "flags": [],
                    }
                )
            spans.append(span)
            cursor = float(span["end_s"])
        if duration - cursor >= FRAME_HOP_S:
            spans.append(
                {
                    "start_s": round(cursor, 6),
                    "end_s": round(duration, 6),
                    "type": TYPE_UNSURE,
                    "speech": False,
                    "source_label": "synthetic_non_speech_unit",
                    "flags": [],
                }
            )
        if not spans:
            stats["synthetic_no_spans"] += 1
            continue

        stats["span_speech"] += len(speech_spans)
        stats["span_synthetic_non_speech"] += len(spans) - len(speech_spans)

        source = meta.get("source_partition") or row.get("source") or base.name
        # The same galgame/anime speech unit is reused across many composites, so
        # the leakage unit is the set of source units, not the batch name.  A
        # composite also carries a background bed lifted from a real definite-drop
        # chunk; that chunk's parent window is a leakage unit against real data.
        source_units = [str(v) for v in (meta.get("source_audio_ids") or []) if str(v)]
        members = [f"unit:{value}" for value in sorted(set(source_units))]
        background = ((meta.get("realism") or {}).get("background") or {})
        background_id = str(background.get("audio_id") or "")
        if background_id:
            members.append(f"bed:{background_id}")
        if not members:
            members = [f"batch:{base.name}:{source}"]

        yield {
            "schema": OUTPUT_SCHEMA,
            "example_id": f"{base.name}:{audio_id}",
            "provenance": "synthetic_hardmix",
            "dataset": base.name,
            "audio": str(audio),
            "audio_sha256": "",  # 116h of synthetic audio; hashed on demand only.
            "duration_s": round(duration, 6),
            "frame_hop_s": float(row.get("frame_hop_s") or FRAME_HOP_S),
            "frame_count": int(round(duration / FRAME_HOP_S)),
            "video_id": "",
            "window_id": audio_id,
            "group_id": f"{base.name}:{source}",
            "group_members": members,
            "coverage": "complete",
            "covered_s": round(duration, 6),
            "spans": spans,
            "boundary_points": [],
        }


def build(
    *,
    output: Path,
    include_synthetic: bool,
    synthetic_limit: int,
    hardmix_datasets: tuple[str, ...] = HARDMIX_DATASETS,
) -> dict[str, Any]:
    stats: dict[str, int] = {}

    def bump(key: str) -> None:
        stats[key] = stats.get(key, 0)

    for key in (
        "window_missing_audio",
        "window_no_spans",
        "chunk_empty",
        "chunk_out_of_range",
        "chunk_shorter_than_frame",
        "chunk_unknown_label",
        "span_overlap_dropped",
        "span_speech",
        "span_unsure",
        f"span_drop_{TYPE_NON_SEMANTIC_VOCAL}",
        f"span_drop_{TYPE_NON_VOCAL}",
        f"span_drop_{TYPE_UNSURE}",
        "span_synthetic_non_speech",
        "synthetic_missing_audio",
        "synthetic_bad_duration",
        "synthetic_segment_too_short",
        "synthetic_overlap_merged",
        "synthetic_no_spans",
    ):
        bump(key)

    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for name in OMNI_DATASETS:
            base = PROJECT_ROOT / "datasets" / "train" / name
            if not base.is_dir():
                continue
            produced = 0
            for example in build_omni_examples(base, stats=stats):
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                produced += 1
                written += 1
            counts[name] = produced
        if include_synthetic:
            for name in hardmix_datasets:
                base = PROJECT_ROOT / "datasets" / "train" / name
                if not base.is_dir():
                    continue
                produced = 0
                for example in build_hardmix_examples(
                    base, stats=stats, limit=synthetic_limit
                ):
                    handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                    produced += 1
                    written += 1
                counts[name] = produced

    return {
        "schema": "typed_span_dataset_build_summary_v1",
        "output": str(output),
        "output_sha256": _sha256(output),
        "example_count": written,
        "examples_per_dataset": counts,
        "type_vocabulary": [
            TYPE_SPEECH,
            TYPE_NON_SEMANTIC_VOCAL,
            TYPE_NON_VOCAL,
            TYPE_UNSURE,
        ],
        "unsure_ignore_index": -100,
        "stats": stats,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Only normalize the real omni-joint windows.",
    )
    parser.add_argument(
        "--hardmix-set",
        choices=["original", "fixed", "both"],
        default="original",
        help="Which synthetic batches to include.",
    )
    parser.add_argument(
        "--synthetic-limit",
        type=int,
        default=0,
        help="Cap synthetic examples per dataset (0 = all).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    summary = build(
        output=Path(args.output).expanduser().resolve(),
        include_synthetic=not args.no_synthetic,
        synthetic_limit=max(0, int(args.synthetic_limit)),
        hardmix_datasets={
            "original": HARDMIX_DATASETS,
            "fixed": HARDMIX_FIXED_DATASETS,
            "both": HARDMIX_DATASETS + HARDMIX_FIXED_DATASETS,
        }[args.hardmix_set],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
