#!/usr/bin/env python3
"""Frame labels for the three-class head: silence / vocalisation / speech.

**Why a class rather than a threshold.** Until now the head had one way to say
"not a word", and moaning had to share it with silence: the stripped targets
left CTC nothing to place over a moan, so blank was the only reading available.
Every attempt to push moaning further into blank pushed breathy speech with it -
2026-08-08 and 08-11 falsified the dose three times, and the measurement that
survived (`measure_blank_class_separation.py`: 67.3% blank on words, 90.7% on
voiced-wordless, 98.6% on silence) says the information is there and unlabelled.
This tool is that labelling.

**No human annotation and no real JAV.** Every label comes from material already
in the training caches:

  L1  anime-nsfw mixed clips. The full script - moans included - is force-aligned
      with the punctuated general head, aggregated into punctuation blocks, and
      each block is put to `is_non_semantic_vocalisation`. A block it calls
      vocalisation labels its own span; the rest label speech. This is the only
      source of new information here, and the only one that can be wrong in the
      expensive direction, so it carries the quality gate below.
  L2  the script-confirmed vocal-blank clips. The script already says the whole
      clip is vocalisation, so energy alone splits it: loud is vocalisation,
      quiet is silence. No alignment is involved and nothing can be mislabelled
      as speech, because there is no speech in these clips to find.
  L3  the Grok-timed corpora. Word islands are speech and long quiet gaps are
      silence - and the voiced-wordless middle is deliberately left `ignore`.
      SFW anime voiced-wordless is breath, SFX and BGM; 2026-08-10 established
      that energy is not a moan detector, and this is where that bites.
  L4  the locally-recovered galgame clips. Same construction as L1 over text a
      local decode produced rather than a script, so it is the least trustworthy;
      `--l4-silence-only` keeps just its silence, which is the part that does not
      depend on the text being right.

**The quality gate exists because L1 can be wrong in one direction.** Labelling a
moan as speech costs a little supervision. Labelling *speech* as vocalisation
teaches the head that a word is a moan, which is the failure this whole line of
work exists to remove - and the filter downstream will then delete it. So a clip
is dropped whole rather than used with doubt, and the checks are on the alignment
rather than on the verdict:

  * character score at or above the corpus median minus one IQR;
  * every semantic block between 1.5 and 14 characters per second;
  * blocks monotonic and non-overlapping;
  * and the one that actually catches the failure - align the *stripped* text as
    well, and require the semantic blocks to land in the same place. If a moaning
    block has dragged its neighbour's boundary, the two alignments disagree.

Outputs an `.npz` of int8 label arrays keyed by `cache/audio_id`, plus a report
carrying the gate outcomes and the audit material (kanji inside vocalisation
blocks, the frequent block texts, the longest spans).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment import (  # noqa: E402
    ENCODER_FRAME_S,
    FRAME_CLASS_SILENCE,
    FRAME_CLASS_SPEECH,
    FRAME_CLASS_VOCALISATION,
    FRAME_CLASSES,
    AlignmentHead,
    normalize_text,
)
from subtitles.vocalisation import is_non_semantic_vocalisation  # noqa: E402
from tools.align.build_vocalisation_stripped_manifest import split_parts  # noqa: E402
from tools.align.frame_teacher_supervision import (  # noqa: E402
    IGNORE_LABEL,
    load_accepted_frame_teachers,
    merge_intervals,
)
from tools.align.measure_blank_class_separation import (  # noqa: E402
    energy_threshold,
    frame_energy,
    resolve_repo_path,
)

SCHEMA = "frame_class_labels_v1"


def sample_rows(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """`limit` rows, drawn at random when a seed is given.

    The index is in manifest order, which groups clips by source work, so its
    head is not a sample of the corpus - a smoke taken from it measures whatever
    that ordering correlates with and reports it as a corpus property.
    """
    if not limit or limit >= len(rows):
        return list(rows)
    if not seed:
        return list(rows[:limit])
    import random

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Shards:
    """Memory-mapped access to one cache's feature shards."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._shards: dict[str, np.ndarray] = {}

    def features(self, row: dict) -> np.ndarray:
        shard = self._shards.get(row["shard"])
        if shard is None:
            shard = np.load(self.cache_dir / row["shard"], mmap_mode="r")
            self._shards[row["shard"]] = shard
        start = int(row["offset"])
        return np.asarray(shard[start : start + int(row["frames"])], dtype=np.float32)


def block_spans(spans, text: str) -> list[tuple[str, float, float]]:
    """Punctuation blocks of `text` with the extent of their characters.

    `align_text` returns one span per character of the normalised text, which is
    the same string `split_parts` walks - so the blocks can be cut by character
    offset without re-deriving anything, and a mismatch in either is a bug rather
    than a rounding difference.
    """
    normalised = normalize_text(text)
    if len(spans) != len(normalised):
        raise ValueError(
            f"alignment returned {len(spans)} spans for {len(normalised)} characters"
        )
    blocks: list[tuple[str, float, float]] = []
    cursor = 0
    for fragment, is_separator in split_parts(normalised):
        width = len(fragment)
        if not is_separator and width:
            chunk = spans[cursor : cursor + width]
            # Zero-width punctuation marks can leave a block whose members all
            # collapse to one instant; those carry no evidence and are skipped
            # rather than labelled over a span they do not occupy.
            starts = [span.start_s for span in chunk]
            ends = [span.end_s for span in chunk]
            if ends and max(ends) > min(starts):
                blocks.append((fragment, float(min(starts)), float(max(ends))))
        cursor += width
    return blocks


def monotonic(blocks: list[tuple[str, float, float]]) -> bool:
    previous_end = -1.0
    for _text, start, end in blocks:
        if start < previous_end - 1e-6 or end <= start:
            return False
        previous_end = end
    return True


def characters_per_second(blocks) -> list[float]:
    return [
        len(text) / (end - start)
        for text, start, end in blocks
        if end > start
    ]


def align_clip(head, features, text: str):
    """`(spans, mean_score)` or None when the text cannot be aligned."""
    from asr.alignment import align_text

    try:
        log_probs = head.log_probs(features)
        spans = align_text(log_probs, text, head.vocab, upsample=head.upsample)
    except (ValueError, RuntimeError):
        return None
    scored = [span.score for span in spans if span.end_s > span.start_s]
    if not scored:
        return None
    return spans, float(np.mean(scored))


def frames_for(row: dict, upsample: int) -> int:
    return int(row["frames"]) * int(upsample)


def energy_for_row(
    row: dict,
    manifest_row: dict,
    *,
    output_frames: int,
    frame_s: float,
    relative_db: float,
    floor_dbfs: float,
):
    """`(energy, threshold)` for a cached row, or None when it cannot be read."""
    from audio.loading import load_audio_16k_mono

    audio_ref = manifest_row.get("audio")
    if not audio_ref:
        return None
    path = resolve_repo_path(str(audio_ref))
    if not path.exists():
        return None
    try:
        audio, rate = load_audio_16k_mono(str(path))
    except Exception:  # noqa: BLE001 - a corpus file that will not open is data
        return None
    audio = np.asarray(audio, dtype=np.float32)
    # The cached row may be a crop of a longer source clip; the labels are
    # indexed from the crop's own zero, so the audio has to be cut the same way
    # before any frame index means the same thing on both sides.
    start = float(row.get("source_start_s") or 0.0)
    end = row.get("source_end_s")
    first = int(round(start * rate))
    last = int(round(float(end) * rate)) if end is not None else audio.size
    audio = audio[max(0, first) : max(0, min(audio.size, last))]
    energy = frame_energy(
        audio, frame_count=output_frames, frame_s=frame_s, sample_rate=int(rate)
    )
    if energy is None:
        return None
    threshold = energy_threshold(
        energy, relative_db=relative_db, floor_dbfs=floor_dbfs
    )
    if threshold is None:
        return None
    return energy, threshold


def mark(labels: np.ndarray, start_s: float, end_s: float, value: int, frame_s: float):
    first = max(0, int(math.floor(start_s / frame_s)))
    last = min(len(labels), int(math.ceil(end_s / frame_s)))
    if last > first:
        labels[first:last] = value


def ignore_boundaries(
    labels: np.ndarray, edges: list[float], *, frame_s: float, ignore_s: float
):
    for edge in edges:
        first = max(0, int(math.floor((edge - ignore_s) / frame_s)))
        last = min(len(labels), int(math.ceil((edge + ignore_s) / frame_s)))
        if last > first:
            labels[first:last] = IGNORE_LABEL


def build_l1(
    *,
    cache_name: str,
    cache_dir: Path,
    manifest_rows: dict[str, dict],
    head,
    limit: int,
    sample_seed: int,
    frame_s: float,
    ignore_s: float,
    relative_db: float,
    floor_dbfs: float,
    boundary_tolerance_s: float,
    cps_range: tuple[float, float],
    report: dict,
    silence_only: bool = False,
):
    """Self-aligned block labels for a mixed-content cache (L1 / L4)."""
    rows = sample_rows(read_jsonl(cache_dir / "index.jsonl"), limit, sample_seed)
    shards = Shards(cache_dir)

    # Two passes: the score gate is relative to this corpus, so the threshold
    # cannot be known until every clip has been aligned once. Alignments are
    # kept rather than recomputed - the second pass is where the labels are
    # written, and realigning would double the only expensive step here.
    prepared: list[tuple[dict, list, list, float]] = []
    rejected = collections.Counter()
    started = time.perf_counter()
    for position, row in enumerate(rows):
        manifest_row = manifest_rows.get(row["audio_id"])
        if manifest_row is None:
            rejected["no_manifest_row"] += 1
            continue
        script = str(manifest_row.get("script_text") or manifest_row.get("text") or "")
        stripped = str(row.get("text") or "")
        if not script.strip():
            rejected["empty_script"] += 1
            continue
        features = shards.features(row)
        full = align_clip(head, features, script)
        if full is None:
            rejected["full_text_unalignable"] += 1
            continue
        spans, score = full
        try:
            blocks = block_spans(spans, script)
        except ValueError:
            rejected["span_count_mismatch"] += 1
            continue
        if not blocks:
            rejected["no_blocks"] += 1
            continue
        prepared.append((row, blocks, spans, score))
        if position % 500 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"  {cache_name}: aligned {position + 1}/{len(rows)} "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    if not prepared:
        report[cache_name] = {"rows": len(rows), "rejected": dict(rejected)}
        return {}

    scores = np.array([item[3] for item in prepared], dtype=np.float64)
    q1, median, q3 = np.percentile(scores, [25, 50, 75])
    score_floor = float(median - (q3 - q1))

    labels_by_key: dict[str, np.ndarray] = {}
    audit_blocks = collections.Counter()
    kanji_blocks: list[str] = []
    longest: list[tuple[float, str, str]] = []
    accepted = 0
    for row, blocks, _spans, score in prepared:
        if score < score_floor:
            rejected["score_below_corpus_floor"] += 1
            continue
        semantic = [b for b in blocks if not is_non_semantic_vocalisation(b[0])]
        vocal = [b for b in blocks if is_non_semantic_vocalisation(b[0])]
        if not monotonic(blocks):
            rejected["blocks_not_monotonic"] += 1
            continue
        rates = characters_per_second(semantic)
        if rates and min(rates) < cps_range[0]:
            rejected["semantic_block_too_slow"] += 1
            continue
        if rates and cps_range[1] > 0.0 and max(rates) > cps_range[1]:
            rejected["semantic_block_too_fast"] += 1
            continue

        # The check that catches the expensive error: the semantic blocks must
        # land in the same place whether or not the moans are in the target. If
        # a moaning block has dragged its neighbour's boundary, they will not.
        if semantic and str(row.get("text") or "").strip():
            features = shards.features(row)
            stripped_aligned = align_clip(head, features, str(row["text"]))
            if stripped_aligned is None:
                rejected["stripped_text_unalignable"] += 1
                continue
            try:
                stripped_blocks = block_spans(stripped_aligned[0], str(row["text"]))
            except ValueError:
                rejected["stripped_span_count_mismatch"] += 1
                continue
            by_text = collections.defaultdict(list)
            for text, start, end in stripped_blocks:
                by_text[text].append((start, end))
            shifts = []
            for text, start, end in semantic:
                candidates = by_text.get(text)
                if not candidates:
                    continue
                best = min(candidates, key=lambda pair: abs(pair[0] - start))
                shifts.append(max(abs(best[0] - start), abs(best[1] - end)))
            if shifts and float(np.median(shifts)) > boundary_tolerance_s:
                rejected["semantic_blocks_moved_without_the_moans"] += 1
                continue

        output_frames = frames_for(row, head.upsample)
        labels = np.full(output_frames, IGNORE_LABEL, dtype=np.int8)

        # Silence first, so any block span overwrites it: the gaps between
        # blocks are only silence where the audio agrees, and where it does not
        # they stay ignored rather than becoming a class by default.
        measured = energy_for_row(
            row,
            manifest_rows[row["audio_id"]],
            output_frames=output_frames,
            frame_s=frame_s,
            relative_db=relative_db,
            floor_dbfs=floor_dbfs,
        )
        if measured is not None:
            energy, threshold = measured
            covered = np.zeros(output_frames, dtype=bool)
            for _text, start, end in blocks:
                first = max(0, int(math.floor(start / frame_s)))
                last = min(output_frames, int(math.ceil(end / frame_s)))
                covered[first:last] = True
            quiet = (~covered) & (energy[:output_frames] < threshold)
            labels[quiet] = FRAME_CLASS_SILENCE

        if not silence_only:
            for text, start, end in vocal:
                mark(labels, start, end, FRAME_CLASS_VOCALISATION, frame_s)
                audit_blocks[text] += 1
                if any(
                    "CJK UNIFIED" in __import__("unicodedata").name(ch, "")
                    for ch in text
                ):
                    kanji_blocks.append(text)
                # Times and source audio too, not just the text: the audit that
                # matters is listening to the span, and a record that cannot be
                # played back can only ever be read - which is the check that
                # already passed.
                longest.append(
                    (
                        end - start,
                        text,
                        row["audio_id"],
                        str(manifest_rows[row["audio_id"]].get("audio") or ""),
                        float(row.get("source_start_s") or 0.0) + start,
                        float(row.get("source_start_s") or 0.0) + end,
                    )
                )
            for text, start, end in semantic:
                mark(labels, start, end, FRAME_CLASS_SPEECH, frame_s)

        edges = sorted({b[1] for b in blocks} | {b[2] for b in blocks})
        ignore_boundaries(labels, edges, frame_s=frame_s, ignore_s=ignore_s)
        if np.any(labels != IGNORE_LABEL):
            labels_by_key[f"{cache_name}/{row['audio_id']}"] = labels
            accepted += 1

    longest.sort(reverse=True)
    report[cache_name] = {
        "source": "L4" if silence_only else "L1",
        "rows": len(rows),
        "aligned": len(prepared),
        "accepted": accepted,
        "gate_pass_rate": round(accepted / max(1, len(prepared)), 4),
        "score_floor": round(score_floor, 4),
        "score_median": round(float(median), 4),
        "rejected": dict(rejected),
        "vocalisation_blocks_with_kanji": len(kanji_blocks),
        "vocalisation_blocks_with_kanji_examples": kanji_blocks[:20],
        "top_vocalisation_blocks": audit_blocks.most_common(40),
        "longest_vocalisation_spans": [
            {
                "seconds": round(seconds, 3),
                "text": text,
                "audio_id": audio_id,
                "audio": audio,
                "start_s": round(begin, 4),
                "end_s": round(finish, 4),
            }
            for seconds, text, audio_id, audio, begin, finish in longest[:120]
        ],
    }
    return labels_by_key


def build_l2(
    *,
    cache_name: str,
    cache_dir: Path,
    manifest_rows: dict[str, dict],
    upsample: int,
    limit: int,
    sample_seed: int,
    frame_s: float,
    relative_db: float,
    floor_dbfs: float,
    report: dict,
):
    """Energy split on clips the script already calls pure vocalisation (L2)."""
    rows = sample_rows(read_jsonl(cache_dir / "index.jsonl"), limit, sample_seed)
    labels_by_key: dict[str, np.ndarray] = {}
    rejected = collections.Counter()
    voiced = silent = 0
    for row in rows:
        manifest_row = manifest_rows.get(row["audio_id"])
        if manifest_row is None:
            rejected["no_manifest_row"] += 1
            continue
        output_frames = frames_for(row, upsample)
        measured = energy_for_row(
            row,
            manifest_row,
            output_frames=output_frames,
            frame_s=frame_s,
            relative_db=relative_db,
            floor_dbfs=floor_dbfs,
        )
        if measured is None:
            rejected["no_usable_energy"] += 1
            continue
        energy, threshold = measured
        labels = np.where(
            energy[:output_frames] >= threshold,
            FRAME_CLASS_VOCALISATION,
            FRAME_CLASS_SILENCE,
        ).astype(np.int8)
        voiced += int(np.sum(labels == FRAME_CLASS_VOCALISATION))
        silent += int(np.sum(labels == FRAME_CLASS_SILENCE))
        labels_by_key[f"{cache_name}/{row['audio_id']}"] = labels
    report[cache_name] = {
        "source": "L2",
        "rows": len(rows),
        "accepted": len(labels_by_key),
        "vocalisation_frames": voiced,
        "silence_frames": silent,
        "rejected": dict(rejected),
    }
    return labels_by_key


def build_l3(
    *,
    cache_name: str,
    cache_dir: Path,
    manifest_rows: dict[str, dict],
    teachers: dict,
    upsample: int,
    limit: int,
    sample_seed: int,
    frame_s: float,
    ignore_s: float,
    negative_minimum_s: float,
    merge_gap_s: float,
    relative_db: float,
    floor_dbfs: float,
    report: dict,
):
    """Word islands as speech, quiet distant gaps as silence (L3).

    The voiced-wordless middle stays `ignore` on purpose. On SFW anime it holds
    breath, sound effects and music, and 2026-08-10 established that energy is
    not a moan detector - labelling it `vocalisation` here would teach the head
    that a door slam is a moan.
    """
    rows = sample_rows(read_jsonl(cache_dir / "index.jsonl"), limit, sample_seed)
    labels_by_key: dict[str, np.ndarray] = {}
    rejected = collections.Counter()
    speech_frames = silence_frames = 0
    for row in rows:
        teacher = teachers.get(str(row.get("source_id") or row["audio_id"]))
        if teacher is None:
            rejected["no_accepted_teacher"] += 1
            continue
        offset = float(row.get("source_start_s") or 0.0)
        output_frames = frames_for(row, upsample)
        duration = output_frames * frame_s
        islands = merge_intervals(
            (
                (start - offset, end - offset)
                for start, end in (teacher.get("lexical_intervals") or ())
            ),
            maximum_gap_s=merge_gap_s,
        )
        islands = [
            (max(0.0, start), min(duration, end))
            for start, end in islands
            if min(duration, end) > max(0.0, start)
        ]
        if not islands:
            rejected["no_word_islands_in_this_crop"] += 1
            continue
        labels = np.full(output_frames, IGNORE_LABEL, dtype=np.int8)

        # Silence only where the teacher says no word AND the audio is quiet.
        # The gap alone is what 2026-08-11 falsified: 55.51% of Grok's empty
        # responses covered script with clear lexical content.
        manifest_row = manifest_rows.get(row["audio_id"])
        if manifest_row is None:
            rejected["no_audio_for_silence"] += 1
        else:
            measured = energy_for_row(
                row,
                manifest_row,
                output_frames=output_frames,
                frame_s=frame_s,
                relative_db=relative_db,
                floor_dbfs=floor_dbfs,
            )
            if measured is not None:
                energy, threshold = measured
                protected = merge_intervals(
                    (
                        (max(0.0, start - ignore_s), min(duration, end + ignore_s))
                        for start, end in islands
                    ),
                    maximum_gap_s=0.0,
                )
                far = np.zeros(output_frames, dtype=bool)
                cursor = 0.0
                gaps: list[tuple[float, float]] = []
                for start, end in protected:
                    if start - cursor >= negative_minimum_s:
                        gaps.append((cursor, start))
                    cursor = max(cursor, end)
                if duration - cursor >= negative_minimum_s:
                    gaps.append((cursor, duration))
                for start, end in gaps:
                    first = max(0, int(math.floor(start / frame_s)))
                    last = min(output_frames, int(math.ceil(end / frame_s)))
                    far[first:last] = True
                labels[far & (energy[:output_frames] < threshold)] = FRAME_CLASS_SILENCE

        for start, end in islands:
            mark(labels, start, end, FRAME_CLASS_SPEECH, frame_s)
        edges = sorted({start for start, _ in islands} | {end for _, end in islands})
        ignore_boundaries(labels, edges, frame_s=frame_s, ignore_s=ignore_s)
        if np.any(labels != IGNORE_LABEL):
            speech_frames += int(np.sum(labels == FRAME_CLASS_SPEECH))
            silence_frames += int(np.sum(labels == FRAME_CLASS_SILENCE))
            labels_by_key[f"{cache_name}/{row['audio_id']}"] = labels
    report[cache_name] = {
        "source": "L3",
        "rows": len(rows),
        "accepted": len(labels_by_key),
        "speech_frames": speech_frames,
        "silence_frames": silence_frames,
        "vocalisation_frames": 0,
        "rejected": dict(rejected),
    }
    return labels_by_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="datasets/train/align-features-v2")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--alignment-head",
        default="models/ctc_aligner.pt",
        help="the punctuated general head. L1 aligns the FULL script including "
        "the moans, which an acoustic-only vocabulary cannot represent.",
    )
    parser.add_argument(
        "--l1-manifest",
        default="agents/temp/20260903_093536_p0-stripped-targets/nsfw/"
        "stripped_text_manifest.jsonl",
        help="carries `script_text` (unstripped) and `audio` for anime-nsfw",
    )
    parser.add_argument(
        "--l1-blank-manifest",
        default="agents/temp/20260903_093536_p0-stripped-targets/nsfw/"
        "stripped_blank_manifest.jsonl",
    )
    parser.add_argument(
        "--l4-manifest",
        default="agents/temp/20260903_093536_p0-stripped-targets/galgame-recovered/"
        "stripped_text_manifest.jsonl",
    )
    parser.add_argument(
        "--l4-blank-manifest",
        default="agents/temp/20260903_093536_p0-stripped-targets/galgame-recovered/"
        "stripped_blank_manifest.jsonl",
    )
    parser.add_argument(
        "--l4-silence-only",
        action="store_true",
        help="keep only L4's silence labels. Its text came from a local decode, "
        "so the part of the construction that depends on the text being right is "
        "the part to leave out.",
    )
    parser.add_argument(
        "--blank-library",
        default="agents/temp/20260811_130000_vocalisation-blank-library/manifest/"
        "vocalisation_blank_manifest.jsonl",
    )
    parser.add_argument("--teacher-results", action="append", default=None)
    parser.add_argument("--teacher-manifest", action="append", default=None)
    parser.add_argument("--teacher-cache", action="append", default=None)
    parser.add_argument(
        "--teacher-source-manifest",
        action="append",
        default=None,
        help="comma-separated; the manifests carrying `audio` for that cache. "
        "Both views are needed where a cache holds them: the crop and full "
        "manifests of one teacher run share `source_id` but not `audio_id`, so "
        "naming only the crop manifest leaves every full row without audio - and "
        "therefore without any silence label - while the run reports success.",
    )
    parser.add_argument(
        "--relative-db",
        type=float,
        default=-35.0,
        help="voicing threshold relative to the clip peak. Sweep -30/-35/-40 "
        "and confirm the conclusion does not turn on it.",
    )
    parser.add_argument("--floor-dbfs", type=float, default=-55.0)
    parser.add_argument("--boundary-ignore-s", type=float, default=0.10)
    parser.add_argument("--negative-minimum-s", type=float, default=0.50)
    parser.add_argument("--positive-merge-gap-s", type=float, default=0.15)
    parser.add_argument("--boundary-tolerance-s", type=float, default=0.10)
    parser.add_argument(
        "--min-chars-per-second",
        type=float,
        default=1.5,
        help="a semantic block slower than this is usually a moan the lexicon "
        "missed, stretched over its own long span - measured, that is exactly "
        "what it catches (`ちゅぶ`, `んっぐ`, `ぢゅぅぅぅ`). It fires on 1.2%% of "
        "clips and every one of them was a lexicon gap.",
    )
    parser.add_argument(
        "--max-chars-per-second",
        type=float,
        default=0.0,
        help="0 disables it, which is the default because the bound measures the "
        "frame grid rather than the alignment. CTC spans are peaky - a character "
        "occupies the frames the model is confident about, not the syllable - so "
        "a block pinned at one frame per character reads as exactly 1/frame_s "
        "= 26.0 c/s no matter how long it is. On 400 sampled clips the p90, p95 "
        "and p99 were all 26.00 and a bound of 14 rejected 40.5%%, none of it "
        "evidence about the alignment. The check that does test the alignment is "
        "--boundary-tolerance-s below.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="rows per cache; 0 is everything. Use a few hundred for a smoke.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="shuffle rows before --limit. The index is in manifest order, so "
        "the head of it is not a sample of the corpus and a smoke taken from it "
        "measures whatever that ordering happens to correlate with.",
    )
    parser.add_argument("--skip-l1", action="store_true")
    parser.add_argument("--skip-l2", action="store_true")
    parser.add_argument("--skip-l3", action="store_true")
    parser.add_argument("--skip-l4", action="store_true")
    args = parser.parse_args()

    cache_root = PROJECT_ROOT / args.cache_root
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    head = AlignmentHead.load(str(PROJECT_ROOT / args.alignment_head))
    if head.vocab.acoustic_only:
        raise SystemExit(
            f"{args.alignment_head} has an acoustic-only vocabulary; L1 aligns "
            "the full script including its punctuation and needs the punctuated "
            "head, or the block boundaries come from a different string than the "
            "one they are cut against"
        )
    frame_s = ENCODER_FRAME_S / head.upsample
    print(
        f"head {args.alignment_head} upsample={head.upsample} "
        f"frame={frame_s * 1000:.2f} ms device={head.device}",
        flush=True,
    )

    report: dict = {}
    labels: dict[str, np.ndarray] = {}

    def manifest_index(*paths: str) -> dict[str, dict]:
        rows: dict[str, dict] = {}
        for relative in paths:
            path = PROJECT_ROOT / relative
            if not path.exists():
                raise SystemExit(f"manifest not found: {relative}")
            for row in read_jsonl(path):
                rows[str(row.get("audio_id") or "")] = row
        return rows

    if not args.skip_l1:
        labels.update(
            build_l1(
                cache_name="anime-nsfw",
                cache_dir=cache_root / "anime-nsfw",
                manifest_rows=manifest_index(args.l1_manifest, args.l1_blank_manifest),
                head=head,
                limit=args.limit,
                sample_seed=args.sample_seed,
                frame_s=frame_s,
                ignore_s=args.boundary_ignore_s,
                relative_db=args.relative_db,
                floor_dbfs=args.floor_dbfs,
                boundary_tolerance_s=args.boundary_tolerance_s,
                cps_range=(args.min_chars_per_second, args.max_chars_per_second),
                report=report,
            )
        )
    if not args.skip_l4:
        labels.update(
            build_l1(
                cache_name="galgame-recovered",
                cache_dir=cache_root / "galgame-recovered",
                manifest_rows=manifest_index(args.l4_manifest, args.l4_blank_manifest),
                head=head,
                limit=args.limit,
                sample_seed=args.sample_seed,
                frame_s=frame_s,
                ignore_s=args.boundary_ignore_s,
                relative_db=args.relative_db,
                floor_dbfs=args.floor_dbfs,
                boundary_tolerance_s=args.boundary_tolerance_s,
                cps_range=(args.min_chars_per_second, args.max_chars_per_second),
                report=report,
                silence_only=args.l4_silence_only,
            )
        )
    if not args.skip_l2:
        blank_rows = manifest_index(args.blank_library)
        for cache_name in (
            "galgame-vocal-blank",
            "anime-sfw-vocal-blank",
            "anime-nsfw-vocal-blank",
        ):
            labels.update(
                build_l2(
                    cache_name=cache_name,
                    cache_dir=cache_root / cache_name,
                    manifest_rows=blank_rows,
                    upsample=head.upsample,
                    limit=args.limit,
                    sample_seed=args.sample_seed,
                    frame_s=frame_s,
                    relative_db=args.relative_db,
                    floor_dbfs=args.floor_dbfs,
                    report=report,
                )
            )
    if not args.skip_l3:
        results = list(args.teacher_results or [])
        manifests = list(args.teacher_manifest or [])
        caches = list(args.teacher_cache or [])
        sources = list(args.teacher_source_manifest or [])
        if len({len(results), len(manifests), len(caches), len(sources)}) != 1:
            raise SystemExit(
                "--teacher-results, --teacher-manifest, --teacher-cache and "
                "--teacher-source-manifest must be given the same number of times"
            )
        for result, manifest, cache_name, source in zip(
            results, manifests, caches, sources
        ):
            teachers, _summary = load_accepted_frame_teachers(
                PROJECT_ROOT / result, PROJECT_ROOT / manifest
            )
            labels.update(
                build_l3(
                    cache_name=cache_name,
                    cache_dir=cache_root / cache_name,
                    manifest_rows=manifest_index(
                        *[part for part in source.split(",") if part.strip()]
                    ),
                    teachers=teachers,
                    upsample=head.upsample,
                    limit=args.limit,
                    sample_seed=args.sample_seed,
                    frame_s=frame_s,
                    ignore_s=args.boundary_ignore_s,
                    negative_minimum_s=args.negative_minimum_s,
                    merge_gap_s=args.positive_merge_gap_s,
                    relative_db=args.relative_db,
                    floor_dbfs=args.floor_dbfs,
                    report=report,
                )
            )

    totals = collections.Counter()
    for array in labels.values():
        for index, name in enumerate(FRAME_CLASSES):
            totals[name] += int(np.sum(array == index))
        totals["ignore"] += int(np.sum(array == IGNORE_LABEL))

    labelled = sum(totals[name] for name in FRAME_CLASSES)
    summary = {
        "schema": SCHEMA,
        "frame_classes": list(FRAME_CLASSES),
        "alignment_head": args.alignment_head,
        "upsample": head.upsample,
        "frame_s": round(frame_s, 8),
        "relative_db": args.relative_db,
        "floor_dbfs": args.floor_dbfs,
        "boundary_ignore_s": args.boundary_ignore_s,
        "boundary_tolerance_s": args.boundary_tolerance_s,
        "limit": args.limit,
        "sample_seed": args.sample_seed,
        "l4_silence_only": bool(args.l4_silence_only),
        "clips": len(labels),
        "frame_totals": dict(totals),
        "class_share_of_labelled": {
            name: round(totals[name] / max(1, labelled), 4) for name in FRAME_CLASSES
        },
        "labelled_frame_share": round(labelled / max(1, labelled + totals["ignore"]), 4),
        "by_cache": report,
    }
    (out_dir / "frame_class_label_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(out_dir / "frame_class_labels.npz", **labels)

    print(json.dumps({k: v for k, v in summary.items() if k != "by_cache"},
                     ensure_ascii=False, indent=2))
    for cache_name, entry in report.items():
        print(f"\n=== {cache_name} ({entry.get('source')}) ===")
        for key in ("rows", "aligned", "accepted", "gate_pass_rate",
                    "vocalisation_blocks_with_kanji"):
            if key in entry:
                print(f"  {key}: {entry[key]}")
        if entry.get("rejected"):
            print(f"  rejected: {entry['rejected']}")
        for text, count in (entry.get("top_vocalisation_blocks") or [])[:15]:
            print(f"    {count:>5}  {text}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
