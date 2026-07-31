#!/usr/bin/env python3
"""Turn the two completed listening audits into a calibration set for a teacher.

146 clips have already been judged by hand on exactly one question - does this
contain words - under the corrected definition. That makes them the only ground
truth in this project for the axis the pipeline actually gates on, and the only
way to find out whether a rewritten teacher prompt asks the same question a
human answers.

Two properties make the comparison mean something, and both are enforced here.

The teacher hears the SAME AUDIO FILE the human heard. The clips are reused from
the audit output directories rather than recut, so the comparison cannot drift
through a different cut, codec or sample rate. It also fixes the listening
condition: the human judged a bare clip with no surrounding context, so the
teacher gets a bare clip too. Handing it context the human lacked would flatter
it, and the number would not transfer to the production run.

The held-out half is frozen before any model is called. A prompt can be tuned
until 146 clips agree with it, and the resulting number would describe the
tuning rather than the prompt, so the set is split in two and only the
development half may be looked at while the wording is being changed. The split
is BY VIDEO, not by clip: the same speaker, room and mixing recur across a
video's windows, and the project already lost a model once to a split that let
provenance leak across it.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.binary_clip_audit import safe_name  # noqa: E402

SCHEMA = "word_definition_calibration_item_v1"
DECISIVE = ("words", "no_words")

# Each completed audit: where its answer key lives, and where its page (and so
# the clips the human actually heard) was written.
SOURCES = (
    (
        "false_drop",
        "agents/temp/20260729_160000_false-drop-audit",
        "agents/audits/20260729_160000_false-drop-ab-audit",
    ),
    (
        "span_position",
        "agents/temp/20260729_170000_span-position-audit",
        "agents/audits/20260729_170000_span-position-ab-audit",
    ),
)


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect(root: Path) -> list[dict[str, Any]]:
    """Every decisively judged clip from both audits, with its audio."""
    items: list[dict[str, Any]] = []
    for audit, key_dir, page_dir in SOURCES:
        key = {row["row_id"]: row for row in _rows(root / key_dir / "audit_key.jsonl")}
        verdicts = _rows(root / page_dir / "manual_verdicts.jsonl")
        for verdict in verdicts:
            row_id = str(verdict.get("row_id") or "")
            answer = str(verdict.get("verdict") or "")
            entry = key.get(row_id)
            if entry is None:
                raise ValueError(f"{audit}: verdict for an unknown row_id {row_id}")
            if answer not in DECISIVE:
                continue
            clip = root / page_dir / "media" / f"{safe_name(row_id)}.mp3"
            if not clip.is_file():
                raise FileNotFoundError(f"{audit}: clip missing for {row_id}: {clip}")
            items.append(
                {
                    "schema": SCHEMA,
                    # Namespaced: the two audits number their rows from zero.
                    "item_id": f"{audit}:{row_id}",
                    "audit": audit,
                    "audio": str(clip),
                    "start_s": float(entry["start_s"]),
                    "end_s": float(entry["end_s"]),
                    "clip_duration_s": round(
                        float(entry["end_s"]) - float(entry["start_s"]), 3
                    ),
                    "human": answer,
                    "stratum": str(entry["stratum"]),
                    "type_label": str(entry.get("type_label") or ""),
                    "video_id": str(entry.get("video_id") or ""),
                    "window_id": str(entry.get("window_id") or ""),
                }
            )
    items.sort(key=lambda row: row["item_id"])
    return items


def split_by_video(
    items: list[dict[str, Any]], *, seed: int
) -> tuple[list[dict], list[dict]]:
    """Halve the set, keeping every clip from one video on one side.

    Videos are dealt into whichever half currently holds fewer WORD-POSITIVE
    clips, most-positive video first. Balancing on total size instead looks
    tidier and is the wrong quantity: the number this whole exercise turns on is
    recall over clips a human heard words in, so it is those clips that have to
    be shared out. An earlier version balanced on size and left one half with 21
    positives, where a flawless 21/21 still yields a Wilson lower bound of 0.845
    - a held-out set that could not have passed its own gate however good the
    teacher was.

    Ties are broken by a seeded digest, so the split is deterministic without
    being alphabetical.
    """
    by_video: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_video[item["video_id"]].append(item)

    def positives(video: str) -> int:
        return sum(1 for item in by_video[video] if item["human"] == "words")

    def rank(video: str) -> tuple[int, int, str]:
        digest = hashlib.sha256(f"{seed}|{video}".encode()).hexdigest()
        return -positives(video), -len(by_video[video]), digest

    development: list[dict] = []
    holdout: list[dict] = []
    for video in sorted(by_video, key=rank):
        counts = [
            sum(1 for item in half if item["human"] == "words")
            for half in (development, holdout)
        ]
        if counts[0] != counts[1]:
            target = development if counts[0] < counts[1] else holdout
        else:
            target = development if len(development) <= len(holdout) else holdout
        target.extend(by_video[video])
    development.sort(key=lambda row: row["item_id"])
    holdout.sort(key=lambda row: row["item_id"])
    return development, holdout


def _write(path: Path, items: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in items
    )
    path.write_text(body, encoding="utf-8")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _profile(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "items": len(items),
        "videos": len({item["video_id"] for item in items}),
        "with_words": sum(1 for item in items if item["human"] == "words"),
        "listening_seconds": round(
            sum(item["clip_duration_s"] for item in items), 1
        ),
        "per_stratum": {
            name: sum(1 for item in items if item["stratum"] == name)
            for name in sorted({item["stratum"] for item in items})
        },
    }


def restrict(items: list[dict], *, min_clip_s: float) -> list[dict]:
    """Keep only clips as long as the ones the teacher will be run on.

    A calibration set drawn from a different population than the deployment
    measures the wrong thing. These clips came from two audits built for other
    questions - one sampled every drop span by duration, the other cut spans
    into thirds - so most of them are shorter than anything the relabelling job
    will ever see. The job is the 456 drop spans of 3 s or more, a scope fixed
    before any model was called, so the calibration is cut to match it.
    """
    return [item for item in items if item["clip_duration_s"] >= min_clip_s]


def videos_touched(paths: list[Path], items: list[dict]) -> set[str]:
    """Videos whose teacher answers have already been read.

    Re-splitting after a run would otherwise slide an item whose answer is
    already known into the held-out half, and the half would no longer be
    held out. Whole videos move, because clips from one video share a speaker
    and a room; keeping the item but moving its sibling would leak most of what
    makes the answer predictable.
    """
    known: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                known.add(str(json.loads(line).get("item_id") or ""))
    return {item["video_id"] for item in items if item["item_id"] in known}


def build(
    *,
    root: Path,
    output_dir: Path,
    seed: int,
    min_clip_s: float = 0.0,
    seen_answers: list[Path] | None = None,
) -> dict[str, Any]:
    items = collect(root)
    if min_clip_s > 0:
        items = restrict(items, min_clip_s=min_clip_s)

    exposed = videos_touched(seen_answers or [], items)
    if exposed:
        # Anything already answered is pinned to the development side; only the
        # untouched videos are eligible to be held out.
        pinned = [item for item in items if item["video_id"] in exposed]
        eligible = [item for item in items if item["video_id"] not in exposed]
        extra, holdout = split_by_video(eligible, seed=seed)
        development = sorted(pinned + extra, key=lambda row: row["item_id"])
    else:
        development, holdout = split_by_video(items, seed=seed)
    overlap = {i["video_id"] for i in development} & {i["video_id"] for i in holdout}
    if overlap:
        raise AssertionError(f"a video reached both halves: {sorted(overlap)}")

    development_path = output_dir / "calibration_development.jsonl"
    holdout_path = output_dir / "calibration_holdout.jsonl"
    summary = {
        "schema": "word_definition_calibration_summary_v1",
        "seed": seed,
        "min_clip_s": min_clip_s,
        "videos_pinned_to_development": sorted(exposed),
        "sources": [{"audit": a, "key": k, "page": p} for a, k, p in SOURCES],
        "total": _profile(items),
        "development": {
            **_profile(development),
            "path": str(development_path),
            "sha256": _write(development_path, development),
        },
        "holdout": {
            **_profile(holdout),
            "path": str(holdout_path),
            "sha256": _write(holdout_path, holdout),
            "status": "frozen before any model call; open once, at the end",
        },
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--min-clip-s",
        type=float,
        default=0.0,
        help="Match the deployment population; 0 keeps every clip.",
    )
    parser.add_argument(
        "--seen-answers",
        action="append",
        default=[],
        help=(
            "Teacher output already read. Videos appearing in it are pinned to "
            "the development half so the holdout stays unseen. Repeatable."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build(
                root=Path(args.root).expanduser().resolve(),
                output_dir=Path(args.output_dir).expanduser().resolve(),
                seed=int(args.seed),
                min_clip_s=float(args.min_clip_s),
                seen_answers=[
                    Path(p).expanduser().resolve() for p in args.seen_answers
                ],
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
