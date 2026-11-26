#!/usr/bin/env python3
"""Build blank-only training rows from clips whose script is pure vocalisation.

The head is wanted to call non-semantic vocalisation blank when it runs free.
Grok cannot supply that supervision - it discards vocalisation, so a clip of
nothing but moaning comes back empty and was thrown away as a failed annotation.
But the game and anime corpora ship a **script line** for every clip, and where
that line is nothing but vocalisation the label needs no annotator at all: the
whole clip is blank.

**The text must be emptied, and that is the entire point.** These clips do have
acoustic characters - `ああぁぁっ` - so passing them through unchanged would
train the head to *align* moaning, which is the opposite of the goal. Emptying
the target turns them into the blank-only rows `--include-empty-targets` already
understands, so no trainer change is needed.

**Why the script can be trusted here.** 2026-08-11 checked it against a third
party: on clips where Grok returned nothing, the local ASR read the script line
back verbatim (16/16), and on a 400-clip sample the local ASR sits at CER median
0.000 against the script. The corpus pairing is sound; it was Grok that failed.

**Two things this deliberately does not do.** It does not use Grok's silence as
evidence - that signal is 55.51% wrong on this very corpus. And it does not
lift the trainer's `--max-empty-train-fraction` cap: 8,782 blank rows against
13,308 text rows is 39.8%, above the 0.30 default, and that cap exists because
too much blank supervision is how a head learns to call quiet speech silence.
The cap does the trimming; this only supplies the candidates.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from subtitles.vocalisation import is_non_semantic_vocalisation  # noqa: E402

SCHEMA = "script_confirmed_vocalisation_blank_manifest_v1"

SOURCES = (
    {
        "name": "galgame-asr-100k-ogg",
        "manifest": "datasets/train/boundary-sources/galgame-asr-100k-ogg/manifest.jsonl",
        "kind": "jsonl",
        "text_keys": ("canonical_text", "text"),
    },
    {
        "name": "japanese-anime-speech-v2-nsfw-60k",
        "manifest": "datasets/train/boundary-sources/japanese-anime-speech-v2-nsfw-60k/hf_audio_manifest.json",
        "kind": "json",
        "text_keys": ("text",),
    },
    {
        "name": "japanese-anime-speech-v2-sfw-40k",
        "manifest": "datasets/train/boundary-sources/japanese-anime-speech-v2-sfw-40k/hf_audio_manifest.json",
        "kind": "json",
        "text_keys": ("text",),
    },
)


def load_rows(spec: dict) -> list[dict]:
    path = PROJECT_ROOT / spec["manifest"]
    if spec["kind"] == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--group-block", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--minimum-duration-s", type=float, default=0.30)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import numpy as np

    # Held-out sources from the existing teacher archive must never reappear
    # here under a different manifest, or the val split stops being held out.
    # The IDs live under `core_spans[].core_id`, not `source_id`: reading the
    # wrong key excluded nothing and looked like it had worked.
    teacher_base = (
        PROJECT_ROOT / "datasets/train/galgame-grok-ctc-teacher-20k-v1"
    )
    held_out: set[str] = set()
    held_out_path = teacher_base / "rebuild" / "heldout_sources.jsonl"
    if held_out_path.exists():
        for line in held_out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            for span in json.loads(line).get("core_spans") or []:
                core_id = str(span.get("core_id") or "")
                if core_id:
                    held_out.add(core_id)
    print(f"held-out sources excluded: {len(held_out)}")

    # A clip already carrying a text target must not also arrive with an empty
    # one: the same audio would be trained toward two contradictory answers.
    # The teacher archive wins, because its target came with word timings.
    teacher_sources: set[str] = set()
    for name in ("ctc_manifest.jsonl", "full_manifest.jsonl"):
        path = teacher_base / "compiled" / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                teacher_sources.add(str(json.loads(line).get("source_id")))
    print(f"teacher sources excluded: {len(teacher_sources)}")
    held_out |= teacher_sources

    rows: list[dict] = []
    per_source = collections.Counter()
    skipped = collections.Counter()
    for spec in SOURCES:
        for index, row in enumerate(load_rows(spec)):
            text = ""
            for key in spec["text_keys"]:
                if row.get(key):
                    text = str(row[key])
                    break
            if not is_non_semantic_vocalisation(text):
                continue
            duration = float(row.get("duration_s") or 0.0)
            if duration < args.minimum_duration_s:
                skipped["too_short"] += 1
                continue
            audio_id = str(row.get("audio_id") or row.get("audio") or f"{spec['name']}-{index}")
            if audio_id in held_out or str(row.get("source_id") or "") in held_out:
                skipped["held_out"] += 1
                continue
            rows.append(
                {
                    "audio": row.get("audio"),
                    "audio_id": f"{audio_id}-vocalisation-blank",
                    "source_id": audio_id,
                    "source_corpus": spec["name"],
                    "duration_s": duration,
                    "script_text": text,
                    # Emptied on purpose - see the module docstring.
                    "text": "",
                    "target_kind": "script_confirmed_vocalisation_blank",
                }
            )
            per_source[spec["name"]] += 1
        print(f"{spec['name']}: {per_source[spec['name']]} vocalisation clips")

    # Grouping mirrors the teacher archive: contiguous blocks of the source
    # order, whole blocks to one partition. Per-clip random assignment was shown
    # on 2026-08-05 to inflate val by +0.020 through near-duplicate neighbours.
    rows.sort(key=lambda row: (row["source_corpus"], row["source_id"]))
    for index, row in enumerate(rows):
        row["source_group"] = f"{row['source_corpus']}-block-{index // args.group_block:05d}"
        row["group"] = row["source_group"]

    groups = sorted({row["source_group"] for row in rows})
    rng = np.random.default_rng(args.seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * args.val_fraction)))
    val_groups = set(shuffled[:val_count])
    for row in rows:
        row["partition"] = "val" if row["source_group"] in val_groups else "train"
        row["schema"] = SCHEMA

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "vocalisation_blank_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    hours = sum(row["duration_s"] for row in rows) / 3600.0
    partitions = collections.Counter(row["partition"] for row in rows)
    summary = {
        "schema": SCHEMA,
        "clips": len(rows),
        "hours": round(hours, 4),
        "by_corpus": dict(per_source),
        "by_partition": dict(partitions),
        "groups": len(groups),
        "val_groups": len(val_groups),
        "skipped": dict(skipped),
        "group_block": args.group_block,
        "seed": args.seed,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {manifest}")


if __name__ == "__main__":
    main()
