#!/usr/bin/env python3
"""Assemble the per-cache manifests for the v2 alignment feature caches.

The trainer caps blank-only rows **per domain**, and a domain with no text rows
gets `maximum_empty = 0` - every blank row in it is dropped. So the blank rows
have to reach the trainer under the same domain label as the text rows they
belong with. `--cache-domain` is that label and it is independent of the
directory, which is why the blank corpora stay in their own caches instead of
being merged into the text manifests.

**Manifests are not merged across runs.** Each run numbered its groups
independently (`...-block-00003` exists in both the selection manifest and the
blank library, over different subsets of clips), so concatenating them would put
one group name on both sides of the train/val split. The feature builder rejects
that outright, and the fix is not to rename - it is to keep each run's own
partition by giving it its own cache.

Within a run, concatenation is safe and used: the crop and full views of the same
teacher output share group names *and* partitions by construction.

`PLAN` and `BLANK_SPLIT` below are not configuration - they are the record of
exactly which manifests went into `datasets/train/align-features-v2/` on
2026-08-11, and which domain each landed in. Editing them describes a different
build; rerunning as-is reproduces that one.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEACHER = "datasets/train/galgame-grok-ctc-teacher-20k-v1/compiled"
STRIPPED = "agents/temp/20260811_150000_stripped-targets"
SFW = "agents/temp/20260811_140000_anime-text-corpus/sfw_full"
BLANKS = (
    "agents/temp/20260811_130000_vocalisation-blank-library/manifest/"
    "vocalisation_blank_manifest.jsonl"
)

# (cache name, domain label, [source manifests], group field)
PLAN = (
    (
        "galgame-teacher",
        "galgame",
        [f"{TEACHER}/ctc_manifest.jsonl", f"{TEACHER}/full_manifest.jsonl"],
        "group",
    ),
    (
        "galgame-recovered",
        "galgame",
        [
            f"{STRIPPED}/galgame-recovered/stripped_text_manifest.jsonl",
            f"{STRIPPED}/galgame-recovered/stripped_blank_manifest.jsonl",
        ],
        "group",
    ),
    ("anime-sfw", "anime-sfw", [f"{SFW}/ctc_manifest.jsonl", f"{SFW}/full_manifest.jsonl"], "group"),
    (
        "anime-nsfw",
        "anime-nsfw",
        [
            f"{STRIPPED}/nsfw/stripped_text_manifest.jsonl",
            f"{STRIPPED}/nsfw/stripped_blank_manifest.jsonl",
        ],
        "source_group",
    ),
)

# The blank library covers three corpora in one file; each slice has to land in
# the domain of the text it will be capped against.
BLANK_SPLIT = (
    ("galgame-vocal-blank", "galgame", "galgame-asr-100k-ogg"),
    ("anime-sfw-vocal-blank", "anime-sfw", "japanese-anime-speech-v2-sfw-40k"),
    ("anime-nsfw-vocal-blank", "anime-nsfw", "japanese-anime-speech-v2-nsfw-60k"),
)


def read(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_groups(name: str, rows: list[dict], field: str) -> None:
    """A group must not straddle train/val, or the split means nothing."""
    seen: dict[str, str] = {}
    for row in rows:
        group = str(row.get(field) or "")
        partition = str(row.get("partition") or "")
        if not group or partition not in {"train", "val"}:
            raise SystemExit(f"{name}: row missing {field!r} or partition: {row.get('audio_id')}")
        previous = seen.setdefault(group, partition)
        if previous != partition:
            raise SystemExit(f"{name}: group {group!r} crosses train/val")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    blank_rows = read(BLANKS)
    by_corpus: dict[str, list[dict]] = collections.defaultdict(list)
    for row in blank_rows:
        by_corpus[str(row.get("source_corpus"))].append(row)

    plan: list[dict] = []
    for name, domain, sources, field in PLAN:
        rows: list[dict] = []
        for source in sources:
            rows.extend(read(source))
        check_groups(name, rows, field)
        plan.append({"name": name, "domain": domain, "rows": rows, "group_field": field})

    for name, domain, corpus in BLANK_SPLIT:
        rows = by_corpus.get(corpus, [])
        if not rows:
            raise SystemExit(f"{name}: no rows for corpus {corpus!r}")
        check_groups(name, rows, "group")
        plan.append({"name": name, "domain": domain, "rows": rows, "group_field": "group"})

    manifest_dir = out_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for entry in plan:
        path = manifest_dir / f"{entry['name']}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in entry["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        blanks = sum(1 for row in entry["rows"] if not str(row.get("text") or ""))
        seconds = 0.0
        for row in entry["rows"]:
            end = row.get("source_end_s")
            seconds += (
                float(end) - float(row.get("source_start_s") or 0.0)
                if end is not None
                else float(row.get("duration_s") or 0.0)
            )
        report.append(
            {
                "cache": entry["name"],
                "domain": entry["domain"],
                "group_field": entry["group_field"],
                "rows": len(entry["rows"]),
                "blank_rows": blanks,
                "hours": round(seconds / 3600.0, 3),
                "val_rows": sum(
                    1 for row in entry["rows"] if row.get("partition") == "val"
                ),
                "manifest": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
        )

    # What the trainer will actually keep, per domain, at the default 0.30 cap.
    import math

    domains: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"nonempty": 0, "empty": 0}
    )
    for entry in plan:
        for row in entry["rows"]:
            if row.get("partition") != "train":
                continue
            key = "nonempty" if str(row.get("text") or "") else "empty"
            domains[entry["domain"]][key] += 1
    cap = {}
    for domain, counts in domains.items():
        maximum = int(math.floor(0.30 * counts["nonempty"] / 0.70))
        keep = min(counts["empty"], maximum)
        cap[domain] = {
            **counts,
            "empty_cap": maximum,
            "empty_kept": keep,
            "empty_dropped": counts["empty"] - keep,
            "blank_share": round(keep / max(counts["nonempty"] + keep, 1), 4),
        }

    (out_dir / "plan.json").write_text(
        json.dumps({"caches": report, "domain_blank_cap": cap}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{'cache':26} {'domain':12} {'rows':>7} {'blank':>7} {'hours':>7} {'val':>7}")
    for entry in report:
        print(
            f"{entry['cache']:26} {entry['domain']:12} {entry['rows']:>7} "
            f"{entry['blank_rows']:>7} {entry['hours']:>7.2f} {entry['val_rows']:>7}"
        )
    print("\n=== blank rows surviving the per-domain 0.30 cap (train only) ===")
    for domain, counts in sorted(cap.items()):
        print(
            f"  {domain:12} nonempty={counts['nonempty']:>6} empty={counts['empty']:>5} "
            f"cap={counts['empty_cap']:>6} kept={counts['empty_kept']:>5} "
            f"dropped={counts['empty_dropped']:>5} blank_share={counts['blank_share']:.1%}"
        )
    print(f"\nwrote {manifest_dir}")


if __name__ == "__main__":
    main()
