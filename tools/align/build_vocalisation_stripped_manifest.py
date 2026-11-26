#!/usr/bin/env python3
"""CTC targets with the vocalisation stripped out of the script line.

For clips that mix moaning with speech, neither obvious target works. Training
on the script as written teaches the head to *align* moaning, which is the
opposite of the capability being built - the free-running argmax would learn to
emit `あ` where it should emit blank. Training the whole clip as blank deletes
the speech in it.

Stripping resolves it. The target keeps only the semantic parts, so CTC aligns
the words and has nothing to place over the moaning - which it can then only
explain as blank. That is direct supervision for "vocalisation is blank" without
a single hand label, and without asking an annotator that discards vocalisation
to describe it.

**Grok-annotated clips do not need this.** The runner's `canonical_text_crop`
already crops to the region where the script and Grok agree, and Grok drops
vocalisation, so those targets arrive stripped by a different route. This is for
the corpora that never went through Grok: the NSFW clips it returns empty for
(72.6% of a pilot), and the galgame clips it rejected that the local ASR cleared.

**The strip direction is the safe one.** `is_non_semantic_vocalisation` keeps
anything it cannot decompose, so an unrecognised moan survives into the target -
a missed strip, which costs only a little supervision. The dangerous mistake
would be stripping real speech, and that needs the classifier to call words
vocalisation, which the protected allow-list exists to prevent.
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

from subtitles.vocalisation import (  # noqa: E402
    _DECORATION,
    is_non_semantic_vocalisation,
)

SCHEMA = "vocalisation_stripped_ctc_manifest_v1"


def split_parts(text: str) -> list[tuple[str, bool]]:
    """`(fragment, is_separator)` in order, so the text can be rebuilt."""
    parts: list[tuple[str, bool]] = []
    current: list[str] = []
    separator: list[str] = []
    for char in str(text or ""):
        if char in _DECORATION or char.isspace():
            if current:
                parts.append(("".join(current), False))
                current = []
            separator.append(char)
        else:
            if separator:
                parts.append(("".join(separator), True))
                separator = []
            current.append(char)
    if current:
        parts.append(("".join(current), False))
    if separator:
        parts.append(("".join(separator), True))
    return parts


def strip_vocalisation(text: str) -> tuple[str, int, int]:
    """Return `(kept_text, kept_parts, stripped_parts)`."""
    kept: list[str] = []
    kept_count = stripped_count = 0
    for fragment, is_separator in split_parts(text):
        if is_separator:
            if kept:
                kept.append(fragment)
            continue
        if is_non_semantic_vocalisation(fragment):
            stripped_count += 1
            continue
        kept.append(fragment)
        kept_count += 1
    return "".join(kept).strip("".join(_DECORATION) + " 　"), kept_count, stripped_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="rows with `text`")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--minimum-acoustic-chars", type=int, default=2)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from asr.alignment import is_acoustic_char

    source = PROJECT_ROOT / args.manifest
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"source rows: {len(rows)}")

    kept_rows: list[dict] = []
    blank_rows: list[dict] = []
    stats = collections.Counter()
    stripped_chars = 0
    original_chars = 0
    for row in rows:
        text = str(row.get(args.text_field) or "")
        stripped, kept_parts, dropped_parts = strip_vocalisation(text)
        original_chars += sum(1 for ch in text if is_acoustic_char(ch))
        stripped_chars += sum(1 for ch in stripped if is_acoustic_char(ch))
        acoustic = sum(1 for ch in stripped if is_acoustic_char(ch))
        record = {
            **row,
            "schema": SCHEMA,
            "script_text": text,
            "text": stripped,
            "vocalisation_parts_stripped": dropped_parts,
            "semantic_parts_kept": kept_parts,
            "target_kind": "vocalisation_stripped_text",
            "label": args.label,
        }
        if acoustic < args.minimum_acoustic_chars:
            # Nothing semantic survived: this is a blank clip, not a text one.
            record["text"] = ""
            record["target_kind"] = "script_confirmed_vocalisation_blank"
            blank_rows.append(record)
            stats["became_blank"] += 1
            continue
        if dropped_parts:
            stats["stripped"] += 1
        else:
            stats["unchanged"] += 1
        kept_rows.append(record)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / "stripped_text_manifest.jsonl"
    blank_path = out_dir / "stripped_blank_manifest.jsonl"
    for path, payload in ((text_path, kept_rows), (blank_path, blank_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for record in payload:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "schema": SCHEMA,
        "source": args.manifest,
        "label": args.label,
        "rows_in": len(rows),
        "text_rows_out": len(kept_rows),
        "blank_rows_out": len(blank_rows),
        "counts": dict(stats),
        "acoustic_chars_before": original_chars,
        "acoustic_chars_after": stripped_chars,
        "acoustic_chars_stripped_share": round(
            1.0 - stripped_chars / max(original_chars, 1), 6
        ),
        "text_hours": round(
            sum(float(row.get("duration_s") or 0.0) for row in kept_rows) / 3600.0, 4
        ),
        "blank_hours": round(
            sum(float(row.get("duration_s") or 0.0) for row in blank_rows) / 3600.0, 4
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n=== examples ===")
    shown = 0
    for record in kept_rows:
        if record["vocalisation_parts_stripped"] and shown < 8:
            print(f"  script={record['script_text'][:48]}")
            print(f"  target={record['text'][:48]}")
            shown += 1
    print(f"\nwrote {text_path}\n      {blank_path}")


if __name__ == "__main__":
    main()
