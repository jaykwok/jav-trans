#!/usr/bin/env python3
"""Cross-tab suspected Split false-cuts against non-speech junction flags.

Reads the per-candidate hard-case labels (omni_split_labels.jsonl) and answers
one question: of the candidates where the current runtime said ``cut`` but the
Omni teacher says ``continue`` (suspected false-cuts), what share sit at a
non-speech junction (breath/moan/laughter flags)? A high share supports the
"scorer lets non-speech in" hypothesis at the verifier-junction level; a low
share points at purely linguistic in-sentence pauses. Read-only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


NONSPEECH_FLAGS = frozenset({"breath", "moan", "laughter"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _bucket(flags: list[str]) -> str:
    normalized = {str(flag) for flag in flags}
    if normalized & NONSPEECH_FLAGS:
        return "nonspeech_junction"
    return "linguistic_or_other"


def analyze(labels: list[dict[str, Any]]) -> dict[str, Any]:
    trainable = [row for row in labels if row.get("trainable")]
    suspected_false_cuts = [
        row
        for row in trainable
        if str(row.get("current_label")) == "cut"
        and str(row.get("label")) == "continue"
    ]
    suspected_missed_cuts = [
        row
        for row in trainable
        if str(row.get("current_label")) == "continue"
        and str(row.get("label")) == "cut"
    ]
    false_cut_buckets = Counter(
        _bucket(row.get("flags") or []) for row in suspected_false_cuts
    )
    false_cut_flag_histogram = Counter(
        str(flag) for row in suspected_false_cuts for flag in row.get("flags") or []
    )
    all_flag_histogram = Counter(
        str(flag) for row in trainable for flag in row.get("flags") or []
    )
    nonspeech_count = false_cut_buckets.get("nonspeech_junction", 0)
    return {
        "schema": "acoustic_split_false_cut_flag_crosstab_v1",
        "trainable_label_count": len(trainable),
        "label_counts": dict(Counter(str(row.get("label")) for row in trainable)),
        "suspected_false_cut_count": len(suspected_false_cuts),
        "suspected_missed_cut_count": len(suspected_missed_cuts),
        "false_cut_buckets": dict(false_cut_buckets),
        "false_cut_nonspeech_share": (
            round(nonspeech_count / len(suspected_false_cuts), 4)
            if suspected_false_cuts
            else None
        ),
        "false_cut_flag_histogram": dict(false_cut_flag_histogram),
        "all_trainable_flag_histogram": dict(all_flag_histogram),
        "nonspeech_flags": sorted(NONSPEECH_FLAGS),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    labels = _read_jsonl(Path(args.labels))
    summary = analyze(labels)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        required=True,
        help="omni_split_labels.jsonl from the per-candidate hard-case labeler",
    )
    parser.add_argument("--output", default="", help="optional JSON summary path")
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
