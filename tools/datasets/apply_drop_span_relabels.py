#!/usr/bin/env python3
"""Write the relabelled long drop spans back into the dataset.

456 spans of 3 s or more were re-judged on one question - does this contain
words - under wording that fixes the four defects in the prompt that produced
the originals. A blind listening pass over 80 of the results accepted them: on
this population the words still being dropped fell from 52% to 20.6%, taking
wrongly-dropped speech from 24.7 to 4.9 minutes.

Only spans the teacher moved to `words` are rewritten. The ones it still calls
`no_words` keep their original labels rather than being re-asserted, because the
audit measured them as 20.6% wrong and stamping them afresh would dress an
unchanged error up as a new decision.

A rewritten span becomes `definite_keep` / `speech` / `speech=True` together.
Splitting them would break the invariant the type head depends on - `speech==1`
iff `type==0`, with no exceptions in 5.99M frames - and a span that reads as
speech on one track and a moan on another would be a third state nothing knows
how to consume. That is a real simplification: these spans hold speech AND a
non-semantic sound, the taxonomy has no word for that, and the question actually
asked was "is there a word in here", not "what is the dominant sound".

Teacher labels are applied uniformly, including to the 70 spans a human also
judged. Substituting the human answers where they exist would give slightly
better data and would confound the experiment this feeds: what is being measured
is what the relabelling process produces, so it has to be the process's own
output. The human verdicts stay held-out truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

SCHEMA = "typed_span_relabel_report_v1"
# Only this direction is written back; see the module docstring.
APPLIED_LABEL = "words"
TOLERANCE_S = 1e-3


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def relabelled_spans(items: Path, answers: Path) -> dict[tuple[str, float, float], str]:
    """(window_id, start, end) -> teacher label, for the spans it moved."""
    by_id = {str(row["item_id"]): row for row in _rows(items)}
    picked: dict[tuple[str, float, float], str] = {}
    for answer in _rows(answers):
        item = by_id.get(str(answer.get("item_id") or ""))
        if item is None or str(answer.get("label")) != APPLIED_LABEL:
            continue
        picked[
            (
                str(item["window_id"]),
                round(float(item["start_s"]), 3),
                round(float(item["end_s"]), 3),
            )
        ] = APPLIED_LABEL
    return picked


def apply_to(
    examples: list[dict[str, Any]], targets: dict[tuple[str, float, float], str]
) -> tuple[list[dict[str, Any]], Counter]:
    stats: Counter = Counter()
    matched: set[tuple[str, float, float]] = set()
    out: list[dict[str, Any]] = []
    for example in examples:
        window_id = str(example.get("window_id") or "")
        spans = []
        for span in example.get("spans") or []:
            key = (
                window_id,
                round(float(span["start_s"]), 3),
                round(float(span["end_s"]), 3),
            )
            if key not in targets:
                spans.append(span)
                continue
            if str(span.get("source_label")) != "definite_drop":
                # The relabel set was built from dropped spans; anything else
                # here means the two files describe different datasets.
                raise ValueError(f"relabel target is not a drop span: {key}")
            matched.add(key)
            stats[f"from_type_{span.get('type')}"] += 1
            stats["rewritten"] += 1
            stats["rewritten_seconds"] += float(span["end_s"]) - float(span["start_s"])
            spans.append(
                {
                    **span,
                    "source_label": "definite_keep",
                    "type": "speech",
                    "speech": True,
                    "relabelled_by": "drop_span_words_v1",
                }
            )
        out.append({**example, "spans": spans})

    missing = set(targets) - matched
    if missing:
        raise ValueError(
            f"{len(missing)} relabelled spans were not found in the dataset; "
            f"first: {sorted(missing)[:3]}"
        )
    return out, stats


def build(*, dataset: Path, items: Path, answers: Path, output: Path) -> dict[str, Any]:
    examples = _rows(dataset)
    targets = relabelled_spans(items, answers)
    rewritten, stats = apply_to(examples, targets)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in rewritten:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    def profile(rows: list[dict]) -> dict[str, Any]:
        seconds: Counter = Counter()
        for example in rows:
            if example.get("provenance") != "real_omni_joint":
                continue
            for span in example.get("spans") or []:
                seconds[
                    f"{span.get('source_label')}|{span.get('type')}"
                ] += float(span["end_s"]) - float(span["start_s"])
        return {k: round(v / 60, 1) for k, v in sorted(seconds.items())}

    return {
        "schema": SCHEMA,
        "dataset_in": str(dataset),
        "dataset_out": str(output),
        "relabel_targets": len(targets),
        "rewritten": stats["rewritten"],
        "rewritten_minutes": round(stats["rewritten_seconds"] / 60, 1),
        "rewritten_from_type": {
            key.replace("from_type_", ""): value
            for key, value in sorted(stats.items())
            if key.startswith("from_type_")
        },
        "minutes_before": profile(examples),
        "minutes_after": profile(rewritten),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build(
                dataset=Path(args.dataset).expanduser().resolve(),
                items=Path(args.items).expanduser().resolve(),
                answers=Path(args.answers).expanduser().resolve(),
                output=Path(args.output).expanduser().resolve(),
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
