#!/usr/bin/env python3
"""Sample what a mixed-cue split deleted, for a listening test.

The split is the only place the vocalisation filter edits text *inside* a cue
rather than keeping or dropping the whole of it. Its criterion is deliberately
borrowed - a fragment goes only when the joint verdict applied to that
fragment's own re-measured frames says drop - but borrowing a criterion is not
the same as validating it at a new granularity. The fragment is chosen by the
text decomposition and cut at a boundary derived from per-character alignment,
and neither of those was ever asked to be right about part of a cue before.

Three strata, and two of them are references:

  * `split_removed`  the seconds a split deleted. Under test. If these come back
                     as "has words", the split is deleting speech and must go.
  * `split_kept`     what survived of the SAME cue. Certainly contains words, and
                     it also localises a failure: a cut placed a moment too early
                     shows up here as a truncated word, not as anything visible in
                     the removed half.
  * `already_dropped` whole cues the shipped text rule already deletes. Says what
                     "no words" scores when the current behaviour is right, so
                     `split_removed` is read against something rather than zero.

Reads a finished job's cue file, which is where `vocalisation_split` records the
seconds a split removed - the shortened cue no longer covers them, so they
cannot be recovered from timings afterwards.

Writes the manifest the audit shell consumes - which may carry nothing but
`row_id`, `audio`, `start_s`, `end_s` - and, separately, the answer key. Two
files, because the shell refuses a manifest with any other field, and that
refusal is the only thing standing between a blind audit and a labelled one.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from subtitles import vocalisation as V  # noqa: E402

MANIFEST_SCHEMA = "vocalisation_verdict_audit_manifest_v1"
KEY_SCHEMA = "vocalisation_verdict_audit_key_v1"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / value


def _report(path: Path) -> str:
    """Project-relative when it can be, absolute otherwise.

    An out-dir on another drive is legitimate - a scratch run, a test - and a
    summary that raises rather than naming the file it just wrote turns that
    into a failure of the whole selection.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _cues(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("blocks", "cues"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    raise SystemExit(f"no cue list in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cues",
        required=True,
        help="a finished job's bilingual.json, produced with the split enabled",
    )
    parser.add_argument(
        "--unfiltered-cues",
        required=True,
        help=(
            "the same film's cues with SUBTITLE_DROP_VOCALISATION_ONLY_CUES=0. "
            "The reference stratum is drawn from here because the cues the "
            "shipped rule deletes are, by construction, absent from --cues."
        ),
    )
    parser.add_argument("--audio", required=True)
    parser.add_argument(
        "--film-alias",
        default="sample-x",
        help="what the page prints. Never the film - the row id is visible.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-stratum", type=int, default=20)
    parser.add_argument("--per-reference-stratum", type=int, default=10)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    audio = Path(args.audio)
    if not audio.is_file():
        raise SystemExit(f"audio not found: {audio}")
    cues = _cues(_resolve(args.cues))

    pools: dict[str, list[dict]] = collections.defaultdict(list)
    for index, cue in enumerate(cues):
        split = cue.get("vocalisation_split")
        text = V.block_text(cue)
        if isinstance(split, dict):
            for order, span in enumerate(split.get("removed_spans") or []):
                start, end = float(span[0]), float(span[1])
                pools["split_removed"].append(
                    {
                        "cue_index": index,
                        "fragment_order": order,
                        "audio": str(audio),
                        "start_s": start,
                        "end_s": end,
                        "stratum": "split_removed",
                        "text": (split.get("removed_prefix") if order == 0 else "")
                        or split.get("removed_suffix")
                        or split.get("removed_prefix"),
                        "kept_text": text,
                    }
                )
            start = cue.get("acoustic_start", cue.get("start"))
            end = cue.get("acoustic_end", cue.get("end"))
            if start is not None and end is not None:
                pools["split_kept"].append(
                    {
                        "cue_index": index,
                        "audio": str(audio),
                        "start_s": float(start),
                        "end_s": float(end),
                        "stratum": "split_kept",
                        "text": text,
                    }
                )

    # The reference stratum: whole cues the shipped text rule deletes. They are
    # absent from the filtered file by construction, so it is read from the
    # unfiltered arm of the same film - same head, same chunking, same layout.
    for index, cue in enumerate(_cues(_resolve(args.unfiltered_cues))):
        text = V.block_text(cue)
        if not V.is_non_semantic_vocalisation(text):
            continue
        start = cue.get("acoustic_start", cue.get("start"))
        end = cue.get("acoustic_end", cue.get("end"))
        if start is None or end is None:
            continue
        pools["already_dropped"].append(
            {
                "cue_index": index,
                "audio": str(audio),
                "start_s": float(start),
                "end_s": float(end),
                "stratum": "already_dropped",
                "text": text,
            }
        )

    # One band for every stratum. A removed fragment runs shorter than a whole
    # cue at the median, and an auditor who notices that is no longer blind.
    for name, rows in pools.items():
        pools[name] = [
            row
            for row in rows
            if args.min_seconds <= row["end_s"] - row["start_s"] <= args.max_seconds
        ]

    rng = random.Random(args.seed)
    wanted = {
        "split_removed": args.per_stratum,
        "split_kept": args.per_reference_stratum,
        "already_dropped": args.per_reference_stratum,
    }
    picked: list[dict] = []
    for name, count in wanted.items():
        pool = sorted(
            pools.get(name, []),
            key=lambda row: (row["cue_index"], row.get("fragment_order", 0)),
        )
        if not pool:
            raise SystemExit(f"stratum {name!r} is empty")
        take = min(count, len(pool))
        picked.extend(rng.sample(pool, take))
    # Shuffled after sampling, so position carries nothing about the stratum.
    rng.shuffle(picked)
    # Row ids are assigned only now, as a running number over the shuffled list.
    # The page prints this string, so anything derived from the cue - an index,
    # a per-stratum suffix - would hand the auditor the grouping the page exists
    # to hide. The answer key keeps `cue_index` for tracing back.
    for position, row in enumerate(picked, start=1):
        row["row_id"] = f"{args.film_alias}-{position:04d}"

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in picked:
            handle.write(
                json.dumps(
                    {
                        "schema": MANIFEST_SCHEMA,
                        "row_id": row["row_id"],
                        "audio": row["audio"],
                        "start_s": row["start_s"],
                        "end_s": row["end_s"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    key_path = out_dir / "answer_key.jsonl"
    with key_path.open("w", encoding="utf-8") as handle:
        for row in picked:
            handle.write(
                json.dumps({"schema": KEY_SCHEMA, **row}, ensure_ascii=False) + "\n"
            )

    summary = {
        "schema": "split_fragment_audit_selection_v1",
        "cues": args.cues,
        "duration_band_s": [args.min_seconds, args.max_seconds],
        "seed": args.seed,
        "pool_sizes": {name: len(rows) for name, rows in sorted(pools.items())},
        "sampled": dict(collections.Counter(row["stratum"] for row in picked)),
        "manifest": _report(manifest_path),
        "answer_key": _report(key_path),
    }
    (out_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
