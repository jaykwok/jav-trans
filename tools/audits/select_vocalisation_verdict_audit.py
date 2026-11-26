#!/usr/bin/env python3
"""Sample the cues the joint verdict would newly delete, for a listening test.

The joint verdict (text decomposition x frame-class posteriors) removes about a
third more cues than the text rule alone. Every statistic available says that is
an improvement, and none of them can say it: the acceptance groups are the text
rule's own verdicts, so "the text rule kept it and the acoustics dropped it" has
no ground truth by construction. Whether those cues contain words is a question
about sound, and only an ear answers it.

Four strata, and two of them exist to make the other two readable:

  * `new_isolated`      a lone vocalisation cue the run rule could not reach.
                        Symptom (e). The acoustics are the only evidence there
                        has ever been for these.
  * `new_kana`          kana the lexicon fails to decompose - `じゅぽっ`,
                        `ごくんっ`, `ちゅぽっ`. Symptom (f), the allow-list's
                        known and unfixable gap.
  * `already_dropped`   the text rule already deletes these. Not under test; it
                        says what "no words" scores when the current shipped
                        behaviour is right, so the two new strata can be read
                        against something rather than against zero.
  * `dialogue_control`  cues carrying kanji. These certainly contain words, and
                        the joint verdict never touches them. If they do not
                        come back as "has words", the ear or the question is
                        wrong and the whole page is void - the same role the
                        `control_early` stratum plays in the onset audit.

**Duration is matched across strata.** `new_kana` runs about a second longer at
the median than `new_isolated`, and an auditor who notices that is no longer
blind. Sampling inside a common band removes it without truncating any cue,
which matters here because the question is about the whole cue's content and a
fixed-length cut would slice words in half.

Writes the manifest the audit shell consumes - which may carry nothing but
`row_id`, `audio`, `start_s`, `end_s` - and, separately, the answer key. The two
are separate files because the shell refuses a manifest with any other field,
and that refusal is the only thing standing between a blind audit and a labelled
one.
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

# The plan's literal §5.4 thresholds. Named here rather than inlined so the key
# file records which rule the sample was drawn under - a sample drawn under one
# threshold and read against another measures nothing.
THRESHOLDS = {
    "vocal_speech_max": 0.10,
    "vocal_speech_run_max_s": 0.30,
    "kana_speech_max": 0.05,
    "kana_vocalisation_min": 0.60,
}


def is_protected(text: str) -> bool:
    """§5.4's last row: `protected | any | keep`, whole and per part."""
    body = V._strip_decoration(text)
    if V._is_protected(body):
        return True
    return any(V._is_protected(part) for part in V._parts(text))


def stratum(cue: dict, head: str) -> str | None:
    speech = cue.get(f"{head}:speech")
    vocal = cue.get(f"{head}:vocalisation")
    run = cue.get(f"{head}:speech_run")
    if speech is None or is_protected(cue["text"]):
        return None
    group = cue["group"]
    if group == "vocalisation_isolated":
        if speech < THRESHOLDS["vocal_speech_max"] and run < THRESHOLDS[
            "vocal_speech_run_max_s"
        ]:
            return "new_isolated"
        return None
    if group == "kana_only_kept":
        if speech < THRESHOLDS["kana_speech_max"] and vocal > THRESHOLDS[
            "kana_vocalisation_min"
        ]:
            return "new_kana"
        return None
    if group == "vocalisation_dropped":
        return "already_dropped"
    if group == "dialogue_lexical":
        return "dialogue_control"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acceptance-dir",
        required=True,
        help="directory of per-film compare_heads_on_film reports (aliased names)",
    )
    parser.add_argument("--head", default="S50")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-new-stratum", type=int, default=20)
    parser.add_argument("--per-reference-stratum", type=int, default=10)
    parser.add_argument("--min-seconds", type=float, default=1.5)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    acceptance = PROJECT_ROOT / args.acceptance_dir
    pools: dict[str, list[dict]] = collections.defaultdict(list)
    for report_path in sorted(acceptance.glob("sample-*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        audio = PROJECT_ROOT / report["audio"]
        blocks = json.loads(
            (PROJECT_ROOT / report["bilingual"]).read_text(encoding="utf-8")
        )["blocks"]
        for cue in report["cues"]:
            name = stratum(cue, args.head)
            if name is None:
                continue
            block = blocks[int(cue["index"])]
            start = block.get("acoustic_start")
            end = block.get("acoustic_end")
            if start is None or end is None:
                start, end = block.get("start"), block.get("end")
            if start is None or end is None:
                continue
            start, end = float(start), float(end)
            if not args.min_seconds <= end - start <= args.max_seconds:
                continue
            pools[name].append(
                {
                    # The alias, never the film. The page shows this string.
                    "row_id": f"{report_path.stem}-{int(cue['index']):05d}",
                    "audio": str(audio),
                    "start_s": start,
                    "end_s": end,
                    "stratum": name,
                    "text": cue["text"],
                    "speech": cue.get(f"{args.head}:speech"),
                    "vocalisation": cue.get(f"{args.head}:vocalisation"),
                    "speech_run_s": cue.get(f"{args.head}:speech_run"),
                }
            )

    rng = random.Random(args.seed)
    wanted = {
        "new_isolated": args.per_new_stratum,
        "new_kana": args.per_new_stratum,
        "already_dropped": args.per_reference_stratum,
        "dialogue_control": args.per_reference_stratum,
    }
    picked: list[dict] = []
    for name, count in wanted.items():
        pool = sorted(pools.get(name, []), key=lambda row: row["row_id"])
        if len(pool) < count:
            raise SystemExit(
                f"stratum {name!r} has only {len(pool)} eligible cues, wanted {count}"
            )
        picked.extend(rng.sample(pool, count))
    # Shuffled after sampling, so position carries nothing about the stratum.
    rng.shuffle(picked)

    out_dir = PROJECT_ROOT / args.out_dir
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
        "schema": "vocalisation_verdict_audit_selection_v1",
        "head": args.head,
        "thresholds": THRESHOLDS,
        "duration_band_s": [args.min_seconds, args.max_seconds],
        "seed": args.seed,
        "pool_sizes": {name: len(rows) for name, rows in sorted(pools.items())},
        "sampled": dict(collections.Counter(row["stratum"] for row in picked)),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "answer_key": str(key_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    (out_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
