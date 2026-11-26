#!/usr/bin/env python3
"""Text-override manifests that put the moaning back into the CTC target (arm F).

The v2 head was trained on stripped targets: the moans were deleted from the
text, so CTC had nothing to place over them and could only explain them as
blank. That was the only way to supervise "vocalisation is not a word" when
blank was the sole non-word class - and it is what conflated moaning with
silence.

With a three-class frame head, that job moves off CTC. Arm F tests whether it
should: the target becomes the full script, moans and all, so CTC aligns them as
text and blank goes back to meaning silence. Arm S keeps the stripped target and
changes only the frame head, which is the smaller change and leaves the pause
reading the chunker depends on exactly as it is.

This writes only the `audio_id` -> `text` mapping the trainer's
`--text-override-manifest` consumes. It creates no features and touches no
cache: arm S and arm F differ in one column of one file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "full_script_ctc_target_manifest_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="repeatable; rows carrying `audio_id` and `script_text`",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    missing_script = 0
    unchanged = 0
    for relative in args.manifest:
        path = PROJECT_ROOT / relative
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            script = str(row.get("script_text") or "")
            if not script:
                # No script to restore. Emitting the row unchanged keeps the
                # override total, which the trainer requires - a manifest that
                # silently skipped rows would leave a mixture of stripped and
                # full targets that trains without complaint.
                missing_script += 1
                script = str(row.get("text") or "")
            if script == str(row.get("text") or ""):
                unchanged += 1
            rows.append(
                {
                    "schema": SCHEMA,
                    "audio_id": row["audio_id"],
                    "text": script,
                    "target_kind": "full_script_text",
                }
            )

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "sources": list(args.manifest),
                "rows": len(rows),
                "rows_without_script_text": missing_script,
                "rows_already_equal_to_the_stripped_target": unchanged,
                "out": args.out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
