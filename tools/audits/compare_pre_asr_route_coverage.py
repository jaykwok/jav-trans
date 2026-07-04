#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse_srt_time(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def read_srt(path: Path) -> list[dict]:
    blocks = re.split(r"\r?\n\r?\n", path.read_text(encoding="utf-8-sig").strip())
    rows = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3 or " --> " not in lines[1]:
            continue
        start, end = lines[1].split(" --> ")
        rows.append(
            {
                "start": _parse_srt_time(start),
                "end": _parse_srt_time(end),
                "text": "\n".join(lines[2:]),
            }
        )
    return rows


def read_route(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compare_coverage(reference: list[dict], routes: list[dict]) -> dict:
    kept = [row for row in routes if row["route"] == "keep_for_asr"]
    uncovered = [
        cue
        for cue in reference
        if not any(
            row["end"] > cue["start"] and row["start"] < cue["end"]
            for row in kept
        )
    ]
    semantic_uncovered = [
        cue
        for cue in uncovered
        if re.search(r"[\w\u3040-\u30ff\u3400-\u9fff]", cue["text"])
    ]
    return {
        "kept_chunks": len(kept),
        "reference_cues": len(reference),
        "uncovered_cues": len(uncovered),
        "semantic_uncovered_cues": len(semantic_uncovered),
        "semantic_uncovered": semantic_uncovered,
    }


def _parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("route must be NAME=PATH")
    name, path = raw.split("=", maxsplit=1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("route must be NAME=PATH")
    return name.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare which reference SRT cues overlap chunks kept by one or more "
            "Pre-ASR route JSONL files. This is a coverage diagnostic, not truth."
        )
    )
    parser.add_argument("--reference-srt", type=Path, required=True)
    parser.add_argument("--route", action="append", type=_parse_named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = read_srt(args.reference_srt)
    result = {
        name: compare_coverage(reference, read_route(path))
        for name, path in args.route
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
