#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _overlap(start: float, end: float, row: dict) -> float:
    return max(
        0.0,
        min(end, float(row["end"])) - max(start, float(row["start"])),
    )


def run(args: argparse.Namespace) -> None:
    old_rows = _read_jsonl(Path(args.omni_labels))
    by_audio: dict[str, list[dict]] = {}
    for row in old_rows:
        audio_id = str(row.get("audio_id") or row.get("video_id") or "")
        by_audio.setdefault(audio_id, []).append(row)
    output_rows: list[dict] = []
    counts: Counter[str] = Counter()
    for raw_path in args.candidates:
        for candidate in _read_jsonl(Path(raw_path)):
            audio_id = str(
                candidate.get("audio_id") or candidate.get("video_id") or ""
            )
            start = float(candidate["start"])
            end = float(candidate["end"])
            duration = max(1e-6, end - start)
            keep_overlap = sum(
                _overlap(start, end, row)
                for row in by_audio.get(audio_id, [])
                if row.get("label") == "definite_keep"
            )
            drop_overlap = sum(
                _overlap(start, end, row)
                for row in by_audio.get(audio_id, [])
                if row.get("label") == "definite_drop"
            )
            if (
                keep_overlap >= args.min_keep_overlap_s
                or keep_overlap / duration >= args.min_keep_ratio
            ):
                label = "definite_keep"
            elif keep_overlap <= 0.0 and drop_overlap / duration >= args.min_drop_coverage:
                label = "definite_drop"
            else:
                label = "ambiguous_ignore"
            counts[label] += 1
            output_rows.append(
                {
                    "schema": "pre_asr_semantic_chunk_inherited_omni_v1",
                    "sample_id": candidate["sample_id"],
                    "candidate_id": candidate["candidate_id"],
                    "audio_id": audio_id,
                    "video_id": audio_id,
                    "chunk_index": int(candidate["chunk_index"]),
                    "start": start,
                    "end": end,
                    "duration_s": duration,
                    "label": label,
                    "display_decision": (
                        "keep"
                        if label == "definite_keep"
                        else "drop"
                        if label == "definite_drop"
                        else "ambiguous_ignore"
                    ),
                    "training_label_included": label != "ambiguous_ignore",
                    "label_source": "inherited:pre_asr_omni_label_v1",
                    "keep_overlap_s": keep_overlap,
                    "drop_overlap_s": drop_overlap,
                    "old_label_coverage": min(
                        1.0, (keep_overlap + drop_overlap) / duration
                    ),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema": "pre_asr_semantic_chunk_inherited_omni_summary_v1",
        "count": len(output_rows),
        "labels": dict(counts),
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--omni-labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-keep-overlap-s", type=float, default=0.08)
    parser.add_argument("--min-keep-ratio", type=float, default=0.25)
    parser.add_argument("--min-drop-coverage", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
