#!/usr/bin/env python3
"""Cut the long dropped spans into clips for relabelling.

456 of the 7,604 dropped spans are 3 s or longer. They are 6.0% of the spans and
hold 79% of the wrongly-dropped speech time, which is why the relabelling job is
scoped to them rather than to everything: the 6,124 spans under a second came
back 0/12 in the false-drop audit, and time spent on them buys nothing.

Clips are cut with the same call, format, bitrate and sample rate as the audit
pages, because the teacher's agreement with the human verdicts was measured on
those files. A different encode would quietly move the teacher off the audio it
was calibrated on.

Spans are NOT truncated. A human cannot hold a 40 s clip in mind, which is why
the audits capped theirs, but the label being rewritten applies to the whole
span, so the whole span is what gets judged. Clip duration is recorded for every
item so the output audit can check whether the long tail behaves differently
from the calibrated 3-10 s range.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.binary_clip_audit import safe_name  # noqa: E402

SCHEMA = "long_drop_span_clip_v1"
MIN_SPAN_S = 3.0


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(dataset: Path, *, min_span_s: float) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for example in _rows(dataset):
        if example.get("provenance") != "real_omni_joint":
            continue
        for span in example.get("spans") or []:
            if str(span.get("source_label")) != "definite_drop":
                continue
            start = float(span["start_s"])
            end = float(span["end_s"])
            if end - start < min_span_s:
                continue
            window_id = str(example.get("window_id") or "")
            picked.append(
                {
                    "schema": SCHEMA,
                    # Stable across reruns: derived from the span, not its index.
                    "item_id": (
                        f"{window_id}@"
                        + hashlib.sha256(
                            f"{window_id}|{start:.6f}|{end:.6f}".encode()
                        ).hexdigest()[:10]
                    ),
                    "window_id": window_id,
                    "video_id": str(example.get("video_id") or ""),
                    "dataset": str(example.get("dataset") or ""),
                    "source_audio": str(example.get("audio") or ""),
                    "start_s": round(start, 6),
                    "end_s": round(end, 6),
                    "clip_duration_s": round(end - start, 3),
                    "type_label": str(span.get("type")),
                    "flags": list(span.get("flags") or []),
                }
            )
    picked.sort(key=lambda row: row["item_id"])
    return picked


def cut(items: list[dict[str, Any]], media_dir: Path) -> list[dict[str, Any]]:
    from tools.omni.openai_compat import slice_audio_clip

    media_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        source = Path(item["source_audio"])
        if not source.is_file():
            raise FileNotFoundError(f"source audio missing: {source}")
        clip = media_dir / f"{safe_name(item['item_id'])}.mp3"
        slice_audio_clip(
            source_audio=source,
            row={
                "start": item["start_s"],
                "end": item["end_s"],
                "duration_s": item["clip_duration_s"],
            },
            output_path=clip,
            fmt="mp3",
            bitrate="64k",
            sample_rate=16000,
            force=False,
        )
        out.append({**item, "audio": str(clip)})
        if index % 50 == 0:
            print(f"  cut {index}/{len(items)}", flush=True)
    return out


def build(
    *, dataset: Path, output_dir: Path, min_span_s: float
) -> dict[str, Any]:
    items = select(dataset, min_span_s=min_span_s)
    output_dir.mkdir(parents=True, exist_ok=True)
    cut_items = cut(items, output_dir / "media")
    items_path = output_dir / "items.jsonl"
    with items_path.open("w", encoding="utf-8") as handle:
        for item in cut_items:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    lengths = sorted(item["clip_duration_s"] for item in cut_items)
    buckets = Counter()
    for value in lengths:
        buckets["3-10s" if value < 10 else "10-30s" if value < 30 else ">=30s"] += 1
    return {
        "schema": "long_drop_span_clip_summary_v1",
        "dataset": str(dataset),
        "items": str(items_path),
        "min_span_s": min_span_s,
        "count": len(cut_items),
        "videos": len({item["video_id"] for item in cut_items}),
        "windows": len({item["window_id"] for item in cut_items}),
        "total_minutes": round(sum(lengths) / 60, 1),
        "median_clip_s": lengths[len(lengths) // 2] if lengths else None,
        "max_clip_s": lengths[-1] if lengths else None,
        "length_buckets": dict(buckets),
        "type_labels": dict(Counter(item["type_label"] for item in cut_items)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-span-s", type=float, default=MIN_SPAN_S)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build(
                dataset=Path(args.dataset).expanduser().resolve(),
                output_dir=Path(args.output_dir).expanduser().resolve(),
                min_span_s=float(args.min_span_s),
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
