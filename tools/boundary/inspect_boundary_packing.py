#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.chunk_packer import PackedChunk  # noqa: E402
from asr import pipeline as asr_pipeline  # noqa: E402


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def stats(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(data),
        "min": round(min(data), 6),
        "p50": round(quantile(data, 0.5), 6),
        "p90": round(quantile(data, 0.9), 6),
        "p95": round(quantile(data, 0.95), 6),
        "max": round(max(data), 6),
        "mean": round(statistics.fmean(data), 6),
    }


def chunk_to_row(index: int, chunk: PackedChunk) -> dict[str, Any]:
    return {
        "chunk_index": index,
        "start": chunk.start,
        "end": chunk.end,
        "duration_s": chunk.duration,
        "core_start": chunk.core_start,
        "core_end": chunk.core_end,
        "core_duration_s": (
            max(0.0, float(chunk.core_end) - float(chunk.core_start))
            if chunk.core_start is not None and chunk.core_end is not None
            else 0.0
        ),
        "speech_island_count": len(chunk.speech_segments),
        "internal_gap_count": chunk.internal_gap_count,
        "internal_gap_max_s": chunk.internal_gap_max_s,
        "split_reason": chunk.split_reason,
        "boundary_score": chunk.boundary_score,
        "boundary_reason": chunk.boundary_reason,
        "boundary_source": chunk.boundary_source,
        "boundary_start_refine_delta_s": chunk.boundary_start_refine_delta_s,
        "boundary_end_refine_delta_s": chunk.boundary_end_refine_delta_s,
        "boundary_decision_source": chunk.boundary_decision_source,
    }


def inspect(audio_path: Path, output_dir: Path) -> dict[str, Any]:
    spans = asr_pipeline._build_processing_spans(str(audio_path))
    if not all(isinstance(item, PackedChunk) for item in spans):
        raise ValueError("boundary packing inspection requires PackedChunk spans")
    chunks = [item for item in spans if isinstance(item, PackedChunk)]
    rows = [chunk_to_row(index, chunk) for index, chunk in enumerate(chunks)]
    summary = {
        "schema": "boundary_packing_inspection_v1",
        "audio": project_rel(audio_path),
        "chunk_count": len(rows),
        "boundary_cache_event": asr_pipeline._LAST_BOUNDARY_CACHE_EVENT,
        "boundary_signature": asr_pipeline._LAST_BOUNDARY_SIGNATURE,
        "duration_s": stats(row["duration_s"] for row in rows),
        "core_duration_s": stats(row["core_duration_s"] for row in rows),
        "speech_island_count": stats(row["speech_island_count"] for row in rows),
        "internal_gap_max_s": stats(row["internal_gap_max_s"] for row in rows),
        "split_reason_counts": dict(Counter(str(row["split_reason"]) for row in rows).most_common()),
        "decision_source_counts": dict(Counter(str(row["boundary_decision_source"] or "none") for row in rows).most_common()),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "chunks.json", rows)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SpeechBoundary-JA packed chunks without ASR.")
    parser.add_argument("--audio", required=True, help="Prepared wav path.")
    parser.add_argument("--output-dir", default="agents/temp/speech-boundary-ja/boundary-packing-inspection")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = inspect(project_path(args.audio), project_path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
