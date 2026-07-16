#!/usr/bin/env python3
"""Export actual Runtime v10 provisional sub-islands for a fixed audit pool."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.pipeline import (  # noqa: E402
    _build_processing_spans,
    _pre_asr_candidates_for_spans,
)
from audio.chunk_packer import PackedChunk  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_audio_path(*, value: str, items_path: Path) -> Path:
    audio = Path(value)
    if audio.is_absolute():
        resolved = audio
    else:
        manifest_relative = items_path.parent / audio
        project_relative = PROJECT_ROOT / audio
        resolved = manifest_relative if manifest_relative.exists() else project_relative
    if not resolved.exists():
        raise FileNotFoundError(f"source audio not found: {resolved}")
    return resolved.resolve()


def run(args: argparse.Namespace) -> None:
    apply_vram_safety_cap(0.95)
    sources: dict[str, tuple[Path, dict[str, Any]]] = {}
    items_path = Path(args.audit_items).resolve()
    for row in _rows(items_path):
        sample_id = str(row["sample_id"])
        try:
            audio = resolve_audio_path(value=str(row["audio"]), items_path=items_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{sample_id}: {exc}") from exc
        sources[sample_id] = (audio, row)
    if args.max_sources > 0:
        sources = dict(list(sorted(sources.items()))[: args.max_sources])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = (
        _rows(output) if args.resume and output.exists() else []
    )
    completed_sources = {str(row["sample_id"]) for row in exported}
    if not args.resume:
        output.write_text("", encoding="utf-8")
    for source_index, (sample_id, source_item) in enumerate(
        sorted(sources.items()), start=1
    ):
        audio, source_row = source_item
        if sample_id in completed_sources:
            print(
                f"runtime_v10_provisional={source_index}/{len(sources)} "
                f"sample={sample_id} status=resume_skip",
                flush=True,
            )
            continue
        print(
            f"runtime_v10_provisional={source_index}/{len(sources)} sample={sample_id}",
            flush=True,
        )
        spans = _build_processing_spans(str(audio))
        if not all(isinstance(span, PackedChunk) for span in spans):
            raise ValueError(f"{sample_id}: Runtime v10 did not return PackedChunk rows")
        chunks = [span for span in spans if isinstance(span, PackedChunk)]
        if any(chunk.boundary_pipeline_version != 10 for chunk in chunks):
            raise ValueError(f"{sample_id}: expected Boundary pipeline version 10")
        candidates = _pre_asr_candidates_for_spans(str(audio), chunks)
        if len(candidates) != len(chunks):
            raise ValueError(f"{sample_id}: candidate/chunk count mismatch")
        source_rows: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            source_rows.append(
                {
                    "schema": "runtime_v10_provisional_subisland_v1",
                    "sample_id": sample_id,
                    "source_partition": str(
                        source_row.get("source_partition") or "train"
                    ),
                    "subisland_id": f"{sample_id}__v10s{index:02d}",
                    "audio": str(audio),
                    "start_s": float(chunk.start),
                    "end_s": float(chunk.end),
                    "duration_s": float(chunk.duration),
                    "left_event_id": (
                        chunk.semantic_event_ids[0]
                        if chunk.semantic_event_ids
                        else None
                    ),
                    "right_event_id": (
                        chunk.semantic_event_ids[-1]
                        if chunk.semantic_event_ids
                        else None
                    ),
                    "semantic_event_ids": list(chunk.semantic_event_ids or []),
                    "semantic_event_probabilities": list(
                        chunk.semantic_event_probabilities or []
                    ),
                    "inner_edge_prediction": dict(chunk.inner_edge_prediction or {}),
                    "boundary_source": chunk.boundary_source,
                    "boundary_decision_source": chunk.boundary_decision_source,
                    "primary_cut_candidates": list(
                        chunk.primary_cut_candidates or []
                    ),
                    "weak_cut_candidates": list(chunk.weak_cut_candidates or []),
                    "pre_asr_candidate": candidates[index],
                }
            )
        with output.open("a", encoding="utf-8") as handle:
            for row in source_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        exported.extend(source_rows)
        completed_sources.add(sample_id)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    summary = {
        "schema": "runtime_v10_provisional_export_summary_v1",
        "source_count": len(sources),
        "subisland_count": len(exported),
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-items", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
