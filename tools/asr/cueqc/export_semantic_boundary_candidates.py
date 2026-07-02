#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.pipeline import (  # noqa: E402
    _build_processing_spans,
    _pre_asr_candidates_for_spans,
    _write_pre_asr_candidates_if_requested,
)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["BOUNDARY_CACHE_ENABLED"] = "0"
    os.environ["PRE_ASR_CUEQC_ENABLED"] = "0"
    os.environ["PRE_ASR_CUEQC_EXPORT_CANDIDATES_PATH"] = str(output)
    os.environ["PRE_ASR_CUEQC_EXPORT_CANDIDATES_APPEND"] = "0"
    if args.split_feature_output:
        os.environ["SEMANTIC_SPLIT_FEATURE_EXPORT_PATH"] = str(
            Path(args.split_feature_output)
        )
    if args.speech_feature_output:
        os.environ["SPEECH_ISLAND_FEATURE_EXPORT_PATH"] = str(
            Path(args.speech_feature_output)
        )
    summary = []
    boundary_audit = Path(args.boundary_audit_output) if args.boundary_audit_output else None
    if boundary_audit is not None:
        boundary_audit.parent.mkdir(parents=True, exist_ok=True)
        boundary_audit.write_text("", encoding="utf-8")
    for audio in args.audio:
        spans = _build_processing_spans(audio)
        if boundary_audit is not None:
            with boundary_audit.open("a", encoding="utf-8") as handle:
                for span_index, span in enumerate(spans):
                    for accepted, candidates in (
                        (True, getattr(span, "primary_cut_candidates", None) or []),
                        (False, getattr(span, "weak_cut_candidates", None) or []),
                    ):
                        for candidate in candidates:
                            handle.write(
                                json.dumps(
                                    {
                                        "audio": audio,
                                        "span_index": span_index,
                                        "span_start": float(span.start),
                                        "span_end": float(span.end),
                                        "accepted": accepted,
                                        **candidate,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
        candidates = _pre_asr_candidates_for_spans(audio, spans)
        log: list[str] = []
        _write_pre_asr_candidates_if_requested(candidates, log=log)
        summary.append(
            {
                "audio": audio,
                "span_count": len(spans),
                "candidate_count": len(candidates),
                "duration_max_s": max(
                    (float(item.get("duration_s") or 0.0) for item in candidates),
                    default=0.0,
                ),
            }
        )
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Pre-ASR candidates from the semantic boundary model chain."
    )
    parser.add_argument("--audio", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--boundary-audit-output")
    parser.add_argument("--split-feature-output")
    parser.add_argument("--speech-feature-output")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
