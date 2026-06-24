#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.base import SpeechSegment  # noqa: E402
from boundary.ja.backend import (  # noqa: E402
    SpeechBoundaryJaConfig,
    _quantile,
    _select_peak_frames,
    _split_peak_candidates_for_segment,
    _split_segment_at_frames,
    decode_frame_boundary_segments,
)


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path) -> str:
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _array_from_payload(payload: dict[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        values = payload.get(key)
        if isinstance(values, list):
            return np.asarray(values, dtype=np.float32).reshape(-1)
    return np.zeros(0, dtype=np.float32)


def _segments_payload(segments: list[SpeechSegment]) -> list[dict[str, float | None]]:
    return [
        {
            "start": round(float(segment.start), 6),
            "end": round(float(segment.end), 6),
            "duration_s": round(float(segment.end) - float(segment.start), 6),
            "score": None if segment.score is None else round(float(segment.score), 6),
        }
        for segment in segments
    ]


def _max_duration(segments: list[SpeechSegment]) -> float:
    if not segments:
        return 0.0
    return float(max(segment.end - segment.start for segment in segments))


def config_from_frame_score_payload(
    payload: dict[str, Any],
    *,
    base: SpeechBoundaryJaConfig | None = None,
    overrides: dict[str, float | None] | None = None,
) -> SpeechBoundaryJaConfig:
    """Build a decoder config from exported frame scores.

    Older frame-score JSON only stored a subset of split parameters. Values in
    the payload win over defaults when present; explicit CLI overrides win last.
    """

    cfg = base or SpeechBoundaryJaConfig()
    audio_stats = payload.get("audio_stats")
    if not isinstance(audio_stats, dict):
        audio_stats = {}
    values: dict[str, Any] = {}
    field_keys = {
        "frame_hop_s": ("frame_hop_s",),
        "threshold": ("threshold",),
        "split_smooth_s": ("split_smooth_s",),
        "split_nms_s": ("split_nms_s",),
        "split_snap_s": ("split_snap_s",),
        "min_segment_s": ("min_segment_s",),
        "min_split_segment_s": ("min_split_segment_s",),
        "frame_dilation_s": ("frame_dilation_s",),
        "split_score_quantile": ("split_score_quantile",),
        "split_prominence_quantile": ("split_prominence_quantile",),
    }
    for field, keys in field_keys.items():
        for key in keys:
            parsed = _float_or_none(payload.get(key))
            if parsed is None:
                parsed = _float_or_none(audio_stats.get(key))
            if parsed is not None:
                values[field] = parsed
                break
    speech_on = _float_or_none(audio_stats.get("speech_on_threshold"))
    speech_off = _float_or_none(audio_stats.get("speech_off_threshold"))
    if speech_on is not None:
        values["speech_on_threshold"] = speech_on
    if speech_off is not None:
        values["speech_off_threshold"] = speech_off

    for key, value in (overrides or {}).items():
        if value is not None:
            values[key] = float(value)
    return replace(cfg, **values)


def _candidate_payload(candidate: Any, *, frame_hop_s: float) -> dict[str, float | int]:
    return {
        "frame": int(candidate.frame),
        "time_s": round(float(candidate.frame) * frame_hop_s, 6),
        "score": round(float(candidate.score), 6),
        "prominence": round(float(candidate.prominence), 6),
    }


def diagnose_split_peaks(
    payload: dict[str, Any],
    *,
    config: SpeechBoundaryJaConfig | None = None,
    min_coarse_duration_s: float = 0.0,
    top_candidates: int = 8,
) -> dict[str, Any]:
    cfg = config or config_from_frame_score_payload(payload)
    speech_probs = _array_from_payload(payload, "scores", "frame_scores")
    split_probs = _array_from_payload(payload, "split_boundary_scores", "split_boundary_frame_scores")
    duration_s = float(payload.get("duration_s") or 0.0)
    if duration_s <= 0.0 and cfg.frame_hop_s > 0.0:
        duration_s = float(max(speech_probs.size, split_probs.size)) * cfg.frame_hop_s

    decode = decode_frame_boundary_segments(
        speech_probabilities=speech_probs,
        split_probabilities=split_probs,
        duration_s=duration_s,
        config=cfg,
    )
    nms_frames = max(1, int(round(max(0.0, cfg.split_nms_s) / cfg.frame_hop_s)))

    islands: list[dict[str, Any]] = []
    for index, segment in enumerate(decode.coarse_segments):
        segment_duration_s = float(segment.end) - float(segment.start)
        if segment_duration_s < float(min_coarse_duration_s):
            continue
        candidates = _split_peak_candidates_for_segment(
            segment,
            speech_probs=speech_probs,
            split_probs=split_probs,
            frame_hop_s=cfg.frame_hop_s,
            split_smooth_s=cfg.split_smooth_s,
            split_snap_s=cfg.split_snap_s,
            min_split_segment_s=cfg.min_split_segment_s,
        )
        score_floor = _quantile((candidate.score for candidate in candidates), cfg.split_score_quantile)
        prominence_floor = _quantile(
            (candidate.prominence for candidate in candidates),
            cfg.split_prominence_quantile,
        )
        adaptive_candidates = [
            candidate
            for candidate in candidates
            if candidate.score >= score_floor and candidate.prominence >= prominence_floor
        ]
        selected_frames = sorted(
            _select_peak_frames(
                adaptive_candidates,
                nms_frames=nms_frames,
                rank_by_prominence=True,
            )
        )
        parts = _split_segment_at_frames(
            segment,
            selected_frames,
            frame_hop_s=cfg.frame_hop_s,
            min_split_segment_s=cfg.min_split_segment_s,
        )
        if not candidates:
            decision = "no_local_prominent_peaks"
        elif not adaptive_candidates:
            decision = "quantile_filter_rejected_all"
        elif not selected_frames:
            decision = "nms_rejected_all"
        elif len(parts) <= 1:
            decision = "min_split_segment_rejected_all"
        else:
            decision = "split_selected"
        ranked_candidates = sorted(
            candidates,
            key=lambda item: (float(item.prominence), float(item.score)),
            reverse=True,
        )
        islands.append(
            {
                "index": index,
                "start": round(float(segment.start), 6),
                "end": round(float(segment.end), 6),
                "duration_s": round(segment_duration_s, 6),
                "candidate_count": len(candidates),
                "adaptive_candidate_count": len(adaptive_candidates),
                "decision": decision,
                "score_floor": round(float(score_floor), 6),
                "prominence_floor": round(float(prominence_floor), 6),
                "selected_frames": [int(frame) for frame in selected_frames],
                "selected_times_s": [round(float(frame) * cfg.frame_hop_s, 6) for frame in selected_frames],
                "parts": _segments_payload(parts),
                "top_candidates": [
                    _candidate_payload(candidate, frame_hop_s=cfg.frame_hop_s)
                    for candidate in ranked_candidates[: max(0, int(top_candidates))]
                ],
            }
        )

    existing_segments = [
        SpeechSegment(
            start=float(row.get("start", 0.0)),
            end=float(row.get("end", 0.0)),
            score=_float_or_none(row.get("score")),
        )
        for row in payload.get("segments", [])
        if isinstance(row, dict)
    ]
    return {
        "schema": "speech_boundary_ja_split_peak_diagnostic_v1",
        "audio_path": payload.get("audio_path") or "",
        "duration_s": round(float(duration_s), 6),
        "frame_count": int(max(speech_probs.size, split_probs.size)),
        "decode_config": {
            "threshold": float(cfg.threshold),
            "speech_on_threshold": float(decode.speech_on_threshold),
            "speech_off_threshold": float(decode.speech_off_threshold),
            "frame_dilation_s": float(cfg.frame_dilation_s),
            "min_segment_s": float(cfg.min_segment_s),
            "split_smooth_s": float(cfg.split_smooth_s),
            "split_nms_s": float(cfg.split_nms_s),
            "split_snap_s": float(cfg.split_snap_s),
            "min_split_segment_s": float(cfg.min_split_segment_s),
            "split_score_quantile": float(cfg.split_score_quantile),
            "split_prominence_quantile": float(cfg.split_prominence_quantile),
            "nms_frames": int(nms_frames),
        },
        "summary": {
            "existing_segment_count": len(existing_segments),
            "existing_max_segment_s": round(_max_duration(existing_segments), 6),
            "redecoded_segment_count": len(decode.segments),
            "redecoded_max_segment_s": round(_max_duration(decode.segments), 6),
            "coarse_segment_count": len(decode.coarse_segments),
            "coarse_max_segment_s": round(_max_duration(decode.coarse_segments), 6),
            "diagnosed_coarse_segment_count": len(islands),
            "islands_with_candidates": sum(1 for item in islands if item["candidate_count"] > 0),
            "islands_with_selected_splits": sum(1 for item in islands if item["selected_frames"]),
        },
        "coarse_segments": _segments_payload(decode.coarse_segments),
        "redecoded_segments": _segments_payload(decode.segments),
        "islands": islands,
    }


def _override_map(args: argparse.Namespace) -> dict[str, float | None]:
    return {
        "threshold": args.threshold,
        "speech_on_threshold": args.speech_on_threshold,
        "speech_off_threshold": args.speech_off_threshold,
        "frame_dilation_s": args.frame_dilation_s,
        "min_segment_s": args.min_segment_s,
        "split_smooth_s": args.split_smooth_s,
        "split_nms_s": args.split_nms_s,
        "split_snap_s": args.split_snap_s,
        "min_split_segment_s": args.min_split_segment_s,
        "split_score_quantile": args.split_score_quantile,
        "split_prominence_quantile": args.split_prominence_quantile,
    }


def run(args: argparse.Namespace) -> None:
    input_path = project_path(args.input)
    output_path = project_path(args.output)
    payload = read_json(input_path)
    cfg = config_from_frame_score_payload(
        payload,
        base=SpeechBoundaryJaConfig(),
        overrides=_override_map(args),
    )
    diagnostic = diagnose_split_peaks(
        payload,
        config=cfg,
        min_coarse_duration_s=args.min_coarse_duration_s,
        top_candidates=args.top_candidates,
    )
    diagnostic["input"] = project_rel(input_path)
    write_json(output_path, diagnostic)
    summary = diagnostic["summary"]
    print(f"split_peak_diagnostic={project_rel(output_path)}")
    print(
        "coarse={coarse} redecoded={decoded} max={max_s:.3f}s selected_islands={selected}".format(
            coarse=summary["coarse_segment_count"],
            decoded=summary["redecoded_segment_count"],
            max_s=summary["redecoded_max_segment_s"],
            selected=summary["islands_with_selected_splits"],
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose adaptive split peak selection from exported SpeechBoundary-JA frame scores."
    )
    parser.add_argument("--input", required=True, help="Frame-score JSON from tools.boundary.ja.export_frame_scores.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / "split-peak-diagnostic.json"),
    )
    parser.add_argument("--min-coarse-duration-s", type=float, default=0.0)
    parser.add_argument("--top-candidates", type=int, default=8)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--speech-on-threshold", type=float)
    parser.add_argument("--speech-off-threshold", type=float)
    parser.add_argument("--frame-dilation-s", type=float)
    parser.add_argument("--min-segment-s", type=float)
    parser.add_argument("--split-smooth-s", type=float)
    parser.add_argument("--split-nms-s", type=float)
    parser.add_argument("--split-snap-s", type=float)
    parser.add_argument("--min-split-segment-s", type=float)
    parser.add_argument("--split-score-quantile", type=float)
    parser.add_argument("--split-prominence-quantile", type=float)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
