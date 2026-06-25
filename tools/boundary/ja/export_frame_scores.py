#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.ja.backend import SpeechBoundaryJaBackend  # noqa: E402


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path) -> str:
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


def export_frame_scores(*, audio_path: Path, output_path: Path) -> dict[str, Any]:
    previous = os.environ.get("SPEECH_BOUNDARY_JA_EXPORT_FRAME_SCORES")
    os.environ["SPEECH_BOUNDARY_JA_EXPORT_FRAME_SCORES"] = "1"
    try:
        result = SpeechBoundaryJaBackend().segment(str(audio_path))
    finally:
        if previous is None:
            os.environ.pop("SPEECH_BOUNDARY_JA_EXPORT_FRAME_SCORES", None)
        else:
            os.environ["SPEECH_BOUNDARY_JA_EXPORT_FRAME_SCORES"] = previous

    scores = result.parameters.get("frame_scores") or []
    split_scores = result.parameters.get("split_boundary_frame_scores") or []
    payload = {
        "audio_path": project_rel(audio_path),
        "backend": result.method,
        "duration_s": float(result.audio_duration_sec),
        "frame_hop_s": float(result.parameters.get("frame_hop_s") or 0.02),
        "frame_count": len(scores),
        "split_boundary_frame_count": len(split_scores),
        "threshold": float(result.parameters.get("threshold") or 0.0),
        "split_strategy": result.parameters.get("split_strategy") or "",
        "split_score_quantile": float(result.parameters.get("split_score_quantile") or 0.0),
        "split_prominence_quantile": float(result.parameters.get("split_prominence_quantile") or 0.0),
        "segments": [
            {"start": float(segment.start), "end": float(segment.end), "score": segment.score}
            for segment in result.segments
        ],
        "audio_stats": result.parameters.get("audio_stats") or {},
        "runtime_device": result.parameters.get("runtime_device") or {},
        "scores": [float(value) for value in scores],
        "split_boundary_scores": [float(value) for value in split_scores],
    }
    write_json(output_path, payload)
    return payload


def run(args: argparse.Namespace) -> None:
    payload = export_frame_scores(
        audio_path=project_path(args.audio),
        output_path=project_path(args.output),
    )
    print(f"frame_scores={project_rel(args.output)}")
    print(
        "frames={frames} duration={duration:.2f}s frame_hop={hop:.6f}s".format(
            frames=payload["frame_count"],
            duration=payload["duration_s"],
            hop=payload["frame_hop_s"],
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SpeechBoundary-JA v6 per-frame speech/split scores."
    )
    parser.add_argument("--audio", required=True, help="16k wav or workflow audio path.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / "frame-scores.json"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
