#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.ja.backend import SpeechBoundaryJaBackend, SpeechBoundaryJaConfig
from boundary.sequence_features import FRAME_SEQUENCE_FRAMES_SCHEMA


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path) -> str:
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def export_silver_sequence_features(
    *,
    silver_labels_path: Path,
    output_dir: Path,
    config: SpeechBoundaryJaConfig,
    limit: int | None = None,
    compressed: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = output_dir / "sequence_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sequence_feature_manifest.jsonl"
    summary_path = output_dir / "sequence_feature_summary.json"
    manifest_path.write_text("", encoding="utf-8")
    cases = _load_unique_cases(silver_labels_path)
    if limit is not None:
        cases = cases[:limit]

    backend = SpeechBoundaryJaBackend(
        SpeechBoundaryJaConfig(
            threshold=config.threshold,
            frame_dilation_s=config.frame_dilation_s,
            frame_hop_s=config.frame_hop_s,
            ptm=config.ptm,
            model_path=config.model_path,
            device=config.device,
            dtype=config.dtype,
            attention=config.attention,
            window_s=config.window_s,
            overlap_s=config.overlap_s,
            min_segment_s=config.min_segment_s,
            max_group_s=config.max_group_s,
            chunk_threshold_s=config.chunk_threshold_s,
            cut_threshold=config.cut_threshold,
            apply_cut_to_speech=config.apply_cut_to_speech,
            export_sequence_features=True,
            sequence_feature_max_ptm_dims=config.sequence_feature_max_ptm_dims,
            no_download=config.no_download,
        )
    )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases, start=1):
            audio_path = project_path(case["source_audio_path"])
            if not audio_path.exists():
                errors.append({**case, "error": "missing_source_audio_path"})
                continue
            print(
                f"sequence_feature_case {index}/{len(cases)} video={case['video']} audio={audio_path}",
                flush=True,
            )
            try:
                result = backend.segment(str(audio_path))
                frames = result.parameters.get("sequence_feature_frames")
                if not isinstance(frames, Mapping):
                    raise ValueError("SpeechBoundary-JA did not export sequence_feature_frames")
                if frames.get("schema") != FRAME_SEQUENCE_FRAMES_SCHEMA:
                    raise ValueError(f"unsupported sequence feature frame schema: {frames.get('schema')!r}")
                ptm = np.asarray(frames.get("ptm"), dtype=np.float32)
                mfcc = np.asarray(frames.get("mfcc"), dtype=np.float32)
                if ptm.ndim != 2 or mfcc.ndim != 2:
                    raise ValueError("sequence feature frames must have [frames, dim] arrays")
                frame_count = min(int(ptm.shape[0]), int(mfcc.shape[0]))
                ptm = np.ascontiguousarray(ptm[:frame_count])
                mfcc = np.ascontiguousarray(mfcc[:frame_count])
                feature_path = feature_dir / f"{_safe_stem(case['video'], audio_path)}.npz"
                payload = {
                    "schema": FRAME_SEQUENCE_FRAMES_SCHEMA,
                    "ptm": ptm,
                    "mfcc": mfcc,
                    "duration_s": np.asarray([float(result.audio_duration_sec)], dtype=np.float32),
                    "frame_hop_s": np.asarray([float(frames.get("frame_hop_s") or config.frame_hop_s)], dtype=np.float32),
                }
                if compressed:
                    np.savez_compressed(feature_path, **payload)
                else:
                    np.savez(feature_path, **payload)
                row = {
                    "schema": FRAME_SEQUENCE_FRAMES_SCHEMA,
                    "video": case["video"],
                    "source_audio_path": case["source_audio_path"],
                    "feature_path": project_rel(feature_path),
                    "duration_s": round(float(result.audio_duration_sec), 6),
                    "frame_hop_s": float(frames.get("frame_hop_s") or config.frame_hop_s),
                    "frame_count": int(frame_count),
                    "ptm_dim": int(ptm.shape[1]),
                    "mfcc_dim": int(mfcc.shape[1]),
                    "compressed": bool(compressed),
                    "ptm": config.ptm,
                    "device": config.device,
                    "dtype": config.dtype,
                }
                rows.append(row)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                with manifest_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            except Exception as exc:
                errors.append({**case, "error": str(exc)})
                print(f"sequence_feature_error video={case['video']} error={exc}", flush=True)
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    errors_path = output_dir / "sequence_feature_errors.jsonl"
    if errors:
        with errors_path.open("w", encoding="utf-8") as handle:
            for row in errors:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema": "speech_boundary_silver_sequence_feature_export_v1",
        "silver_labels": str(silver_labels_path),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "cases": len(cases),
        "exported": len(rows),
        "errors": len(errors),
        "compressed": bool(compressed),
        "config": asdict(config),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")
    print(f"exported={len(rows)} errors={len(errors)}")
    return summary


def _load_unique_cases(path: Path) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    cases: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"silver label row must be an object: {path}:{line_number}")
            video = str(row.get("video") or "")
            source_audio_path = str(row.get("source_audio_path") or "")
            if not video or not source_audio_path:
                continue
            key = (video, project_rel(project_path(source_audio_path)))
            if key in seen:
                continue
            seen.add(key)
            cases.append({"video": video, "source_audio_path": key[1]})
    return cases


def _safe_stem(video: str, audio_path: Path) -> str:
    digest = hashlib.sha1(str(audio_path).encode("utf-8")).hexdigest()[:8]
    safe_video = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in video).strip("_")
    return f"{safe_video or 'audio'}-{digest}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export full-audio sequence feature frames for forced-aligner silver labels."
    )
    parser.add_argument("--silver-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.200)
    parser.add_argument("--frame-dilation-s", type=float, default=0.2)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--ptm", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--overlap-s", type=float, default=1.0)
    parser.add_argument("--min-segment-s", type=float, default=0.05)
    parser.add_argument("--max-group-s", type=float, default=6.0)
    parser.add_argument("--chunk-threshold-s", type=float, default=1.0)
    parser.add_argument("--cut-threshold", type=float, default=0.500)
    parser.add_argument("--no-apply-cut-to-speech", action="store_true")
    parser.add_argument("--sequence-feature-max-ptm-dims", type=int, default=64)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.frame_hop_s <= 0.0:
        parser.error("--frame-hop-s must be positive")
    if args.window_s <= 0.0:
        parser.error("--window-s must be positive")
    if args.overlap_s < 0.0:
        parser.error("--overlap-s must be non-negative")
    if args.sequence_feature_max_ptm_dims <= 0:
        parser.error("--sequence-feature-max-ptm-dims must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    env_config = SpeechBoundaryJaConfig.from_env()
    config = SpeechBoundaryJaConfig(
        threshold=args.threshold,
        frame_dilation_s=args.frame_dilation_s,
        frame_hop_s=args.frame_hop_s,
        ptm=args.ptm.strip() or env_config.ptm,
        model_path=args.model_path.strip() or env_config.model_path,
        device=args.device,
        dtype=args.dtype,
        attention=args.attention,
        window_s=args.window_s,
        overlap_s=args.overlap_s,
        min_segment_s=args.min_segment_s,
        max_group_s=args.max_group_s,
        chunk_threshold_s=args.chunk_threshold_s,
        cut_threshold=args.cut_threshold,
        apply_cut_to_speech=not args.no_apply_cut_to_speech,
        export_sequence_features=True,
        sequence_feature_max_ptm_dims=args.sequence_feature_max_ptm_dims,
        no_download=args.no_download,
    )
    export_silver_sequence_features(
        silver_labels_path=Path(args.silver_labels),
        output_dir=Path(args.output_dir),
        config=config,
        limit=args.limit,
        compressed=args.compress,
    )


if __name__ == "__main__":
    main()
