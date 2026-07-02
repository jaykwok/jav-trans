#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402
from boundary.ja.features import FeatureConfig, build_ptm_feature_extractor, extract_mfcc  # noqa: E402
from tools.boundary.ja.build_feature_cache import (  # noqa: E402
    _combine_workflow_window_features,
    _extract_ptm_window_features,
    _workflow_window_starts,
)


def run(args: argparse.Namespace) -> None:
    audio, sample_rate = load_audio_16k_mono(args.audio)
    config = FeatureConfig(
        ptm=args.ptm,
        frame_hop_s=args.frame_hop_s,
        window_s=args.window_s,
        overlap_s=args.overlap_s,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        feature_dim=args.ptm_dim,
        device=args.device,
        dtype=args.dtype,
        model_path=args.model_path,
        download=False,
        attention=args.attention,
        language=args.language,
    )
    window_samples = max(1, int(round(args.window_s * sample_rate)))
    windows: list[dict] = []
    for window_index, start_sample in enumerate(
        _workflow_window_starts(
            sample_count=len(audio),
            sample_rate=sample_rate,
            window_s=args.window_s,
            overlap_s=args.overlap_s,
        )
    ):
        end_sample = min(len(audio), start_sample + window_samples)
        chunk = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
        windows.append(
            {
                "window_index": window_index,
                "start_sample": int(start_sample),
                "audio": chunk,
                "mfcc": extract_mfcc(chunk, sample_rate=sample_rate, config=config),
            }
        )

    extractor = build_ptm_feature_extractor(config)
    try:
        ptm_features, batch_count = _extract_ptm_window_features(
            ptm_extractor=extractor,
            window_audios=[window["audio"] for window in windows],
            sample_rate=sample_rate,
            ptm_window_batch_size=args.batch_size,
        )
    finally:
        extractor.close()
    bundle = _combine_workflow_window_features(
        windows=windows,
        ptm_features=ptm_features,
        duration_s=len(audio) / sample_rate,
        sample_rate=sample_rate,
        config=config,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ptm=np.asarray(bundle["ptm"][:, : args.ptm_dim], dtype=np.float32),
        mfcc=np.asarray(bundle["mfcc"], dtype=np.float32),
        frame_hop_s=np.asarray([args.frame_hop_s], dtype=np.float32),
    )
    summary = {
        "schema": "audio_sequence_features_v1",
        "audio": args.audio,
        "ptm_repo_id": args.ptm,
        "model_path": args.model_path,
        "frame_count": int(bundle["ptm"].shape[0]),
        "ptm_dim": int(args.ptm_dim),
        "mfcc_dim": int(bundle["mfcc"].shape[1]),
        "window_count": len(windows),
        "ptm_batch_count": batch_count,
        "feature_coverage_ratio": float(bundle["feature_coverage_ratio"]),
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one full-audio PTM+MFCC sequence.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ptm", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ptm-dim", type=int, default=128)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--overlap-s", type=float, default=5.0)
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--language", default="Japanese")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
