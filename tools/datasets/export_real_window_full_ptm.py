#!/usr/bin/env python3
"""Re-export the real omni windows at full PTM width, model loaded once.

The cached `speech_sequence_features.npz` for the 810 real windows stores PTM
truncated to the leading 128 dims. Two things need the full 2048:

  * the learned 2048->128 projector (`learned_ptm_projection.npz`) cannot be
    applied to an already-truncated array - 10.5% of its component energy comes
    from dims outside the first 128
  * the leading 128 dims hold only ~6.7% of total PTM energy, near the 6.25% a
    uniform spread would give, so "does width matter" is an open question that
    truncated features cannot answer

`export_audio_sequence_features.py` does exactly this for ONE file, but reloads
1.7B weights per invocation; across 810 windows that is almost entirely model
loading. This keeps the extractor open and streams the windows through it.

Extraction settings match the synthetic cache (`feature-cache-17b-hf-bf16`):
bfloat16, sdpa, 30 s windows, 5 s overlap. The existing real cache does not
reproduce bit-exactly under these settings (r=0.94 on the shared 128 dims), so
anything comparing real against synthetic should prefer features from this tool
on both sides rather than mixing the two extraction paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402
from boundary.gpu_safety import (  # noqa: E402
    apply_host_memory_cap,
    apply_vram_safety_cap,
)
from boundary.ja.features import (  # noqa: E402
    FeatureConfig,
    build_ptm_feature_extractor,
    extract_mfcc,
)
from tools.boundary.ja.build_feature_cache import (  # noqa: E402
    _combine_workflow_window_features,
    _extract_ptm_window_features,
    _workflow_window_starts,
)

OMNI_DATASETS = (
    "omni-joint-boundary-preasr-v1",
    "omni-joint-boundary-preasr-v2",
    "omni-joint-boundary-preasr-v3",
)


def _window_features(
    audio_path: Path, *, config: FeatureConfig, extractor, batch_size: int
) -> dict:
    audio, sample_rate = load_audio_16k_mono(str(audio_path))
    window_samples = max(1, int(round(config.window_s * sample_rate)))
    windows: list[dict] = []
    for window_index, start_sample in enumerate(
        _workflow_window_starts(
            sample_count=len(audio),
            sample_rate=sample_rate,
            window_s=config.window_s,
            overlap_s=config.overlap_s,
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
    ptm_features, _batches = _extract_ptm_window_features(
        ptm_extractor=extractor,
        window_audios=[window["audio"] for window in windows],
        sample_rate=sample_rate,
        ptm_window_batch_size=batch_size,
    )
    return _combine_workflow_window_features(
        windows=windows,
        ptm_features=ptm_features,
        duration_s=len(audio) / sample_rate,
        sample_rate=sample_rate,
        config=config,
    )


def export(args: argparse.Namespace) -> dict:
    vram_ratio = apply_vram_safety_cap()
    guard = apply_host_memory_cap()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[str, Path]] = []
    for name in OMNI_DATASETS:
        audio_dir = PROJECT_ROOT / "datasets" / "train" / name / "audio_wav"
        if not audio_dir.is_dir():
            continue
        for wav in sorted(audio_dir.glob("*.wav")):
            targets.append((f"{name}:{wav.stem}", wav))
    if args.limit:
        targets = targets[: args.limit]
    if not targets:
        raise SystemExit("no real omni windows found")

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

    manifest_path = output_root / "feature_manifest.jsonl"
    done: set[str] = set()
    if args.resume and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(str(json.loads(line)["example_id"]))
        print(f"resume: {len(done)} already exported")

    extractor = build_ptm_feature_extractor(config)
    written = 0
    failed = 0
    started = time.time()
    try:
        with manifest_path.open("a", encoding="utf-8") as manifest:
            for position, (example_id, wav) in enumerate(targets, start=1):
                if example_id in done:
                    continue
                target = output_root / "features" / f"{example_id.replace(':', '__')}.npz"
                try:
                    bundle = _window_features(
                        wav,
                        config=config,
                        extractor=extractor,
                        batch_size=args.batch_size,
                    )
                except Exception as error:  # noqa: BLE001 - recorded, not fatal
                    failed += 1
                    print(f"  FAILED {example_id}: {error}", flush=True)
                    continue

                ptm = np.asarray(bundle["ptm"][:, : args.ptm_dim], dtype=np.float16)
                mfcc = np.asarray(bundle["mfcc"], dtype=np.float16)
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez(target, ptm=ptm, mfcc=mfcc)
                manifest.write(
                    json.dumps(
                        {
                            "example_id": example_id,
                            "audio": str(wav),
                            "feature_path": str(target),
                            "frame_count": int(ptm.shape[0]),
                            "ptm_dim": int(ptm.shape[1]),
                            "mfcc_dim": int(mfcc.shape[1]),
                            "frame_hop_s": float(args.frame_hop_s),
                            "dtype": "float16",
                            "coverage": float(bundle["feature_coverage_ratio"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                manifest.flush()
                written += 1
                if written % 25 == 0:
                    rate = written / max(1e-6, time.time() - started)
                    remaining = (len(targets) - position) / max(rate, 1e-6)
                    # Print the memory BEFORE checking it, so a trip leaves the
                    # trend visible in the log instead of just a final number.
                    print(
                        f"  {position}/{len(targets)}  {rate:.2f} win/s  "
                        f"eta {remaining / 60:.1f} min  "
                        f"host {guard.process_bytes() / 2**30:.2f}/"
                        f"{guard.budget_bytes / 2**30:.2f} GiB",
                        flush=True,
                    )
                    guard.check()
    finally:
        extractor.close()
        guard.stop()

    summary = {
        "schema": "real_window_full_ptm_v1",
        "output": str(output_root),
        "manifest": str(manifest_path),
        "targets": len(targets),
        "written": written,
        "failed": failed,
        "skipped_resume": len(done),
        "ptm_dim": int(args.ptm_dim),
        "dtype_stored": "float16",
        "extraction_dtype": args.dtype,
        "window_s": args.window_s,
        "overlap_s": args.overlap_s,
        "vram_ratio": vram_ratio,
        "host_peak_gib": round(guard.peak_bytes / 2**30, 3),
        "seconds": round(time.time() - started, 1),
    }
    (output_root / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ptm", default="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf")
    parser.add_argument(
        "--model-path", default="models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    )
    parser.add_argument("--ptm-dim", type=int, default=2048)
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    export(parse_args())
