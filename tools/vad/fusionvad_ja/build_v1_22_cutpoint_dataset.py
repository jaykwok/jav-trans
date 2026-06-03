#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.vad.fusionvad_ja import build_galgame_synthetic_timeline as timeline  # noqa: E402


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "datasets"
    / "train"
    / "fusionvad-ja"
    / "v1-19"
    / "galgame-synthetic-timeline-v8-speaker-random-touch4096-train"
    / "manifest.json"
)
DEFAULT_NEGATIVE_MANIFEST = (
    PROJECT_ROOT
    / "datasets"
    / "train"
    / "fusionvad-ja"
    / "v1-negative"
    / "musan-clips-20s"
    / "clip_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "train"
    / "fusionvad-ja"
    / "v1-22"
    / "galgame-cutpoint-supervised"
)


def _summary_value(path: Path, key: str, default: Any = None) -> Any:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return payload.get(key, default) if isinstance(payload, dict) else default


def build_timeline_args(args: argparse.Namespace) -> list[str]:
    output_dir = Path(args.output_dir)
    manifest = Path(args.manifest)
    values = [
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--count",
        str(args.count),
        "--seed",
        str(args.seed),
        "--speech-clips-per-example",
        str(args.speech_clips_per_example),
        "--source",
        args.source,
        "--audio-id-prefix",
        args.audio_id_prefix,
        "--frame-hop-s",
        str(args.frame_hop_s),
        "--reuse-sources",
        "--shuffle",
        "--randomize-speech-order",
        "--speaker-proxy-mode",
        args.speaker_proxy_mode,
        "--speaker-proxy-retry-count",
        str(args.speaker_proxy_retry_count),
        "--max-speech-s",
        str(args.max_speech_s),
        "--min-speech-s",
        str(args.min_speech_s),
        "--trim-head-s",
        str(args.trim_head_s),
        "--trim-tail-s",
        str(args.trim_tail_s),
        "--touch-gap-prob",
        str(args.touch_gap_prob),
        "--short-gap-prob",
        str(args.short_gap_prob),
        "--short-gap-max-s",
        str(args.short_gap_max_s),
        "--gap-min-s",
        str(args.gap_min_s),
        "--gap-max-s",
        str(args.gap_max_s),
        "--cut-point-max-gap-s",
        str(args.cut_point_max_gap_s),
        "--cut-drop-min-gap-s",
        str(args.cut_drop_min_gap_s),
        "--leading-gap-min-s",
        str(args.leading_gap_min_s),
        "--leading-gap-max-s",
        str(args.leading_gap_max_s),
        "--trailing-gap-min-s",
        str(args.trailing_gap_min_s),
        "--trailing-gap-max-s",
        str(args.trailing_gap_max_s),
        "--speech-label-pad-s",
        str(args.speech_label_pad_s),
        "--crossfade-ms-min",
        str(args.crossfade_ms_min),
        "--crossfade-ms-max",
        str(args.crossfade_ms_max),
        "--crossfade-curve",
        args.crossfade_curve,
        "--noise-rms",
        str(args.noise_rms),
        "--hum-rms",
        str(args.hum_rms),
        "--negative-gap-prob",
        str(args.negative_gap_prob),
        "--background-mix-prob",
        str(args.background_mix_prob),
        "--background-snr-db-min",
        str(args.background_snr_db_min),
        "--background-snr-db-max",
        str(args.background_snr_db_max),
        "--gain-db-min",
        str(args.gain_db_min),
        "--gain-db-max",
        str(args.gain_db_max),
        "--filter-prob",
        str(args.filter_prob),
        "--filter-mode",
        args.filter_mode,
        "--codec-prob",
        str(args.codec_prob),
        "--codec-aug",
        args.codec_aug,
        "--overlap-speech-prob",
        str(args.overlap_speech_prob),
        "--overlap-snr-db-min",
        str(args.overlap_snr_db_min),
        "--overlap-snr-db-max",
        str(args.overlap_snr_db_max),
        "--overlap-max-speech-s",
        str(args.overlap_max_speech_s),
    ]
    if args.negative_manifest:
        for item in args.negative_manifest:
            values.extend(["--negative-manifest", str(Path(item))])
    if args.background_manifest:
        for item in args.background_manifest:
            values.extend(["--background-manifest", str(Path(item))])
    if args.limit is not None:
        values.extend(["--limit", str(args.limit)])
    return values


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_args = build_timeline_args(args)
    parsed = timeline.parse_args(timeline_args)
    timeline.build_synthetic_timeline(parsed)

    summary_path = output_dir / "synthetic_timeline_summary.json"
    wrapper_summary = {
        "version": "v1.22",
        "purpose": "supervised cutpoint/boundary dataset for fallback-safe speech-island packing",
        "timeline_summary": str(summary_path),
        "labels": str(output_dir / "labels.jsonl"),
        "manifest": str(output_dir / "manifest.json"),
        "boundary_manifest": str(output_dir / "boundary_manifest.jsonl"),
        "records": _summary_value(summary_path, "records", 0),
        "duration_s_total": _summary_value(summary_path, "duration_s_total", 0.0),
        "cut_point_segment_count": _summary_value(summary_path, "cut_point_segment_count", 0),
        "cut_drop_zone_count": _summary_value(summary_path, "cut_drop_zone_count", 0),
        "speaker_turn_boundary_count": _summary_value(summary_path, "speaker_turn_boundary_count", 0),
        "internal_gap_policy_counts": _summary_value(summary_path, "internal_gap_policy_counts", {}),
        "config": vars(args),
    }
    wrapper_summary_path = output_dir / "v1_22_cutpoint_dataset_summary.json"
    wrapper_summary_path.write_text(
        json.dumps(wrapper_summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"v1_22_summary={wrapper_summary_path}")
    print(
        "records={records} cut_points={cut_points} cut_drops={cut_drops} gap_policy={gap_policy}".format(
            records=wrapper_summary["records"],
            cut_points=wrapper_summary["cut_point_segment_count"],
            cut_drops=wrapper_summary["cut_drop_zone_count"],
            gap_policy=wrapper_summary["internal_gap_policy_counts"],
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v1.22 supervised cutpoint dataset. This is a preset wrapper "
            "around build_galgame_synthetic_timeline.py focused on touching/short-gap "
            "speech-island boundaries."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--negative-manifest", action="append", default=[str(DEFAULT_NEGATIVE_MANIFEST)])
    parser.add_argument("--background-manifest", action="append", default=[str(DEFAULT_NEGATIVE_MANIFEST)])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--source", default="galgame-cutpoint-supervised-v1-22")
    parser.add_argument("--audio-id-prefix", default="galgame-v122-cut")
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--speech-clips-per-example", type=int, default=5)
    parser.add_argument("--speaker-proxy-mode", choices=("none", "auto", "audio_id", "character", "voice_actor"), default="auto")
    parser.add_argument("--speaker-proxy-retry-count", type=int, default=16)
    parser.add_argument("--max-speech-s", type=float, default=7.0)
    parser.add_argument("--min-speech-s", type=float, default=0.05)
    parser.add_argument("--trim-head-s", type=float, default=0.0)
    parser.add_argument("--trim-tail-s", type=float, default=0.0)
    parser.add_argument("--touch-gap-prob", type=float, default=0.30)
    parser.add_argument("--short-gap-prob", type=float, default=0.45)
    parser.add_argument("--short-gap-max-s", type=float, default=0.35)
    parser.add_argument("--gap-min-s", type=float, default=0.60)
    parser.add_argument("--gap-max-s", type=float, default=3.50)
    parser.add_argument("--cut-point-max-gap-s", type=float, default=0.35)
    parser.add_argument("--cut-drop-min-gap-s", type=float, default=0.60)
    parser.add_argument("--leading-gap-min-s", type=float, default=0.2)
    parser.add_argument("--leading-gap-max-s", type=float, default=2.0)
    parser.add_argument("--trailing-gap-min-s", type=float, default=0.2)
    parser.add_argument("--trailing-gap-max-s", type=float, default=2.0)
    parser.add_argument("--speech-label-pad-s", type=float, default=0.04)
    parser.add_argument("--crossfade-ms-min", type=float, default=5.0)
    parser.add_argument("--crossfade-ms-max", type=float, default=25.0)
    parser.add_argument("--crossfade-curve", choices=("equal_power", "linear"), default="equal_power")
    parser.add_argument("--noise-rms", type=float, default=0.01)
    parser.add_argument("--hum-rms", type=float, default=0.02)
    parser.add_argument("--negative-gap-prob", type=float, default=0.60)
    parser.add_argument("--background-mix-prob", type=float, default=0.40)
    parser.add_argument("--background-snr-db-min", type=float, default=8.0)
    parser.add_argument("--background-snr-db-max", type=float, default=22.0)
    parser.add_argument("--gain-db-min", type=float, default=-3.0)
    parser.add_argument("--gain-db-max", type=float, default=3.0)
    parser.add_argument("--filter-prob", type=float, default=0.20)
    parser.add_argument("--filter-mode", choices=("none", "random", "lowpass", "bandpass"), default="random")
    parser.add_argument("--codec-prob", type=float, default=0.03)
    parser.add_argument("--codec-aug", choices=("none", "random", "pcm16", "mulaw"), default="random")
    parser.add_argument("--overlap-speech-prob", type=float, default=0.08)
    parser.add_argument("--overlap-snr-db-min", type=float, default=0.0)
    parser.add_argument("--overlap-snr-db-max", type=float, default=10.0)
    parser.add_argument("--overlap-max-speech-s", type=float, default=2.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
