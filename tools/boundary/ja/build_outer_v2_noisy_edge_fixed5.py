#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from audio.loading import load_audio_16k_mono  # noqa: E402
from tools.boundary.ja.build_galgame_synthetic_timeline import (  # noqa: E402
    append_timeline_part,
)


SCHEMA = "outer_v2_empirical_noisy_edge_fixed5_v1"
SAMPLE_RATE = 16000
FRAME_HOP_S = 0.02
VOCAL_FLAGS = {
    "breathing",
    "moaning",
    "groaning",
    "groan",
    "kissing",
    "kiss",
    "crying",
    "non_speech_vocalization",
    "non_verbal_vocalization",
}
VOCAL_CATEGORY_FLAGS = (
    {"moaning", "groaning", "groan"},
    {"kissing", "kissing_sound", "kiss", "kiss_sound"},
    {"breathing", "heavy_breathing", "short_breath"},
    {"crying"},
    {"non_speech_vocalization", "non_verbal_vocalization", "non_verbal"},
)
NOISE_FLAGS = {
    "noise",
    "music",
    "bgm",
    "movement",
    "impact_sound",
    "impact",
    "vehicle_noise",
    "environmental_noise",
    "rustling",
    "heavy_machinery",
}
CANONICAL_NEGATIVE_FLAGS = {
    "crying": {"crying"},
    "moaning": {"moaning", "groaning", "groan"},
    "kissing": {"kissing", "kissing_sound", "kiss", "kiss_sound"},
    "breathing": {"breathing", "heavy_breathing", "short_breath"},
    "nonverbal_vocalization": {
        "non_speech_vocalization",
        "non_verbal_vocalization",
        "non_verbal",
        "grunt",
    },
    "music": {"music", "bgm"},
    "noise": NOISE_FLAGS,
    "silence": {"silence", "short_silence"},
}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _flag_values(row: dict[str, Any]) -> set[str]:
    return {
        str(row.get("background_type") or "").lower(),
        *(str(value).lower() for value in row.get("omni_flags") or []),
    }


def _matches(row: dict[str, Any], flags: set[str]) -> bool:
    values = _flag_values(row)
    return any(any(flag in value for value in values) for flag in flags)


def canonical_negative_categories(row: dict[str, Any]) -> tuple[str, ...]:
    categories = tuple(
        category
        for category, flags in CANONICAL_NEGATIVE_FLAGS.items()
        if _matches(row, flags)
    )
    return categories or ("other",)


def select_empirical_edge_assets(
    rows: list[dict[str, Any]], *, count_per_kind: int = 5
) -> dict[str, list[dict[str, Any]]]:
    if count_per_kind != len(VOCAL_CATEGORY_FLAGS):
        raise ValueError("noisy-edge fixed-5 requires one asset per vocal category")
    valid = [
        row
        for row in rows
        if str(row.get("source") or "") == "omni_definite_drop"
        and str(row.get("source_partition") or "train") == "train"
        and Path(str(row.get("audio") or "")).is_file()
        and float(row.get("duration_s") or 0.0) > 0.0
    ]
    noise = [
        row
        for row in valid
        if _matches(row, NOISE_FLAGS) and not _matches(row, VOCAL_FLAGS)
    ]

    def quantile_pick(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(
            values,
            key=lambda row: (
                float(row["duration_s"]),
                str(row.get("audio_id") or row["audio"]),
            ),
        )
        if len(ordered) < count_per_kind:
            raise ValueError(
                f"not enough definite-drop assets: {len(ordered)} < {count_per_kind}"
            )
        quantiles = np.linspace(0.25, 0.85, count_per_kind)
        indexes = np.rint(quantiles * (len(ordered) - 1)).astype(np.int64)
        selected = [ordered[int(index)] for index in indexes]
        if len({str(row["audio_id"]) for row in selected}) != count_per_kind:
            raise RuntimeError("duration-quantile edge asset selection produced duplicates")
        return selected

    vocal: list[dict[str, Any]] = []
    used_vocal_ids: set[str] = set()
    vocal_quantiles = np.linspace(0.35, 0.85, count_per_kind)
    for flags, quantile in zip(VOCAL_CATEGORY_FLAGS, vocal_quantiles, strict=True):
        candidates = sorted(
            (
                row
                for row in valid
                if _matches(row, flags)
                and str(row["audio_id"]) not in used_vocal_ids
            ),
            key=lambda row: (
                float(row["duration_s"]),
                str(row.get("audio_id") or row["audio"]),
            ),
        )
        if not candidates:
            raise ValueError(f"definite-drop manifest has no vocal asset for {sorted(flags)}")
        selected_index = int(round(float(quantile) * (len(candidates) - 1)))
        selected_row = candidates[selected_index]
        vocal.append(selected_row)
        used_vocal_ids.add(str(selected_row["audio_id"]))
    selected = {"vocal": vocal, "noise": quantile_pick(noise)}
    all_ids = [
        str(row["audio_id"])
        for values in selected.values()
        for row in values
    ]
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("edge assets must be unique across kinds")
    return selected


def _load(path: str | Path) -> np.ndarray:
    audio, sample_rate = load_audio_16k_mono(str(path))
    if sample_rate != SAMPLE_RATE or not len(audio):
        raise ValueError(f"invalid 16 kHz edge audio: {path}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def _asset_metadata(row: dict[str, Any], *, side: str, kind: str) -> dict[str, Any]:
    return {
        "side": side,
        "kind": kind,
        "audio_id": str(row["audio_id"]),
        "audio": str(row["audio"]),
        "background_type": str(row.get("background_type") or ""),
        "omni_flags": [str(value) for value in row.get("omni_flags") or []],
        "duration_s": float(row["duration_s"]),
        "label_source": str(row.get("label_source") or ""),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    semantic_rows = _rows(Path(args.semantic_timeline_labels))
    if len(semantic_rows) != 5:
        raise ValueError(f"noisy-edge fixed audit requires exactly 5 semantic sources; got {len(semantic_rows)}")
    sample_ids = [str(row["sample_id"]) for row in semantic_rows]
    if len(set(sample_ids)) != 5:
        raise ValueError("semantic sources must be unique")
    assets = select_empirical_edge_assets(_rows(Path(args.negative_manifest)))
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    timeline_rows: list[dict[str, Any]] = []
    feature_label_rows: list[dict[str, Any]] = []
    used_assets: list[str] = []
    for index, semantic in enumerate(semantic_rows):
        leading_kind = "vocal" if index % 2 == 0 else "noise"
        trailing_kind = "noise" if index % 2 == 0 else "vocal"
        leading_row = assets[leading_kind][index]
        trailing_row = assets[trailing_kind][index]
        used_assets.extend([str(leading_row["audio_id"]), str(trailing_row["audio_id"])])
        leading = _load(leading_row["audio"])
        core = _load(semantic["audio"])
        trailing = _load(trailing_row["audio"])
        composite = np.zeros(0, dtype=np.float32)
        transitions: list[dict[str, Any]] = []
        composite, _leading_start, _leading_end, transition = append_timeline_part(
            composite,
            leading,
            kind="edge_negative",
            previous_kind=None,
            rng=rng,
            sample_rate=SAMPLE_RATE,
            crossfade_ms_min=args.crossfade_ms_min,
            crossfade_ms_max=args.crossfade_ms_max,
            crossfade_curve="equal_power",
        )
        if transition:
            transitions.append(transition)
        composite, core_start, core_end, transition = append_timeline_part(
            composite,
            core,
            kind="semantic_core",
            previous_kind="edge_negative",
            rng=rng,
            sample_rate=SAMPLE_RATE,
            crossfade_ms_min=args.crossfade_ms_min,
            crossfade_ms_max=args.crossfade_ms_max,
            crossfade_curve="equal_power",
        )
        if transition:
            transitions.append(transition)
        composite, _trailing_start, _trailing_end, transition = append_timeline_part(
            composite,
            trailing,
            kind="edge_negative",
            previous_kind="semantic_core",
            rng=rng,
            sample_rate=SAMPLE_RATE,
            crossfade_ms_min=args.crossfade_ms_min,
            crossfade_ms_max=args.crossfade_ms_max,
            crossfade_curve="equal_power",
        )
        if transition:
            transitions.append(transition)
        audio_id = f"outerv2-edge-noise-{index + 1:02d}"
        audio_path = audio_dir / f"{audio_id}.wav"
        sf.write(str(audio_path), composite, SAMPLE_RATE, subtype="PCM_16")
        duration_s = len(composite) / SAMPLE_RATE
        core_start_s = core_start / SAMPLE_RATE
        core_end_s = core_end / SAMPLE_RATE
        edge_noise = {
            "contract": "empirical_definite_drop_edge_pair_v1",
            "leading": _asset_metadata(
                leading_row, side="leading", kind=leading_kind
            ),
            "trailing": _asset_metadata(
                trailing_row, side="trailing", kind=trailing_kind
            ),
            "transitions": transitions,
        }
        timeline_rows.append(
            {
                "schema": SCHEMA,
                "sample_id": audio_id,
                "source_sample_id": str(semantic["sample_id"]),
                "audio": str(audio_path),
                "duration_s": duration_s,
                "reference_text": str(semantic.get("reference_text") or ""),
                "semantic_alignments": [
                    {
                        "unit_id": "semantic_core",
                        "status": "matched",
                        "start_s": core_start_s,
                        "end_s": core_end_s,
                    }
                ],
                "semantic_events": [],
                "semantic_core_span": {
                    "start_s": core_start_s,
                    "end_s": core_end_s,
                },
                "edge_noise": edge_noise,
                "audio_contract": "confirmed_semantic_core_with_empirical_edge_negatives_v1",
            }
        )
        frame_count = int(math.ceil((duration_s / FRAME_HOP_S) - 1e-9))
        speech_frames = [
            int(core_start_s <= (frame + 0.5) * FRAME_HOP_S < core_end_s)
            for frame in range(frame_count)
        ]
        feature_label_rows.append(
            {
                "audio_id": audio_id,
                "source": SCHEMA,
                "duration_s": duration_s,
                "frame_hop_s": FRAME_HOP_S,
                "label_quality": "supervised",
                "speech_frames": speech_frames,
                "text": str(semantic.get("reference_text") or ""),
            }
        )
    if len(set(used_assets)) != 10:
        raise RuntimeError("each definite-drop edge asset must be used once")
    timeline_path = output_dir / "timeline_labels.jsonl"
    feature_labels_path = output_dir / "feature_labels.jsonl"
    _write_jsonl(timeline_path, timeline_rows)
    _write_jsonl(feature_labels_path, feature_label_rows)
    summary = {
        "schema": SCHEMA,
        "sample_count": len(timeline_rows),
        "semantic_source_count": len(set(sample_ids)),
        "max_semantic_source_use_count": 1,
        "edge_asset_count": len(used_assets),
        "max_edge_asset_use_count": 1,
        "edge_selection": "empirical_vocal_category_stratified_plus_noise_quantiles_v1",
        "crossfade_ms": [args.crossfade_ms_min, args.crossfade_ms_max],
        "timeline_labels": str(timeline_path),
        "feature_labels": str(feature_labels_path),
        "audio_root": str(audio_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Outer v2 empirical noisy-edge fixed-5.")
    parser.add_argument("--semantic-timeline-labels", required=True)
    parser.add_argument("--negative-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--crossfade-ms-min", type=float, default=5.0)
    parser.add_argument("--crossfade-ms-max", type=float, default=30.0)
    args = parser.parse_args()
    if args.crossfade_ms_min < 0.0 or args.crossfade_ms_max < args.crossfade_ms_min:
        parser.error("invalid crossfade range")
    return args


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False))
