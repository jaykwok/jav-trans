#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402
from tools.boundary.ja.build_galgame_synthetic_timeline import (  # noqa: E402
    crop_or_tile_audio,
)


SCHEMA = "semantic_split_multicore_composite_smoke_v1"
SUMMARY_SCHEMA = "semantic_split_multicore_composite_smoke_summary_v1"
SAMPLE_RATE = 16000


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_span(path: Path, start_s: float, end_s: float) -> tuple[np.ndarray, int, int]:
    audio, sample_rate = load_audio_16k_mono(str(path))
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected 16 kHz normalized audio: {path}")
    start = max(0, int(round(start_s * sample_rate)))
    end = min(len(audio), int(round(end_s * sample_rate)))
    if end <= start:
        raise ValueError(f"empty semantic span: {path} {start_s:.3f}-{end_s:.3f}")
    return np.ascontiguousarray(audio[start:end], dtype=np.float32), start, end


def load_semantic_cores(path: Path) -> list[dict[str, Any]]:
    cores: list[dict[str, Any]] = []
    for source in _rows(path):
        units = {str(unit["unit_id"]): unit for unit in source["text_units"]}
        for alignment in source["semantic_alignments"]:
            if alignment["status"] != "matched":
                continue
            unit_id = str(alignment["unit_id"])
            unit = units[unit_id]
            audio, source_start_sample, source_end_sample = _load_span(
                Path(source["audio"]),
                float(alignment["start_s"]),
                float(alignment["end_s"]),
            )
            cores.append(
                {
                    "core_id": f"{source['sample_id']}:{unit_id}",
                    "source_sample_id": str(source["sample_id"]),
                    "source_audio": str(source["audio"]),
                    "source_start_s": source_start_sample / SAMPLE_RATE,
                    "source_end_s": source_end_sample / SAMPLE_RATE,
                    "teacher_start_s": float(alignment["start_s"]),
                    "teacher_end_s": float(alignment["end_s"]),
                    "source_start_sample": source_start_sample,
                    "source_end_sample": source_end_sample,
                    "text": str(unit["text"]),
                    "audio": audio,
                }
            )
    cores.sort(key=lambda row: row["core_id"])
    if len(cores) < 4:
        raise ValueError("multi-core smoke requires at least four matched semantic cores")
    return cores


def load_gap_quantiles(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    values = np.asarray(payload.get("durations_s") or [], dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise ValueError("gap duration pool is empty")
    return {
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "source": str(path),
        "count": int(values.size),
    }


def _negative_matches(row: dict[str, Any], flags: set[str]) -> bool:
    values = {
        str(row.get("background_type") or "").lower(),
        *(str(value).lower() for value in row.get("omni_flags") or []),
    }
    return any(any(flag in value for value in values) for flag in flags)


def select_negative_rows(
    path: Path, *, min_duration_s: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        row
        for row in _rows(path)
        if Path(str(row.get("audio") or "")).exists()
        and str(row.get("source_partition") or "train") == "train"
        and float(row.get("duration_s") or 0.0) >= float(min_duration_s)
    ]
    music = sorted(
        (row for row in rows if _negative_matches(row, {"music", "bgm"})),
        key=lambda row: str(row.get("audio_id") or row["audio"]),
    )
    vocal = sorted(
        (
            row
            for row in rows
            if _negative_matches(row, {"breathing", "moaning", "kissing"})
        ),
        key=lambda row: str(row.get("audio_id") or row["audio"]),
    )
    if len(music) < 2 or not vocal:
        raise ValueError("negative manifest needs two music rows and one vocal row")
    return [music[0], music[1], vocal[0]], {
        "manifest": str(path),
        "music_audio_ids": [str(music[0]["audio_id"]), str(music[1]["audio_id"])],
        "vocal_audio_id": str(vocal[0]["audio_id"]),
    }


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64, copy=False)))))


def _scale_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    current = _rms(audio)
    if current <= 1e-8 or target_rms <= 0.0:
        return audio.astype(np.float32, copy=False)
    return (audio * (target_rms / current)).astype(np.float32, copy=False)


def _limit(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _negative_audio(
    row: dict[str, Any], *, duration_s: float, rng: np.random.Generator
) -> np.ndarray:
    audio, sample_rate = load_audio_16k_mono(str(row["audio"]))
    if sample_rate != SAMPLE_RATE:
        raise ValueError("negative audio must normalize to 16 kHz")
    values, _offset = crop_or_tile_audio(
        audio,
        samples=max(1, int(round(duration_s * sample_rate))),
        rng=rng,
    )
    return values


def _core_copy(core: dict[str, Any], *, start_sample: int, end_sample: int) -> dict[str, Any]:
    return {
        "core_id": core["core_id"],
        "text": core["text"],
        "start_sample": start_sample,
        "end_sample": end_sample,
        "start_s": start_sample / SAMPLE_RATE,
        "end_s": end_sample / SAMPLE_RATE,
        "source_audio": core["source_audio"],
        "core_library_audio": core.get("core_library_audio"),
        "source_start_sample": core["source_start_sample"],
        "source_end_sample": core["source_end_sample"],
        "source_start_s": core["source_start_s"],
        "source_end_s": core["source_end_s"],
    }


def _safe_composite(
    *,
    sample_id: str,
    cores: list[dict[str, Any]],
    gaps: list[tuple[str, np.ndarray, dict[str, Any]]],
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(gaps) != len(cores) - 1:
        raise ValueError("safe composite needs one gap between adjacent cores")
    parts: list[np.ndarray] = []
    core_spans: list[dict[str, Any]] = []
    gap_spans: list[dict[str, Any]] = []
    cursor = 0
    for index, core in enumerate(cores):
        values = np.asarray(core["audio"], dtype=np.float32)
        start = cursor
        cursor += values.size
        core_spans.append(_core_copy(core, start_sample=start, end_sample=cursor))
        parts.append(values)
        if index >= len(gaps):
            continue
        kind, gap, detail = gaps[index]
        gap_start = cursor
        cursor += gap.size
        gap_spans.append(
            {
                "gap_id": f"{sample_id}:g{index:02d}",
                "kind": kind,
                "start_sample": gap_start,
                "end_sample": cursor,
                "start_s": gap_start / SAMPLE_RATE,
                "end_s": cursor / SAMPLE_RATE,
                "source": detail,
            }
        )
        parts.append(gap)
    audio = _limit(np.concatenate(parts))
    events = []
    for index, (left, right, gap) in enumerate(zip(core_spans, core_spans[1:], gap_spans)):
        events.append(
            {
                "event_id": f"{sample_id}:e{index:02d}",
                "left_core_id": left["core_id"],
                "right_core_id": right["core_id"],
                "semantic_decision": "cut",
                "representative_sample": (int(left["end_sample"]) + int(right["start_sample"])) // 2,
                "representative_s": ((int(left["end_sample"]) + int(right["start_sample"])) // 2) / SAMPLE_RATE,
                "event_interval_start_sample": int(left["end_sample"]),
                "event_interval_end_sample": int(right["start_sample"]),
                "event_interval_start_s": float(left["end_s"]),
                "event_interval_end_s": float(right["start_s"]),
                "inner_target": {
                    "status": "safe",
                    "left_speech_end_sample": int(left["end_sample"]),
                    "right_speech_start_sample": int(right["start_sample"]),
                    "left_speech_end_s": float(left["end_s"]),
                    "right_speech_start_s": float(right["start_s"]),
                    "removed_gap_start_sample": int(gap["start_sample"]),
                    "removed_gap_end_sample": int(gap["end_sample"]),
                    "removed_gap_start_s": float(gap["start_s"]),
                    "removed_gap_end_s": float(gap["end_s"]),
                    "gap_kind": gap["kind"],
                    "reason": "独立 semantic cores 之间存在已知可移除的非语义区间。",
                },
            }
        )
    return audio, {
        "core_spans": core_spans,
        "gap_spans": gap_spans,
        "semantic_events": events,
        "continue_control": False,
    }


def _overlap_composite(
    *, sample_id: str, left: dict[str, Any], right: dict[str, Any], overlap_s: float
) -> tuple[np.ndarray, dict[str, Any]]:
    left_audio = np.asarray(left["audio"], dtype=np.float32)
    right_audio = np.asarray(right["audio"], dtype=np.float32)
    overlap = min(
        max(1, int(round(overlap_s * SAMPLE_RATE))),
        left_audio.size - 1,
        right_audio.size - 1,
    )
    phase = np.linspace(0.0, np.pi / 2.0, overlap, dtype=np.float32)
    mixed = left_audio[-overlap:] * np.cos(phase) + right_audio[:overlap] * np.sin(phase)
    audio = _limit(np.concatenate((left_audio[:-overlap], mixed, right_audio[overlap:])))
    left_span = _core_copy(left, start_sample=0, end_sample=left_audio.size)
    right_start = left_audio.size - overlap
    right_span = _core_copy(
        right,
        start_sample=right_start,
        end_sample=right_start + right_audio.size,
    )
    event = {
        "event_id": f"{sample_id}:e00",
        "left_core_id": left_span["core_id"],
        "right_core_id": right_span["core_id"],
        "semantic_decision": "cut",
        "representative_sample": (int(left_span["end_sample"]) + int(right_span["start_sample"])) // 2,
        "representative_s": ((int(left_span["end_sample"]) + int(right_span["start_sample"])) // 2) / SAMPLE_RATE,
        "event_interval_start_sample": int(right_span["start_sample"]),
        "event_interval_end_sample": int(left_span["end_sample"]),
        "event_interval_start_s": right_span["start_s"],
        "event_interval_end_s": left_span["end_s"],
        "inner_target": {
            "status": "abstain",
            "left_speech_end_sample": None,
            "right_speech_start_sample": None,
            "left_speech_end_s": None,
            "right_speech_start_s": None,
            "removed_gap_start_sample": None,
            "removed_gap_end_sample": None,
            "removed_gap_start_s": None,
            "removed_gap_end_s": None,
            "gap_kind": "overlap",
            "reason": "独立 semantic cores 发生重叠，没有连续安全区，不得强制切。",
        },
    }
    return audio, {
        "core_spans": [left_span, right_span],
        "gap_spans": [
            {
                "gap_id": f"{sample_id}:g00",
                "kind": "overlap",
                "start_sample": int(right_span["start_sample"]),
                "end_sample": int(left_span["end_sample"]),
                "start_s": right_span["start_s"],
                "end_s": left_span["end_s"],
                "source": {"overlap_s": overlap / SAMPLE_RATE},
            }
        ],
        "semantic_events": [event],
        "continue_control": False,
    }


def _background_switch(
    signal: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    crossfade_s: float,
    snr_db: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    size = signal.size
    first = np.resize(first, size).astype(np.float32, copy=False)
    second = np.resize(second, size).astype(np.float32, copy=False)
    midpoint = size // 2
    fade = min(max(1, int(round(crossfade_s * SAMPLE_RATE))), midpoint, size - midpoint)
    start = midpoint - fade // 2
    end = start + fade
    bed = first.copy()
    phase = np.linspace(0.0, np.pi / 2.0, fade, dtype=np.float32)
    bed[start:end] = first[start:end] * np.cos(phase) + second[start:end] * np.sin(phase)
    bed[end:] = second[end:]
    target = _rms(signal) / (10.0 ** (snr_db / 20.0))
    bed = _scale_rms(bed, target)
    return _limit(signal + bed), {
        "switch_s": midpoint / SAMPLE_RATE,
        "crossfade_s": fade / SAMPLE_RATE,
        "snr_db": snr_db,
    }


def build_smoke(
    *,
    semantic_labels: Path,
    negative_manifest: Path,
    gap_duration_pool: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cores = load_semantic_cores(semantic_labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    core_dir = output_dir / "semantic_cores"
    core_dir.mkdir(parents=True, exist_ok=True)
    core_library_rows: list[dict[str, Any]] = []
    for core in cores:
        filename = core["core_id"].replace(":", "__") + ".wav"
        core_path = core_dir / filename
        sf.write(str(core_path), core["audio"], SAMPLE_RATE, subtype="PCM_16")
        core["core_library_audio"] = str(core_path)
        core_library_rows.append(
            {
                "schema": "semantic_core_seed_library_v1",
                "core_id": core["core_id"],
                "audio": str(core_path),
                "sample_rate": SAMPLE_RATE,
                "sample_count": int(len(core["audio"])),
                "duration_s": len(core["audio"]) / SAMPLE_RATE,
                "text": core["text"],
                "source_audio": core["source_audio"],
                "source_start_sample": core["source_start_sample"],
                "source_end_sample": core["source_end_sample"],
                "source_start_s": core["source_start_s"],
                "source_end_s": core["source_end_s"],
                "teacher_start_s": core["teacher_start_s"],
                "teacher_end_s": core["teacher_end_s"],
            }
        )
    core_library_path = output_dir / "semantic_core_library.jsonl"
    _write_jsonl(core_library_path, core_library_rows)
    quantiles = load_gap_quantiles(gap_duration_pool)
    negative_rows, negative_selection = select_negative_rows(
        negative_manifest,
        min_duration_s=quantiles["q50"],
    )
    music_a, music_b, vocal = negative_rows
    core_rms = float(np.median([_rms(core["audio"]) for core in cores]))
    longest_index = max(range(len(cores)), key=lambda index: len(cores[index]["audio"]))
    long_core = cores[longest_index]
    remaining = [core for index, core in enumerate(cores) if index != longest_index]
    shortest_index = min(
        range(len(remaining)), key=lambda index: len(remaining[index]["audio"])
    )
    short_core = remaining[shortest_index]
    medium_cores = [
        core for index, core in enumerate(remaining) if index != shortest_index
    ]
    medium_left, medium_right = medium_cores[:2]

    silence = rng.normal(
        0.0,
        max(core_rms * 0.003, 1e-6),
        max(1, int(round(quantiles["q25"] * SAMPLE_RATE))),
    ).astype(np.float32)
    short_room = rng.normal(
        0.0,
        max(core_rms * 0.006, 1e-6),
        max(1, int(round(quantiles["q25"] * SAMPLE_RATE))),
    ).astype(np.float32)
    music_gap = _scale_rms(
        _negative_audio(music_a, duration_s=quantiles["q50"], rng=rng),
        core_rms * 0.35,
    )
    vocal_gap = _scale_rms(
        _negative_audio(vocal, duration_s=quantiles["q50"], rng=rng),
        core_rms * 0.65,
    )

    recipes: list[tuple[str, str, np.ndarray, dict[str, Any], dict[str, Any]]] = []
    audio, truth = _safe_composite(
        sample_id="mc01_silence_safe",
        cores=[medium_left, medium_right],
        gaps=[("low_room_tone", silence, {"duration_quantile": "q25"})],
    )
    recipes.append(
        (
            "mc01_silence_safe",
            "两个独立 semantic cores，中间为低 room tone；Split 应 cut，Inner 应给连续 safe gap。",
            audio,
            truth,
            {"core_count": 2, "gap_axis": "low_room_tone", "speaker_axis": "cross_source"},
        )
    )

    audio, truth = _safe_composite(
        sample_id="mc02_music_gap_three_core",
        cores=[short_core, medium_left, medium_right],
        gaps=[
            (
                "music",
                music_gap,
                {
                    "duration_quantile": "q50",
                    "audio_id": music_a["audio_id"],
                    "source_audio": music_a["audio"],
                },
            ),
            ("low_room_tone", short_room, {"duration_quantile": "q25"}),
        ],
    )
    recipes.append(
        (
            "mc02_music_gap_three_core",
            "三个独立 cores，分别由真实 music gap 与低 room tone 分隔；验证多 event 与两段 safe gap。",
            audio,
            truth,
            {"core_count": 3, "gap_axis": "music+room", "speaker_axis": "cross_source"},
        )
    )

    audio, truth = _safe_composite(
        sample_id="mc03_nonsemantic_vocal_gap",
        cores=[medium_right, short_core],
        gaps=[
            (
                "nonsemantic_vocal",
                vocal_gap,
                {
                    "duration_quantile": "q50",
                    "audio_id": vocal["audio_id"],
                    "background_type": vocal.get("background_type"),
                    "source_audio": vocal["audio"],
                },
            )
        ],
    )
    recipes.append(
        (
            "mc03_nonsemantic_vocal_gap",
            "两个独立 cores，中间为真实 definite-drop 呼吸/呻吟；按产品语义应移除，但必须人工确认无清楚词语。",
            audio,
            truth,
            {"core_count": 2, "gap_axis": "nonsemantic_vocal", "speaker_axis": "cross_source"},
        )
    )

    audio, truth = _overlap_composite(
        sample_id="mc04_overlap_abstain",
        left=medium_left,
        right=medium_right,
        overlap_s=0.18,
    )
    recipes.append(
        (
            "mc04_overlap_abstain",
            "两个独立 cores 发生 equal-power overlap；Split 语义上应 cut，但 Inner 必须 abstain 并保持单 chunk。",
            audio,
            truth,
            {"core_count": 2, "gap_axis": "overlap", "speaker_axis": "cross_source"},
        )
    )

    single = np.asarray(long_core["audio"], dtype=np.float32)
    bg_a = _negative_audio(music_a, duration_s=len(single) / SAMPLE_RATE, rng=rng)
    bg_b = _negative_audio(music_b, duration_s=len(single) / SAMPLE_RATE, rng=rng)
    audio, bg_detail = _background_switch(
        single,
        bg_a,
        bg_b,
        crossfade_s=quantiles["q25"],
        snr_db=12.0,
    )
    truth = {
        "core_spans": [_core_copy(long_core, start_sample=0, end_sample=len(single))],
        "gap_spans": [],
        "semantic_events": [],
        "continue_control": True,
        "background_switch": {
            **bg_detail,
            "first_audio_id": music_a["audio_id"],
            "second_audio_id": music_b["audio_id"],
        },
    }
    recipes.append(
        (
            "mc05_single_core_bgm_switch_continue",
            "单一 maximal semantic core 内连续 BGM 发生 crossfade 切换；所有候选均应 continue，不得把背景变化当语义边界。",
            audio,
            truth,
            {"core_count": 1, "gap_axis": "bgm_switch_inside_core", "speaker_axis": "single_source"},
        )
    )

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for sample_id, focus, audio, truth, axes in recipes:
        audio_path = audio_dir / f"{sample_id}.wav"
        sf.write(str(audio_path), audio, SAMPLE_RATE, subtype="PCM_16")
        manifest_rows.append(
            {
                "schema": SCHEMA,
                "sample_id": sample_id,
                "audio": str(audio_path),
                "sample_rate": SAMPLE_RATE,
                "sample_count": int(len(audio)),
                "duration_s": len(audio) / SAMPLE_RATE,
                "audit_focus": focus,
                "sampling_axes": axes,
                "semantic_core_source": str(semantic_labels),
                "negative_manifest": str(negative_manifest),
                **truth,
            }
        )
    manifest_path = output_dir / "multicore_composite_smoke.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "seed": int(seed),
        "sample_count": len(manifest_rows),
        "core_count_values": sorted({int(row["sampling_axes"]["core_count"]) for row in manifest_rows}),
        "gap_axes": [str(row["sampling_axes"]["gap_axis"]) for row in manifest_rows],
        "semantic_event_count": sum(len(row["semantic_events"]) for row in manifest_rows),
        "inner_safe_count": sum(
            event["inner_target"]["status"] == "safe"
            for row in manifest_rows
            for event in row["semantic_events"]
        ),
        "inner_abstain_count": sum(
            event["inner_target"]["status"] == "abstain"
            for row in manifest_rows
            for event in row["semantic_events"]
        ),
        "continue_control_count": sum(bool(row["continue_control"]) for row in manifest_rows),
        "gap_duration_quantiles": quantiles,
        "negative_selection": negative_selection,
        "semantic_core_library": str(core_library_path),
        "semantic_core_count": len(core_library_rows),
        "coordinate_contract": "dual_coordinates_float_seconds_model_interface_exact_sample_offsets_v1",
        "training_ready": False,
        "training_blocker": "fixed five-sample manual audit is required before proposer replay or dataset expansion",
        "manifest": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed five-sample semantic Split/Inner multi-core composition smoke."
    )
    parser.add_argument("--semantic-labels", required=True)
    parser.add_argument("--negative-manifest", required=True)
    parser.add_argument("--gap-duration-pool", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build_smoke(
                semantic_labels=Path(args.semantic_labels),
                negative_manifest=Path(args.negative_manifest),
                gap_duration_pool=Path(args.gap_duration_pool),
                output_dir=Path(args.output_dir),
                seed=int(args.seed),
            ),
            ensure_ascii=False,
        )
    )
