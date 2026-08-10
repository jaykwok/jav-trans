"""Observation-only comparison between the production and candidate CTC heads.

The shadow head may measure and persist disagreements, but it never supplies
the words returned to the subtitle pipeline.  Keeping the comparison in this
small module makes that one-way dependency explicit and testable.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any

from asr.subtitle_timing import build_aligned_word_timestamps
from utils.model_paths import PROJECT_ROOT


SHADOW_COMPARISON_SCHEMA = "ctc_alignment_shadow_comparison_v1"
SHADOW_RUN_SCHEMA = "ctc_alignment_shadow_run_v1"
SHADOW_HEAD_PATH_ENV = "ASR_ALIGNMENT_SHADOW_HEAD_PATH"
SHADOW_ROOT_ENV = "ASR_ALIGNMENT_SHADOW_ROOT"
SHADOW_MIN_DELTA_MS_ENV = "ASR_ALIGNMENT_SHADOW_MIN_DELTA_MS"
DEFAULT_SHADOW_ROOT = "./tmp/cache/alignment_shadow"
DEFAULT_MIN_DELTA_MS = 20.0


def shadow_head_reference() -> str:
    return os.getenv(SHADOW_HEAD_PATH_ENV, "").strip()


def shadow_enabled() -> bool:
    return bool(shadow_head_reference())


def shadow_min_delta_ms() -> float:
    try:
        value = float(os.getenv(SHADOW_MIN_DELTA_MS_ENV, str(DEFAULT_MIN_DELTA_MS)))
    except (TypeError, ValueError):
        return DEFAULT_MIN_DELTA_MS
    return max(0.0, value)


def compare_alignment_heads(
    *,
    primary_words: list[dict[str, Any]],
    primary_timing_meta: dict[str, Any],
    shadow_head: Any,
    features: Any,
    text: str,
    window_start: float,
    window_end: float,
) -> dict[str, Any]:
    """Return a boundary-only comparison without mutating ``primary_words``."""

    base: dict[str, Any] = {
        "schema": SHADOW_COMPARISON_SCHEMA,
        "status": "declined",
    }
    if features is None:
        return {**base, "status": "features_unavailable"}
    if not primary_words:
        return {**base, "status": "primary_unavailable"}
    try:
        aligned = shadow_head.align_extent(features, text)
    except Exception as error:  # instrumentation may never fail production timing
        return {**base, "status": "error", "error": str(error)}
    if not aligned:
        return base
    spans, extent_start, extent_end = aligned
    shadow_words, shadow_mode, shadow_meta = build_aligned_word_timestamps(
        text,
        spans,
        window_start,
        window_end,
        (extent_start, extent_end),
    )
    if shadow_mode != "ctc_forced_alignment" or not shadow_words:
        return {**base, "status": "shadow_timing_unavailable"}

    primary_start = float(primary_words[0]["start"])
    primary_end = float(primary_words[-1]["end"])
    shadow_start = float(shadow_words[0]["start"])
    shadow_end = float(shadow_words[-1]["end"])
    onset_delta_ms = (shadow_start - primary_start) * 1000.0
    end_delta_ms = (shadow_end - primary_end) * 1000.0
    return {
        **base,
        "status": "ok",
        "primary_start_s": round(primary_start, 6),
        "primary_end_s": round(primary_end, 6),
        "shadow_start_s": round(shadow_start, 6),
        "shadow_end_s": round(shadow_end, 6),
        "onset_delta_ms": round(onset_delta_ms, 3),
        "end_delta_ms": round(end_delta_ms, 3),
        "max_abs_delta_ms": round(max(abs(onset_delta_ms), abs(end_delta_ms)), 3),
        "primary_alignment_score": primary_timing_meta.get("alignment_score"),
        "shadow_alignment_score": shadow_meta.get("alignment_score"),
        "primary_boundary_trimmed": bool(primary_timing_meta.get("boundary_trimmed")),
        "primary_boundary_edged": bool(primary_timing_meta.get("boundary_edged")),
        "shadow_boundary_trimmed": bool(shadow_meta.get("boundary_trimmed")),
        "shadow_boundary_edged": bool(shadow_meta.get("boundary_edged")),
    }


def build_run_details(
    chunk_infos: list[dict[str, Any]],
    prepared_results: list[tuple[dict[str, Any], list[str]]],
) -> dict[str, Any] | None:
    """Lift chunk-relative comparisons into the source-audio timeline."""

    reference = shadow_head_reference()
    if not reference:
        return None
    comparisons: list[dict[str, Any]] = []
    for position, (chunk, prepared) in enumerate(zip(chunk_infos, prepared_results)):
        result, _log = prepared
        meta = result.get("timing_meta")
        shadow = meta.get("alignment_shadow") if isinstance(meta, dict) else None
        if not isinstance(shadow, dict):
            continue
        chunk_start = float(chunk.get("start", 0.0))
        item = {
            **shadow,
            "chunk_index": int(chunk.get("index", position)),
            "chunk_start_s": round(chunk_start, 6),
            "chunk_end_s": round(float(chunk.get("end", chunk_start)), 6),
            "text": str(result.get("text") or result.get("raw_text") or ""),
        }
        if shadow.get("status") == "ok":
            for key in ("primary_start_s", "primary_end_s", "shadow_start_s", "shadow_end_s"):
                item[key.removesuffix("_s") + "_abs_s"] = round(
                    chunk_start + float(shadow[key]), 6
                )
        comparisons.append(item)
    if not comparisons:
        return None
    threshold = shadow_min_delta_ms()
    statuses = Counter(str(row.get("status") or "unknown") for row in comparisons)
    eligible = sum(
        1
        for row in comparisons
        if row.get("status") == "ok"
        and float(row.get("max_abs_delta_ms") or 0.0) >= threshold
    )
    return {
        "schema": SHADOW_RUN_SCHEMA,
        "primary_head": os.getenv("ASR_ALIGNMENT_HEAD_PATH", "").strip(),
        "shadow_head": reference,
        "minimum_disagreement_ms": threshold,
        "comparison_count": len(comparisons),
        "eligible_disagreement_count": eligible,
        "status_counts": dict(statuses),
        "comparisons": comparisons,
    }


def _runtime_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    return cleaned.strip("._-") or "run"


def persist_shadow_run(
    *,
    run_details: dict[str, Any] | None,
    video_path: str | Path,
    video_duration_s: float | None,
    job_id: str,
    audio_cache_key: str,
    root: str | Path | None = None,
) -> Path | None:
    """Persist one job atomically outside its disposable job directory."""

    if not isinstance(run_details, dict) or not run_details.get("comparisons"):
        return None
    output_root = _runtime_path(
        root or os.getenv(SHADOW_ROOT_ENV, DEFAULT_SHADOW_ROOT) or DEFAULT_SHADOW_ROOT
    )
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = _safe_component(audio_cache_key)[:16]
    target = output_root / f"{_safe_component(job_id)}_{suffix}.json"
    payload = {
        **run_details,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "job_id": str(job_id),
        "audio_cache_key": str(audio_cache_key),
        "source_video_path": str(Path(video_path).resolve()),
        "source_video_duration_s": (
            None if video_duration_s is None else float(video_duration_s)
        ),
    }
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
