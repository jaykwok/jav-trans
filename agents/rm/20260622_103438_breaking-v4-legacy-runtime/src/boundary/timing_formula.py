from __future__ import annotations

from typing import Any


def recommend_boundary_timing_params(
    duration_stats: dict[str, Any],
    *,
    target_domain_speedup: float = 1.5,
) -> dict[str, Any]:
    """Derive boundary/subtitle timing defaults from clean speech-island durations.

    The input distribution is a source-domain clean utterance distribution, such as
    Galgame ASR clips. The speech-core values are scaled by target_domain_speedup
    because JAV dialogue is usually shorter/faster than Galgame dialogue. The
    Boundary Refiner v4 uses speech-core-only ASR chunks; there is no learned
    ASR padding budget in the runtime formula.
    """
    if target_domain_speedup <= 0:
        raise ValueError("target_domain_speedup must be positive")

    percentiles = duration_stats.get("percentiles_s") or {}
    p5 = _required_number(percentiles, "p5")
    p50 = _required_number(percentiles, "p50")
    p80 = _required_number(percentiles, "p80")

    target_core_s = _round_step(
        _clamp(p50 / target_domain_speedup, 2.0, 3.5),
        step=0.1,
    )
    min_chunk_s = _round_step(
        _clamp((p5 / target_domain_speedup) * 0.60, 0.25, 0.50),
        step=0.05,
    )
    max_core_s = _floor_step(
        _clamp(p80 / target_domain_speedup, target_core_s + 1.0, 5.5),
        step=0.5,
    )
    subtitle_soft_max_s = _round_step(
        _clamp(max_core_s + 0.5, 4.5, 5.5),
        step=0.1,
    )
    max_subtitle_duration_s = _round_step(
        _clamp(max_core_s + 1.5, subtitle_soft_max_s + 0.5, 6.5),
        step=0.1,
    )

    return {
        "inputs": {
            "target_domain_speedup": target_domain_speedup,
            "source_p5_s": p5,
            "source_p50_s": p50,
            "source_p80_s": p80,
        },
        "formula": {
            "boundary_planner_target_chunk_s": "clamp(p50 / speedup, 2.0, 3.5), round 0.1s",
            "boundary_planner_max_core_chunk_s": "clamp(p80 / speedup, target + 1.0, 5.5), floor 0.5s",
            "boundary_planner_min_chunk_s": "clamp(p5 / speedup * 0.60, 0.25, 0.50), round 0.05s",
            "subtitle_soft_max_s": "clamp(max_core + 0.5, 4.5, 5.5), round 0.1s",
            "max_subtitle_duration_s": "clamp(max_core + 1.5, soft_max + 0.5, 6.5), round 0.1s",
        },
        "env": {
            "BOUNDARY_PLANNER_TARGET_CHUNK_S": target_core_s,
            "BOUNDARY_PLANNER_MAX_CORE_CHUNK_S": max_core_s,
            "BOUNDARY_PLANNER_MIN_CHUNK_S": min_chunk_s,
            "SUBTITLE_SOFT_MAX_S": subtitle_soft_max_s,
            "MAX_SUBTITLE_DURATION": max_subtitle_duration_s,
            "ASR_SEGMENT_HARD_MAX_DURATION": max_subtitle_duration_s,
        },
        "rationale": {
            "target_core_source_equivalent_s": target_core_s * target_domain_speedup,
            "max_core_source_equivalent_s": max_core_s * target_domain_speedup,
            "notes": [
                "target_core follows source p50 after target-domain speed scaling.",
                "max_core follows source p80 after speed scaling and is the fallback/subtitle timing window.",
                "Boundary Refiner v4 does not add learned ASR padding; ASR sees the refined speech core.",
                "subtitle caps stay above max_core because timing polish may adjust cue windows, but they remain below the industry-style long-cue ceiling.",
            ],
        },
    }


def _required_number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"duration_stats.percentiles_s.{key} is required")
    if value <= 0:
        raise ValueError(f"duration_stats.percentiles_s.{key} must be positive")
    return float(value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _round_step(value: float, *, step: float) -> float:
    return round(round(value / step) * step, 6)


def _floor_step(value: float, *, step: float) -> float:
    return round(int(value / step) * step, 6)
