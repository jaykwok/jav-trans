from __future__ import annotations

from typing import Any


def recommend_boundary_timing_params(
    duration_stats: dict[str, Any],
    *,
    target_domain_speedup: float = 1.5,
    max_padded_cap_s: float = 6.5,
) -> dict[str, Any]:
    """Derive boundary/subtitle timing defaults from clean speech-island durations.

    The input distribution is a source-domain clean utterance distribution, such as
    Galgame ASR clips. The speech-core values are scaled by target_domain_speedup
    because JAV dialogue is usually shorter/faster than Galgame dialogue. The
    padded ASR context cap is not scaled down as aggressively because it is
    recognition context, not the fallback subtitle window.
    """
    if target_domain_speedup <= 0:
        raise ValueError("target_domain_speedup must be positive")
    if max_padded_cap_s <= 0:
        raise ValueError("max_padded_cap_s must be positive")

    percentiles = duration_stats.get("percentiles_s") or {}
    p5 = _required_number(percentiles, "p5")
    p50 = _required_number(percentiles, "p50")
    p80 = _required_number(percentiles, "p80")
    p90 = _required_number(percentiles, "p90")

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
    context_max_padding_s = _round_step(
        _clamp((p90 / target_domain_speedup - max_core_s) / 2.0, 0.8, 1.5),
        step=0.1,
    )
    context_max_speech_overlap_s = 0.25
    max_padded_s = _floor_step(
        min(max_core_s + 2.0 * context_max_padding_s, max_padded_cap_s),
        step=0.5,
    )
    max_padded_s = max(max_core_s, max_padded_s)

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
            "max_padded_cap_s": max_padded_cap_s,
            "source_p5_s": p5,
            "source_p50_s": p50,
            "source_p80_s": p80,
            "source_p90_s": p90,
        },
        "formula": {
            "boundary_planner_target_chunk_s": "clamp(p50 / speedup, 2.0, 3.5), round 0.1s",
            "boundary_planner_max_core_chunk_s": "clamp(p80 / speedup, target + 1.0, 5.5), floor 0.5s",
            "boundary_context_max_padding_s": "clamp((p90 / speedup - max_core) / 2, 0.8, 1.5), round 0.1s",
            "boundary_context_max_speech_overlap_s": "fixed 0.25s hard cap for within-speech splits",
            "boundary_planner_max_padded_chunk_s": "min(max_core + 2 * context_cap, cap), floor 0.5s",
            "boundary_planner_min_chunk_s": "clamp(p5 / speedup * 0.60, 0.25, 0.50), round 0.05s",
            "subtitle_soft_max_s": "clamp(max_core + 0.5, 4.5, 5.5), round 0.1s",
            "max_subtitle_duration_s": "clamp(max_core + 1.5, soft_max + 0.5, 6.5), round 0.1s",
        },
        "env": {
            "BOUNDARY_PLANNER_TARGET_CHUNK_S": target_core_s,
            "BOUNDARY_PLANNER_MAX_CORE_CHUNK_S": max_core_s,
            "BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S": max_padded_s,
            "BOUNDARY_PLANNER_MIN_CHUNK_S": min_chunk_s,
            "BOUNDARY_CONTEXT_MAX_PADDING_S": context_max_padding_s,
            "BOUNDARY_CONTEXT_MAX_SPEECH_OVERLAP_S": context_max_speech_overlap_s,
            "SUBTITLE_SOFT_MAX_S": subtitle_soft_max_s,
            "MAX_SUBTITLE_DURATION": max_subtitle_duration_s,
            "ASR_MERGE_HARD_MAX_DURATION": max_padded_s,
        },
        "rationale": {
            "target_core_source_equivalent_s": target_core_s * target_domain_speedup,
            "max_core_source_equivalent_s": max_core_s * target_domain_speedup,
            "max_padded_source_cap_s": max_padded_s,
            "notes": [
                "target_core follows source p50 after target-domain speed scaling.",
                "max_core follows source p80 after speed scaling and is the fallback/subtitle timing window.",
                "context caps are hard upper bounds; learned refiner predicts actual ASR context budget.",
                "max_padded keeps ASR context bounded so failed alignments do not inherit padded timing.",
                "subtitle caps stay above max_core because timing polish may extend/merge short cues, but they remain below the industry-style long-cue ceiling.",
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
