import os
import re

from subtitles.options import BASE_FPS
from subtitles.zh_style import (
    count_banned_punctuation,
    normalize_zh_subtitle_text,
    wrap_zh_subtitle_text,
    zh_display_units,
)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _append_asr_generation_warnings(
    warnings: list[str],
    *,
    asr_generation_error_count: int,
    asr_generation_overflow_count: int,
) -> None:
    if asr_generation_error_count > _env_float("QC_MAX_ASR_GENERATION_ERRORS", 0.0):
        warnings.append(
            f"asr_generation_error_count={asr_generation_error_count} > QC_MAX_ASR_GENERATION_ERRORS={_env_float('QC_MAX_ASR_GENERATION_ERRORS', 0.0):.0f}"
        )
    if asr_generation_overflow_count > _env_float("QC_MAX_ASR_GENERATION_OVERFLOWS", 0.0):
        warnings.append(
            f"asr_generation_overflow_count={asr_generation_overflow_count} > QC_MAX_ASR_GENERATION_OVERFLOWS={_env_float('QC_MAX_ASR_GENERATION_OVERFLOWS', 0.0):.0f}"
        )


def _subtitle_overlap_stats(segments: list[dict]) -> dict:
    ordered: list[dict] = []
    for segment in segments:
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            continue
        ordered.append({"start": start, "end": max(start, end)})
    ordered.sort(key=lambda item: (item["start"], item["end"]))

    count = 0
    total_s = 0.0
    max_s = 0.0
    examples: list[dict] = []
    for previous, current in zip(ordered, ordered[1:]):
        overlap_s = previous["end"] - current["start"]
        if overlap_s <= 0:
            continue
        count += 1
        total_s += overlap_s
        max_s = max(max_s, overlap_s)
        if len(examples) < 5:
            examples.append(
                {
                    "previous_start": round(previous["start"], 3),
                    "previous_end": round(previous["end"], 3),
                    "current_start": round(current["start"], 3),
                    "current_end": round(current["end"], 3),
                    "overlap_s": round(overlap_s, 3),
                }
            )

    return {
        "subtitle_overlap_count": count,
        "subtitle_overlap_total_s": round(total_s, 3),
        "subtitle_overlap_max_s": round(max_s, 3),
        "subtitle_overlap_examples": examples,
    }


def _percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = max(0.0, min(1.0, ratio)) * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _subtitle_duration_stats(segments: list[dict]) -> dict:
    durations: list[float] = []
    short_count = 0
    micro_count = 0
    long_count = 0
    for segment in segments:
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            continue
        duration = max(0.0, end - start)
        durations.append(duration)
        if duration < 0.8:
            short_count += 1
        if duration < 0.5:
            micro_count += 1
        if duration > 5.0:
            long_count += 1

    durations.sort()
    return {
        "subtitle_duration_p50_s": round(_percentile(durations, 0.50), 3),
        "subtitle_duration_p90_s": round(_percentile(durations, 0.90), 3),
        "subtitle_duration_p95_s": round(_percentile(durations, 0.95), 3),
        "subtitle_duration_max_s": round(durations[-1], 3) if durations else 0.0,
        "short_segment_count": short_count,
        "micro_segment_count": micro_count,
        "long_segment_count": long_count,
    }


_KANA_ONLY_RE = re.compile(r"^[ぁ-ゟァ-ヿ\s、。！？…ー～「」『』・\(\)（）]+$")
_COMPACT_TEXT_RE = re.compile(r"\s+")

# Netflix CHS TTSG hard limits the writer is expected to satisfy; QC measures
# the rendered output (same normalize+wrap pass as write_srt) so a regression
# in either layer shows up here.
_SPEC_LINE_MAX_UNITS = 16.0
_SPEC_MAX_LINES = 2
_SPEC_MAX_ZH_CPS = 9.0
_SPEC_MAX_DURATION_S = 7.0
# The last two are a known, deliberate deviation, not a target. Layout v3 ends a
# cue at the last character actually spoken, so reaching either would mean
# moving a boundary to a time with no speech evidence behind it. Measured over
# eight real films: 7.0% of cues are under the minimum and 553 neighbour pairs
# are under the gap, while zero pairs overlap. Their counters below are a scale
# to watch for movement, not a pass/fail line - the spec metrics that do act as
# regression signals are the text ones (line units, CPS, banned punctuation).
_SPEC_MIN_DURATION_S = 5.0 / 6.0
_SPEC_MIN_GAP_S = 2.0 / BASE_FPS


def _rendered_zh_lines(segment: dict) -> list[str]:
    rendered = wrap_zh_subtitle_text(
        normalize_zh_subtitle_text(str(segment.get("zh") or ""))
    )
    return [line for line in rendered.split("\n") if line]


def _subtitle_spec_compliance_stats(segments: list[dict]) -> dict:
    line_over_count = 0
    lines_over_count = 0
    banned_punct_count = 0
    raw_banned_punct_count = 0
    cps_over_count = 0
    max_cps = 0.0
    duration_over_count = 0
    duration_under_count = 0
    gap_under_count = 0
    examples: list[dict] = []
    windows: list[tuple[float, float]] = []

    for index, segment in enumerate(segments):
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            continue
        end = max(start, end)
        duration = end - start
        windows.append((start, end))

        issues: list[str] = []
        if duration > _SPEC_MAX_DURATION_S + 1e-6:
            duration_over_count += 1
            issues.append(f"duration>{_SPEC_MAX_DURATION_S:.0f}s")
        if duration < _SPEC_MIN_DURATION_S - 1e-6:
            duration_under_count += 1
            issues.append("duration<5/6s")

        lines = _rendered_zh_lines(segment)
        raw_banned_punct_count += count_banned_punctuation(str(segment.get("zh") or ""))
        if lines:
            if len(lines) > _SPEC_MAX_LINES:
                lines_over_count += 1
                issues.append(f"lines={len(lines)}")
            over_lines = sum(
                1 for line in lines if zh_display_units(line) > _SPEC_LINE_MAX_UNITS + 1e-6
            )
            if over_lines:
                line_over_count += over_lines
                issues.append("line>16")
            banned = count_banned_punctuation("\n".join(lines))
            if banned:
                banned_punct_count += banned
                issues.append(f"banned_punct={banned}")
            reading_units = zh_display_units(
                "".join(lines).replace(" ", "")
            )
            cps = reading_units / max(duration, 0.001)
            max_cps = max(max_cps, cps)
            if cps > _SPEC_MAX_ZH_CPS + 1e-6:
                cps_over_count += 1
                issues.append(f"cps={cps:.1f}")

        if issues and len(examples) < 20:
            examples.append(
                {
                    "index": index,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "issues": issues,
                }
            )

    windows.sort()
    for (_, previous_end), (next_start, _) in zip(windows, windows[1:]):
        gap = next_start - previous_end
        if 0.0 <= gap < _SPEC_MIN_GAP_S - 1e-6:
            gap_under_count += 1

    cue_count = len(windows)
    neighbour_count = max(0, cue_count - 1)
    return {
        "spec_zh_line_over_16_count": line_over_count,
        "spec_zh_lines_over_2_count": lines_over_count,
        "spec_zh_banned_punct_count": banned_punct_count,
        "spec_zh_raw_banned_punct_count": raw_banned_punct_count,
        "spec_zh_cps_over_9_count": cps_over_count,
        "spec_zh_cps_max": round(max_cps, 3),
        "spec_duration_over_7s_count": duration_over_count,
        "spec_duration_under_min_count": duration_under_count,
        "spec_gap_under_2frames_count": gap_under_count,
        "spec_cue_count": cue_count,
        # Shares, because the two deviating metrics scale with film length: on
        # eight real films the same 5-10% rate reads as 21 cues on a short film
        # and 97 on a long one, so a count threshold that passes the long one
        # would hide a fourfold regression on the short one.
        "spec_duration_under_min_share": (
            round(duration_under_count / cue_count, 4) if cue_count else 0.0
        ),
        "spec_gap_under_2frames_share": (
            round(gap_under_count / neighbour_count, 4) if neighbour_count else 0.0
        ),
        "spec_review_examples": examples,
    }


# Counts that must stay at zero: nothing in the layout is allowed to produce
# them, so any occurrence is a defect.
_SPEC_COUNT_THRESHOLDS = {
    "spec_zh_line_over_16_count": "QC_MAX_SPEC_LINE_OVER",
    "spec_zh_lines_over_2_count": "QC_MAX_SPEC_LINES_OVER",
    "spec_zh_banned_punct_count": "QC_MAX_SPEC_BANNED_PUNCT",
    "spec_zh_cps_over_9_count": "QC_MAX_SPEC_CPS_OVER",
    "spec_duration_over_7s_count": "QC_MAX_SPEC_DURATION_OVER",
}

# The two TTSG timing rules Layout v3 deliberately does not satisfy (see
# `_SPEC_MIN_DURATION_S`). Zero is unreachable by construction, so they are
# watched as rates instead: the defaults sit above the highest per-film rate
# measured on the shipped head (10.6% and 9.7% over eight films) with room for
# ordinary variation, so a warning here means the cue shape moved, not that the
# layout is doing what it was designed to do. A head change can move the whole
# distribution - the retained punctuated head runs 12.6-23.8% on the gap rate -
# so these are recalibrated when the head is, not treated as universal.
_SPEC_SHARE_THRESHOLDS = {
    "spec_duration_under_min_share": ("QC_MAX_SPEC_DURATION_UNDER_SHARE", 0.15),
    "spec_gap_under_2frames_share": ("QC_MAX_SPEC_GAP_UNDER_SHARE", 0.15),
}


def _append_spec_warnings(warnings: list[str], spec_stats: dict) -> None:
    for key, env_name in _SPEC_COUNT_THRESHOLDS.items():
        limit = _env_float(env_name, 0.0)
        value = spec_stats.get(key, 0)
        if value > limit:
            warnings.append(f"{key}={value} > {env_name}={limit:.0f}")
    for key, (env_name, default) in _SPEC_SHARE_THRESHOLDS.items():
        limit = _env_float(env_name, default)
        value = float(spec_stats.get(key, 0.0))
        if value > limit:
            warnings.append(f"{key}={value:.3f} > {env_name}={limit:.3f}")


# Chunk-boundary provenance from `asr.chunking.plan_chunk_cuts`, and the cue
# continuity counts from the layout pass. Both are copied into the report rather
# than recomputed: neither can be recovered from the finished subtitles, and both
# are the only run-over-run record of decisions taken far upstream of them.
#
# Nothing here has a threshold. `max_chunk_fallback_share` - boundaries the pause
# search could not place, so the chunker cut at 30s regardless - measures 0.7% to
# 53% across eight real films, and the high ones are films that are mostly
# continuous vocalisation rather than films that were cut badly. A limit that
# passed those would be met by anything.
_CHUNK_CUT_REPORT_KEYS = {
    "policy": "chunk_cut_policy",
    "source": "chunk_cut_source",
    "chunk_count": "chunk_count",
    "cut_count": "chunk_cut_count",
    "pause_cut_count": "chunk_cut_at_pause_count",
    "max_chunk_fallback_count": "chunk_cut_max_fallback_count",
    "max_chunk_fallback_share": "chunk_cut_max_fallback_share",
    "cut_pause_width_median_s": "chunk_cut_pause_width_median_s",
    "cut_pause_width_min_s": "chunk_cut_pause_width_min_s",
    "chunk_duration_median_s": "chunk_duration_median_s",
    "chunk_duration_min_s": "chunk_duration_min_s",
    "chunk_duration_max_s": "chunk_duration_max_s",
}


def _chunk_cut_stats(chunk_cuts: dict | None) -> dict:
    if not isinstance(chunk_cuts, dict) or not chunk_cuts:
        return {}
    return {
        report_key: chunk_cuts[source_key]
        for source_key, report_key in _CHUNK_CUT_REPORT_KEYS.items()
        if source_key in chunk_cuts
    }


# The layout's own cut points, and the continuation claims they produce. The
# two are one subject: `continues_into_next` is set on every cue whose boundary
# is not a written sentence end, so the break-type mix below is the reason the
# continuation counts are what they are.
_CUE_CONTINUITY_REPORT_KEYS = {
    "subtitle_layout_break_type": "layout_break_type_counts",
    "layout_word_gap_cut_count": "layout_word_gap_cut_count",
    "layout_word_gap_cut_under_0p2s": "layout_word_gap_cut_under_0p2s",
    "layout_word_gap_median_s": "layout_word_gap_median_s",
    # How much reading time was taken from silence that was already empty. Read
    # against `spec_duration_under_min_share`: this is the mechanism that moves
    # it, so a run where the share jumps and this is zero means the pass did not
    # run rather than that the cues changed shape.
    "display_linger_applied_count": "display_linger_applied_count",
    "display_linger_total_s": "display_linger_total_s",
    "continues_from_previous": "cue_continues_from_previous_count",
    "continues_into_next": "cue_continues_into_next_count",
    "vocalisation_cues_dropped": "vocalisation_cues_dropped",
    "vocalisation_runs_dropped": "vocalisation_runs_dropped",
    "vocalisation_continuity_flags_cleared": "vocalisation_continuity_flags_cleared",
    "postgate_flagged_cues": "postgate_flagged_cue_count",
    "postgate_cue_flags": "postgate_cue_flag_counts",
}


# `asr.postgate` reviews every decoded chunk and marks the ones the audio does
# not support, and until now the marks stopped at `aligned_segments.json`. Both
# layers are reported because they answer different questions: the chunk counts
# say what the detector saw, and the cue counts above say how much of it reached
# the viewer after the layout and the vocalisation filter had their turn.
#
# No thresholds. `repeated_unit` alone runs around 10% of chunks on this domain
# and most of it is genuine repeated interjection, so a limit would have to be
# invented rather than measured - and what to do about the rest is a decision to
# take from these numbers, not before them.
_POSTGATE_CHUNK_REPORT_KEYS = {
    "reviewed": "postgate_chunks_reviewed",
    "flagged": "postgate_chunks_flagged",
    "flags": "postgate_chunk_flag_counts",
    "alignment_score_checked": "postgate_alignment_score_checked",
}


def _postgate_chunk_stats(postgate: dict | None) -> dict:
    if not isinstance(postgate, dict) or not postgate:
        return {}
    stats = {
        report_key: postgate[source_key]
        for source_key, report_key in _POSTGATE_CHUNK_REPORT_KEYS.items()
        if source_key in postgate
    }
    reviewed = int(stats.get("postgate_chunks_reviewed") or 0)
    if reviewed > 0 and "postgate_chunks_flagged" in stats:
        stats["postgate_chunks_flagged_share"] = round(
            int(stats["postgate_chunks_flagged"]) / reviewed, 4
        )
    return stats


def _cue_continuity_stats(cue_plan: dict | None) -> dict:
    if not isinstance(cue_plan, dict) or not cue_plan:
        return {}
    diagnostics = cue_plan.get("layout_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    stats = {
        report_key: diagnostics[source_key]
        for source_key, report_key in _CUE_CONTINUITY_REPORT_KEYS.items()
        if source_key in diagnostics
    }
    # A count of continuation claims means nothing without how many cues could
    # have made one, and the layout's own cue count is the denominator - not the
    # segment count below, which is taken after translation.
    cues = int(cue_plan.get("cues_after") or 0)
    if cues > 0 and "cue_continues_from_previous_count" in stats:
        stats["cue_continues_from_previous_share"] = round(
            int(stats["cue_continues_from_previous_count"]) / cues, 4
        )
    if cues > 0 and "postgate_flagged_cue_count" in stats:
        stats["postgate_flagged_cue_share"] = round(
            int(stats["postgate_flagged_cue_count"]) / cues, 4
        )
    if cues > 0:
        stats["cue_plan_cue_count"] = cues
    return stats


def _subtitle_text_units(segment: dict) -> int:
    text = str(segment.get("text") or segment.get("ja") or "")
    return len(_COMPACT_TEXT_RE.sub("", text))


def _subtitle_density_audit_stats(
    segments: list[dict],
    *,
    cps_threshold: float = 4.0,
) -> dict:
    cues: list[dict] = []
    for index, segment in enumerate(segments):
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            continue
        end = max(start, end)
        duration = max(0.001, end - start)
        units = _subtitle_text_units(segment)
        cps = units / duration if units > 0 else 0.0
        cues.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "duration": duration,
                "units": units,
                "cps": cps,
            }
        )
    cues.sort(key=lambda item: (item["start"], item["end"], item["index"]))

    over_threshold = [cue for cue in cues if cue["cps"] > cps_threshold]
    max_cps = max((cue["cps"] for cue in cues), default=0.0)
    cps_values = sorted(cue["cps"] for cue in cues)

    def window_stats(window_s: float) -> dict:
        max_count = 0
        max_active_ratio = 0.0
        max_window_cps = 0.0
        min_gap: float | None = None
        median_gap_values: list[float] = []
        for cue in cues:
            window_start = cue["start"]
            window_end = window_start + window_s
            members = [
                item
                for item in cues
                if item["end"] > window_start and item["start"] < window_end
            ]
            if not members:
                continue
            max_count = max(max_count, len(members))
            active_s = sum(
                max(
                    0.0,
                    min(item["end"], window_end) - max(item["start"], window_start),
                )
                for item in members
            )
            max_active_ratio = max(max_active_ratio, active_s / max(window_s, 0.001))
            max_window_cps = max(
                max_window_cps,
                sum(item["units"] for item in members) / max(window_s, 0.001),
            )
            gaps = [
                max(0.0, right["start"] - left["end"])
                for left, right in zip(members, members[1:])
            ]
            if gaps:
                gap_sorted = sorted(gaps)
                min_gap = min(gap_sorted[0], min_gap) if min_gap is not None else gap_sorted[0]
                median_gap_values.append(_percentile(gap_sorted, 0.50))
        median_gap_values.sort()
        return {
            f"subtitle_density_window_{int(window_s)}s_max_cue_count": max_count,
            f"subtitle_density_window_{int(window_s)}s_max_active_ratio": round(
                max_active_ratio,
                3,
            ),
            f"subtitle_density_window_{int(window_s)}s_max_cps": round(max_window_cps, 3),
            f"subtitle_density_window_{int(window_s)}s_min_gap_s": round(min_gap, 3)
            if min_gap is not None
            else None,
            f"subtitle_density_window_{int(window_s)}s_median_gap_s": round(
                _percentile(median_gap_values, 0.50),
                3,
            )
            if median_gap_values
            else None,
        }

    examples = [
        {
            "index": cue["index"],
            "start": round(cue["start"], 3),
            "end": round(cue["end"], 3),
            "duration_s": round(cue["duration"], 3),
            "ja_units": cue["units"],
            "ja_cps": round(cue["cps"], 3),
        }
        for cue in sorted(over_threshold, key=lambda item: (-item["cps"], item["start"]))[:20]
    ]
    return {
        "subtitle_density_cps_threshold": round(cps_threshold, 3),
        "subtitle_density_over_4cps_count": len(over_threshold),
        "subtitle_density_over_4cps_ratio": round(len(over_threshold) / max(1, len(cues)), 6),
        "subtitle_density_max_ja_cps": round(max_cps, 3),
        "subtitle_density_p90_ja_cps": round(_percentile(cps_values, 0.90), 3),
        "subtitle_density_p95_ja_cps": round(_percentile(cps_values, 0.95), 3),
        "subtitle_density_review_examples": examples,
        **window_stats(10.0),
        **window_stats(30.0),
    }


def compute_quality_report(
    segments: list[dict],
    video_duration_s: float,
    glossary_pairs: list[tuple],
    alignment_issue_count: int,
    total_segments: int,
    asr_generation: dict | None = None,
    *,
    chunk_cuts: dict | None = None,
    cue_plan: dict | None = None,
    postgate: dict | None = None,
) -> dict:
    """Compute SRT quality metrics and flag threshold violations."""
    asr_generation = asr_generation or {}
    asr_generation_error_count = int(asr_generation.get("generation_error_count") or 0)
    asr_generation_overflow_count = int(asr_generation.get("generation_overflow_count") or 0)
    asr_timeout_count = int(asr_generation.get("timeout_count") or 0)
    asr_quarantined_count = int(asr_generation.get("quarantined_count") or 0)
    overlap_stats = _subtitle_overlap_stats(segments)
    duration_stats = _subtitle_duration_stats(segments)
    density_stats = _subtitle_density_audit_stats(segments)
    spec_stats = _subtitle_spec_compliance_stats(segments)
    chunk_cut_stats = _chunk_cut_stats(chunk_cuts)
    cue_continuity_stats = _cue_continuity_stats(cue_plan)
    postgate_stats = _postgate_chunk_stats(postgate)
    alignment_issue_total = max(int(total_segments or 0), 0)

    n = len(segments)
    if n == 0:
        warnings: list[str] = []
        _append_asr_generation_warnings(
            warnings,
            asr_generation_error_count=asr_generation_error_count,
            asr_generation_overflow_count=asr_generation_overflow_count,
        )
        return {
            "empty_zh_ratio": 0.0,
            "repetition_ratio": 0.0,
            "kana_only_ratio": 0.0,
            "short_segment_ratio": 0.0,
            "per_min_subtitle_count": 0.0,
            "glossary_hit_rate": None,
            "alignment_issue_count": alignment_issue_count,
            "alignment_issue_total": alignment_issue_total,
            "alignment_issue_ratio": 0.0,
            "asr_generation_error_count": asr_generation_error_count,
            "asr_generation_overflow_count": asr_generation_overflow_count,
            "asr_timeout_count": asr_timeout_count,
            "asr_quarantined_count": asr_quarantined_count,
            **overlap_stats,
            **duration_stats,
            **density_stats,
            **spec_stats,
            **chunk_cut_stats,
            **cue_continuity_stats,
            **postgate_stats,
            "warnings": warnings,
        }

    # 1. empty_zh_ratio
    empty_zh = sum(1 for s in segments if not (s.get("zh") or "").strip())
    empty_zh_ratio = empty_zh / n

    # 2. repetition_ratio — consecutive identical zh lines
    repeat = 0
    for i in range(1, n):
        prev = (segments[i - 1].get("zh") or "").strip()
        curr = (segments[i].get("zh") or "").strip()
        if curr and curr == prev:
            repeat += 1
    repetition_ratio = repeat / n

    # 3. kana_only_ratio — empty zh or ja contains only kana/punctuation.
    kana_only = sum(
        1
        for s in segments
        if not (s.get("zh") or "").strip()
        or _KANA_ONLY_RE.fullmatch((s.get("text") or s.get("ja") or "").strip())
    )
    kana_only_ratio = kana_only / n

    # 4. short_segment_ratio
    short = sum(1 for s in segments if (s.get("end", 0) - s.get("start", 0)) < 0.8)
    short_segment_ratio = short / n

    # 5. per_min_subtitle_count
    minutes = max(video_duration_s / 60.0, 0.001)
    per_min_subtitle_count = n / minutes

    # 6. glossary_hit_rate — bilateral: ja term in original AND zh term in translation
    glossary_hit_rate = None
    if glossary_pairs:
        hits = 0
        checks = 0
        for ja_term, zh_term in glossary_pairs:
            if not ja_term or not zh_term:
                continue
            for s in segments:
                ja_text = s.get("text") or s.get("ja") or ""
                zh_text = s.get("zh") or ""
                if ja_term in ja_text:
                    checks += 1
                    if zh_term in zh_text:
                        hits += 1
        glossary_hit_rate = (hits / checks) if checks > 0 else None

    # 7. Subtitle timing/alignment issue observation.
    alignment_issue_ratio = alignment_issue_count / max(alignment_issue_total, 1)

    # Threshold checks
    warnings: list[str] = []
    if empty_zh_ratio > _env_float("QC_MAX_EMPTY_ZH", 0.02):
        warnings.append(
            f"empty_zh_ratio={empty_zh_ratio:.3f} > QC_MAX_EMPTY_ZH={_env_float('QC_MAX_EMPTY_ZH', 0.02)}"
        )
    if repetition_ratio > _env_float("QC_MAX_REPETITION", 0.05):
        warnings.append(
            f"repetition_ratio={repetition_ratio:.3f} > QC_MAX_REPETITION={_env_float('QC_MAX_REPETITION', 0.05)}"
        )
    if kana_only_ratio > _env_float("QC_MAX_KANA_ONLY", 0.30):
        warnings.append(
            f"kana_only_ratio={kana_only_ratio:.3f} > QC_MAX_KANA_ONLY={_env_float('QC_MAX_KANA_ONLY', 0.30)}"
        )
    if short_segment_ratio > _env_float("QC_MAX_SHORT_SEG", 0.15):
        warnings.append(
            f"short_segment_ratio={short_segment_ratio:.3f} > QC_MAX_SHORT_SEG={_env_float('QC_MAX_SHORT_SEG', 0.15)}"
        )
    if per_min_subtitle_count > _env_float("QC_MAX_PER_MIN", 8.0):
        warnings.append(
            f"per_min_subtitle_count={per_min_subtitle_count:.1f} > QC_MAX_PER_MIN={_env_float('QC_MAX_PER_MIN', 8.0)}"
        )
    if glossary_hit_rate is not None and glossary_hit_rate < _env_float("QC_MIN_GLOSSARY_HIT", 0.80):
        warnings.append(
            f"glossary_hit_rate={glossary_hit_rate:.3f} < QC_MIN_GLOSSARY_HIT={_env_float('QC_MIN_GLOSSARY_HIT', 0.80)}"
        )
    if overlap_stats["subtitle_overlap_count"] > 0:
        warnings.append(
            f"subtitle_overlap_count={overlap_stats['subtitle_overlap_count']} after timeline normalization"
        )
    _append_asr_generation_warnings(
        warnings,
        asr_generation_error_count=asr_generation_error_count,
        asr_generation_overflow_count=asr_generation_overflow_count,
    )
    _append_spec_warnings(warnings, spec_stats)

    report = {
        "empty_zh_ratio": empty_zh_ratio,
        "repetition_ratio": repetition_ratio,
        "kana_only_ratio": kana_only_ratio,
        "short_segment_ratio": short_segment_ratio,
        "per_min_subtitle_count": round(per_min_subtitle_count, 2),
        "glossary_hit_rate": glossary_hit_rate,
        "alignment_issue_count": alignment_issue_count,
        "alignment_issue_total": alignment_issue_total,
        "alignment_issue_ratio": alignment_issue_ratio,
        "asr_generation_error_count": asr_generation_error_count,
        "asr_generation_overflow_count": asr_generation_overflow_count,
        "asr_timeout_count": asr_timeout_count,
        "asr_quarantined_count": asr_quarantined_count,
        **overlap_stats,
        **duration_stats,
        **density_stats,
        **spec_stats,
        **chunk_cut_stats,
        **cue_continuity_stats,
        **postgate_stats,
        "warnings": warnings,
    }
    return report

