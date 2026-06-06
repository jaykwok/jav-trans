import os
import re


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


def compute_quality_report(
    segments: list[dict],
    video_duration_s: float,
    glossary_pairs: list[tuple],
    alignment_fallback_count: int,
    total_segments: int,
    asr_qc: dict | None = None,
) -> dict:
    """Compute SRT quality metrics and flag threshold violations."""
    asr_qc = asr_qc or {}
    asr_generation_error_count = int(asr_qc.get("generation_error_count") or 0)
    asr_generation_overflow_count = int(asr_qc.get("generation_overflow_count") or 0)
    asr_timeout_count = int(asr_qc.get("timeout_count") or 0)
    asr_quarantined_count = int(asr_qc.get("quarantined_count") or 0)
    asr_empty_text_for_speech_count = int(asr_qc.get("empty_text_for_speech_count") or 0)
    asr_review_uncertain_count = int(asr_qc.get("review_uncertain_count") or 0)
    asr_review_uncertain_items = (
        list(asr_qc.get("review_uncertain_items") or [])
        if isinstance(asr_qc.get("review_uncertain_items"), list)
        else []
    )
    overlap_stats = _subtitle_overlap_stats(segments)
    duration_stats = _subtitle_duration_stats(segments)

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
            "alignment_fallback_ratio": 0.0,
            "asr_generation_error_count": asr_generation_error_count,
            "asr_generation_overflow_count": asr_generation_overflow_count,
            "asr_timeout_count": asr_timeout_count,
            "asr_quarantined_count": asr_quarantined_count,
            "asr_empty_text_for_speech_count": asr_empty_text_for_speech_count,
            "asr_review_uncertain_count": asr_review_uncertain_count,
            "asr_review_uncertain_items": asr_review_uncertain_items,
            **overlap_stats,
            **duration_stats,
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

    # 7. alignment_fallback_ratio
    alignment_fallback_ratio = alignment_fallback_count / max(total_segments, 1)

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
    if alignment_fallback_ratio > _env_float("QC_MAX_ALIGN_FALLBACK", 0.20):
        warnings.append(
            f"alignment_fallback_ratio={alignment_fallback_ratio:.3f} > QC_MAX_ALIGN_FALLBACK={_env_float('QC_MAX_ALIGN_FALLBACK', 0.20)}"
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

    report = {
        "empty_zh_ratio": empty_zh_ratio,
        "repetition_ratio": repetition_ratio,
        "kana_only_ratio": kana_only_ratio,
        "short_segment_ratio": short_segment_ratio,
        "per_min_subtitle_count": round(per_min_subtitle_count, 2),
        "glossary_hit_rate": glossary_hit_rate,
        "alignment_fallback_ratio": alignment_fallback_ratio,
        "asr_generation_error_count": asr_generation_error_count,
        "asr_generation_overflow_count": asr_generation_overflow_count,
        "asr_timeout_count": asr_timeout_count,
        "asr_quarantined_count": asr_quarantined_count,
        "asr_empty_text_for_speech_count": asr_empty_text_for_speech_count,
        "asr_review_uncertain_count": asr_review_uncertain_count,
        "asr_review_uncertain_items": asr_review_uncertain_items,
        **overlap_stats,
        **duration_stats,
        "warnings": warnings,
    }
    return report

