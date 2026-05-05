import os
import re


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


_KANA_ONLY_RE = re.compile(r"^[ぁ-ゟァ-ヿ\s、。！？…ー～「」『』・\(\)（）]+$")


def compute_quality_report(
    segments: list[dict],
    video_duration_s: float,
    glossary_pairs: list[tuple],
    alignment_fallback_count: int,
    total_segments: int,
    f0_filtered_count: int = 0,
    f0_failure: bool = False,
) -> dict:
    """Compute SRT quality metrics and flag threshold violations."""
    n = len(segments)
    if n == 0:
        return {
            "empty_zh_ratio": 0.0,
            "repetition_ratio": 0.0,
            "kana_only_ratio": 0.0,
            "short_segment_ratio": 0.0,
            "per_min_subtitle_count": 0.0,
            "glossary_hit_rate": None,
            "alignment_fallback_ratio": 0.0,
            "f0_filtered_count": f0_filtered_count,
            "f0_failure": f0_failure,
            "warnings": [],
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

    # 8. gender ratios (optional — only present when 'gender' field populated)
    gendered = [s for s in segments if "gender" in s]
    if gendered:
        male_ratio = sum(1 for s in gendered if s["gender"] == "M") / n
        female_ratio = sum(1 for s in gendered if s["gender"] == "F") / n
        gender_none_ratio = sum(1 for s in gendered if s["gender"] is None) / n
    else:
        male_ratio = female_ratio = gender_none_ratio = None

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

    report = {
        "empty_zh_ratio": empty_zh_ratio,
        "repetition_ratio": repetition_ratio,
        "kana_only_ratio": kana_only_ratio,
        "short_segment_ratio": short_segment_ratio,
        "per_min_subtitle_count": round(per_min_subtitle_count, 2),
        "glossary_hit_rate": glossary_hit_rate,
        "alignment_fallback_ratio": alignment_fallback_ratio,
        "f0_filtered_count": f0_filtered_count,
        "f0_failure": f0_failure,
        "warnings": warnings,
    }
    if male_ratio is not None:
        report["male_ratio"] = male_ratio
        report["female_ratio"] = female_ratio
        report["gender_none_ratio"] = gender_none_ratio
    return report

