"""CueQC decision-threshold calibration helpers."""
from __future__ import annotations

from typing import Any, Mapping

PUNCT_OR_EMPTY_CHARS = {".", "!", "?", "。", "、", " "}


def text_bucket(text: Any) -> str:
    value = str(text or "").strip()
    compact = value.replace("…", ".").replace("・", "").strip()
    if not compact or set(compact) <= PUNCT_OR_EMPTY_CHARS:
        return "punct_or_empty"
    if len(value) <= 4:
        return "short_text"
    if len(value) <= 16:
        return "medium_text"
    return "long_text"


def resolve_drop_threshold(
    decision_config: Mapping[str, Any] | None,
    *,
    text: Any,
    default: float = 0.85,
) -> tuple[float, dict[str, Any]]:
    """Resolve the conservative drop threshold for one candidate.

    Adaptive profiles are calibration-only: they may raise the threshold for a
    risk bucket, but they cannot lower it below the base threshold.
    """
    config = dict(decision_config or {})
    try:
        base = float(config.get("drop_threshold", default))
    except (TypeError, ValueError):
        base = float(default)
    profile = config.get("drop_threshold_profile")
    bucket = text_bucket(text)
    if not isinstance(profile, Mapping) or profile.get("enabled", True) is False:
        return base, {"mode": "fixed", "text_bucket": bucket, "base_threshold": base}

    threshold = base
    text_bucket_thresholds = profile.get("text_bucket")
    if isinstance(text_bucket_thresholds, Mapping) and bucket in text_bucket_thresholds:
        try:
            threshold = max(base, float(text_bucket_thresholds[bucket]))
        except (TypeError, ValueError):
            threshold = base
    return threshold, {
        "mode": str(profile.get("mode") or "text_bucket_profile_v1"),
        "text_bucket": bucket,
        "base_threshold": base,
        "threshold": threshold,
    }
