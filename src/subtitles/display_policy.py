from __future__ import annotations

import os
import unicodedata
from collections import Counter


DISPLAY_POLICY_VERSION = 1
_OFF_MODES = {"0", "false", "no", "off", "disabled", "none"}
_DECISIONS = ("keep", "drop", "compact", "review")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def settings_from_env() -> dict:
    mode = os.getenv("SUBTITLE_DISPLAY_POLICY", "readability").strip().lower()
    if not mode:
        mode = "readability"
    if mode in _OFF_MODES:
        mode = "off"
    return {
        "version": DISPLAY_POLICY_VERSION,
        "mode": mode,
        "repeat_run_min": max(
            2,
            _env_int("SUBTITLE_DISPLAY_POLICY_REPEAT_RUN_MIN", 3),
        ),
        "max_run_gap_s": max(
            0.0,
            _env_float("SUBTITLE_DISPLAY_POLICY_MAX_RUN_GAP_S", 1.0),
        ),
        "short_kana_max_chars": max(
            1,
            _env_int("SUBTITLE_DISPLAY_POLICY_SHORT_KANA_MAX_CHARS", 4),
        ),
        "stable_min_chars": max(
            2,
            _env_int("SUBTITLE_DISPLAY_POLICY_STABLE_MIN_CHARS", 5),
        ),
    }


def signature(settings: dict | None = None) -> dict:
    cfg = dict(settings or settings_from_env())
    return {
        "version": DISPLAY_POLICY_VERSION,
        "mode": str(cfg.get("mode") or "readability"),
        "repeat_run_min": int(cfg.get("repeat_run_min") or 3),
        "max_run_gap_s": float(cfg.get("max_run_gap_s") or 0.0),
        "short_kana_max_chars": int(cfg.get("short_kana_max_chars") or 4),
        "stable_min_chars": int(cfg.get("stable_min_chars") or 5),
    }


def _is_punctuation_or_symbol(char: str) -> bool:
    category = unicodedata.category(char)
    return char.isspace() or category.startswith("P") or category.startswith("S")


def _compact_text(text: str) -> str:
    return "".join(char for char in str(text or "") if not _is_punctuation_or_symbol(char))


def _is_kana(char: str) -> bool:
    return (
        "\u3040" <= char <= "\u30ff"
        or "\uff66" <= char <= "\uff9f"
        or char in {"\u30fc", "\uff70"}
    )


def _is_cjk(char: str) -> bool:
    return "\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff"


def _max_unit_repeat(compact: str) -> dict:
    if not compact:
        return {"run": 0, "unit_len": 0, "ratio": 0.0}
    best = {"run": 1, "unit_len": 1, "ratio": 1 / max(1, len(compact))}
    max_unit_len = min(6, max(1, len(compact) // 2))
    for unit_len in range(1, max_unit_len + 1):
        index = 0
        while index <= len(compact) - unit_len:
            unit = compact[index : index + unit_len]
            run = 1
            next_index = index + unit_len
            while (
                next_index + unit_len <= len(compact)
                and compact[next_index : next_index + unit_len] == unit
            ):
                run += 1
                next_index += unit_len
            ratio = (run * unit_len) / max(1, len(compact))
            if run > int(best["run"]) or (
                run == int(best["run"]) and ratio > float(best["ratio"])
            ):
                best = {"run": run, "unit_len": unit_len, "ratio": round(ratio, 4)}
            index += max(1, unit_len * run if run > 1 else 1)
    return best


def _max_char_run(compact: str) -> int:
    if not compact:
        return 0
    best = 1
    current = 1
    previous = compact[0]
    for char in compact[1:]:
        if char == previous:
            current += 1
        else:
            best = max(best, current)
            current = 1
            previous = char
    return max(best, current)


def _source_segment_ids(cue: dict, fallback_id: int) -> list[int]:
    raw_ids = cue.get("source_segment_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        return [fallback_id]
    ids: list[int] = []
    for item in raw_ids:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return list(dict.fromkeys(ids)) or [fallback_id]


def _source_segments_for_cue(
    cue: dict,
    *,
    source_segments: list[dict],
    fallback_id: int,
) -> list[dict]:
    ids = _source_segment_ids(cue, fallback_id)
    sources: list[dict] = []
    for source_id in ids:
        if 0 <= source_id < len(source_segments):
            segment = source_segments[source_id]
            if isinstance(segment, dict):
                sources.append(segment)
    return sources


def _source_signals(sources: list[dict]) -> dict:
    values: list[str] = []
    severities: list[str] = []
    for segment in sources:
        for key in (
            "alignment_quality",
            "fallback_type",
            "fallback_subtype",
            "alignment_mode",
        ):
            value = str(segment.get(key) or "").strip()
            if value:
                values.append(value)
        severity = str(segment.get("asr_qc_severity") or "").strip().lower()
        if severity:
            severities.append(severity)
            values.append(severity)
        for key in ("alignment_quality_reasons", "asr_qc_reasons"):
            for reason in segment.get(key) or []:
                value = str(reason or "").strip()
                if value:
                    values.append(value)
    joined = " ".join(values).lower()
    return {
        "nonlexical_signal": "nonlexical" in joined,
        "repetition_signal": "repeat" in joined or "repetition" in joined,
        "fallback_signal": "fallback" in joined or "proportional" in joined,
        "qc_reject": "reject" in severities,
        "qc_warn": "warn" in severities,
    }


def _text_profile(text: str, settings: dict) -> dict:
    compact = _compact_text(text)
    char_count = len(compact)
    unique_count = len(set(compact))
    kana_count = sum(1 for char in compact if _is_kana(char))
    has_cjk = any(_is_cjk(char) for char in compact)
    has_latin_or_digit = any(char.isascii() and char.isalnum() for char in compact)
    unique_ratio = unique_count / max(1, char_count)
    kana_ratio = kana_count / max(1, char_count)
    repeat = _max_unit_repeat(compact)
    char_run = _max_char_run(compact)
    repeated = (
        int(repeat["run"]) >= 2 and float(repeat["ratio"]) >= 0.5
    ) or char_run >= 3
    short_kana = (
        bool(compact)
        and char_count <= int(settings["short_kana_max_chars"])
        and kana_ratio >= 0.75
        and not has_cjk
        and not has_latin_or_digit
    )
    stable = (
        has_cjk
        or has_latin_or_digit
        or (
            char_count >= int(settings["stable_min_chars"])
            and unique_ratio >= 0.55
            and not repeated
        )
    )
    return {
        "compact": compact,
        "char_count": char_count,
        "unique_chars": unique_count,
        "unique_ratio": round(unique_ratio, 4),
        "kana_ratio": round(kana_ratio, 4),
        "has_stable_vocabulary": stable,
        "short_kana": short_kana,
        "repeat_profile": {
            "unit_run": int(repeat["run"]),
            "unit_len": int(repeat["unit_len"]),
            "ratio": float(repeat["ratio"]),
            "char_run": char_run,
            "repeated": repeated,
        },
    }


def _cue_features(
    cue: dict,
    *,
    cue_id: int,
    source_segments: list[dict],
    settings: dict,
) -> dict:
    start = float(cue.get("start", 0.0))
    end = max(start, float(cue.get("end", start)))
    text = str(cue.get("ja_text") or cue.get("text") or "").strip()
    sources = _source_segments_for_cue(cue, source_segments=source_segments, fallback_id=cue_id)
    source_signals = _source_signals(sources)
    text_profile = _text_profile(text, settings)
    low_information = (
        text_profile["char_count"] == 0
        or bool(source_signals["nonlexical_signal"])
        or bool(source_signals["repetition_signal"])
        or (
            bool(text_profile["short_kana"])
            and not bool(text_profile["has_stable_vocabulary"])
        )
        or (
            bool(text_profile["short_kana"])
            and bool(text_profile["repeat_profile"]["repeated"])
        )
    )
    return {
        "cue_id": cue_id,
        "start": start,
        "end": end,
        "duration_s": round(max(0.0, end - start), 3),
        "text": text,
        "source_segment_ids": _source_segment_ids(cue, cue_id),
        "source_signals": source_signals,
        **text_profile,
        "low_information": low_information,
    }


def _base_reasons(features: dict) -> list[str]:
    reasons: list[str] = []
    if int(features["char_count"]) == 0:
        reasons.append("empty_text")
    if bool(features["short_kana"]):
        reasons.append("short_kana")
    if bool(features["repeat_profile"]["repeated"]):
        reasons.append("repetition_profile")
    source_signals = features["source_signals"]
    if source_signals["nonlexical_signal"]:
        reasons.append("nonlexical_signal")
    if source_signals["repetition_signal"]:
        reasons.append("repetition_signal")
    if source_signals["qc_reject"]:
        reasons.append("qc_reject")
    elif source_signals["qc_warn"]:
        reasons.append("qc_warn")
    return reasons


def _same_low_information_run(left: dict, right: dict) -> bool:
    if not left["low_information"] or not right["low_information"]:
        return False
    if left["has_stable_vocabulary"] or right["has_stable_vocabulary"]:
        return False
    if left["compact"] and left["compact"] == right["compact"]:
        return True
    return (
        bool(left["short_kana"])
        and bool(right["short_kana"])
        and int(left["char_count"]) <= 2
        and int(right["char_count"]) <= 2
    )


def _summarizable_features(features: dict) -> dict:
    return {
        "duration_s": features["duration_s"],
        "char_count": features["char_count"],
        "kana_ratio": features["kana_ratio"],
        "unique_ratio": features["unique_ratio"],
        "has_stable_vocabulary": features["has_stable_vocabulary"],
        "low_information": features["low_information"],
        "repeat_profile": features["repeat_profile"],
        "source_signals": features["source_signals"],
    }


def _decision_item(
    features: dict,
    *,
    decision: str,
    display_text: str,
    reasons: list[str],
    compacted_into: int | None = None,
) -> dict:
    item = {
        "cue_id": features["cue_id"],
        "start": round(float(features["start"]), 3),
        "end": round(float(features["end"]), 3),
        "source_segment_ids": list(features["source_segment_ids"]),
        "raw_text": features["text"],
        "display_text": display_text,
        "display_decision": decision,
        "reasons": list(dict.fromkeys(reasons)),
        "features": _summarizable_features(features),
    }
    if compacted_into is not None:
        item["compacted_into"] = compacted_into
    return item


def _copy_display_cue(cue: dict, features: dict, decision: str, reasons: list[str]) -> dict:
    item = dict(cue)
    item["raw_text"] = features["text"]
    item["display_decision"] = decision
    item["display_policy_reasons"] = list(dict.fromkeys(reasons))
    item["display_policy_features"] = _summarizable_features(features)
    return item


def _compact_run(cues: list[dict], features: list[dict], start: int, end: int) -> dict:
    first = features[start]
    compacted = _copy_display_cue(
        cues[start],
        first,
        "compact",
        _base_reasons(first) + ["repeated_low_information_run"],
    )
    compacted["start"] = float(features[start]["start"])
    compacted["end"] = max(float(item["end"]) for item in features[start:end])
    compacted["raw_texts"] = [item["text"] for item in features[start:end]]
    compacted["display_policy_reasons"] = list(
        dict.fromkeys(compacted["display_policy_reasons"])
    )
    source_ids: list[int] = []
    words: list[dict] = []
    for index in range(start, end):
        source_ids.extend(int(item) for item in features[index]["source_segment_ids"])
        words.extend(dict(word) for word in (cues[index].get("words") or []) if isinstance(word, dict))
    compacted["source_segment_ids"] = list(dict.fromkeys(source_ids))
    compacted["words"] = words
    return compacted


def _summary(
    *,
    settings: dict,
    cues_before: int,
    cues_after: int,
    decisions: list[dict],
) -> dict:
    counter = Counter(str(item.get("display_decision") or "keep") for item in decisions)
    return {
        "version": DISPLAY_POLICY_VERSION,
        "mode": settings["mode"],
        "policy": "readability_priority",
        "cues_before": cues_before,
        "cues_after": cues_after,
        "counts": {decision: int(counter.get(decision, 0)) for decision in _DECISIONS},
        "settings": signature(settings),
        "decisions": decisions,
    }


def apply_display_policy(
    cues: list[dict],
    *,
    source_segments: list[dict] | None = None,
    settings: dict | None = None,
) -> tuple[list[dict], dict]:
    cfg = dict(settings or settings_from_env())
    normalized_cues = [dict(cue) for cue in cues]
    sources = [dict(segment) for segment in (source_segments or []) if isinstance(segment, dict)]
    features = [
        _cue_features(cue, cue_id=index, source_segments=sources, settings=cfg)
        for index, cue in enumerate(normalized_cues)
    ]

    if str(cfg.get("mode") or "readability").lower() == "off":
        displayed: list[dict] = []
        decisions: list[dict] = []
        for cue, item in zip(normalized_cues, features):
            reasons = ["policy_off"]
            displayed.append(_copy_display_cue(cue, item, "keep", reasons))
            decisions.append(
                _decision_item(
                    item,
                    decision="keep",
                    display_text=item["text"],
                    reasons=reasons,
                )
            )
        return displayed, _summary(
            settings=cfg,
            cues_before=len(normalized_cues),
            cues_after=len(displayed),
            decisions=decisions,
        )

    displayed = []
    decisions = []
    index = 0
    while index < len(normalized_cues):
        item = features[index]
        if int(item["char_count"]) == 0:
            reasons = _base_reasons(item)
            decisions.append(
                _decision_item(
                    item,
                    decision="drop",
                    display_text="",
                    reasons=reasons,
                )
            )
            index += 1
            continue

        run_end = index + 1
        while run_end < len(features):
            gap_s = max(0.0, float(features[run_end]["start"]) - float(features[run_end - 1]["end"]))
            if gap_s > float(cfg["max_run_gap_s"]):
                break
            if not _same_low_information_run(features[run_end - 1], features[run_end]):
                break
            run_end += 1

        run_length = run_end - index
        if run_length >= int(cfg["repeat_run_min"]):
            compacted = _compact_run(normalized_cues, features, index, run_end)
            displayed.append(compacted)
            first = features[index]
            decisions.append(
                _decision_item(
                    first,
                    decision="compact",
                    display_text=first["text"],
                    reasons=compacted["display_policy_reasons"],
                )
            )
            for drop_index in range(index + 1, run_end):
                dropped = features[drop_index]
                decisions.append(
                    _decision_item(
                        dropped,
                        decision="drop",
                        display_text="",
                        reasons=_base_reasons(dropped)
                        + ["repeated_low_information_run"],
                        compacted_into=int(first["cue_id"]),
                    )
                )
            index = run_end
            continue

        reasons = _base_reasons(item)
        decision = "keep"
        if item["low_information"] and (
            item["source_signals"]["nonlexical_signal"]
            or item["source_signals"]["repetition_signal"]
            or item["source_signals"]["qc_reject"]
            or item["source_signals"]["qc_warn"]
        ):
            decision = "review"
            reasons = reasons or ["low_information_review"]
        displayed.append(_copy_display_cue(normalized_cues[index], item, decision, reasons))
        decisions.append(
            _decision_item(
                item,
                decision=decision,
                display_text=item["text"],
                reasons=reasons,
            )
        )
        index += 1

    return displayed, _summary(
        settings=cfg,
        cues_before=len(normalized_cues),
        cues_after=len(displayed),
        decisions=decisions,
    )
