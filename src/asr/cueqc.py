from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from asr.qc import evaluate_asr_chunk_qc


CUEQC_FEATURE_SCHEMA_VERSION = 1
CUEQC_SHADOW_SCHEMA_VERSION = 1
CUEQC_TAXONOMY_VERSION = "cluster_first_v1"
CUEQC_MODEL_VERSION = "rules_shadow_v1"
DEFAULT_CLUSTER_COUNT_MIN = 12
DEFAULT_CLUSTER_COUNT_MAX = 20

_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_COMPACT_TEXT_RE = re.compile(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u9fff]+")
_PUNCT_RE = re.compile(
    r"[\s\u3000\u3001\u3002\uff01\uff1f\uff0c\uff0e,.!?~\u301c"
    r"\u30fc\u2010-\u2015\u300c\u300d\u300e\u300f()\[\]{}]+"
)
_KANA_RE = re.compile(r"^[\u3040-\u30ff]+$")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in _FALSE_VALUES:
        return False
    if value in _TRUE_VALUES:
        return True
    return default


def _env_text(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return round(float(value), digits)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _compact_text(value: Any) -> str:
    return _COMPACT_TEXT_RE.sub("", str(value or ""))


def _strip_punctuation(value: Any) -> str:
    return _PUNCT_RE.sub("", str(value or ""))


def _text_sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def _audio_ref(path_value: Any) -> dict[str, Any]:
    text = str(path_value or "").strip()
    if not text:
        return {"path": "", "exists": False, "sha1": ""}
    path = Path(text)
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    digest = ""
    exists = resolved.exists()
    if exists and resolved.is_file():
        try:
            hasher = hashlib.sha1()
            with resolved.open("rb") as handle:
                hasher.update(handle.read(1024 * 1024))
            digest = hasher.hexdigest()
        except Exception:
            digest = ""
    return {
        "path": str(resolved),
        "exists": bool(exists),
        "sha1": digest,
    }


def _chunk_duration(chunk: Mapping[str, Any], text_result: Mapping[str, Any]) -> float:
    duration = _optional_float(text_result.get("duration"))
    if duration is None:
        duration = _optional_float(chunk.get("duration"))
    if duration is None:
        duration = max(0.0, _safe_float(chunk.get("end")) - _safe_float(chunk.get("start")))
    return max(0.0, duration)


def _max_unit_repeat(compact: str) -> dict[str, Any]:
    if not compact:
        return {"unit": "", "unit_len": 0, "run": 0, "chars": 0, "ratio": 0.0}
    best = {"unit": "", "unit_len": 0, "run": 1, "chars": 0, "ratio": 0.0}
    max_unit_len = min(12, max(1, len(compact) // 2))
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
            repeated_chars = unit_len * run
            ratio = repeated_chars / max(1, len(compact))
            if run > 1 and (run, ratio, repeated_chars) > (
                int(best["run"]),
                float(best["ratio"]),
                int(best["chars"]),
            ):
                best = {
                    "unit": unit,
                    "unit_len": unit_len,
                    "run": run,
                    "chars": repeated_chars,
                    "ratio": round(ratio, 4),
                }
            index += max(1, repeated_chars if run > 1 else 1)
    return best


def _char_class_counts(compact: str) -> dict[str, int]:
    return {
        "chars": len(compact),
        "unique_chars": len(set(compact)),
        "kana_chars": sum(1 for char in compact if "\u3040" <= char <= "\u30ff"),
        "kanji_chars": sum(1 for char in compact if "\u3400" <= char <= "\u9fff"),
        "latin_digit_chars": sum(1 for char in compact if char.isascii() and char.isalnum()),
    }


def text_features(raw_text: Any, text: Any, *, duration_s: float) -> dict[str, Any]:
    raw = _clean_text(raw_text)
    display = _clean_text(text or raw)
    compact = _strip_punctuation(display)
    context_compact = _compact_text(display)
    counts = _char_class_counts(compact)
    repeat = _max_unit_repeat(context_compact)
    chars = int(counts["chars"])
    unique_chars = int(counts["unique_chars"])
    kana_ratio = int(counts["kana_chars"]) / max(1, chars)
    kanji_ratio = int(counts["kanji_chars"]) / max(1, chars)
    unique_ratio = unique_chars / max(1, chars)
    has_stable_vocabulary = (
        int(counts["kanji_chars"]) > 0
        or int(counts["latin_digit_chars"]) > 0
        or (chars >= 5 and unique_ratio >= 0.55 and int(repeat["run"]) <= 2)
    )
    kana_only = bool(compact) and bool(_KANA_RE.fullmatch(compact))
    return {
        "raw_text": raw,
        "text": display,
        "text_sha1": _text_sha1(display),
        "raw_text_sha1": _text_sha1(raw),
        "compact_text": compact,
        "context_compact_text": context_compact,
        "char_count": chars,
        "raw_char_count": len(_strip_punctuation(raw)),
        "unique_chars": unique_chars,
        "unique_ratio": round(unique_ratio, 4),
        "kana_ratio": round(kana_ratio, 4),
        "kanji_ratio": round(kanji_ratio, 4),
        "kana_only": kana_only,
        "has_kanji": int(counts["kanji_chars"]) > 0,
        "has_latin_or_digit": int(counts["latin_digit_chars"]) > 0,
        "has_stable_vocabulary": has_stable_vocabulary,
        "chars_per_sec": round(chars / duration_s, 4) if duration_s > 0 else 0.0,
        "repeat_profile": repeat,
    }


def _neighbor_features(
    index: int,
    chunks: list[Mapping[str, Any]],
    text_results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    chunk = chunks[index]
    start = _safe_float(chunk.get("start"))
    end = _safe_float(chunk.get("end"), start)
    prev_gap = None
    next_gap = None
    prev_text_same = False
    next_text_same = False
    current_text = _compact_text(
        str(text_results[index].get("text") or text_results[index].get("raw_text") or "")
    )
    if index > 0:
        prev = chunks[index - 1]
        prev_gap = max(0.0, start - _safe_float(prev.get("end"), start))
        prev_text = _compact_text(
            str(text_results[index - 1].get("text") or text_results[index - 1].get("raw_text") or "")
        )
        prev_text_same = bool(current_text and current_text == prev_text)
    if index + 1 < len(chunks):
        nxt = chunks[index + 1]
        next_gap = max(0.0, _safe_float(nxt.get("start"), end) - end)
        next_text = _compact_text(
            str(text_results[index + 1].get("text") or text_results[index + 1].get("raw_text") or "")
        )
        next_text_same = bool(current_text and current_text == next_text)
    return {
        "prev_gap_s": _round(prev_gap, 3),
        "next_gap_s": _round(next_gap, 3),
        "prev_text_same": prev_text_same,
        "next_text_same": next_text_same,
    }


def _run_length_features(
    index: int,
    chunks: list[Mapping[str, Any]],
    text_results: list[Mapping[str, Any]],
    *,
    max_gap_s: float = 1.0,
) -> dict[str, Any]:
    compact_values = [
        _compact_text(str(result.get("text") or result.get("raw_text") or ""))
        for result in text_results
    ]
    current = compact_values[index] if 0 <= index < len(compact_values) else ""
    if not current:
        return {"same_text_run_length": 0, "same_text_run_position": 0}
    left = index
    while left > 0:
        gap = max(
            0.0,
            _safe_float(chunks[left].get("start")) - _safe_float(chunks[left - 1].get("end")),
        )
        if gap > max_gap_s or compact_values[left - 1] != current:
            break
        left -= 1
    right = index
    while right + 1 < len(chunks):
        gap = max(
            0.0,
            _safe_float(chunks[right + 1].get("start")) - _safe_float(chunks[right].get("end")),
        )
        if gap > max_gap_s or compact_values[right + 1] != current:
            break
        right += 1
    return {
        "same_text_run_length": right - left + 1,
        "same_text_run_position": index - left,
    }


def _qc_item_by_chunk(qc_report: Mapping[str, Any] | None) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(qc_report, Mapping):
        return out
    for item in qc_report.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        try:
            out[int(item.get("chunk_index"))] = dict(item)
        except (TypeError, ValueError):
            continue
    return out


def build_candidate(
    *,
    chunk: Mapping[str, Any],
    text_result: Mapping[str, Any],
    position: int,
    chunks: list[Mapping[str, Any]],
    text_results: list[Mapping[str, Any]],
    qc_item: Mapping[str, Any] | None = None,
    audio_id: str = "",
    video_id: str = "",
) -> dict[str, Any]:
    chunk_index = _safe_int(chunk.get("index"), position)
    start = _safe_float(chunk.get("start"))
    end = max(start, _safe_float(chunk.get("end"), start))
    duration = _chunk_duration(chunk, text_result)
    raw_text = str(text_result.get("raw_text") or "")
    text = str(text_result.get("text") or raw_text)
    features = text_features(raw_text, text, duration_s=duration)
    qc_payload: dict[str, Any]
    if isinstance(qc_item, Mapping):
        qc_payload = dict(qc_item)
    else:
        qc_payload = evaluate_asr_chunk_qc(dict(chunk), dict(text_result))
    metrics = qc_payload.get("metrics") if isinstance(qc_payload.get("metrics"), Mapping) else {}
    text_density = metrics.get("text_density") if isinstance(metrics.get("text_density"), Mapping) else {}
    vocalization = (
        metrics.get("vocalization_repetition")
        if isinstance(metrics.get("vocalization_repetition"), Mapping)
        else {}
    )
    signal_quality = (
        metrics.get("signal_quality")
        if isinstance(metrics.get("signal_quality"), Mapping)
        else {}
    )
    sample_id = f"cueqc-{video_id or audio_id or 'audio'}-chunk{chunk_index:05d}"
    return {
        "schema": "cueqc_candidate_v1",
        "schema_version": CUEQC_FEATURE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "audio_id": audio_id,
        "video_id": video_id,
        "chunk_index": chunk_index,
        "position": position,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration_s": round(duration, 3),
        "audio": _audio_ref(text_result.get("normalized_path") or chunk.get("path")),
        "source_audio_path": str(chunk.get("source_audio_path") or ""),
        "text": features["text"],
        "raw_text": features["raw_text"],
        "text_features": {
            key: value
            for key, value in features.items()
            if key not in {"text", "raw_text", "compact_text", "context_compact_text"}
        },
        "text_preview": features["text"][:120],
        "compact_text": features["compact_text"],
        "boundary": {
            key: chunk.get(key)
            for key in (
                "speech_segment_count",
                "boundary_split_reason",
                "boundary_parent_chunk_id",
                "speech_island_id",
                "speech_island_count",
                "speech_internal_gap_count",
                "speech_internal_gap_max_s",
                "boundary_score",
                "boundary_reason",
                "boundary_source",
                "boundary_start_refine_delta_s",
                "boundary_end_refine_delta_s",
                "boundary_decision_source",
            )
            if key in chunk
        },
        "adjacency": {
            **_neighbor_features(position, chunks, text_results),
            **_run_length_features(position, chunks, text_results),
        },
        "qc": {
            "severity": str(qc_payload.get("severity") or "ok"),
            "reasons": list(qc_payload.get("reasons") or []),
            "ok": bool(qc_payload.get("ok", True)),
            "text_density": dict(text_density),
            "vocalization_repetition": dict(vocalization),
            "signal_quality": dict(signal_quality),
            "max_repeat": dict(metrics.get("max_repeat") or {}),
            "repetition_repair": dict(metrics.get("repetition_repair") or {}),
            "low_value": bool(metrics.get("low_value")),
            "mojibake": bool(metrics.get("mojibake")),
            "chars_per_sec": metrics.get("chars_per_sec"),
            "compact_chars": metrics.get("compact_chars"),
        },
        "asr_signals": {
            "avg_logprob": text_result.get("avg_logprob"),
            "no_speech_prob": text_result.get("no_speech_prob"),
            "compression_ratio": text_result.get("compression_ratio"),
            "language": text_result.get("language", ""),
            "generation": dict(text_result.get("asr_generation") or {})
            if isinstance(text_result.get("asr_generation"), Mapping)
            else {},
        },
        "alignment_diagnostics": {
            "fallback_window_start_s": text_result.get("alignment_fallback_start_s"),
            "fallback_window_end_s": text_result.get("alignment_fallback_end_s"),
            "fallback_window_source": text_result.get("alignment_fallback_source", ""),
        },
        "existing_diagnostics": {
            "qc_item": dict(qc_payload),
        },
        "labels": {},
    }


def build_candidates(
    chunks: list[Mapping[str, Any]],
    text_results: list[Mapping[str, Any]],
    *,
    qc_report: Mapping[str, Any] | None = None,
    audio_id: str = "",
    video_id: str = "",
) -> list[dict[str, Any]]:
    qc_by_chunk = _qc_item_by_chunk(qc_report)
    candidates: list[dict[str, Any]] = []
    for position, (chunk, text_result) in enumerate(zip(chunks, text_results)):
        chunk_index = _safe_int(chunk.get("index"), position)
        candidates.append(
            build_candidate(
                chunk=chunk,
                text_result=text_result,
                position=position,
                chunks=chunks,
                text_results=text_results,
                qc_item=qc_by_chunk.get(chunk_index),
                audio_id=audio_id,
                video_id=video_id,
            )
        )
    return candidates


def heuristic_shadow_decision(candidate: Mapping[str, Any]) -> dict[str, Any]:
    qc = candidate.get("qc") if isinstance(candidate.get("qc"), Mapping) else {}
    text_features_payload = (
        candidate.get("text_features")
        if isinstance(candidate.get("text_features"), Mapping)
        else {}
    )
    adjacency = candidate.get("adjacency") if isinstance(candidate.get("adjacency"), Mapping) else {}
    reasons: list[str] = []
    content_type = "dialogue"
    display_hint = "keep"
    qc_decision = "keep"
    alignment_policy = "align"
    confidence = 0.55

    severity = str(qc.get("severity") or "ok")
    qc_reasons = [str(item) for item in qc.get("reasons") or []]
    text_density = qc.get("text_density") if isinstance(qc.get("text_density"), Mapping) else {}
    density_level = str(text_density.get("level") or "")
    repeat_profile = (
        text_features_payload.get("repeat_profile")
        if isinstance(text_features_payload.get("repeat_profile"), Mapping)
        else {}
    )
    char_count = _safe_int(text_features_payload.get("char_count"))
    has_stable = bool(text_features_payload.get("has_stable_vocabulary"))
    same_run_length = _safe_int(adjacency.get("same_text_run_length"))
    if density_level in {
        "empty_or_punctuation",
        "short_vocalization_candidate",
        "repeated_vocalization_candidate",
        "long_sparse_text",
    }:
        content_type = "non_dialogue"
        display_hint = "review"
        qc_decision = "review"
        reasons.append(f"text_density:{density_level}")
        confidence = max(confidence, 0.62)
    elif severity in {"warn", "reject"}:
        content_type = "uncertain"
        display_hint = "review"
        qc_decision = "review"
        reasons.append(f"asr_qc:{severity}")
    if "repeated_nonlexical_vocalization" in qc_reasons:
        content_type = "non_dialogue"
        display_hint = "compact" if same_run_length >= 3 else "review"
        qc_decision = "review"
        reasons.append("repeated_nonlexical_vocalization")
        confidence = max(confidence, 0.68)
    if same_run_length >= 3 and not has_stable and char_count <= 4:
        content_type = "non_dialogue"
        display_hint = "compact"
        qc_decision = "review"
        reasons.append("repeated_low_information_run")
        confidence = max(confidence, 0.7)
    if severity == "reject":
        qc_decision = "review"
        display_hint = "review"
        reasons.extend(qc_reasons)
        confidence = max(confidence, 0.72)
    if char_count == 0:
        content_type = "non_dialogue"
        display_hint = "drop"
        qc_decision = "review"
        reasons.append("empty_text")
        confidence = max(confidence, 0.8)
    if has_stable and severity == "ok":
        content_type = "dialogue"
        display_hint = "keep"
        qc_decision = "keep"
        reasons = [reason for reason in reasons if reason.startswith("cluster:")]
        confidence = max(confidence, 0.78)
    if int(repeat_profile.get("run") or 0) >= 4 and not has_stable:
        content_type = "non_dialogue"
        display_hint = "compact" if same_run_length >= 2 else "review"
        qc_decision = "review"
        reasons.append("repeat_profile")
        confidence = max(confidence, 0.68)

    return {
        "schema": "cueqc_shadow_v1",
        "schema_version": CUEQC_SHADOW_SCHEMA_VERSION,
        "model_version": CUEQC_MODEL_VERSION,
        "taxonomy_version": CUEQC_TAXONOMY_VERSION,
        "mode": "shadow",
        "qc_decision": qc_decision,
        "content_type": content_type,
        "cluster_id": candidate.get("cluster_id", "unclustered"),
        "alignment_policy": alignment_policy,
        "display_hint": display_hint,
        "confidence": round(min(0.99, confidence), 4),
        "reasons": list(dict.fromkeys(reasons or ["shadow_conservative"])),
    }


def build_shadow_report(candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    decisions = []
    for candidate in candidates:
        decision = heuristic_shadow_decision(candidate)
        decisions.append(
            {
                "sample_id": candidate.get("sample_id", ""),
                "chunk_index": candidate.get("chunk_index"),
                "start": candidate.get("start"),
                "end": candidate.get("end"),
                "text_preview": candidate.get("text_preview", ""),
                **decision,
            }
        )
    return {
        "schema": "cueqc_shadow_report_v1",
        "enabled": cueqc_enabled(),
        "shadow_only": True,
        "feature_schema_version": CUEQC_FEATURE_SCHEMA_VERSION,
        "taxonomy_version": CUEQC_TAXONOMY_VERSION,
        "model_version": CUEQC_MODEL_VERSION,
        "candidate_count": len(candidates),
        "decisions": decisions,
        "counts": {
            "qc_decision": dict(Counter(str(item.get("qc_decision") or "") for item in decisions)),
            "content_type": dict(Counter(str(item.get("content_type") or "") for item in decisions)),
            "alignment_policy": dict(Counter(str(item.get("alignment_policy") or "") for item in decisions)),
            "display_hint": dict(Counter(str(item.get("display_hint") or "") for item in decisions)),
        },
    }


def cueqc_enabled() -> bool:
    return _env_bool("CUEQC_SHADOW_ENABLED", True)


def runtime_signature() -> dict[str, Any]:
    checkpoint = _env_text("CUEQC_MODEL_PATH", "")
    checkpoint_hash = ""
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if path.exists() and path.is_file():
            try:
                hasher = hashlib.sha1()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        hasher.update(block)
                checkpoint_hash = hasher.hexdigest()
            except Exception:
                checkpoint_hash = ""
    return {
        "schema_version": CUEQC_SHADOW_SCHEMA_VERSION,
        "feature_schema_version": CUEQC_FEATURE_SCHEMA_VERSION,
        "enabled": cueqc_enabled(),
        "shadow_only": True,
        "policy": "cluster_first_shadow",
        "taxonomy_version": _env_text("CUEQC_TAXONOMY_VERSION", CUEQC_TAXONOMY_VERSION)
        or CUEQC_TAXONOMY_VERSION,
        "model_version": _env_text("CUEQC_MODEL_VERSION", CUEQC_MODEL_VERSION)
        or CUEQC_MODEL_VERSION,
        "model_path": checkpoint,
        "checkpoint_sha1": checkpoint_hash,
        "prealign_skip_enabled": _env_bool("CUEQC_PREALIGN_SKIP_ENABLED", False),
        "prealign_skip_min_confidence": _env_text(
            "CUEQC_PREALIGN_SKIP_MIN_CONFIDENCE",
            "0.92",
        ),
        "shadow_embed_candidates": _env_bool("CUEQC_SHADOW_EMBED_CANDIDATES", False),
    }


def numeric_feature_vector(candidate: Mapping[str, Any]) -> list[float]:
    text_features_payload = (
        candidate.get("text_features")
        if isinstance(candidate.get("text_features"), Mapping)
        else {}
    )
    qc = candidate.get("qc") if isinstance(candidate.get("qc"), Mapping) else {}
    adjacency = candidate.get("adjacency") if isinstance(candidate.get("adjacency"), Mapping) else {}
    repeat = (
        text_features_payload.get("repeat_profile")
        if isinstance(text_features_payload.get("repeat_profile"), Mapping)
        else {}
    )
    density = qc.get("text_density") if isinstance(qc.get("text_density"), Mapping) else {}
    vocal = (
        qc.get("vocalization_repetition")
        if isinstance(qc.get("vocalization_repetition"), Mapping)
        else {}
    )
    severity = str(qc.get("severity") or "ok")
    severity_num = {"ok": 0.0, "warn": 0.5, "reject": 1.0}.get(severity, 0.25)
    density_level = str(density.get("level") or "")
    density_num = {
        "normal_dialogue": 0.0,
        "short_kana_dialogue_candidate": 0.25,
        "short_vocalization_candidate": 0.5,
        "repeated_vocalization_candidate": 0.7,
        "long_sparse_text": 0.8,
        "empty_or_punctuation": 1.0,
    }.get(density_level, 0.25)
    return [
        _safe_float(candidate.get("duration_s")),
        _safe_float(text_features_payload.get("char_count")),
        _safe_float(text_features_payload.get("unique_chars")),
        _safe_float(text_features_payload.get("unique_ratio")),
        _safe_float(text_features_payload.get("kana_ratio")),
        _safe_float(text_features_payload.get("kanji_ratio")),
        _safe_float(text_features_payload.get("chars_per_sec")),
        _safe_float(repeat.get("run")),
        _safe_float(repeat.get("unit_len")),
        _safe_float(repeat.get("ratio")),
        _safe_float(adjacency.get("prev_gap_s"), 5.0),
        _safe_float(adjacency.get("next_gap_s"), 5.0),
        _safe_float(adjacency.get("same_text_run_length")),
        severity_num,
        density_num,
        1.0 if bool(vocal.get("preserve_candidate")) else 0.0,
        1.0 if bool(text_features_payload.get("has_stable_vocabulary")) else 0.0,
    ]


def normalize_feature_matrix(rows: Iterable[Mapping[str, Any]]) -> list[list[float]]:
    matrix = [numeric_feature_vector(row) for row in rows]
    if not matrix:
        return []
    width = len(matrix[0])
    means: list[float] = []
    stds: list[float] = []
    for col in range(width):
        values = [row[col] for row in matrix]
        mean = sum(values) / max(1, len(values))
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        means.append(mean)
        stds.append(math.sqrt(variance) or 1.0)
    return [
        [(row[col] - means[col]) / stds[col] for col in range(width)]
        for row in matrix
    ]
