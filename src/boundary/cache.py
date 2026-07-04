from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import wave
from pathlib import Path
from typing import Any, Sequence

from audio.chunk_packer import PackedChunk
from boundary.base import SpeechSegment

log = logging.getLogger(__name__)

# v17 invalidates v16 artifacts after the semantic Split batching and shared
# sequence-feature matrix changes. Those changes intentionally preserve model
# semantics, but reusing a cache produced by an intermediate implementation
# makes a fresh job skip PTM entirely and prevents the new path from being
# verified.
BOUNDARY_CACHE_VERSION = 17
_AUDIO_SAMPLE_BYTES = 2 * 1024 * 1024
_AUDIO_KEY_RE = re.compile(r"^[0-9a-fA-F]{8,40}$")

_BOUNDARY_BACKEND_ENV_KEYS = (
    "ASR_BOUNDARY_BACKEND",
    "SPEECH_BOUNDARY_JA_EXPORT_FRAME_SCORES",
    "SPEECH_BOUNDARY_JA_THRESHOLD",
    "SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD",
    "SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD",
    "SPEECH_BOUNDARY_JA_FRAME_DILATION_S",
    "SPEECH_BOUNDARY_JA_PTM",
    "SPEECH_BOUNDARY_JA_MODEL_PATH",
    "SPEECH_BOUNDARY_JA_DEVICE",
    "SPEECH_BOUNDARY_JA_DTYPE",
    "SPEECH_BOUNDARY_JA_ATTENTION",
    "SPEECH_BOUNDARY_JA_WINDOW_S",
    "SPEECH_BOUNDARY_JA_OVERLAP_S",
    "SPEECH_BOUNDARY_JA_MIN_SEGMENT_S",
    "SPEECH_BOUNDARY_JA_SPLIT_SCORE_QUANTILE",
    "SPEECH_BOUNDARY_JA_SPLIT_PROMINENCE_QUANTILE",
    "SPEECH_BOUNDARY_JA_SPLIT_SMOOTH_S",
    "SPEECH_BOUNDARY_JA_SPLIT_NMS_S",
    "SPEECH_BOUNDARY_JA_SPLIT_SNAP_S",
    "SPEECH_BOUNDARY_JA_MIN_SPLIT_SEGMENT_S",
    "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO",
    "SPEECH_BOUNDARY_JA_SCORER_DEVICE",
)
_BOUNDARY_ENV_KEYS = (
    "BOUNDARY_FEATURE_FRAME_HOP_S",
    "OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
    "SEMANTIC_SPLIT_MODEL_PATH_BY_REPO",
    "CUT_EDGE_REFINER_MODEL_PATH_BY_REPO",
    "OUTER_EDGE_REFINER_DEVICE",
    "SEMANTIC_SPLIT_DEVICE",
    "CUT_EDGE_REFINER_DEVICE",
    "BOUNDARY_FRAME_SEQUENCE_LEFT_CONTEXT_S",
    "BOUNDARY_FRAME_SEQUENCE_RIGHT_CONTEXT_S",
    "BOUNDARY_FRAME_SEQUENCE_MAX_PTM_DIMS",
    "BOUNDARY_FRAME_SEQUENCE_INCLUDE_MFCC",
)

# Model checkpoint paths whose *content* must invalidate the cache when a file is
# overwritten at the same path. Keys live in either boundary_config or
# boundary_signature; they are fingerprinted by sha256 so a silent in-place model
# swap can no longer reuse a stale cache entry.
_BOUNDARY_CONFIG_CHECKPOINT_KEYS = (
    "outer_edge_refiner_model_path",
    "semantic_split_model_path",
    "cut_edge_refiner_model_path",
)
_BOUNDARY_SIGNATURE_CHECKPOINT_KEYS = (
    "model_path",
    "scorer_checkpoint",
)


def cache_enabled() -> bool:
    return os.getenv("BOUNDARY_CACHE_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def cache_root() -> Path:
    return Path(os.getenv("BOUNDARY_CACHE_DIR", Path("tmp") / "cache" / "boundary")).resolve()


def delete_for_audio_cache_key(audio_cache_key: str) -> int:
    """Delete every Boundary cache variant produced for one extracted audio."""
    key = str(audio_cache_key or "").strip().lower()
    if _AUDIO_KEY_RE.fullmatch(key) is None:
        return 0
    root = cache_root()
    if not root.is_dir():
        return 0
    removed = 0
    for path in root.glob(f"{key}.*"):
        if not path.is_file():
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            log.warning("[boundary-cache] delete failed: %s", path, exc_info=True)
    return removed


def build_cache_lookup(
    audio_path: str,
    *,
    boundary_signature: dict,
    boundary_config: dict,
) -> dict:
    audio = _audio_metadata(audio_path)
    signature = {
        "boundary_cache_version": BOUNDARY_CACHE_VERSION,
        "audio": audio,
        "boundary_backend": _jsonable(boundary_signature),
        "boundary_backend_env": _env_signature(_BOUNDARY_BACKEND_ENV_KEYS),
        "boundary_config": _jsonable(boundary_config),
        "boundary_env": _env_signature(_BOUNDARY_ENV_KEYS),
        "model_content": _model_content_fingerprint(boundary_config, boundary_signature),
    }
    digest = hashlib.sha1(_stable_json(signature).encode("utf-8")).hexdigest()[:16]
    audio_key = _safe_cache_component(str(audio["key"]))
    return {
        "path": cache_root() / f"{audio_key}.{digest}.json",
        "signature": signature,
        "audio": audio,
        "digest": digest,
    }


def load_processing_spans(
    audio_path: str,
    *,
    boundary_signature: dict,
    boundary_config: dict,
) -> tuple[list[tuple[float, float]] | list[PackedChunk], dict, dict] | None:
    if not cache_enabled():
        return None
    try:
        lookup = build_cache_lookup(
            audio_path,
            boundary_signature=boundary_signature,
            boundary_config=boundary_config,
        )
        path = Path(lookup["path"])
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("boundary_cache_version") != BOUNDARY_CACHE_VERSION:
            return None
        if _stable_json(payload.get("signature")) != _stable_json(lookup["signature"]):
            return None
        span_kind = payload.get("span_kind")
        raw_spans = payload.get("processing_spans")
        if not isinstance(raw_spans, list):
            return None
        if span_kind == "packed":
            spans = [_packed_chunk_from_dict(item) for item in raw_spans]
        elif span_kind == "spans":
            spans = [_span_from_dict(item) for item in raw_spans]
        else:
            return None
        runtime_signature = payload.get("runtime_boundary_signature")
        if not isinstance(runtime_signature, dict):
            return None
        event = {
            "status": "hit",
            "path": str(path),
            "digest": lookup["digest"],
        }
        return spans, runtime_signature, event
    except Exception as exc:
        log.warning("[boundary-cache] load failed: %s", exc)
        return None


def save_processing_spans(
    audio_path: str,
    *,
    boundary_signature: dict,
    boundary_config: dict,
    processing_spans: Sequence[tuple[float, float]] | Sequence[PackedChunk],
    runtime_boundary_signature: dict,
    speech_segments: Sequence[SpeechSegment] | None = None,
    speech_groups: Sequence[Sequence[SpeechSegment]] | None = None,
) -> dict | None:
    if not cache_enabled():
        return None
    try:
        lookup = build_cache_lookup(
            audio_path,
            boundary_signature=boundary_signature,
            boundary_config=boundary_config,
        )
        path = Path(lookup["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        span_kind = _span_kind(processing_spans)
        payload = {
            "boundary_cache_version": BOUNDARY_CACHE_VERSION,
            "created_at": time.time(),
            "signature": lookup["signature"],
            "runtime_boundary_signature": _jsonable(runtime_boundary_signature),
            "span_kind": span_kind,
            "processing_spans": _processing_spans_to_payload(processing_spans),
            "speech_segments": _segments_to_payload(speech_segments or []),
            "speech_groups": [
                _segments_to_payload(group)
                for group in (speech_groups or [])
            ],
        }
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            tmp_path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        return {
            "status": "miss",
            "path": str(path),
            "digest": lookup["digest"],
        }
    except Exception as exc:
        log.warning("[boundary-cache] save failed: %s", exc)
        return None


def _audio_metadata(audio_path: str) -> dict:
    path = Path(audio_path)
    with wave.open(str(path), "rb") as reader:
        frames = reader.getnframes()
        frame_rate = reader.getframerate()
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
    duration = frames / frame_rate if frame_rate else 0.0
    return {
        "key": _audio_cache_key(path),
        "frames": int(frames),
        "sample_rate": int(frame_rate),
        "channels": int(channels),
        "sample_width": int(sample_width),
        "duration_s": round(float(duration), 6),
        "size": path.stat().st_size,
    }


def _audio_cache_key(path: Path) -> str:
    suffix = path.stem.rsplit(".", 1)[-1]
    if _AUDIO_KEY_RE.match(suffix):
        return suffix.lower()
    return _sample_hash(path)


def _sample_hash(path: Path) -> str:
    hasher = hashlib.sha1()
    size = path.stat().st_size
    with path.open("rb") as reader:
        if size <= _AUDIO_SAMPLE_BYTES * 2:
            hasher.update(reader.read())
        else:
            hasher.update(reader.read(_AUDIO_SAMPLE_BYTES))
            reader.seek(size - _AUDIO_SAMPLE_BYTES)
            hasher.update(reader.read(_AUDIO_SAMPLE_BYTES))
    return hasher.hexdigest()[:16]


def _env_signature(names: Sequence[str]) -> dict[str, str]:
    return {name: os.getenv(name, "") for name in names if os.getenv(name, "") != ""}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_content_fingerprint(boundary_config: dict, boundary_signature: dict) -> dict[str, str]:
    """sha256 of each referenced model checkpoint file.

    The cache key historically depended only on model *paths*, so overwriting a
    checkpoint at the same path silently reused a stale cache entry. Folding the
    content hash of every checkpoint into the lookup signature makes such silent
    swaps invalidate the cache. Missing paths (e.g. test fixtures) fall back to a
    stable, path-sensitive sentinel so lookups remain deterministic.
    """
    config = _jsonable(boundary_config) if isinstance(boundary_config, dict) else {}
    signature = _jsonable(boundary_signature) if isinstance(boundary_signature, dict) else {}
    fingerprint: dict[str, str] = {}
    for key in _BOUNDARY_CONFIG_CHECKPOINT_KEYS:
        value = config.get(key)
        if isinstance(value, str) and value:
            fingerprint[key] = _checkpoint_content_token(value)
    for key in _BOUNDARY_SIGNATURE_CHECKPOINT_KEYS:
        value = signature.get(key)
        if isinstance(value, str) and value:
            fingerprint[key] = _checkpoint_content_token(value)
    return fingerprint


def _checkpoint_content_token(raw_path: str) -> str:
    try:
        path = Path(raw_path)
        if path.is_file():
            return _file_sha256(path)
    except OSError:
        pass
    return f"missing:{raw_path}"


def _stable_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_cache_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80] or "audio"


def _span_kind(
    spans: Sequence[tuple[float, float]] | Sequence[PackedChunk],
) -> str:
    if spans and isinstance(spans[0], PackedChunk):
        return "packed"
    return "spans"


def _processing_spans_to_payload(
    spans: Sequence[tuple[float, float]] | Sequence[PackedChunk],
) -> list[dict]:
    if _span_kind(spans) == "packed":
        return [_packed_chunk_to_dict(span) for span in spans]  # type: ignore[arg-type]
    return [_span_to_dict(span) for span in spans]  # type: ignore[arg-type]


def _packed_chunk_to_dict(chunk: PackedChunk) -> dict:
    return {
        "start": float(chunk.start),
        "end": float(chunk.end),
        "duration": float(chunk.duration),
        "split_reason": chunk.split_reason,
        "source_abs_start": chunk.source_abs_start,
        "source_abs_end": chunk.source_abs_end,
        "parent_chunk_id": chunk.parent_chunk_id,
        "island_id": chunk.island_id,
        "island_count": chunk.island_count,
        "core_start": chunk.core_start,
        "core_end": chunk.core_end,
        "raw_start": chunk.raw_start,
        "raw_end": chunk.raw_end,
        "raw_duration": chunk.raw_duration,
        "acoustic_start": chunk.acoustic_start,
        "acoustic_end": chunk.acoustic_end,
        "acoustic_duration": chunk.acoustic_duration,
        "internal_gap_count": int(chunk.internal_gap_count),
        "internal_gap_max_s": float(chunk.internal_gap_max_s),
        "boundary_score": (
            None if chunk.boundary_score is None else float(chunk.boundary_score)
        ),
        "boundary_reason": chunk.boundary_reason,
        "boundary_source": chunk.boundary_source,
        "boundary_start_refine_delta_s": (
            None
            if chunk.boundary_start_refine_delta_s is None
            else float(chunk.boundary_start_refine_delta_s)
        ),
        "boundary_end_refine_delta_s": (
            None
            if chunk.boundary_end_refine_delta_s is None
            else float(chunk.boundary_end_refine_delta_s)
        ),
        "boundary_decision_source": chunk.boundary_decision_source,
        "refiner_pred_start_delta_s": chunk.refiner_pred_start_delta_s,
        "refiner_pred_end_delta_s": chunk.refiner_pred_end_delta_s,
        "refiner_applied_start_delta_s": chunk.refiner_applied_start_delta_s,
        "refiner_applied_end_delta_s": chunk.refiner_applied_end_delta_s,
        "refiner_start_confidence": chunk.refiner_start_confidence,
        "refiner_end_confidence": chunk.refiner_end_confidence,
        "refiner_start_source": chunk.refiner_start_source,
        "refiner_end_source": chunk.refiner_end_source,
        "refiner_safety_action": chunk.refiner_safety_action,
        "refiner_safety_reason": chunk.refiner_safety_reason,
        "refiner_effective_start_delta_max_s": chunk.refiner_effective_start_delta_max_s,
        "refiner_effective_end_delta_max_s": chunk.refiner_effective_end_delta_max_s,
        "refiner_fallback_used": bool(chunk.refiner_fallback_used),
        "refiner_shared_boundary_adjusted": bool(chunk.refiner_shared_boundary_adjusted),
        "scorer_speech_mean": chunk.scorer_speech_mean,
        "scorer_speech_max": chunk.scorer_speech_max,
        "scorer_speech_p90": chunk.scorer_speech_p90,
        "scorer_speech_p10": chunk.scorer_speech_p10,
        "scorer_speech_p50": chunk.scorer_speech_p50,
        "scorer_speech_std": chunk.scorer_speech_std,
        "scorer_speech_active_ratio_05": chunk.scorer_speech_active_ratio_05,
        "scorer_speech_active_ratio_07": chunk.scorer_speech_active_ratio_07,
        "scorer_speech_active_ratio_09": chunk.scorer_speech_active_ratio_09,
        "scorer_split_mean": chunk.scorer_split_mean,
        "scorer_split_max": chunk.scorer_split_max,
        "scorer_split_p90": chunk.scorer_split_p90,
        "scorer_split_std": chunk.scorer_split_std,
        "subtitle_min_duration_s": chunk.subtitle_min_duration_s,
        "below_subtitle_min_duration": bool(chunk.below_subtitle_min_duration),
        "micro_chunk_candidate": bool(chunk.micro_chunk_candidate),
        "micro_resolve_action": chunk.micro_resolve_action,
        "micro_resolve_reason": chunk.micro_resolve_reason,
        "left_split_score": chunk.left_split_score,
        "right_split_score": chunk.right_split_score,
        "left_split_prominence": chunk.left_split_prominence,
        "right_split_prominence": chunk.right_split_prominence,
        "left_split_speech_valley": chunk.left_split_speech_valley,
        "right_split_speech_valley": chunk.right_split_speech_valley,
        "primary_cut_candidates": _jsonable(chunk.primary_cut_candidates or []),
        "weak_cut_candidates": _jsonable(chunk.weak_cut_candidates or []),
        "pre_asr_ptm_pooling_schema": chunk.pre_asr_ptm_pooling_schema,
        "pre_asr_ptm_pooling_bins": chunk.pre_asr_ptm_pooling_bins,
        "pre_asr_ptm_pooling_dim": chunk.pre_asr_ptm_pooling_dim,
        "pre_asr_ptm_pooled_features": _jsonable(chunk.pre_asr_ptm_pooled_features or []),
        "speech_segments": _segments_to_payload(chunk.speech_segments),
    }


def _packed_chunk_from_dict(item: Any) -> PackedChunk:
    if not isinstance(item, dict):
        raise ValueError("invalid packed chunk")
    segments = [_segment_from_dict(segment) for segment in item.get("speech_segments", [])]
    return PackedChunk(
        start=float(item["start"]),
        end=float(item["end"]),
        duration=float(item.get("duration", float(item["end"]) - float(item["start"]))),
        speech_segments=segments,
        split_reason=str(item.get("split_reason") or "unknown"),
        source_abs_start=(
            float(item["start"])
            if item.get("source_abs_start") is None
            else float(item["source_abs_start"])
        ),
        source_abs_end=(
            float(item["end"])
            if item.get("source_abs_end") is None
            else float(item["source_abs_end"])
        ),
        parent_chunk_id=(
            None
            if item.get("parent_chunk_id") is None
            else int(item.get("parent_chunk_id"))
        ),
        island_id=None if item.get("island_id") is None else int(item.get("island_id")),
        island_count=(
            None if item.get("island_count") is None else int(item.get("island_count"))
        ),
        core_start=None if item.get("core_start") is None else float(item.get("core_start")),
        core_end=None if item.get("core_end") is None else float(item.get("core_end")),
        raw_start=None if item.get("raw_start") is None else float(item.get("raw_start")),
        raw_end=None if item.get("raw_end") is None else float(item.get("raw_end")),
        raw_duration=None if item.get("raw_duration") is None else float(item.get("raw_duration")),
        acoustic_start=(
            None if item.get("acoustic_start") is None else float(item.get("acoustic_start"))
        ),
        acoustic_end=None if item.get("acoustic_end") is None else float(item.get("acoustic_end")),
        acoustic_duration=(
            None
            if item.get("acoustic_duration") is None
            else float(item.get("acoustic_duration"))
        ),
        internal_gap_count=int(item.get("internal_gap_count") or 0),
        internal_gap_max_s=float(item.get("internal_gap_max_s") or 0.0),
        boundary_score=(
            None
            if item.get("boundary_score") is None
            else float(item.get("boundary_score"))
        ),
        boundary_reason=str(item.get("boundary_reason") or ""),
        boundary_source=str(item.get("boundary_source") or ""),
        boundary_start_refine_delta_s=(
            None
            if item.get("boundary_start_refine_delta_s") is None
            else float(item.get("boundary_start_refine_delta_s"))
        ),
        boundary_end_refine_delta_s=(
            None
            if item.get("boundary_end_refine_delta_s") is None
            else float(item.get("boundary_end_refine_delta_s"))
        ),
        boundary_decision_source=str(item.get("boundary_decision_source") or ""),
        refiner_pred_start_delta_s=(
            None
            if item.get("refiner_pred_start_delta_s") is None
            else float(item.get("refiner_pred_start_delta_s"))
        ),
        refiner_pred_end_delta_s=(
            None
            if item.get("refiner_pred_end_delta_s") is None
            else float(item.get("refiner_pred_end_delta_s"))
        ),
        refiner_applied_start_delta_s=(
            None
            if item.get("refiner_applied_start_delta_s") is None
            else float(item.get("refiner_applied_start_delta_s"))
        ),
        refiner_applied_end_delta_s=(
            None
            if item.get("refiner_applied_end_delta_s") is None
            else float(item.get("refiner_applied_end_delta_s"))
        ),
        refiner_start_confidence=(
            None
            if item.get("refiner_start_confidence") is None
            else float(item.get("refiner_start_confidence"))
        ),
        refiner_end_confidence=(
            None
            if item.get("refiner_end_confidence") is None
            else float(item.get("refiner_end_confidence"))
        ),
        refiner_start_source=str(item.get("refiner_start_source") or ""),
        refiner_end_source=str(item.get("refiner_end_source") or ""),
        refiner_safety_action=str(item.get("refiner_safety_action") or ""),
        refiner_safety_reason=str(item.get("refiner_safety_reason") or ""),
        refiner_effective_start_delta_max_s=(
            None
            if item.get("refiner_effective_start_delta_max_s") is None
            else float(item.get("refiner_effective_start_delta_max_s"))
        ),
        refiner_effective_end_delta_max_s=(
            None
            if item.get("refiner_effective_end_delta_max_s") is None
            else float(item.get("refiner_effective_end_delta_max_s"))
        ),
        refiner_fallback_used=bool(item.get("refiner_fallback_used", False)),
        refiner_shared_boundary_adjusted=bool(
            item.get("refiner_shared_boundary_adjusted", False)
        ),
        scorer_speech_mean=(
            None if item.get("scorer_speech_mean") is None else float(item.get("scorer_speech_mean"))
        ),
        scorer_speech_max=(
            None if item.get("scorer_speech_max") is None else float(item.get("scorer_speech_max"))
        ),
        scorer_speech_p90=(
            None if item.get("scorer_speech_p90") is None else float(item.get("scorer_speech_p90"))
        ),
        scorer_speech_p10=(
            None if item.get("scorer_speech_p10") is None else float(item.get("scorer_speech_p10"))
        ),
        scorer_speech_p50=(
            None if item.get("scorer_speech_p50") is None else float(item.get("scorer_speech_p50"))
        ),
        scorer_speech_std=(
            None if item.get("scorer_speech_std") is None else float(item.get("scorer_speech_std"))
        ),
        scorer_speech_active_ratio_05=(
            None
            if item.get("scorer_speech_active_ratio_05") is None
            else float(item.get("scorer_speech_active_ratio_05"))
        ),
        scorer_speech_active_ratio_07=(
            None
            if item.get("scorer_speech_active_ratio_07") is None
            else float(item.get("scorer_speech_active_ratio_07"))
        ),
        scorer_speech_active_ratio_09=(
            None
            if item.get("scorer_speech_active_ratio_09") is None
            else float(item.get("scorer_speech_active_ratio_09"))
        ),
        scorer_split_mean=(
            None if item.get("scorer_split_mean") is None else float(item.get("scorer_split_mean"))
        ),
        scorer_split_max=(
            None if item.get("scorer_split_max") is None else float(item.get("scorer_split_max"))
        ),
        scorer_split_p90=(
            None if item.get("scorer_split_p90") is None else float(item.get("scorer_split_p90"))
        ),
        scorer_split_std=(
            None if item.get("scorer_split_std") is None else float(item.get("scorer_split_std"))
        ),
        subtitle_min_duration_s=(
            None
            if item.get("subtitle_min_duration_s") is None
            else float(item.get("subtitle_min_duration_s"))
        ),
        below_subtitle_min_duration=bool(item.get("below_subtitle_min_duration", False)),
        micro_chunk_candidate=bool(item.get("micro_chunk_candidate", False)),
        micro_resolve_action=str(item.get("micro_resolve_action") or ""),
        micro_resolve_reason=str(item.get("micro_resolve_reason") or ""),
        left_split_score=(
            None if item.get("left_split_score") is None else float(item.get("left_split_score"))
        ),
        right_split_score=(
            None if item.get("right_split_score") is None else float(item.get("right_split_score"))
        ),
        left_split_prominence=(
            None
            if item.get("left_split_prominence") is None
            else float(item.get("left_split_prominence"))
        ),
        right_split_prominence=(
            None
            if item.get("right_split_prominence") is None
            else float(item.get("right_split_prominence"))
        ),
        left_split_speech_valley=(
            None
            if item.get("left_split_speech_valley") is None
            else float(item.get("left_split_speech_valley"))
        ),
        right_split_speech_valley=(
            None
            if item.get("right_split_speech_valley") is None
            else float(item.get("right_split_speech_valley"))
        ),
        primary_cut_candidates=_cut_candidates_from_payload(item.get("primary_cut_candidates")),
        weak_cut_candidates=_cut_candidates_from_payload(item.get("weak_cut_candidates")),
        pre_asr_ptm_pooling_schema=str(item.get("pre_asr_ptm_pooling_schema") or ""),
        pre_asr_ptm_pooling_bins=(
            None
            if item.get("pre_asr_ptm_pooling_bins") is None
            else int(item.get("pre_asr_ptm_pooling_bins"))
        ),
        pre_asr_ptm_pooling_dim=(
            None
            if item.get("pre_asr_ptm_pooling_dim") is None
            else int(item.get("pre_asr_ptm_pooling_dim"))
        ),
        pre_asr_ptm_pooled_features=_float_list_from_payload(
            item.get("pre_asr_ptm_pooled_features")
        ),
    )


def _span_to_dict(span: tuple[float, float]) -> dict:
    return {"start": float(span[0]), "end": float(span[1])}


def _span_from_dict(item: Any) -> tuple[float, float]:
    if not isinstance(item, dict):
        raise ValueError("invalid span")
    return (float(item["start"]), float(item["end"]))


def _segments_to_payload(segments: Sequence[SpeechSegment]) -> list[dict]:
    return [_segment_to_dict(segment) for segment in segments]


def _segment_to_dict(segment: SpeechSegment) -> dict:
    return {
        "start": float(segment.start),
        "end": float(segment.end),
        "score": None if segment.score is None else float(segment.score),
        "subtitle_min_duration_s": segment.subtitle_min_duration_s,
        "below_subtitle_min_duration": bool(segment.below_subtitle_min_duration),
        "micro_chunk_candidate": bool(segment.micro_chunk_candidate),
        "micro_resolve_action": segment.micro_resolve_action,
        "micro_resolve_reason": segment.micro_resolve_reason,
        "left_split_score": segment.left_split_score,
        "right_split_score": segment.right_split_score,
        "left_split_prominence": segment.left_split_prominence,
        "right_split_prominence": segment.right_split_prominence,
        "left_split_speech_valley": segment.left_split_speech_valley,
        "right_split_speech_valley": segment.right_split_speech_valley,
        "primary_cut_candidates": _jsonable(segment.primary_cut_candidates or []),
        "weak_cut_candidates": _jsonable(segment.weak_cut_candidates or []),
    }


def _segment_from_dict(item: Any) -> SpeechSegment:
    if not isinstance(item, dict):
        raise ValueError("invalid speech segment")
    score = item.get("score")
    return SpeechSegment(
        start=float(item["start"]),
        end=float(item["end"]),
        score=None if score is None else float(score),
        subtitle_min_duration_s=(
            None
            if item.get("subtitle_min_duration_s") is None
            else float(item.get("subtitle_min_duration_s"))
        ),
        below_subtitle_min_duration=bool(item.get("below_subtitle_min_duration", False)),
        micro_chunk_candidate=bool(item.get("micro_chunk_candidate", False)),
        micro_resolve_action=str(item.get("micro_resolve_action") or ""),
        micro_resolve_reason=str(item.get("micro_resolve_reason") or ""),
        left_split_score=(
            None if item.get("left_split_score") is None else float(item.get("left_split_score"))
        ),
        right_split_score=(
            None if item.get("right_split_score") is None else float(item.get("right_split_score"))
        ),
        left_split_prominence=(
            None
            if item.get("left_split_prominence") is None
            else float(item.get("left_split_prominence"))
        ),
        right_split_prominence=(
            None
            if item.get("right_split_prominence") is None
            else float(item.get("right_split_prominence"))
        ),
        left_split_speech_valley=(
            None
            if item.get("left_split_speech_valley") is None
            else float(item.get("left_split_speech_valley"))
        ),
        right_split_speech_valley=(
            None
            if item.get("right_split_speech_valley") is None
            else float(item.get("right_split_speech_valley"))
        ),
        primary_cut_candidates=_cut_candidates_from_payload(item.get("primary_cut_candidates")),
        weak_cut_candidates=_cut_candidates_from_payload(item.get("weak_cut_candidates")),
    )


def _cut_candidates_from_payload(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    candidates: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            candidate = {
                "kind": str(item.get("kind") or ""),
                "time_s": float(item["time_s"]),
                "frame": int(item["frame"]),
                "score": float(item.get("score") or 0.0),
                "prominence": float(item.get("prominence") or 0.0),
                "speech_valley": float(item.get("speech_valley") or 0.0),
                "strength": float(item.get("strength") or 0.0),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if item.get("downgraded_from") is not None:
            candidate["downgraded_from"] = str(item.get("downgraded_from") or "")
        for key in ("proposal_time_s", "p_cut", "p_continue", "p_unsure"):
            if item.get(key) is not None:
                candidate[key] = float(item[key])
        if item.get("label") is not None:
            candidate["label"] = str(item.get("label") or "")
        if item.get("shared_absolute_timestamp") is not None:
            candidate["shared_absolute_timestamp"] = bool(item["shared_absolute_timestamp"])
        candidates.append(candidate)
    return candidates


def _float_list_from_payload(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            continue
        out.append(parsed)
    return out
