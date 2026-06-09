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

BOUNDARY_CACHE_VERSION = 4
_AUDIO_SAMPLE_BYTES = 2 * 1024 * 1024
_AUDIO_KEY_RE = re.compile(r"^[0-9a-fA-F]{8,40}$")

_BOUNDARY_BACKEND_ENV_KEYS = (
    "ASR_BOUNDARY_BACKEND",
    "ASR_CHUNK_MIN_DURATION_S",
    "SPEECH_BOUNDARY_JA_EXPORT_FRAME_SCORES",
    "SPEECH_BOUNDARY_JA_THRESHOLD",
    "SPEECH_BOUNDARY_JA_FRAME_DILATION_S",
    "SPEECH_BOUNDARY_JA_PTM",
    "SPEECH_BOUNDARY_JA_MODEL_PATH",
    "SPEECH_BOUNDARY_JA_DEVICE",
    "SPEECH_BOUNDARY_JA_DTYPE",
    "SPEECH_BOUNDARY_JA_ATTENTION",
    "SPEECH_BOUNDARY_JA_WINDOW_S",
    "SPEECH_BOUNDARY_JA_OVERLAP_S",
    "SPEECH_BOUNDARY_JA_MIN_SEGMENT_S",
    "SPEECH_BOUNDARY_JA_MAX_GROUP_S",
    "SPEECH_BOUNDARY_JA_CHUNK_THRESHOLD_S",
    "SPEECH_BOUNDARY_JA_CUT_THRESHOLD",
    "SPEECH_BOUNDARY_JA_APPLY_CUT_TO_SPEECH",
    "SPEECH_BOUNDARY_JA_EXPORT_SEQUENCE_FEATURES",
)
_BOUNDARY_ENV_KEYS = (
    "BOUNDARY_FEATURE_FRAME_HOP_S",
    "BOUNDARY_REFINER_ENABLED",
    "BOUNDARY_REFINER_MODEL_PATH",
    "BOUNDARY_REFINER_BACKBONE",
    "BOUNDARY_REFINER_DEVICE",
    "BOUNDARY_REFINER_THRESHOLD",
    "BOUNDARY_FRAME_SEQUENCE_LEFT_CONTEXT_S",
    "BOUNDARY_FRAME_SEQUENCE_RIGHT_CONTEXT_S",
    "BOUNDARY_FRAME_SEQUENCE_MAX_PTM_DIMS",
    "BOUNDARY_FRAME_SEQUENCE_INCLUDE_MFCC",
    "BOUNDARY_PLANNER_MAX_CORE_CHUNK_S",
    "BOUNDARY_PLANNER_TARGET_CHUNK_S",
    "BOUNDARY_PLANNER_MIN_CHUNK_S",
    "BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT",
    "BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE",
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
                json.dumps(payload, ensure_ascii=False, indent=2),
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
        "parent_chunk_id": chunk.parent_chunk_id,
        "island_id": chunk.island_id,
        "island_count": chunk.island_count,
        "core_start": chunk.core_start,
        "core_end": chunk.core_end,
        "internal_gap_count": int(chunk.internal_gap_count),
        "internal_gap_max_s": float(chunk.internal_gap_max_s),
        "boundary_score": (
            None if chunk.boundary_score is None else float(chunk.boundary_score)
        ),
        "boundary_reason": chunk.boundary_reason,
        "boundary_source": chunk.boundary_source,
        "boundary_decision_merge": chunk.boundary_decision_merge,
        "boundary_merge_prob": (
            None if chunk.boundary_merge_prob is None else float(chunk.boundary_merge_prob)
        ),
        "boundary_split_prob": (
            None if chunk.boundary_split_prob is None else float(chunk.boundary_split_prob)
        ),
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
        internal_gap_count=int(item.get("internal_gap_count") or 0),
        internal_gap_max_s=float(item.get("internal_gap_max_s") or 0.0),
        boundary_score=(
            None
            if item.get("boundary_score") is None
            else float(item.get("boundary_score"))
        ),
        boundary_reason=str(item.get("boundary_reason") or ""),
        boundary_source=str(item.get("boundary_source") or ""),
        boundary_decision_merge=(
            None
            if item.get("boundary_decision_merge") is None
            else bool(item.get("boundary_decision_merge"))
        ),
        boundary_merge_prob=(
            None
            if item.get("boundary_merge_prob") is None
            else float(item.get("boundary_merge_prob"))
        ),
        boundary_split_prob=(
            None
            if item.get("boundary_split_prob") is None
            else float(item.get("boundary_split_prob"))
        ),
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
    }


def _segment_from_dict(item: Any) -> SpeechSegment:
    if not isinstance(item, dict):
        raise ValueError("invalid speech segment")
    score = item.get("score")
    return SpeechSegment(
        start=float(item["start"]),
        end=float(item["end"]),
        score=None if score is None else float(score),
    )
