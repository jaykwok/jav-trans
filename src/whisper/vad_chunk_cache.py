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
from vad.base import SpeechSegment

log = logging.getLogger(__name__)

_VERSION = 5
_AUDIO_SAMPLE_BYTES = 2 * 1024 * 1024
_AUDIO_KEY_RE = re.compile(r"^[0-9a-fA-F]{8,40}$")

_VAD_ENV_KEYS = (
    "ASR_VAD_BACKEND",
    "ASR_VAD_PRIMARY",
    "ASR_VAD_GATE",
    "ASR_VAD_ADAPTIVE",
    "WHISPERSEG_THRESHOLD",
    "WHISPERSEG_MIN_SPEECH_MS",
    "WHISPERSEG_MIN_SILENCE_MS",
    "WHISPERSEG_PAD_MS",
    "WHISPERSEG_MAX_SPEECH_S",
    "WHISPERSEG_MAX_GROUP_S",
    "WHISPERSEG_CHUNK_THRESHOLD_S",
    "WHISPERSEG_FORCE_CPU",
    "WHISPERSEG_NEG_THRESHOLD_OFFSET",
    "SILERO_VAD_THRESHOLD",
    "SILERO_VAD_MIN_SPEECH_MS",
    "SILERO_VAD_MIN_SILENCE_MS",
    "SILERO_VAD_PAD_MS",
    "SILERO_VAD_MAX_SPEECH_S",
    "SILERO_VAD_MAX_GROUP_S",
    "SILERO_VAD_CHUNK_THRESHOLD_S",
    "SILERO_VAD_ONNX",
    "FUSION_VAD_PRIMARY_WEIGHT",
    "FUSION_VAD_GATE_WEIGHT",
    "FUSION_VAD_RMS_WEIGHT",
    "FUSION_VAD_SPECTRAL_FLUX_WEIGHT",
    "FUSION_VAD_DURATION_WEIGHT",
    "FUSION_VAD_MIN_SCORE",
    "FUSION_VAD_MIN_GATE_OVERLAP_RATIO",
    "FUSION_VAD_GATE_PAD_S",
    "FUSION_VAD_DEFAULT_PRIMARY_SCORE",
    "FUSION_VAD_RMS_FLOOR_DBFS",
    "FUSION_VAD_RMS_FULL_DBFS",
    "FUSION_VAD_SPECTRAL_FLUX_FLOOR",
    "FUSION_VAD_SPECTRAL_FLUX_FULL",
    "FUSION_VAD_DURATION_MIN_S",
    "FUSION_VAD_DURATION_FULL_S",
    "FUSION_VAD_DURATION_MAX_S",
    "VAD_MIN_OFF",
    "VAD_PAD",
    "SEGMENT_MIN_SILENCE",
    "SEGMENT_MIN_CHUNK",
    "SEGMENT_MAX_CHUNK",
    "SEGMENT_TARGET_CHUNK",
    "SEGMENT_MIN_SPEECH",
    "SEGMENT_PAD",
    "FUSIONVAD_JA_EXPORT_FRAME_SCORES",
    "FUSIONVAD_JA_CHECKPOINT",
    "FUSIONVAD_JA_THRESHOLD",
    "FUSIONVAD_JA_PAD_S",
    "FUSIONVAD_JA_PTM",
    "FUSIONVAD_JA_MODEL_PATH",
    "FUSIONVAD_JA_DEVICE",
    "FUSIONVAD_JA_DTYPE",
    "FUSIONVAD_JA_ATTENTION",
    "FUSIONVAD_JA_WINDOW_S",
    "FUSIONVAD_JA_OVERLAP_S",
    "FUSIONVAD_JA_MIN_SEGMENT_S",
    "FUSIONVAD_JA_MERGE_GAP_S",
    "FUSIONVAD_JA_MAX_GROUP_S",
    "FUSIONVAD_JA_CHUNK_THRESHOLD_S",
    "FUSIONVAD_JA_CUT_THRESHOLD",
    "FUSIONVAD_JA_APPLY_CUT_TO_SPEECH",
)
_CHUNK_ENV_KEYS = (
    "ASR_CHUNK_PACKING_ENABLED",
    "ASR_CHUNK_PACK_FRAME_HOP_S",
    "ASR_CHUNK_PACK_WINDOW_FRAMES",
    "ASR_CHUNK_PACK_RESERVE_FRAMES",
    "ASR_CHUNK_PACK_TARGET_PADDING_FRAMES",
    "ASR_CHUNK_PACK_GAP_MERGE_FRAMES",
    "ASR_CHUNK_PACK_MAX_CORE_FRAMES",
    "ASR_PRE_ASR_ISLAND_SPLIT_ENABLED",
    "ASR_PRE_ASR_ISLAND_SPLIT_MIN_CORE_FRAMES",
    "ASR_PRE_ASR_ISLAND_SPLIT_MIN_GAP_FRAMES",
    "ASR_PRE_ASR_ISLAND_SPLIT_MIN_ISLAND_FRAMES",
    "ASR_PRE_ASR_ISLAND_SPLIT_MAX_CHILDREN",
    "ASR_PRE_ASR_VALLEY_SPLIT_ENABLED",
    "ASR_PRE_ASR_VALLEY_SPLIT_MIN_CORE_FRAMES",
    "ASR_PRE_ASR_VALLEY_SPLIT_TARGET_CORE_FRAMES",
    "ASR_PRE_ASR_VALLEY_SPLIT_MIN_VALLEY_FRAMES",
    "ASR_PRE_ASR_VALLEY_SPLIT_MIN_CHILD_FRAMES",
    "ASR_PRE_ASR_VALLEY_SPLIT_MAX_CHILDREN",
    "ASR_PRE_ASR_VALLEY_SPLIT_THRESHOLD",
    "ASR_CHUNK_DROP_ENABLED",
    "ASR_CHUNK_DROP_MIN_DURATION_S",
    "ASR_CHUNK_DROP_RMS_DBFS",
    "VAD_MERGE_SHORT_MAX_S",
    "VAD_MERGE_GAP_MAX_S",
)


def cache_enabled() -> bool:
    return os.getenv("VAD_CHUNK_CACHE_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def cache_root() -> Path:
    return Path(os.getenv("VAD_CHUNK_CACHE_DIR", Path("temp") / "vad-cache")).resolve()


def build_cache_lookup(
    audio_path: str,
    *,
    vad_signature: dict,
    chunk_config: dict,
) -> dict:
    audio = _audio_metadata(audio_path)
    signature = {
        "version": _VERSION,
        "audio": audio,
        "vad": _jsonable(vad_signature),
        "vad_env": _env_signature(_VAD_ENV_KEYS),
        "chunk": _jsonable(chunk_config),
        "chunk_env": _env_signature(_CHUNK_ENV_KEYS),
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
    vad_signature: dict,
    chunk_config: dict,
) -> tuple[list[tuple[float, float]] | list[PackedChunk], dict, dict] | None:
    if not cache_enabled():
        return None
    try:
        lookup = build_cache_lookup(
            audio_path,
            vad_signature=vad_signature,
            chunk_config=chunk_config,
        )
        path = Path(lookup["path"])
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != _VERSION:
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
        runtime_signature = payload.get("runtime_vad_signature")
        if not isinstance(runtime_signature, dict):
            return None
        event = {
            "status": "hit",
            "path": str(path),
            "digest": lookup["digest"],
        }
        return spans, runtime_signature, event
    except Exception as exc:
        log.warning("[vad-cache] load failed: %s", exc)
        return None


def save_processing_spans(
    audio_path: str,
    *,
    vad_signature: dict,
    chunk_config: dict,
    processing_spans: Sequence[tuple[float, float]] | Sequence[PackedChunk],
    runtime_vad_signature: dict,
    vad_segments: Sequence[SpeechSegment] | None = None,
    vad_groups: Sequence[Sequence[SpeechSegment]] | None = None,
) -> dict | None:
    if not cache_enabled():
        return None
    try:
        lookup = build_cache_lookup(
            audio_path,
            vad_signature=vad_signature,
            chunk_config=chunk_config,
        )
        path = Path(lookup["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        span_kind = _span_kind(processing_spans)
        payload = {
            "version": _VERSION,
            "created_at": time.time(),
            "signature": lookup["signature"],
            "runtime_vad_signature": _jsonable(runtime_vad_signature),
            "span_kind": span_kind,
            "processing_spans": _processing_spans_to_payload(processing_spans),
            "vad_segments": _segments_to_payload(vad_segments or []),
            "vad_groups": [
                _segments_to_payload(group)
                for group in (vad_groups or [])
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
        log.warning("[vad-cache] save failed: %s", exc)
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
        "left_padding_s": float(chunk.left_padding_s),
        "right_padding_s": float(chunk.right_padding_s),
        "split_reason": chunk.split_reason,
        "parent_chunk_id": chunk.parent_chunk_id,
        "island_id": chunk.island_id,
        "island_count": chunk.island_count,
        "core_start": chunk.core_start,
        "core_end": chunk.core_end,
        "internal_gap_count": int(chunk.internal_gap_count),
        "internal_gap_max_s": float(chunk.internal_gap_max_s),
        "split_policy": chunk.split_policy,
        "valley_split_count": int(chunk.valley_split_count),
        "valley_score_min": (
            None if chunk.valley_score_min is None else float(chunk.valley_score_min)
        ),
        "vad_segments": _segments_to_payload(chunk.vad_segments),
    }


def _packed_chunk_from_dict(item: Any) -> PackedChunk:
    if not isinstance(item, dict):
        raise ValueError("invalid packed chunk")
    segments = [_segment_from_dict(segment) for segment in item.get("vad_segments", [])]
    return PackedChunk(
        start=float(item["start"]),
        end=float(item["end"]),
        duration=float(item.get("duration", float(item["end"]) - float(item["start"]))),
        vad_segments=segments,
        left_padding_s=float(item.get("left_padding_s", 0.0)),
        right_padding_s=float(item.get("right_padding_s", 0.0)),
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
        split_policy=str(item.get("split_policy") or ""),
        valley_split_count=int(item.get("valley_split_count") or 0),
        valley_score_min=(
            None
            if item.get("valley_score_min") is None
            else float(item.get("valley_score_min"))
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
