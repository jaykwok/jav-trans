"""ASR worker environment forwarding whitelist, aligned-cache signatures, and sidecar compaction."""

from __future__ import annotations

import os

from core.config import NETWORK_PROXY_ENV_KEYS
from core.job_context import JobContext
from subtitles.options import BASE_FPS

# Prefixes whose settings reach the ASR stage worker. The boundary/CueQC
# prefixes were dropped on 2026-07-31 with the chain they configured - a
# forwarded setting for a stage that no longer runs is not neutral, because it
# still participates in the stage's cache signature.
_ASR_STAGE_ADVANCED_PREFIXES = (
    "ASR_NATIVE_",
    "ASR_CHUNK_",
    "ASR_ALIGNMENT_",
    # Decode budget and loop guard. Every one of them changes where a sequence
    # stops, so they are all cache-relevant and none belongs in the neutral set.
    "ASR_DECODE_",
)
_ASR_STAGE_ADVANCED_KEYS = {
    "ASR_STAGE_WORKER_TIMEOUT_S",
    "ASR_STAGE_WORKER_READY_TIMEOUT_S",
    "ASR_STAGE_WORKER_KILL_GRACE_S",
    "ASR_STAGE_WORKER_OOM_RETRY_LIMIT",
    "ASR_STAGE_WORKER_VRAM_BUDGET_MB",
    "ASR_STAGE_WORKER_VRAM_RATIO",
    "ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO",
    "ASR_STAGE_WORKER_RAM_RATIO",
    "ASR_STAGE_WORKER_HEARTBEAT_S",
    "ASR_LANGUAGE",
    "ASR_FORCE_LANGUAGE",
    "ASR_MODEL_PATH",
    "ASR_MODEL_ID",
    "ASR_DTYPE",
    "ASR_ATTENTION",
    "ASR_MAX_NEW_TOKENS",
    "ASR_REPETITION_PENALTY",
    "TRANSCRIPTION_TIMEOUT_S",
    "ASR_BATCH_SIZE",
    "ASR_CHUNK_ROOT",
    "KEEP_ASR_CHUNKS",
    # These two move segment end times, so they change the value the
    # aligned-segments cache stores. `_asr_stage_config_signature_for_env` is
    # built from this same predicate, so being absent from it meant absent from
    # that signature too: changing either one replayed the old segments.
    "ASR_INVALID_SEGMENT_DURATION",
    "ASR_MIN_REPAIRED_SEGMENT_DURATION",
    # Batch sizes for the two GPU passes that are not the text decode. Cost-only,
    # like ASR_BATCH_SIZE, and the ones a user reaches for after an OOM.
    "ASR_ALIGN_BATCH_SIZE",
    "ASR_FEATURE_BATCH_SIZE",
    # Operational siblings of the worker settings above; left out by oversight.
    "ASR_STAGE_WORKER_MAX_IDLE_S",
    "ASR_SHARED_VRAM_SPILL_TOLERANCE_MB",
    # Already declared cache-neutral below, which only means anything if they can
    # reach the stage in the first place.
    "ASR_RESULT_CACHE_ENABLED",
    "ASR_RESULT_CACHE_ROOT",
}
_ASR_STAGE_CACHE_NEUTRAL_KEYS = {
    "ASR_BATCH_SIZE",
    "ASR_STAGE_WORKER_KILL_GRACE_S",
    "ASR_STAGE_WORKER_OOM_RETRY_LIMIT",
    "ASR_STAGE_WORKER_READY_TIMEOUT_S",
    "ASR_STAGE_WORKER_TIMEOUT_S",
    "ASR_STAGE_WORKER_VRAM_BUDGET_MB",
    "ASR_STAGE_WORKER_VRAM_RATIO",
    "ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO",
    "ASR_STAGE_WORKER_RAM_RATIO",
    "ASR_STAGE_WORKER_HEARTBEAT_S",
    "ASR_CHUNK_ROOT",
    "KEEP_ASR_CHUNKS",
    "ASR_RESULT_CACHE_ENABLED",
    "ASR_RESULT_CACHE_ROOT",
    "TRANSCRIPTION_TIMEOUT_S",
    "ASR_ALIGN_BATCH_SIZE",
    "ASR_FEATURE_BATCH_SIZE",
    "ASR_STAGE_WORKER_MAX_IDLE_S",
    "ASR_SHARED_VRAM_SPILL_TOLERANCE_MB",
}

_ASR_STAGE_ENV_SNAPSHOT_KEYS = {
    "ASR_BACKEND",
    "HF_HOME",
    "HF_ENDPOINT",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "TORCH_HOME",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUDA_VISIBLE_DEVICES",
}

_ASR_DETAILS_LARGE_SIDECAR_KEYS = {
    "transcript_chunks",
}


def _is_asr_stage_advanced_key(name: str) -> bool:
    return name in _ASR_STAGE_ADVANCED_KEYS or any(
        name.startswith(prefix) for prefix in _ASR_STAGE_ADVANCED_PREFIXES
    )


def _asr_stage_env_snapshot_for_ctx(_ctx: JobContext | None = None) -> dict[str, str]:
    names = {
        name
        for name in os.environ
        if _is_asr_stage_advanced_key(name) or name in _ASR_STAGE_ENV_SNAPSHOT_KEYS
    }
    # The GPU process is persistent. Empty values also have to travel so
    # switching a proxy off clears the settings inherited at process start.
    names.update(NETWORK_PROXY_ENV_KEYS)
    return {
        name: os.getenv(name, "")
        for name in sorted(names)
        if name in os.environ or name in NETWORK_PROXY_ENV_KEYS
    }


def _asr_runtime_signature_for_env(*, asr_module=None) -> dict:
    if asr_module is None:
        from asr import pipeline as asr_module
    try:
        return dict(asr_module._get_asr_runtime_signature(last_boundary_signature={}))
    except Exception:
        return {}


def _asr_stage_config_signature_for_env() -> dict:
    names = {
        name
        for name in os.environ
        if _is_asr_stage_advanced_key(name) or name in {"ASR_BACKEND"}
    }
    return {
        name: os.getenv(name, "")
        for name in sorted(names)
        if name not in _ASR_STAGE_CACHE_NEUTRAL_KEYS
    }


def _aligned_cache_signature_for_ctx(
    ctx: JobContext,
    *,
    backend_label: str,
    subtitle_options=None,
    asr_module=None,
    subtitle_module=None,
) -> dict:
    if subtitle_options is None:
        if subtitle_module is None:
            from subtitles import writer as subtitle_module
        subtitle_options = subtitle_module.SubtitleOptions.from_env()
    asr_signature = _asr_runtime_signature_for_env(asr_module=asr_module)
    subtitle_signature = dict(subtitle_options.signature())
    subtitle_signature["base_fps"] = BASE_FPS
    subtitle_signature["frame_duration_s"] = subtitle_options.frame_duration_s
    subtitle_signature["frame_gap_s"] = subtitle_options.frame_gap_s
    subtitle_signature["frame_min_duration_s"] = subtitle_options.frame_min_duration_s
    # The head path reaches this through `asr_stage_config`, but only as the
    # configured REFERENCE. The edge caps are source constants, so nothing here
    # saw them at all until version 10 - retuning one left every job's cached
    # aligned segments serving the previous boundaries.
    from asr.alignment import CODA_EXTEND_MAX_S, ONSET_BACKOFF_MAX_S

    # Version 11: `normalize_word_dicts` stopped dropping the zero-width spans an
    # acoustic-only head gives punctuation, so the words - and the segment text
    # rebuilt from them - now keep their commas.
    return {
        "version": 11,
        "backend_label": backend_label,
        "asr": asr_signature,
        "asr_stage_config": _asr_stage_config_signature_for_env(),
        "subtitle": subtitle_signature,
        "alignment_edge_caps": {
            "onset_backoff_max_s": round(float(ONSET_BACKOFF_MAX_S), 6),
            "coda_extend_max_s": round(float(CODA_EXTEND_MAX_S), 6),
        },
    }


def aligned_cache_expectations_for_ctx(
    ctx: JobContext,
    *,
    backend_label: str | None = None,
    asr_module=None,
    subtitle_module=None,
) -> tuple[str, dict]:
    if asr_module is None:
        from asr import pipeline as asr_module
    if subtitle_module is None:
        from subtitles import writer as subtitle_module
    resolved_backend_label = backend_label or asr_module.get_backend_label()
    subtitle_options = subtitle_module.SubtitleOptions.from_env()
    signature = _aligned_cache_signature_for_ctx(
        ctx,
        backend_label=resolved_backend_label,
        subtitle_options=subtitle_options,
        asr_module=asr_module,
        subtitle_module=subtitle_module,
    )
    return resolved_backend_label, signature


def _compact_asr_details_for_sidecar(
    asr_details: dict | None,
    *,
    strip_frame_class_track: bool = False,
) -> dict:
    if not isinstance(asr_details, dict):
        return {}
    excluded = set(_ASR_DETAILS_LARGE_SIDECAR_KEYS)
    if strip_frame_class_track:
        excluded.add("frame_class_track")
    compact = {
        key: value
        for key, value in asr_details.items()
        if key not in excluded
    }
    transcript_chunks = asr_details.get("transcript_chunks")
    if isinstance(transcript_chunks, list):
        compact["transcript_chunk_count"] = len(transcript_chunks)
    if strip_frame_class_track:
        track = asr_details.get("frame_class_track")
        if isinstance(track, dict):
            compact["frame_class_track_meta"] = {
                key: value
                for key, value in track.items()
                if key != "data_b64"
            }
    return compact


def _aligned_segments_payload(
    *,
    backend_label: str,
    audio_path: str,
    audio_cache_key: str,
    segments: list[dict],
    asr_details: dict,
    asr_log: list[str],
    cache_signature: dict | None,
    subtitle_options,
    cache_stage: str = "ready",
) -> dict:
    payload = {
        "backend": backend_label,
        "audio_path": audio_path,
        "audio_cache_key": audio_cache_key,
        "segments": segments,
        "asr_details": _compact_asr_details_for_sidecar(
            asr_details,
            strip_frame_class_track=False,
        ),
        "asr_log": asr_log,
        "timeline_mode": subtitle_options.timeline_mode,
        "cache_stage": cache_stage,
    }
    if cache_signature is not None:
        payload["cache_signature"] = cache_signature
    return payload
