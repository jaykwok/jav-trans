import importlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Callable

from audio.chunk_packer import PackedChunk, pack_speech_segments
from boundary import cache as _boundary_cache_module
from boundary.candidates import CANDIDATE_EXTRACTOR_VERSION
from boundary.sequence_features import (
    FRAME_SEQUENCE_FRAMES_SCHEMA,
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
)
from boundary.refiner import (
    file_sha1 as _boundary_refiner_file_sha1,
    load_frame_sequence_refiner_checkpoint,
)
from asr import checkpoint as _checkpoint_module
from asr import chunking as _chunking_module
from asr import cueqc as _cueqc_module
from asr import transcribe as _transcribe_module
from asr.backends.qwen import (
    DEFAULT_BOUNDARY_REFINER_CHECKPOINT_BY_REPO,
    DEFAULT_CUEQC_CHECKPOINT_BY_REPO,
    checkpoint_path_for_repo_env,
    validate_checkpoint_repo_id,
)
from asr.backends import registry as _registry_module

warnings.filterwarnings("ignore")

_registry_module = importlib.reload(_registry_module)
_chunking_module = importlib.reload(_chunking_module)
_checkpoint_module = importlib.reload(_checkpoint_module)
_cueqc_module = importlib.reload(_cueqc_module)
_transcribe_module = importlib.reload(_transcribe_module)
_boundary_cache_module = importlib.reload(_boundary_cache_module)

ASR_BACKEND = _registry_module.current_asr_backend()
_ASR_WORKER_MODE = _registry_module.current_asr_worker_mode()
_QWEN_BACKENDS = _registry_module._QWEN_BACKENDS
_VALID_ASR_BACKENDS = _registry_module._VALID_ASR_BACKENDS
_VALID_ASR_WORKER_MODES = _registry_module._VALID_ASR_WORKER_MODES

_ASR_CHUNK_ROOT = _chunking_module._ASR_CHUNK_ROOT
_KEEP_ASR_CHUNKS = _chunking_module._KEEP_ASR_CHUNKS
_LAST_BOUNDARY_SIGNATURE: dict = _chunking_module._LAST_BOUNDARY_SIGNATURE
_LAST_BOUNDARY_CACHE_EVENT: dict | None = None
_ASR_SLIDING_CONTEXT_SEGS = _transcribe_module._ASR_SLIDING_CONTEXT_SEGS
_ASR_CHECKPOINT_ENABLED = _transcribe_module._ASR_CHECKPOINT_ENABLED
CUEQC_DECISION_VERSION = _cueqc_module.CUEQC_DECISION_VERSION
CUEQC_MODEL_VERSION = _cueqc_module.CUEQC_MODEL_VERSION

# CueQC Mamba v3-Fusion model is lazily loaded and cached per process.
# ASR-internals capture is delegated to the backend (inline or subprocess
# worker) via capture_asr_internals(), so the model is reused wherever it is
# loaded — no second Qwen3-ASR in VRAM (v3-Fusion §5.2).
_CUEQC_REFINER_CACHE: dict[str, object] = {}


def _cueqc_refiner_for(path: str, *, expected_asr_repo_id: str | None = None):
    """Lazily load + cache the CueQC v3-Fusion refiner for ``path``."""
    if not path:
        return None
    cached = _CUEQC_REFINER_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        from asr.cueqc_refiner import load_cueqc_mamba_checkpoint

        device = os.getenv("CUEQC_DEVICE", "auto").strip() or "auto"
        refiner = load_cueqc_mamba_checkpoint(
            path,
            device=device,
            expected_asr_repo_id=expected_asr_repo_id,
        )
        _CUEQC_REFINER_CACHE[path] = refiner
        return refiner
    except Exception as exc:  # noqa: BLE001 - surface checkpoint/config errors clearly
        raise RuntimeError(f"CueQC v3-Fusion checkpoint load failed for {path}: {exc!r}") from exc

def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: str) -> int:
    return int(float(os.getenv(name, default)))


def _boundary_config() -> dict:
    refiner_path = checkpoint_path_for_repo_env(
        repo_id=ASR_BACKEND,
        mapping_env="BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        default_mapping=DEFAULT_BOUNDARY_REFINER_CHECKPOINT_BY_REPO,
    )
    refiner_path_obj = Path(refiner_path).expanduser() if refiner_path else None
    runtime_adapter = _boundary_refiner_runtime_adapter(refiner_path_obj)
    return {
        "feature_frame_hop_s": _env_float("BOUNDARY_FEATURE_FRAME_HOP_S", "0.02"),
        "boundary_refiner_model_path": refiner_path,
        "boundary_refiner_model_sha1": (
            _boundary_refiner_file_sha1(refiner_path_obj)
            if refiner_path_obj is not None and refiner_path_obj.exists()
            else ""
        ),
        "boundary_refiner_runtime_adapter": runtime_adapter,
        "boundary_refiner_device": os.getenv("BOUNDARY_REFINER_DEVICE", "auto").strip()
        or "auto",
        "boundary_candidate_extractor_version": CANDIDATE_EXTRACTOR_VERSION,
        "boundary_planner_max_core_chunk_s": _env_float(
            "BOUNDARY_PLANNER_MAX_CORE_CHUNK_S",
            "5.0",
        ),
        "boundary_planner_target_chunk_s": _env_float("BOUNDARY_PLANNER_TARGET_CHUNK_S", "3.0"),
        "boundary_planner_min_chunk_s": _env_float("BOUNDARY_PLANNER_MIN_CHUNK_S", "0.4"),
        "boundary_planner_max_splits_per_segment": _env_int(
            "BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT", "16"
        ),
        "boundary_planner_sequence_batch_size": _env_int(
            "BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE", "256"
        ),
    }


def _boundary_refiner_runtime_adapter(path: Path | None) -> str:
    if path is None or not path.exists():
        raise FileNotFoundError(
            "Boundary Refiner checkpoint is required for the selected ASR repo id"
        )
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Boundary refiner checkpoint must be a dict")
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        validate_checkpoint_repo_id(
            metadata.get("ptm_repo_id"),
            ASR_BACKEND,
            checkpoint_kind="Boundary Refiner",
            metadata_key="metadata.ptm_repo_id",
        )
        adapter = str(metadata.get("runtime_adapter") or "").strip()
        if adapter == "frame_sequence_v1":
            return adapter
    raise ValueError("Boundary Refiner checkpoint must use metadata.runtime_adapter='frame_sequence_v1'")


def _sequence_feature_provider_from_result(
    payload,
    *,
    duration_s: float,
    target_chunk_s: float,
) -> FrameSequenceFeatureProvider | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != FRAME_SEQUENCE_FRAMES_SCHEMA:
        return None
    ptm = payload.get("ptm")
    mfcc = payload.get("mfcc")
    frame_hop_s = payload.get("frame_hop_s")
    if not isinstance(ptm, list) or not isinstance(mfcc, list):
        return None
    try:
        hop = float(frame_hop_s)
    except (TypeError, ValueError):
        return None
    if hop <= 0.0:
        return None
    return FrameSequenceFeatureProvider(
        duration_s=float(duration_s),
        frame_hop_s=hop,
        ptm=ptm,
        mfcc=mfcc,
        config=FrameSequenceFeatureConfig(
            left_context_s=_env_float("BOUNDARY_FRAME_SEQUENCE_LEFT_CONTEXT_S", "0.60"),
            right_context_s=_env_float("BOUNDARY_FRAME_SEQUENCE_RIGHT_CONTEXT_S", "0.60"),
            max_ptm_dims=_env_int("BOUNDARY_FRAME_SEQUENCE_MAX_PTM_DIMS", "64"),
            include_mfcc=_env_bool("BOUNDARY_FRAME_SEQUENCE_INCLUDE_MFCC", "1"),
            target_chunk_s=target_chunk_s,
        ),
    )


def _required_sequence_feature_provider_from_result(
    payload,
    *,
    duration_s: float,
    target_chunk_s: float,
) -> FrameSequenceFeatureProvider:
    provider = _sequence_feature_provider_from_result(
        payload,
        duration_s=duration_s,
        target_chunk_s=target_chunk_s,
    )
    if provider is None:
        raise ValueError(
            "frame_sequence_v1 Boundary Refiner requires "
            f"{FRAME_SEQUENCE_FRAMES_SCHEMA} in SpeechBoundary-JA output"
        )
    return provider


get_backend_label = _registry_module.get_backend_label
_resolve_asr_backend = _registry_module._resolve_asr_backend
_create_asr_backend = _registry_module._create_asr_backend
_is_subprocess_backend = _registry_module._is_subprocess_backend

_is_timed_out_result = _checkpoint_module._is_timed_out_result
_checkpointable_text_results = _checkpoint_module._checkpointable_text_results
_delete_path_for_cleanup = _checkpoint_module._delete_path_for_cleanup
_get_asr_checkpoint_source = _checkpoint_module._get_asr_checkpoint_source
_build_quarantined_text_result = _checkpoint_module._build_quarantined_text_result
_quarantine_failed_chunks = _checkpoint_module._quarantine_failed_chunks

_get_wav_duration = _chunking_module._get_wav_duration
_extract_wav_chunks = _chunking_module._extract_wav_chunks
_chunk_duration = _chunking_module._chunk_duration

ASRWorkerSystemError = _transcribe_module.ASRWorkerSystemError
_strip_punctuation = _transcribe_module._strip_punctuation
_compact_text_units = _transcribe_module._compact_text_units
_collapse_repeated_noise = _transcribe_module._collapse_repeated_noise
_is_low_value_text = _transcribe_module._is_low_value_text
_clean_segment_text = _transcribe_module._clean_segment_text
_with_alignment_fallback_window = _transcribe_module._with_alignment_fallback_window
_alignment_outcome_for_chunk = _transcribe_module._alignment_outcome_for_chunk
_transcribe_asr_chunks_text_only = _transcribe_module._transcribe_asr_chunks_text_only
_is_empty_segment_text_result = _transcribe_module._is_empty_segment_text_result
_empty_alignment_placeholder = _transcribe_module._empty_alignment_placeholder
_empty_segments_quarantine_placeholder = _transcribe_module._empty_segments_quarantine_placeholder
_align_TRANSCRIPTION_results = _transcribe_module._align_TRANSCRIPTION_results
_build_transcript_chunks = _transcribe_module._build_transcript_chunks
_build_ASR_CONTEXT_for_chunk = _transcribe_module._build_ASR_CONTEXT_for_chunk
_backend_accepts_initial_prompts = _transcribe_module._backend_accepts_initial_prompts
_should_reset_sliding_context = _transcribe_module._should_reset_sliding_context
_sliding_context_result_text = _transcribe_module._sliding_context_result_text
_build_initial_prompt_for_chunk = _transcribe_module._build_initial_prompt_for_chunk
_postprocess_segments = _transcribe_module._postprocess_segments
_pick_postprocess_split_index = _transcribe_module._pick_postprocess_split_index
_split_long_postprocessed_segment = _transcribe_module._split_long_postprocessed_segment
_split_long_postprocessed_segments = _transcribe_module._split_long_postprocessed_segments
_repair_postprocessed_segment_windows = _transcribe_module._repair_postprocessed_segment_windows
_group_words_to_segments = _transcribe_module._group_words_to_segments

build_cueqc_candidates = _cueqc_module.build_candidates
cueqc_enabled = _cueqc_module.cueqc_enabled


def _sync_checkpoint_state() -> None:
    _checkpoint_module._ASR_CHUNK_ROOT = _ASR_CHUNK_ROOT
    _checkpoint_module._LAST_BOUNDARY_SIGNATURE = _LAST_BOUNDARY_SIGNATURE


def _get_asr_generation_checkpoint_signature() -> dict:
    _sync_checkpoint_state()
    return _checkpoint_module._get_asr_runtime_signature(
        last_boundary_signature=_LAST_BOUNDARY_SIGNATURE,
        sliding_context_segs=_ASR_SLIDING_CONTEXT_SEGS
    )


def _get_asr_runtime_signature(last_boundary_signature: dict | None = None) -> dict:
    _sync_checkpoint_state()
    return _checkpoint_module._get_asr_runtime_signature(
        last_boundary_signature=_LAST_BOUNDARY_SIGNATURE if last_boundary_signature is None else last_boundary_signature,
        sliding_context_segs=_ASR_SLIDING_CONTEXT_SEGS,
    )


def _get_asr_checkpoint_path(audio_path: str) -> Path:
    _sync_checkpoint_state()
    return _checkpoint_module._get_asr_checkpoint_path(
        audio_path,
        last_boundary_signature=_LAST_BOUNDARY_SIGNATURE,
        chunk_root=_ASR_CHUNK_ROOT,
        sliding_context_segs=_ASR_SLIDING_CONTEXT_SEGS,
    )


def _chunk_checkpoint_signature(chunks: list[dict]) -> dict[str, dict[str, float | str]]:
    _sync_checkpoint_state()
    return _checkpoint_module._chunk_checkpoint_signature(
        chunks,
        last_boundary_signature=_LAST_BOUNDARY_SIGNATURE,
    )


def _load_asr_checkpoint(
    checkpoint_path: Path,
    checkpoint_source: str,
    chunks: list[dict],
    run_id: str | None = None,
) -> dict[int, dict]:
    _sync_checkpoint_state()
    return _checkpoint_module._load_asr_checkpoint(
        checkpoint_path,
        checkpoint_source,
        chunks,
        run_id=run_id,
        last_boundary_signature=_LAST_BOUNDARY_SIGNATURE,
        checkpoint_enabled=_ASR_CHECKPOINT_ENABLED,
    )


def _save_asr_checkpoint(
    checkpoint_path: Path,
    checkpoint_source: str,
    chunks: list[dict],
    text_results_by_index: dict[int, dict],
    run_id: str | None = None,
) -> None:
    _sync_checkpoint_state()
    return _checkpoint_module._save_asr_checkpoint(
        checkpoint_path,
        checkpoint_source,
        chunks,
        text_results_by_index,
        run_id=run_id,
        last_boundary_signature=_LAST_BOUNDARY_SIGNATURE,
        checkpoint_enabled=_ASR_CHECKPOINT_ENABLED,
    )


def aggregate_timeout_fragments(job_id: str) -> Path | None:
    _sync_checkpoint_state()
    return _checkpoint_module.aggregate_timeout_fragments(job_id)


import logging as _logging
_pipeline_logger = _logging.getLogger(__name__)


def _set_last_boundary_signature(signature: dict) -> None:
    global _LAST_BOUNDARY_SIGNATURE
    _LAST_BOUNDARY_SIGNATURE = dict(signature)
    _chunking_module._LAST_BOUNDARY_SIGNATURE = _LAST_BOUNDARY_SIGNATURE
    _sync_checkpoint_state()


def _set_last_boundary_cache_event(event: dict | None) -> None:
    global _LAST_BOUNDARY_CACHE_EVENT
    _LAST_BOUNDARY_CACHE_EVENT = dict(event) if isinstance(event, dict) else None


def _display_cache_path(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def _boundary_cache_log_entry(event: dict | None) -> str | None:
    if not event:
        return None
    status = str(event.get("status") or "")
    path = _display_cache_path(str(event.get("path") or ""))
    digest = str(event.get("digest") or "")
    if status == "hit":
        return f"Boundary cache hit: path={path} digest={digest}"
    if status == "miss":
        return f"Boundary cache saved: path={path} digest={digest}"
    return None


def _build_processing_spans(
    audio_path: str,
) -> list[tuple[float, float]] | list[PackedChunk]:
    cfg = _boundary_config()
    _set_last_boundary_cache_event(None)

    needs_sequence_features = cfg["boundary_refiner_runtime_adapter"] == "frame_sequence_v1"
    restore_sequence_export = (
        os.environ.get("SPEECH_BOUNDARY_JA_EXPORT_SEQUENCE_FEATURES")
        if needs_sequence_features
        else None
    )
    if needs_sequence_features:
        os.environ["SPEECH_BOUNDARY_JA_EXPORT_SEQUENCE_FEATURES"] = "1"
    try:
        from boundary import get_boundary_backend

        boundary_backend = get_boundary_backend()
        boundary_signature = boundary_backend.signature()
        cached = _boundary_cache_module.load_processing_spans(
            audio_path,
            boundary_signature=boundary_signature,
            boundary_config=cfg,
        )
        if cached is not None:
            spans, runtime_boundary_signature, event = cached
            _set_last_boundary_signature(runtime_boundary_signature)
            _pipeline_logger.info(
                "[boundary-cache] hit path=%s digest=%s",
                event["path"],
                event["digest"],
            )
            _set_last_boundary_cache_event(event)
            return spans

        result = boundary_backend.segment(audio_path)
    finally:
        if restore_sequence_export is not None:
            os.environ["SPEECH_BOUNDARY_JA_EXPORT_SEQUENCE_FEATURES"] = restore_sequence_export
        elif needs_sequence_features:
            os.environ.pop("SPEECH_BOUNDARY_JA_EXPORT_SEQUENCE_FEATURES", None)
    frame_scores = result.parameters.get("frame_scores")
    cut_frame_scores = result.parameters.get("cut_frame_scores")
    score_frame_hop_s = result.parameters.get("frame_hop_s")
    sequence_feature_frames = result.parameters.get("sequence_feature_frames")
    sequence_boundary_refiner = load_frame_sequence_refiner_checkpoint(
        Path(cfg["boundary_refiner_model_path"]),
        device=cfg["boundary_refiner_device"],
        expected_ptm_repo_id=ASR_BACKEND,
    )
    sequence_feature_provider = _required_sequence_feature_provider_from_result(
        sequence_feature_frames,
        duration_s=result.audio_duration_sec,
        target_chunk_s=cfg["boundary_planner_target_chunk_s"],
    )
    sequence_feature_provider.validate_for_checkpoint(
        sequence_boundary_refiner.feature_names,
        sequence_boundary_refiner.feature_schema_hash,
    )
    result_parameters = {
        key: value
        for key, value in result.parameters.items()
        if key not in {"frame_scores", "cut_frame_scores", "sequence_feature_frames"}
    }
    runtime_boundary_signature = {
        **result_parameters,
        "boundary_pipeline": {
            "version": 5,
            "refiner_schema": "boundary_refiner_v5",
            "feature_frame_hop_s": cfg["feature_frame_hop_s"],
            "score_frame_hop_s": score_frame_hop_s,
            "feature_sources": {
                "speech_scores": frame_scores is not None,
                "cut_scores": cut_frame_scores is not None,
            },
            "sequence_boundary_refiner": (
                sequence_boundary_refiner.signature()
                if sequence_boundary_refiner is not None
                else None
            ),
            "sequence_feature_provider": (
                sequence_feature_provider.signature()
                if sequence_feature_provider is not None
                else None
            ),
            "boundary_planner": {
                "candidate_extractor_version": cfg[
                    "boundary_candidate_extractor_version"
                ],
                "max_core_chunk_s": cfg["boundary_planner_max_core_chunk_s"],
                "target_chunk_s": cfg["boundary_planner_target_chunk_s"],
                "min_chunk_s": cfg["boundary_planner_min_chunk_s"],
                "max_splits_per_segment": cfg["boundary_planner_max_splits_per_segment"],
                "sequence_batch_size": cfg["boundary_planner_sequence_batch_size"],
            },
        },
    }
    segments = result.segments
    _set_last_boundary_signature(runtime_boundary_signature)
    if not segments:
        event = _boundary_cache_module.save_processing_spans(
            audio_path,
            boundary_signature=boundary_signature,
            boundary_config=cfg,
            processing_spans=[],
            runtime_boundary_signature=runtime_boundary_signature,
            speech_segments=result.segments,
            speech_groups=result.groups,
        )
        if event is not None:
            _pipeline_logger.info(
                "[boundary-cache] saved path=%s digest=%s",
                event["path"],
                event["digest"],
            )
            _set_last_boundary_cache_event(event)
        return []
    packed = pack_speech_segments(
        segments,
        frame_hop_s=cfg["feature_frame_hop_s"],
        max_core_chunk_s=cfg["boundary_planner_max_core_chunk_s"],
        target_chunk_s=cfg["boundary_planner_target_chunk_s"],
        min_chunk_s=cfg["boundary_planner_min_chunk_s"],
        frame_scores=frame_scores,
        score_frame_hop_s=score_frame_hop_s,
        cut_frame_scores=cut_frame_scores,
        sequence_boundary_refiner=sequence_boundary_refiner,
        sequence_feature_provider=sequence_feature_provider,
        max_splits_per_segment=cfg["boundary_planner_max_splits_per_segment"],
        sequence_batch_size=cfg["boundary_planner_sequence_batch_size"],
    )
    event = _boundary_cache_module.save_processing_spans(
        audio_path,
        boundary_signature=boundary_signature,
        boundary_config=cfg,
        processing_spans=packed,
        runtime_boundary_signature=runtime_boundary_signature,
        speech_segments=result.segments,
        speech_groups=result.groups,
    )
    if event is not None:
        _pipeline_logger.info(
            "[boundary-cache] saved path=%s digest=%s",
            event["path"],
            event["digest"],
        )
        _set_last_boundary_cache_event(event)
    return packed


def _span_boundaries(
    spans: list[tuple[float, float]] | list[PackedChunk],
) -> list[tuple[float, float]]:
    return [
        (span.start, span.end) if isinstance(span, PackedChunk) else span
        for span in spans
    ]


def _annotate_packed_chunks(
    chunk_infos: list[dict],
    spans: list[tuple[float, float]] | list[PackedChunk],
    log: list[str],
) -> None:
    packed_spans = [span for span in spans if isinstance(span, PackedChunk)]
    if not packed_spans:
        return
    for idx, chunk in enumerate(chunk_infos):
        span_index = int(chunk.get("source_span_index", idx))
        if span_index < 0 or span_index >= len(packed_spans):
            continue
        packed = packed_spans[span_index]
        chunk["speech_segment_count"] = len(packed.speech_segments)
        chunk["boundary_split_reason"] = packed.split_reason
        chunk["boundary_parent_chunk_id"] = packed.parent_chunk_id
        chunk["speech_island_id"] = packed.island_id
        chunk["speech_island_count"] = packed.island_count
        chunk["speech_internal_gap_count"] = packed.internal_gap_count
        chunk["speech_internal_gap_max_s"] = packed.internal_gap_max_s
        chunk["boundary_score"] = packed.boundary_score
        chunk["boundary_reason"] = packed.boundary_reason
        chunk["boundary_source"] = packed.boundary_source
        chunk["boundary_start_refine_delta_s"] = packed.boundary_start_refine_delta_s
        chunk["boundary_end_refine_delta_s"] = packed.boundary_end_refine_delta_s
        chunk["boundary_decision_source"] = packed.boundary_decision_source
        log.append(
            "[chunk] idx={idx} dur={duration:.1f} speech_segment_count={count} "
            "reason={reason} "
            "parent={parent} island={island}/{islands} gap_max={gap:.2f} "
            "boundary={boundary_reason} source={boundary_source} score={boundary_score} "
            "delta=({start_delta},{end_delta}) "
            "decision_source={decision_source}".format(
                idx=idx,
                duration=packed.duration,
                count=len(packed.speech_segments),
                reason=packed.split_reason,
                parent=packed.parent_chunk_id,
                island=packed.island_id,
                islands=packed.island_count,
                gap=packed.internal_gap_max_s,
                boundary_reason=packed.boundary_reason,
                boundary_source=packed.boundary_source,
                boundary_score=packed.boundary_score,
                start_delta=packed.boundary_start_refine_delta_s,
                end_delta=packed.boundary_end_refine_delta_s,
                decision_source=packed.boundary_decision_source,
            )
        )


def _record_stage_timing(
    log: list[str],
    timings: dict[str, float],
    key: str,
    label: str,
    elapsed_s: float,
) -> None:
    timings[key] = elapsed_s
    log.append(f"ASR 阶段耗时: {label}={elapsed_s:.2f}s")


def _empty_cueqc_shadow_report() -> dict:
    return _cueqc_module.build_shadow_report([])


def _write_cueqc_candidates_if_requested(
    candidates: list[dict],
    *,
    log: list[str],
) -> None:
    if not candidates:
        return
    output_path_raw = os.getenv("CUEQC_EXPORT_CANDIDATES_PATH", "").strip()
    if not output_path_raw:
        return
    output_path = Path(output_path_raw).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    append = os.getenv("CUEQC_EXPORT_CANDIDATES_APPEND", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    log.append(
        "CueQC shadow: exported candidates path={path} count={count}".format(
            path=_display_cache_path(str(output_path)),
            count=len(candidates),
        )
    )


def _apply_cueqc_drop_filter(
    chunk_infos: list[dict],
    text_results: list[dict],
    cueqc_shadow_by_chunk: dict[int, dict],
) -> tuple[list[dict], list[dict], str]:
    """Remove high-confidence model-drop chunks from the parallel arrays.

    Only decisions with ``mode == "cueqc_mamba_v3_fusion"`` and
    ``display_hint == "drop"`` are removed; fallback / keep decisions are never
    dropped. All parallel lists (chunk_infos, text_results) are kept
    index-aligned by filtering on the same kept positions.
    """
    drop_indices: set[int] = set()
    for chunk_index, dec in cueqc_shadow_by_chunk.items():
        if (
            dec.get("display_hint") == "drop"
            and dec.get("mode") == "cueqc_mamba_v3_fusion"
        ):
            drop_indices.add(int(chunk_index))
    if not drop_indices:
        return chunk_infos, text_results, ""

    kept_chunks: list[dict] = []
    kept_texts: list[dict] = []
    removed = 0
    for pos, (chunk, text_result) in enumerate(zip(chunk_infos, text_results)):
        raw_index = chunk.get("index") if hasattr(chunk, "get") else None
        try:
            chunk_index = int(raw_index) if raw_index is not None else pos
        except (TypeError, ValueError):
            chunk_index = pos
        if chunk_index in drop_indices:
            removed += 1
            continue
        kept_chunks.append(chunk)
        kept_texts.append(text_result)
    log_msg = f"CueQC v3-Fusion: dropped {removed} candidate(s) before subtitle timing"
    return kept_chunks, kept_texts, log_msg


def _candidate_capture_window(
    candidate: dict,
    fallback_audio_path: str,
) -> tuple[str, float, float]:
    """Resolve the wav path and local capture window for a candidate.

    Candidate ``audio.path`` normally points at the extracted chunk wav. In that
    case the ASR-internals capture must use the chunk-local window
    ``0..duration``. Passing the global movie timestamp into a chunk wav slices
    past EOF and forces every v3 sample into fallback-keep.
    """
    audio = candidate.get("audio") if isinstance(candidate.get("audio"), dict) else {}
    own = audio.get("path") or candidate.get("source_audio_path") or ""
    if own and os.path.exists(str(own)):
        duration = float(candidate.get("duration_s") or 0.0)
        if duration <= 0.0:
            start = float(candidate.get("start", 0.0))
            end = float(candidate.get("end", start))
            duration = max(0.0, end - start)
        return str(own), 0.0, duration
    start_s = float(candidate.get("start", 0.0))
    end_s = float(candidate.get("end", start_s))
    return str(fallback_audio_path) if fallback_audio_path else "", start_s, end_s


def _top_counts(items: list[str], *, limit: int = 3) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item or "").strip() or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))[:limit])


def _cueqc_fallback_summary(decisions: list[dict]) -> dict:
    fallback = [
        item for item in decisions
        if item.get("mode") != "cueqc_mamba_v3_fusion"
    ]
    return {
        "count": len(fallback),
        "stages": _top_counts([str(item.get("fallback_stage") or "") for item in fallback]),
        "reasons": _top_counts([
            str((item.get("reasons") or [""])[0])
            if isinstance(item.get("reasons"), list)
            else str(item.get("reasons") or "")
            for item in fallback
        ]),
        "details": _top_counts([str(item.get("fallback_detail") or "") for item in fallback if item.get("fallback_detail")]),
    }


def _apply_cueqc_v3_model(
    *,
    refiner,
    candidates: list[dict],
    backend,
    audio_path: str,
    log: list[str],
) -> list[dict] | None:
    """Run the v3-Fusion refiner over candidates, reusing the shared ASR model.

    Captures ASR internals via ``backend.capture_asr_internals()`` which works in
    both inline and subprocess worker modes (the capture runs where the model is
    loaded). Returns one decision dict per candidate. Per-candidate capture or
    inference failures are handled inside the refiner, but model-level capture
    unavailability is fatal to the job.
    """
    if not candidates:
        return []
    if backend is None or not hasattr(backend, "capture_asr_internals"):
        log.append("CueQC v3-Fusion: backend has no capture_asr_internals; cannot continue")
        return None
    capture_chunks = []
    for cand in candidates:
        path, start_s, end_s = _candidate_capture_window(cand, audio_path)
        capture_chunks.append({
            "path": path,
            "text": str(cand.get("text") or ""),
            "start_s": start_s,
            "end_s": end_s,
        })
    try:
        asr_internals = backend.capture_asr_internals(capture_chunks)
    except Exception as exc:  # noqa: BLE001
        log.append(f"CueQC v3-Fusion: capture failed ({exc!r}); cannot continue")
        return None
    if not isinstance(asr_internals, list) or len(asr_internals) != len(candidates):
        log.append("CueQC v3-Fusion: capture count mismatch; cannot continue")
        return None
    capture_failed = [
        str(item.get("error") or item.get("detail") or "")
        for item in asr_internals
        if not (isinstance(item, dict) and item.get("ok"))
    ]
    if capture_failed:
        log.append(
            "CueQC v3-Fusion: capture fallback candidates={failed}/{total} top_errors={errors}".format(
                failed=len(capture_failed),
                total=len(candidates),
                errors=_top_counts(capture_failed),
            )
        )
    try:
        decisions = refiner.decide(candidates, asr_internals=asr_internals)
    except Exception:
        raise
    drops = sum(1 for d in decisions if d.get("display_hint") == "drop")
    fallback_summary = _cueqc_fallback_summary(decisions)
    log.append(
        "CueQC v3-Fusion: model decisions={decisions} drops={drops} fallback={fallback} "
        "fallback_stages={stages} fallback_reasons={reasons} fallback_details={details}".format(
            decisions=len(decisions),
            drops=drops,
            fallback=fallback_summary["count"],
            stages=fallback_summary["stages"],
            reasons=fallback_summary["reasons"],
            details=fallback_summary["details"],
        )
    )
    return decisions


def _merge_cueqc_v3_decisions(
    report: dict,
    candidates: list[dict],
    model_decisions: list[dict],
) -> dict:
    """Replace report decisions with v3 model decisions, keyed by chunk_index."""
    decisions = []
    for cand, dec in zip(candidates, model_decisions):
        item = dict(dec)
        item["chunk_index"] = cand.get("chunk_index")
        decisions.append(item)
    report = dict(report)
    report["decisions"] = decisions
    # Recompute display_hint counts for the shadow log line.
    counts: dict[str, int] = {}
    for dec in decisions:
        key = str(dec.get("display_hint") or "keep")
        counts[key] = counts.get(key, 0) + 1
    existing_counts = dict(report.get("counts") or {})
    existing_counts["display_hint"] = counts
    fallback_summary = _cueqc_fallback_summary(decisions)
    existing_counts["fallback_stage"] = fallback_summary["stages"]
    existing_counts["fallback_reason"] = fallback_summary["reasons"]
    report["counts"] = existing_counts
    report["decision_source"] = "cueqc_mamba_v3_fusion"
    report["fallback_summary"] = fallback_summary
    return report


def _run_cueqc_shadow(
    *,
    audio_path: str,
    chunk_infos: list[dict],
    text_results: list[dict],
    log: list[str],
    backend=None,
) -> tuple[dict, dict[int, dict]]:
    if not cueqc_enabled():
        return _empty_cueqc_shadow_report(), {}
    try:
        audio_id = Path(audio_path).stem
        candidates = build_cueqc_candidates(
            chunk_infos,
            text_results,
            audio_id=audio_id,
            video_id=audio_id,
        )
        report = _cueqc_module.build_shadow_report(candidates)
        if os.getenv("CUEQC_SHADOW_EMBED_CANDIDATES", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            report["candidates"] = candidates
        _write_cueqc_candidates_if_requested(candidates, log=log)

        # v3-Fusion is required when CueQC is enabled; checkpoint mapping/load
        # failures should stop the job instead of falling back to old rules.
        model_path = checkpoint_path_for_repo_env(
            repo_id=ASR_BACKEND,
            mapping_env="CUEQC_MODEL_PATH_BY_REPO",
            default_mapping=DEFAULT_CUEQC_CHECKPOINT_BY_REPO,
        )
        refiner = _cueqc_refiner_for(model_path, expected_asr_repo_id=ASR_BACKEND) if model_path else None
        if refiner is not None:
            model_decisions = _apply_cueqc_v3_model(
                refiner=refiner,
                candidates=candidates,
                backend=backend,
                audio_path=audio_path,
                log=log,
            )
            if model_decisions is None:
                raise RuntimeError("CueQC v3-Fusion produced no model decisions")
            report = _merge_cueqc_v3_decisions(report, candidates, model_decisions)

        decision_by_chunk = {
            int(item["chunk_index"]): {
                key: value
                for key, value in item.items()
                if key
                in {
                    "schema",
                    "schema_version",
                    "model_version",
                    "decision_version",
                    "mode",
                    "cluster_id",
                    "display_hint",
                    "confidence",
                    "reasons",
                    "display_prob_keep",
                    "display_prob_drop",
                    "drop_threshold",
                    "threshold_profile",
                    "fallback_stage",
                    "fallback_detail",
                }
            }
            for item in report.get("decisions", [])
            if item.get("chunk_index") is not None
        }
        log.append(
            "CueQC: candidates={count} display={display}".format(
                count=report.get("candidate_count", 0),
                display=dict((report.get("counts") or {}).get("display_hint") or {}),
            )
        )
        return report, decision_by_chunk
    except Exception as exc:
        message = f"CueQC v3-Fusion failed; job cannot continue ({exc!r})"
        log.append(message)
        raise RuntimeError(message) from exc


def _segment_alignment_outcome(segment: dict, outcomes: dict[int, dict]) -> dict:
    chunk_indices: list[int] = []
    for word in segment.get("words") or []:
        try:
            chunk_indices.append(int(word.get("source_chunk_index")))
        except (TypeError, ValueError):
            continue
    if not chunk_indices:
        try:
            chunk_indices.append(int(segment.get("source_chunk_index")))
        except (TypeError, ValueError):
            pass
    unique_indices = list(dict.fromkeys(chunk_indices))
    if not unique_indices:
        return {}
    members = [outcomes[index] for index in unique_indices if index in outcomes]
    if not members:
        return {}
    qualities = [str(item.get("alignment_quality") or "") for item in members]
    if any(quality in {"drop_or_review", "partial"} for quality in qualities):
        selected = next(
            item
            for item in members
            if str(item.get("alignment_quality") or "")
            in {"drop_or_review", "partial"}
        )
    else:
        selected = members[0]
    return {
        "source_chunk_indices": unique_indices,
        "alignment_mode": selected.get("alignment_mode", ""),
        "alignment_quality": selected.get("alignment_quality", ""),
        "fallback_type": selected.get("fallback_type", ""),
        "fallback_subtype": selected.get("fallback_subtype", ""),
        "alignment_quality_reasons": list(selected.get("alignment_quality_reasons") or []),
        "fallback_active": bool(selected.get("fallback_active")),
    }


def _annotate_segments_with_alignment_outcomes(
    segments: list[dict],
    outcomes: dict[int, dict],
) -> list[dict]:
    annotated: list[dict] = []
    for segment in segments:
        item = dict(segment)
        outcome = _segment_alignment_outcome(item, outcomes)
        if outcome:
            item.update(outcome)
        annotated.append(item)
    return annotated


def _transcribe_and_align_local(
    audio_path: str,
    device: str,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[str], dict]:
    def _notify(message: str) -> None:
        if on_stage:
            on_stage(message)

    log: list[str] = [f"ASR backend: {get_backend_label()}"]
    timings: dict[str, float] = {}
    transcript_chunks: list[dict] = []
    chunk_dir: Path | None = None
    total_started = time.perf_counter()

    try:
        _notify("分析静音并切分音频...")
        split_started = time.perf_counter()
        chunk_spans = _build_processing_spans(audio_path)
        cache_log_entry = _boundary_cache_log_entry(_LAST_BOUNDARY_CACHE_EVENT)
        if cache_log_entry:
            log.append(cache_log_entry)
        chunk_dir, chunk_infos = _extract_wav_chunks(
            audio_path, _span_boundaries(chunk_spans), on_stage=on_stage
        )
        _annotate_packed_chunks(chunk_infos, chunk_spans, log)
        split_elapsed = time.perf_counter() - split_started
        log.append(f"切分完成：共 {len(chunk_infos)} 个处理块")
        _record_stage_timing(log, timings, "split_s", "静音分析与切块", split_elapsed)

        if not chunk_infos:
            timings.update(
                {
                    "asr_model_load_s": 0.0,
                    "asr_text_transcribe_s": 0.0,
                    "asr_model_unload_s": 0.0,
                    "alignment_s": 0.0,
                    "alignment_model_unload_s": 0.0,
                    "subtitle_segment_s": 0.0,
                }
            )
            total_elapsed = time.perf_counter() - total_started
            _record_stage_timing(
                log,
                timings,
                "asr_alignment_total_s",
                "ASR与字幕时间轴总计",
                total_elapsed,
            )
            log.append("边界系统未检测到可处理语音块，跳过 ASR")
            details = {
                "backend": get_backend_label(),
                "audio_path": audio_path,
                "device": device,
                "chunk_count": 0,
                "transcript_chunks": [],
                "stage_timings": timings,
                "word_count": 0,
                "segment_count": 0,
                "boundary_no_speech": True,
                "boundary_signature": dict(_LAST_BOUNDARY_SIGNATURE),
                "cueqc_shadow": _empty_cueqc_shadow_report(),
            }
            return [], log, details

        backend = _resolve_asr_backend(device)
        word_dicts: list[dict] = []
        try:
            load_started = time.perf_counter()
            backend.load(on_stage=on_stage)
            load_elapsed = time.perf_counter() - load_started
            _record_stage_timing(
                log, timings, "asr_model_load_s", "ASR模型加载", load_elapsed
            )

            text_results, text_timings = _transcribe_asr_chunks_text_only(
                backend,
                chunk_infos,
                "ASR 文本转写",
                on_stage=on_stage,
            )
            _record_stage_timing(
                log,
                timings,
                "asr_text_transcribe_s",
                "ASR文本转写",
                text_timings["text_transcribe_s"],
            )
            for timing_key, timing_value in text_timings.items():
                if timing_key == "text_transcribe_s":
                    continue
                timings[timing_key] = timing_value

            text_results = [
                _with_alignment_fallback_window(chunk, text_result)
                for chunk, text_result in zip(chunk_infos, text_results)
            ]
            cueqc_shadow_report, cueqc_shadow_by_chunk = _run_cueqc_shadow(
                audio_path=audio_path,
                chunk_infos=chunk_infos,
                text_results=text_results,
                log=log,
                backend=backend,
            )

            unload_started = time.perf_counter()
            backend.unload_model(on_stage=on_stage)
            unload_elapsed = time.perf_counter() - unload_started
            _record_stage_timing(
                log,
                timings,
                "asr_model_unload_s",
                "ASR模型卸载",
                unload_elapsed,
            )

            # v3-Fusion drop filter: remove model-confirmed drop candidates from
            # the parallel arrays before subtitle timing. Only high-confidence
            # model drops (mode=cueqc_mamba_v3_fusion) are removed; fallback
            # decisions never drop. Indices are aligned across all three lists.
            if _env_bool("CUEQC_DROP_APPLY_ENABLED", "1"):
                chunk_infos, text_results, _drop_log = _apply_cueqc_drop_filter(
                    chunk_infos,
                    text_results,
                    cueqc_shadow_by_chunk,
                )
                if _drop_log:
                    log.append(_drop_log)
                    timings["cueqc_drop_count"] = int(
                        sum(1 for v in cueqc_shadow_by_chunk.values()
                            if v.get("display_hint") == "drop" and v.get("mode") == "cueqc_mamba_v3_fusion")
                    )

            prepared_results, align_timings = _align_TRANSCRIPTION_results(
                backend,
                text_results,
                on_stage=on_stage,
            )
            _record_stage_timing(
                log,
                timings,
                "alignment_s",
                "字幕时间轴生成",
                align_timings["alignment_s"],
            )

            alignment_outcomes: dict[int, dict] = {}
            for idx, chunk, (chunk_result, chunk_log) in zip(
                range(1, len(chunk_infos) + 1),
                chunk_infos,
                prepared_results,
            ):
                chunk_index = int(chunk.get("index", idx - 1))
                chunk_log = list(chunk_log)
                chunk_words = list(chunk_result.get("words", []))
                outcome = _alignment_outcome_for_chunk(
                    chunk=chunk,
                    chunk_result=chunk_result,
                    chunk_words=chunk_words,
                    chunk_log=chunk_log,
                )
                alignment_outcomes[chunk_index] = outcome
                for entry in chunk_log:
                    if entry.startswith(
                        (
                            "ASR 输出",
                            "Subtitle timing",
                            "Subtitle timing word count",
                            "ASR 原始文本长度",
                            "speech-island",
                        )
                    ):
                        log.append(f"chunk {idx}: {entry}")

                for word in chunk_words:
                    word_dicts.append(
                        {
                            "start": chunk["start"] + float(word["start"]),
                            "end": chunk["start"] + float(word["end"]),
                            "word": word["word"],
                            "source_chunk_index": chunk.get("index", idx - 1),
                            "alignment_mode": outcome.get("alignment_mode", ""),
                            "alignment_quality": outcome.get("alignment_quality", ""),
                            "fallback_type": outcome.get("fallback_type", ""),
                            "fallback_subtype": outcome.get("fallback_subtype", ""),
                        }
                    )
            transcript_chunks = _build_transcript_chunks(
                chunk_infos,
                text_results,
                alignment_outcomes,
            )
            for transcript_chunk in transcript_chunks:
                try:
                    chunk_index = int(transcript_chunk.get("index"))
                except (TypeError, ValueError):
                    continue
                cueqc_shadow = cueqc_shadow_by_chunk.get(chunk_index)
                if cueqc_shadow:
                    transcript_chunk["cueqc_shadow"] = dict(cueqc_shadow)
        finally:
            backend.close()

        segment_started = time.perf_counter()
        word_dicts.sort(key=lambda item: (item["start"], item["end"]))
        log.append(f"字幕时间轴汇总词数: {len(word_dicts)}")
        segments = _group_words_to_segments(word_dicts)
        segments = _postprocess_segments(segments)
        segments = _annotate_segments_with_alignment_outcomes(segments, alignment_outcomes)
        for segment in segments:
            chunk_indices: list[int] = []
            for word in segment.get("words") or []:
                try:
                    chunk_indices.append(int(word.get("source_chunk_index")))
                except (TypeError, ValueError):
                    continue
            if not chunk_indices:
                try:
                    chunk_indices.append(int(segment.get("source_chunk_index")))
                except (TypeError, ValueError):
                    pass
            shadows = [
                cueqc_shadow_by_chunk[index]
                for index in list(dict.fromkeys(chunk_indices))
                if index in cueqc_shadow_by_chunk
            ]
            if shadows:
                segment["cueqc_shadow"] = shadows[0] if len(shadows) == 1 else shadows
        segment_elapsed = time.perf_counter() - segment_started
        _record_stage_timing(
            log, timings, "subtitle_segment_s", "字幕分段", segment_elapsed
        )

        total_elapsed = time.perf_counter() - total_started
        _record_stage_timing(
            log,
            timings,
            "asr_alignment_total_s",
            "ASR与字幕时间轴总计",
            total_elapsed,
        )
        log.append(f"过滤后保留字幕: {len(segments)}")

        details = {
            "backend": get_backend_label(),
            "audio_path": audio_path,
            "device": device,
            "chunk_count": len(chunk_infos),
            "transcript_chunks": transcript_chunks,
            "cueqc_shadow": cueqc_shadow_report,
            "stage_timings": timings,
            "word_count": len(word_dicts),
            "segment_count": len(segments),
            "boundary_signature": dict(_LAST_BOUNDARY_SIGNATURE),
            "fallback_count": sum(
                1
                for outcome in alignment_outcomes.values()
                if bool(outcome.get("fallback_active"))
            ),
            "alignment_outcome_counts": {
                key: sum(
                    1
                    for outcome in alignment_outcomes.values()
                    if str(outcome.get("alignment_quality") or "") == key
                )
                for key in sorted(
                    {
                        str(outcome.get("alignment_quality") or "")
                        for outcome in alignment_outcomes.values()
                        if outcome.get("alignment_quality")
                    }
                )
            },
            "fallback_subtype_counts": {
                key: sum(
                    1
                    for outcome in alignment_outcomes.values()
                    if str(outcome.get("fallback_subtype") or "") == key
                )
                for key in sorted(
                    {
                        str(outcome.get("fallback_subtype") or "")
                        for outcome in alignment_outcomes.values()
                        if outcome.get("fallback_subtype")
                    }
                )
            },
        }
        return segments, log, details
    finally:
        if chunk_dir is not None and chunk_dir.exists() and not _KEEP_ASR_CHUNKS:
            _delete_path_for_cleanup(chunk_dir)


def transcribe_and_align(
    audio_path: str,
    device: str,
    on_stage: Callable[[str], None] | None = None,
    include_details: bool = False,
) -> tuple[list[dict], list[str]] | tuple[list[dict], list[str], dict]:
    segments, log, details = _transcribe_and_align_local(
        audio_path,
        device,
        on_stage=on_stage,
    )
    if include_details:
        return segments, log, details
    return segments, log
