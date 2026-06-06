import importlib
import os
import time
import warnings
from pathlib import Path
from typing import Callable

from audio.chunk_packer import PackedChunk, pack_speech_segments
from boundary import load_boundary_refiner
from boundary import cache as _boundary_cache_module
from boundary.candidates import CANDIDATE_EXTRACTOR_VERSION
from boundary.sequence_features import (
    FRAME_SEQUENCE_FRAMES_SCHEMA,
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
)
from boundary.backbones import normalize_boundary_backbone
from boundary.refiner import (
    file_sha1 as _boundary_refiner_file_sha1,
    load_frame_sequence_refiner_checkpoint,
)
from asr import checkpoint as _checkpoint_module
from asr import chunking as _chunking_module
from asr import qc_stage as _qc_stage_module
from asr import transcribe as _transcribe_module
from asr.qc import collect_adaptive_precision_review
from asr.backends import registry as _registry_module

warnings.filterwarnings("ignore")

_registry_module = importlib.reload(_registry_module)
_chunking_module = importlib.reload(_chunking_module)
_checkpoint_module = importlib.reload(_checkpoint_module)
_transcribe_module = importlib.reload(_transcribe_module)
_qc_stage_module = importlib.reload(_qc_stage_module)
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

def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: str) -> int:
    return int(float(os.getenv(name, default)))


def _boundary_config() -> dict:
    refiner_path = os.getenv("BOUNDARY_REFINER_MODEL_PATH", "").strip()
    refiner_path_obj = Path(refiner_path).expanduser() if refiner_path else None
    runtime_adapter = _boundary_refiner_runtime_adapter(refiner_path_obj)
    return {
        "feature_frame_hop_s": _env_float("BOUNDARY_FEATURE_FRAME_HOP_S", "0.02"),
        "boundary_refiner_enabled": _env_bool("BOUNDARY_REFINER_ENABLED", "1"),
        "boundary_refiner_model_path": refiner_path,
        "boundary_refiner_model_sha1": (
            _boundary_refiner_file_sha1(refiner_path_obj)
            if refiner_path_obj is not None and refiner_path_obj.exists()
            else ""
        ),
        "boundary_refiner_runtime_adapter": runtime_adapter,
        "boundary_refiner_backbone": normalize_boundary_backbone(
            os.getenv("BOUNDARY_REFINER_BACKBONE", "transformers.Mamba2Model")
        ),
        "boundary_refiner_device": os.getenv("BOUNDARY_REFINER_DEVICE", "auto").strip()
        or "auto",
        "boundary_refiner_threshold": _env_float("BOUNDARY_REFINER_THRESHOLD", "0.5"),
        "boundary_candidate_extractor_version": CANDIDATE_EXTRACTOR_VERSION,
        "boundary_planner_max_core_chunk_s": _env_float(
            "BOUNDARY_PLANNER_MAX_CORE_CHUNK_S",
            "5.0",
        ),
        "boundary_planner_max_padded_chunk_s": _env_float(
            "BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S",
            "9.0",
        ),
        "boundary_planner_target_chunk_s": _env_float("BOUNDARY_PLANNER_TARGET_CHUNK_S", "3.0"),
        "boundary_planner_min_chunk_s": _env_float("BOUNDARY_PLANNER_MIN_CHUNK_S", "0.4"),
        "boundary_planner_start_weight": _env_float("BOUNDARY_PLANNER_START_WEIGHT", "1.5"),
        "boundary_planner_target_padding_s": _env_float("BOUNDARY_PLANNER_TARGET_PADDING_S", "2.0"),
        "boundary_planner_max_splits_per_segment": _env_int(
            "BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT", "16"
        ),
        "boundary_planner_sequence_batch_size": _env_int(
            "BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE", "256"
        ),
        "boundary_dp_chunk_base_cost": _env_float("BOUNDARY_DP_CHUNK_BASE_COST", "0.04"),
        "boundary_dp_over_target_weight": _env_float("BOUNDARY_DP_OVER_TARGET_WEIGHT", "0.30"),
        "boundary_dp_far_over_target_weight": _env_float("BOUNDARY_DP_FAR_OVER_TARGET_WEIGHT", "1.50"),
        "boundary_dp_under_min_weight": _env_float("BOUNDARY_DP_UNDER_MIN_WEIGHT", "0.20"),
        "boundary_dp_long_gap_weight": _env_float("BOUNDARY_DP_LONG_GAP_WEIGHT", "0.35"),
        "boundary_dp_split_merge_weight": _env_float("BOUNDARY_DP_SPLIT_MERGE_WEIGHT", "0.35"),
    }


def _boundary_refiner_runtime_adapter(path: Path | None) -> str:
    if path is None or not path.exists():
        return "refiner_input_v1"
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Boundary refiner checkpoint must be a dict")
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        adapter = str(metadata.get("runtime_adapter") or "").strip()
        if adapter:
            return adapter
    return "refiner_input_v1"


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
        ),
        target_chunk_s=target_chunk_s,
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
_chunk_original_boundaries = _chunking_module._chunk_original_boundaries

ASRWorkerSystemError = _transcribe_module.ASRWorkerSystemError
_strip_punctuation = _transcribe_module._strip_punctuation
_compact_text_units = _transcribe_module._compact_text_units
_collapse_repeated_noise = _transcribe_module._collapse_repeated_noise
_is_low_value_text = _transcribe_module._is_low_value_text
_clean_segment_text = _transcribe_module._clean_segment_text
_build_timestamp_fallback = _transcribe_module._build_timestamp_fallback
_with_alignment_fallback_window = _transcribe_module._with_alignment_fallback_window
_looks_like_alignment_failure = _transcribe_module._looks_like_alignment_failure
_alignment_failure_reasons = _transcribe_module._alignment_failure_reasons
_split_span_evenly = _transcribe_module._split_span_evenly
_prepare_asr_chunk_results = _transcribe_module._prepare_asr_chunk_results
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
_should_skip_alignment_retry = _transcribe_module._should_skip_alignment_retry
_needs_alignment_fallback = _transcribe_module._needs_alignment_fallback
_split_alignment_sentinel_with_speech_islands = (
    _transcribe_module._split_alignment_sentinel_with_speech_islands
)
_split_alignment_sentinels_with_speech_islands_batch = (
    _transcribe_module._split_alignment_sentinels_with_speech_islands_batch
)
_finalize_aligned_chunk_without_asr_retry = _transcribe_module._finalize_aligned_chunk_without_asr_retry
_refine_chunk_with_subchunks = _transcribe_module._refine_chunk_with_subchunks
_transcribe_asr_chunk_with_retry = _transcribe_module._transcribe_asr_chunk_with_retry
_postprocess_segments = _transcribe_module._postprocess_segments
_ends_sentence = _transcribe_module._ends_sentence
_same_source_chunk = _transcribe_module._same_source_chunk
_should_merge_fragment = _transcribe_module._should_merge_fragment
_join_segment_text = _transcribe_module._join_segment_text
_merge_fragment_segments = _transcribe_module._merge_fragment_segments
_pick_postprocess_split_index = _transcribe_module._pick_postprocess_split_index
_split_long_postprocessed_segment = _transcribe_module._split_long_postprocessed_segment
_split_long_postprocessed_segments = _transcribe_module._split_long_postprocessed_segments
_repair_postprocessed_segment_windows = _transcribe_module._repair_postprocessed_segment_windows
_merge_words_to_segments = _transcribe_module._merge_words_to_segments

_run_TRANSCRIPTION_qc = _qc_stage_module._run_TRANSCRIPTION_qc


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
    if cfg["boundary_refiner_runtime_adapter"] == "frame_sequence_v1":
        boundary_refiner = None
        sequence_boundary_refiner = load_frame_sequence_refiner_checkpoint(
            Path(cfg["boundary_refiner_model_path"]),
            threshold=cfg["boundary_refiner_threshold"],
            backbone_override=cfg["boundary_refiner_backbone"],
            device=cfg["boundary_refiner_device"],
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
    else:
        boundary_refiner = load_boundary_refiner(
            enabled=cfg["boundary_refiner_enabled"],
            model_path=cfg["boundary_refiner_model_path"],
            backbone=cfg["boundary_refiner_backbone"],
            device=cfg["boundary_refiner_device"],
            merge_threshold=cfg["boundary_refiner_threshold"],
            max_merge_gap_s=None,
            target_core_s=cfg["boundary_planner_target_chunk_s"],
        )
        sequence_boundary_refiner = None
        sequence_feature_provider = None
    result_parameters = {
        key: value
        for key, value in result.parameters.items()
        if key not in {"frame_scores", "cut_frame_scores", "sequence_feature_frames"}
    }
    runtime_boundary_signature = {
        **result_parameters,
        "boundary_pipeline": {
            "version": 2,
            "feature_frame_hop_s": cfg["feature_frame_hop_s"],
            "score_frame_hop_s": score_frame_hop_s,
            "feature_sources": {
                "speech_scores": frame_scores is not None,
                "cut_scores": cut_frame_scores is not None,
            },
            "boundary_refiner": (
                boundary_refiner.signature() if boundary_refiner is not None else None
            ),
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
                "max_padded_chunk_s": cfg["boundary_planner_max_padded_chunk_s"],
                "target_chunk_s": cfg["boundary_planner_target_chunk_s"],
                "min_chunk_s": cfg["boundary_planner_min_chunk_s"],
                "start_weight": cfg["boundary_planner_start_weight"],
                "target_padding_s": cfg["boundary_planner_target_padding_s"],
                "max_splits_per_segment": cfg["boundary_planner_max_splits_per_segment"],
                "sequence_batch_size": cfg["boundary_planner_sequence_batch_size"],
                "dp_chunk_base_cost": cfg["boundary_dp_chunk_base_cost"],
                "dp_over_target_weight": cfg["boundary_dp_over_target_weight"],
                "dp_far_over_target_weight": cfg["boundary_dp_far_over_target_weight"],
                "dp_under_min_weight": cfg["boundary_dp_under_min_weight"],
                "dp_long_gap_weight": cfg["boundary_dp_long_gap_weight"],
                "dp_split_merge_weight": cfg["boundary_dp_split_merge_weight"],
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
        max_padded_chunk_s=cfg["boundary_planner_max_padded_chunk_s"],
        target_chunk_s=cfg["boundary_planner_target_chunk_s"],
        min_chunk_s=cfg["boundary_planner_min_chunk_s"],
        target_padding_s=cfg["boundary_planner_target_padding_s"],
        start_weight=cfg["boundary_planner_start_weight"],
        frame_scores=frame_scores,
        score_frame_hop_s=score_frame_hop_s,
        cut_frame_scores=cut_frame_scores,
        boundary_refiner=boundary_refiner,
        sequence_boundary_refiner=sequence_boundary_refiner,
        sequence_feature_provider=sequence_feature_provider,
        max_splits_per_segment=cfg["boundary_planner_max_splits_per_segment"],
        sequence_batch_size=cfg["boundary_planner_sequence_batch_size"],
        dp_chunk_base_cost=cfg["boundary_dp_chunk_base_cost"],
        dp_over_target_weight=cfg["boundary_dp_over_target_weight"],
        dp_far_over_target_weight=cfg["boundary_dp_far_over_target_weight"],
        dp_under_min_weight=cfg["boundary_dp_under_min_weight"],
        dp_long_gap_weight=cfg["boundary_dp_long_gap_weight"],
        dp_split_merge_weight=cfg["boundary_dp_split_merge_weight"],
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
        chunk["speech_left_padding_s"] = packed.left_padding_s
        chunk["speech_right_padding_s"] = packed.right_padding_s
        chunk["boundary_split_reason"] = packed.split_reason
        chunk["boundary_parent_chunk_id"] = packed.parent_chunk_id
        chunk["speech_island_id"] = packed.island_id
        chunk["speech_island_count"] = packed.island_count
        chunk["speech_internal_gap_count"] = packed.internal_gap_count
        chunk["speech_internal_gap_max_s"] = packed.internal_gap_max_s
        chunk["boundary_score"] = packed.boundary_score
        chunk["boundary_reason"] = packed.boundary_reason
        chunk["boundary_source"] = packed.boundary_source
        chunk["boundary_decision_merge"] = packed.boundary_decision_merge
        chunk["boundary_merge_prob"] = packed.boundary_merge_prob
        chunk["boundary_split_prob"] = packed.boundary_split_prob
        chunk["boundary_refine_delta_s"] = packed.boundary_refine_delta_s
        chunk["boundary_decision_source"] = packed.boundary_decision_source
        log.append(
            "[chunk] idx={idx} dur={duration:.1f} speech_segment_count={count} "
            "pad=({left:.2f},{right:.2f}) reason={reason} "
            "parent={parent} island={island}/{islands} gap_max={gap:.2f} "
            "boundary={boundary_reason} source={boundary_source} score={boundary_score} "
            "decision_source={decision_source} merge={decision_merge}".format(
                idx=idx,
                duration=packed.duration,
                count=len(packed.speech_segments),
                left=packed.left_padding_s,
                right=packed.right_padding_s,
                reason=packed.split_reason,
                parent=packed.parent_chunk_id,
                island=packed.island_id,
                islands=packed.island_count,
                gap=packed.internal_gap_max_s,
                boundary_reason=packed.boundary_reason,
                boundary_source=packed.boundary_source,
                boundary_score=packed.boundary_score,
                decision_source=packed.boundary_decision_source,
                decision_merge=packed.boundary_decision_merge,
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


def _alignment_fallback_count_from_log(log: list[str]) -> int:
    markers = (
        "aligner_vad_fallback",
        "even_fallback",
        "Alignment 回退",
        "Alignment 快速回退",
        "Alignment 降级失败",
        "Alignment 降级后仍异常",
        "边界回退",
        "等比分配时间戳",
    )
    chunk_ids: set[str] = set()
    unscoped_count = 0
    for entry in log:
        if not any(marker in entry for marker in markers):
            continue
        if entry.startswith("chunk ") and ":" in entry:
            chunk_id = entry.split(":", 1)[0].replace("chunk", "", 1).strip()
            if chunk_id:
                chunk_ids.add(chunk_id)
                continue
        unscoped_count += 1
    return len(chunk_ids) + unscoped_count


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
                    "asr_qc_s": 0.0,
                    "asr_model_unload_s": 0.0,
                    "alignment_s": 0.0,
                    "alignment_model_unload_s": 0.0,
                    "subtitle_merge_s": 0.0,
                }
            )
            total_elapsed = time.perf_counter() - total_started
            _record_stage_timing(
                log,
                timings,
                "asr_alignment_total_s",
                "ASR与Alignment总计",
                total_elapsed,
            )
            log.append("边界系统未检测到可处理语音块，跳过 ASR")
            details = {
                "backend": get_backend_label(),
                "audio_path": audio_path,
                "device": device,
                "chunk_count": 0,
                "transcript_chunks": [],
                "asr_qc": {
                    "enabled": True,
                    "chunk_count": 0,
                    "reject_count": 0,
                    "warning_count": 0,
                    "generation_error_count": 0,
                    "generation_overflow_count": 0,
                    "timeout_count": 0,
                    "quarantined_count": 0,
                    "empty_text_for_speech_count": 0,
                    "review_uncertain_count": 0,
                    "review_uncertain_items": [],
                    "items": [],
                    "rejected_indices": [],
                },
                "stage_timings": timings,
                "word_count": 0,
                "segment_count": 0,
                "boundary_no_speech": True,
                "boundary_signature": dict(_LAST_BOUNDARY_SIGNATURE),
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

            qc_report, qc_timings = _run_TRANSCRIPTION_qc(
                chunk_infos,
                text_results,
                log,
                on_stage=on_stage,
            )
            _record_stage_timing(
                log,
                timings,
                "asr_qc_s",
                "ASR质检",
                qc_timings["asr_qc_s"],
            )

            text_results, qc_report, adaptive_review_log = collect_adaptive_precision_review(
                chunk_infos,
                text_results,
                qc_report,
            )
            if adaptive_review_log:
                timings["asr_adaptive_review_chunks"] = len(adaptive_review_log)
                log.append(
                    "ASR Adaptive Precision: review_uncertain={count}".format(
                        count=len(adaptive_review_log)
                    )
                )
                log.extend(adaptive_review_log[:8])
                remaining = len(adaptive_review_log) - 8
                if remaining > 0:
                    log.append(
                        "ASR Adaptive Precision: "
                        f"{remaining} additional review chunks omitted from log"
                    )

            text_results = [
                _with_alignment_fallback_window(chunk, text_result)
                for chunk, text_result in zip(chunk_infos, text_results)
            ]
            transcript_chunks = _build_transcript_chunks(chunk_infos, text_results)

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

            prepared_results, align_timings = _align_TRANSCRIPTION_results(
                backend,
                text_results,
                on_stage=on_stage,
            )
            _record_stage_timing(
                log,
                timings,
                "alignment_s",
                "Alignment对齐",
                align_timings["alignment_s"],
            )

            aligner_unload_started = time.perf_counter()
            backend.unload_forced_aligner(on_stage=on_stage)
            aligner_unload_elapsed = time.perf_counter() - aligner_unload_started
            _record_stage_timing(
                log,
                timings,
                "alignment_model_unload_s",
                "Alignment模型卸载",
                aligner_unload_elapsed,
            )

            island_split_words, island_split_logs, island_split_attempted = (
                _split_alignment_sentinels_with_speech_islands_batch(
                    backend,
                    audio_path,
                    chunk_infos,
                    prepared_results,
                    on_stage=on_stage,
                )
            )
            if island_split_attempted:
                log.append(
                    "Alignment speech-island split batch: "
                    f"success_chunks={len(island_split_words)} "
                    f"candidate_chunks={len(island_split_logs)}"
                )

            for idx, chunk, (chunk_result, chunk_log) in zip(
                range(1, len(chunk_infos) + 1),
                chunk_infos,
                prepared_results,
            ):
                chunk_index = int(chunk.get("index", idx - 1))
                chunk_log = list(chunk_log)
                if chunk_index in island_split_logs:
                    chunk_log.extend(island_split_logs[chunk_index])
                if chunk_index in island_split_words:
                    chunk_words = island_split_words[chunk_index]
                else:
                    chunk_words, chunk_log = _finalize_aligned_chunk_without_asr_retry(
                        chunk,
                        chunk_result,
                        chunk_log,
                        backend=None if island_split_attempted else backend,
                        source_audio_path=None if island_split_attempted else audio_path,
                        on_stage=on_stage,
                    )
                for entry in chunk_log:
                    if entry.startswith(
                        (
                            "ASR 输出",
                            "Alignment 策略",
                            "Alignment 哨兵",
                            "Alignment 回退",
                            "Alignment 词数",
                            "Alignment 输入文本长度",
                            "ASR 原始文本长度",
                            "Alignment 模式",
                            "Alignment 异常",
                            "Alignment 边界回退",
                            "Alignment speech-island",
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
                        }
                    )
        finally:
            backend.close()

        merge_started = time.perf_counter()
        word_dicts.sort(key=lambda item: (item["start"], item["end"]))
        log.append(f"Alignment 汇总词数: {len(word_dicts)}")
        segments = _merge_words_to_segments(word_dicts)
        segments = _postprocess_segments(segments)
        merge_elapsed = time.perf_counter() - merge_started
        _record_stage_timing(
            log, timings, "subtitle_merge_s", "字幕合并", merge_elapsed
        )

        total_elapsed = time.perf_counter() - total_started
        _record_stage_timing(
            log,
            timings,
            "asr_alignment_total_s",
            "ASR与Alignment总计",
            total_elapsed,
        )
        log.append(f"过滤后保留字幕: {len(segments)}")

        details = {
            "backend": get_backend_label(),
            "audio_path": audio_path,
            "device": device,
            "chunk_count": len(chunk_infos),
            "transcript_chunks": transcript_chunks,
            "asr_qc": qc_report,
            "stage_timings": timings,
            "word_count": len(word_dicts),
            "segment_count": len(segments),
            "fallback_count": _alignment_fallback_count_from_log(log),
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
