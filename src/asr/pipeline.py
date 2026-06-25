import importlib
import json
import os
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from audio.chunk_packer import PackedChunk, pack_speech_segments
from boundary import cache as _boundary_cache_module
from boundary.sequence_features import (
    CHUNK_POOLED_PTM_SCHEMA,
    DEFAULT_CHUNK_POOLED_PTM_BINS,
    FRAME_SEQUENCE_FRAMES_SCHEMA,
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
)
from boundary.refiner import (
    file_sha1 as _boundary_refiner_file_sha1,
    load_edge_sequence_refiner_v6_checkpoint,
)
from asr import checkpoint as _checkpoint_module
from asr import chunking as _chunking_module
from asr import cueqc as _cueqc_module
from asr import pre_asr_cueqc as _pre_asr_cueqc_module
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
_pre_asr_cueqc_module = importlib.reload(_pre_asr_cueqc_module)
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

# CueQC Mamba v4 binary model is lazily loaded and cached per process.
# ASR-internals capture is delegated to the backend (inline or subprocess
# worker) via capture_asr_internals(), so the model is reused wherever it is
# loaded — no second Qwen3-ASR in VRAM (v4 binary §5.2).
_CUEQC_REFINER_CACHE: dict[str, object] = {}


def _cueqc_refiner_for(path: str, *, expected_asr_repo_id: str | None = None):
    """Lazily load + cache the CueQC v4 binary refiner for ``path``."""
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
        raise RuntimeError(f"CueQC v4 binary checkpoint load failed for {path}: {exc!r}") from exc

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
        if adapter == "edge_sequence_v1":
            return adapter
    raise ValueError("Boundary Refiner checkpoint must use metadata.runtime_adapter='edge_sequence_v1'")


def _sequence_feature_provider_from_result(
    payload,
    *,
    duration_s: float,
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
    )


def _required_sequence_feature_provider_from_result(
    payload,
    *,
    duration_s: float,
) -> FrameSequenceFeatureProvider:
    provider = _sequence_feature_provider_from_result(
        payload,
        duration_s=duration_s,
    )
    if provider is None:
        raise ValueError(
            "edge_sequence_v1 Boundary Refiner requires "
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
_collapse_repeated_noise = _transcribe_module._collapse_repeated_noise
_is_low_value_text = _transcribe_module._is_low_value_text
_clean_segment_text = _transcribe_module._clean_segment_text
_with_alignment_window = _transcribe_module._with_alignment_window
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

    needs_sequence_features = cfg["boundary_refiner_runtime_adapter"] == "edge_sequence_v1"
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
    split_boundary_frame_scores = result.parameters.get("split_boundary_frame_scores")
    score_frame_hop_s = result.parameters.get("frame_hop_s")
    sequence_feature_frames = result.parameters.get("sequence_feature_frames")
    sequence_boundary_refiner = load_edge_sequence_refiner_v6_checkpoint(
        Path(cfg["boundary_refiner_model_path"]),
        device=cfg["boundary_refiner_device"],
        expected_ptm_repo_id=ASR_BACKEND,
    )
    sequence_feature_provider = _required_sequence_feature_provider_from_result(
        sequence_feature_frames,
        duration_s=result.audio_duration_sec,
    )
    sequence_feature_provider.validate_for_checkpoint(
        sequence_boundary_refiner.feature_names,
        sequence_boundary_refiner.feature_schema_hash,
    )
    result_parameters = {
        key: value
        for key, value in result.parameters.items()
        if key not in {"frame_scores", "split_boundary_frame_scores", "sequence_feature_frames"}
    }
    runtime_boundary_signature = {
        **result_parameters,
        "boundary_pipeline": {
            "version": 6,
            "refiner_schema": "boundary_edge_refiner_v6",
            "feature_frame_hop_s": cfg["feature_frame_hop_s"],
            "score_frame_hop_s": score_frame_hop_s,
            "feature_sources": {
                "speech_scores": frame_scores is not None,
                "split_boundary_scores": split_boundary_frame_scores is not None,
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
                "planner": "edge_sequence_island_planner_v6",
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
        sequence_boundary_refiner=sequence_boundary_refiner,
        sequence_feature_provider=sequence_feature_provider,
        sequence_batch_size=cfg["boundary_planner_sequence_batch_size"],
    )
    packed = _annotate_scorer_stats_on_packed_chunks(
        packed,
        frame_scores=frame_scores,
        split_scores=split_boundary_frame_scores,
        frame_hop_s=float(score_frame_hop_s or cfg["feature_frame_hop_s"]),
    )
    packed = _annotate_pre_asr_ptm_pooling_on_packed_chunks(
        packed,
        sequence_feature_provider=sequence_feature_provider,
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


def _pre_asr_audio_id(audio_path: str) -> str:
    stem = Path(audio_path).stem
    if "." in stem:
        prefix, suffix = stem.rsplit(".", 1)
        if len(suffix) == 8 and all(char in "0123456789abcdefABCDEF" for char in suffix):
            return prefix
    return stem


def _pre_asr_candidates_for_spans(
    audio_path: str,
    spans: list[tuple[float, float]] | list[PackedChunk],
) -> list[dict]:
    audio_id = _pre_asr_audio_id(audio_path)
    candidates: list[dict] = []
    for index in range(len(spans)):
        candidate = _pre_asr_cueqc_module.candidate_from_span(
            spans,
            index,
            require_ptm_pooling=_pre_asr_cueqc_module.enabled(),
        )
        chunk_index = int(candidate.get("index", index))
        start = float(candidate.get("start", 0.0))
        end = float(candidate.get("end", start))
        sample_id = f"preasr-{audio_id}-chunk{chunk_index:05d}"
        candidates.append(
            {
                **candidate,
                "sample_id": sample_id,
                "candidate_id": sample_id,
                "audio_id": audio_id,
                "video_id": audio_id,
                "chunk_index": chunk_index,
                "duration_s": round(max(0.0, end - start), 6),
            }
        )
    return candidates


def _pre_asr_candidates_with_decisions(candidates: list[dict], report: dict) -> list[dict]:
    decisions: dict[int, dict] = {}
    for decision in report.get("decisions") or []:
        try:
            decisions[int(decision.get("index"))] = dict(decision)
        except (TypeError, ValueError):
            continue
    annotated: list[dict] = []
    for candidate in candidates:
        item = dict(candidate)
        try:
            index = int(candidate.get("index"))
        except (TypeError, ValueError):
            index = int(candidate.get("chunk_index", len(annotated)))
        decision = decisions.get(index)
        if decision:
            item["pre_asr_cueqc"] = decision
            item["pre_asr_route"] = str(decision.get("route") or "")
            item["pre_asr_prob_drop"] = decision.get("prob_drop")
            item["pre_asr_prob_keep"] = decision.get("prob_keep")
        annotated.append(item)
    return annotated


def _score_stats_for_span(
    values: list[float] | tuple[float, ...] | None,
    *,
    start_s: float,
    end_s: float,
    frame_hop_s: float,
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    hop = max(1e-6, float(frame_hop_s))
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    start = max(0, min(data.size, int(float(start_s) / hop)))
    end = max(start, min(data.size, int(np.ceil(float(end_s) / hop))))
    if end <= start:
        return None, None, None
    window = data[start:end]
    if window.size == 0:
        return None, None, None
    return float(window.mean()), float(window.max()), float(np.quantile(window, 0.90))


def _annotate_scorer_stats_on_packed_chunks(
    spans: list[tuple[float, float]] | list[PackedChunk],
    *,
    frame_scores: list[float] | None,
    split_scores: list[float] | None,
    frame_hop_s: float,
) -> list[tuple[float, float]] | list[PackedChunk]:
    if not spans or not all(isinstance(span, PackedChunk) for span in spans):
        return spans
    annotated: list[PackedChunk] = []
    for span in spans:
        speech_mean, speech_max, speech_p90 = _score_stats_for_span(
            frame_scores,
            start_s=span.start,
            end_s=span.end,
            frame_hop_s=frame_hop_s,
        )
        split_mean, split_max, split_p90 = _score_stats_for_span(
            split_scores,
            start_s=span.start,
            end_s=span.end,
            frame_hop_s=frame_hop_s,
        )
        annotated.append(
            replace(
                span,
                scorer_speech_mean=speech_mean,
                scorer_speech_max=speech_max,
                scorer_speech_p90=speech_p90,
                scorer_split_mean=split_mean,
                scorer_split_max=split_max,
                scorer_split_p90=split_p90,
            )
        )
    return annotated


def _annotate_pre_asr_ptm_pooling_on_packed_chunks(
    spans: list[tuple[float, float]] | list[PackedChunk],
    *,
    sequence_feature_provider: FrameSequenceFeatureProvider,
) -> list[tuple[float, float]] | list[PackedChunk]:
    if not spans or not all(isinstance(span, PackedChunk) for span in spans):
        return spans
    feature_names = sequence_feature_provider.chunk_pooled_ptm_feature_names(
        bins=DEFAULT_CHUNK_POOLED_PTM_BINS
    )
    annotated: list[PackedChunk] = []
    for span in spans:
        values = sequence_feature_provider.chunk_pooled_ptm_features(
            start_s=span.start,
            end_s=span.end,
            bins=DEFAULT_CHUNK_POOLED_PTM_BINS,
        )
        annotated.append(
            replace(
                span,
                pre_asr_ptm_pooling_schema=CHUNK_POOLED_PTM_SCHEMA,
                pre_asr_ptm_pooling_bins=DEFAULT_CHUNK_POOLED_PTM_BINS,
                pre_asr_ptm_pooling_dim=len(feature_names),
                pre_asr_ptm_pooled_features=values,
            )
        )
    return annotated


def _apply_pre_asr_cueqc(
    spans: list[tuple[float, float]] | list[PackedChunk],
    *,
    log: list[str],
    candidates: list[dict] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[tuple[float, float]] | list[PackedChunk], dict]:
    def _progress(message: str) -> None:
        log.append(message)
        if on_stage:
            on_stage(message)
        print(message, flush=True)

    report = {
        "schema": "pre_asr_cueqc_report_v1",
        "enabled": _pre_asr_cueqc_module.enabled(),
        "candidate_count": len(spans),
        "drop_count": 0,
        "decisions": [],
    }
    if not spans:
        return spans, report
    if not _pre_asr_cueqc_module.enabled():
        _progress(
            f"Pre-ASR CueQC disabled candidates={len(spans)} pass_to_asr={len(spans)}"
        )
        return spans, report
    started = time.perf_counter()
    model = _pre_asr_cueqc_module.load_active(expected_asr_repo_id=ASR_BACKEND)
    candidates = candidates or [
        _pre_asr_cueqc_module.candidate_from_span(
            spans,
            index,
            require_ptm_pooling=True,
        )
        for index in range(len(spans))
    ]
    _progress(
        "Pre-ASR CueQC keep/drop route start candidates={candidates} drop_threshold={threshold}".format(
            candidates=len(candidates),
            threshold=getattr(model, "drop_threshold", ""),
        )
    )
    decisions = model.decide(candidates)
    candidate_by_index = {
        int(candidate.get("index", index)): candidate
        for index, candidate in enumerate(candidates)
    }
    for decision in decisions:
        try:
            decision_index = int(decision.get("index"))
        except (TypeError, ValueError):
            continue
        candidate = candidate_by_index.get(decision_index)
        if candidate is None:
            continue
        for key in ("sample_id", "candidate_id", "audio_id", "video_id", "chunk_index"):
            if key in candidate:
                decision[key] = candidate[key]
    drop_indexes = {
        int(decision.get("index"))
        for decision in decisions
        if decision.get("route") == "drop_before_asr"
    }
    kept = [span for index, span in enumerate(spans) if index not in drop_indexes]
    keep_count = len(kept)
    drop_count = len(drop_indexes)
    confidences = sorted(
        float(decision.get("confidence"))
        for decision in decisions
        if decision.get("confidence") is not None
    )
    confidence_stats = {}
    if confidences:
        confidence_stats = {
            "confidence_min": round(confidences[0], 4),
            "confidence_p10": round(
                confidences[min(len(confidences) - 1, int(len(confidences) * 0.10))],
                4,
            ),
            "confidence_p50": round(confidences[len(confidences) // 2], 4),
            "confidence_mean": round(sum(confidences) / len(confidences), 4),
        }
    report.update(
        {
            "enabled": True,
            "candidate_count": len(spans),
            "drop_count": drop_count,
            "keep_count": keep_count,
            **confidence_stats,
            "model": model.signature(),
            "decisions": decisions,
        }
    )
    elapsed = time.perf_counter() - started
    _progress(
        (
            "Pre-ASR CueQC keep/drop route done candidates={candidates} decisions={decisions} "
            "keep_for_asr={keep} drop_before_asr={drop} pass_to_asr={pass_to_asr} "
            "confidence_min={confidence_min} confidence_p10={confidence_p10} "
            "confidence_p50={confidence_p50} confidence_mean={confidence_mean} "
            "elapsed={elapsed:.2f}s"
        ).format(
            candidates=len(spans),
            decisions=len(decisions),
            keep=keep_count,
            drop=drop_count,
            pass_to_asr=keep_count,
            confidence_min=confidence_stats.get("confidence_min", ""),
            confidence_p10=confidence_stats.get("confidence_p10", ""),
            confidence_p50=confidence_stats.get("confidence_p50", ""),
            confidence_mean=confidence_stats.get("confidence_mean", ""),
            elapsed=elapsed,
        )
    )
    return kept, report


def _normalize_cut_candidates(value: Any) -> list[dict[str, Any]]:
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
        candidates.append(candidate)
    return _dedupe_cut_candidates(candidates)


def _dedupe_cut_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in candidates:
        try:
            time_s = float(candidate["time_s"])
        except (KeyError, TypeError, ValueError):
            continue
        kind = str(candidate.get("kind") or "")
        key = (kind, int(round(time_s * 1000.0)))
        existing = by_key.get(key)
        strength = float(candidate.get("strength") or 0.0)
        existing_strength = float(existing.get("strength") or 0.0) if existing else -1.0
        if existing is None or strength > existing_strength:
            by_key[key] = dict(candidate)
    return [
        by_key[key]
        for key in sorted(
            by_key,
            key=lambda item: (float(by_key[item].get("time_s") or 0.0), item[0]),
        )
    ]


def _cut_candidates_for_segment(
    segment: dict,
    chunks_by_index: dict[int, dict],
    *,
    key: str,
) -> list[dict[str, Any]]:
    start = float(segment.get("start", 0.0))
    end = max(start, float(segment.get("end", start)))
    chunk_indexes: list[int] = []
    for word in segment.get("words") or []:
        try:
            chunk_indexes.append(int(word.get("source_chunk_index")))
        except (TypeError, ValueError):
            continue
    if not chunk_indexes:
        try:
            chunk_indexes.append(int(segment.get("source_chunk_index")))
        except (TypeError, ValueError):
            pass
    selected: list[dict[str, Any]] = []
    for chunk_index in list(dict.fromkeys(chunk_indexes)):
        chunk = chunks_by_index.get(chunk_index)
        if not chunk:
            continue
        for candidate in _normalize_cut_candidates(chunk.get(key)):
            try:
                time_s = float(candidate["time_s"])
            except (KeyError, TypeError, ValueError):
                continue
            if start < time_s < end:
                selected.append(candidate)
    return _dedupe_cut_candidates(selected)


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
        chunk["scorer_speech_mean"] = packed.scorer_speech_mean
        chunk["scorer_speech_max"] = packed.scorer_speech_max
        chunk["scorer_speech_p90"] = packed.scorer_speech_p90
        chunk["scorer_split_mean"] = packed.scorer_split_mean
        chunk["scorer_split_max"] = packed.scorer_split_max
        chunk["scorer_split_p90"] = packed.scorer_split_p90
        chunk["subtitle_min_duration_s"] = packed.subtitle_min_duration_s
        chunk["below_subtitle_min_duration"] = packed.below_subtitle_min_duration
        chunk["micro_chunk_candidate"] = packed.micro_chunk_candidate
        chunk["micro_resolve_action"] = packed.micro_resolve_action
        chunk["micro_resolve_reason"] = packed.micro_resolve_reason
        chunk["left_split_score"] = packed.left_split_score
        chunk["right_split_score"] = packed.right_split_score
        chunk["left_split_prominence"] = packed.left_split_prominence
        chunk["right_split_prominence"] = packed.right_split_prominence
        chunk["left_split_speech_valley"] = packed.left_split_speech_valley
        chunk["right_split_speech_valley"] = packed.right_split_speech_valley
        chunk["primary_cut_candidates"] = _normalize_cut_candidates(
            packed.primary_cut_candidates
        )
        chunk["weak_cut_candidates"] = _normalize_cut_candidates(
            packed.weak_cut_candidates
        )
        chunk["pre_asr_ptm_pooling_schema"] = packed.pre_asr_ptm_pooling_schema
        chunk["pre_asr_ptm_pooling_bins"] = packed.pre_asr_ptm_pooling_bins
        chunk["pre_asr_ptm_pooling_dim"] = packed.pre_asr_ptm_pooling_dim
        chunk["pre_asr_ptm_pooled_features"] = list(packed.pre_asr_ptm_pooled_features or [])
        log.append(
            "[chunk] idx={idx} dur={duration:.1f} speech_segment_count={count} "
            "reason={reason} "
            "parent={parent} island={island}/{islands} gap_max={gap:.2f} "
            "boundary={boundary_reason} source={boundary_source} score={boundary_score} "
            "micro={micro_action} below_subtitle_min={below_min} "
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
                micro_action=packed.micro_resolve_action,
                below_min=packed.below_subtitle_min_duration,
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


def _cuda_memory_snapshot(stage: str, *, elapsed_s: float | None = None) -> dict:
    snapshot: dict[str, object] = {
        "stage": stage,
        "cuda_available": False,
    }
    if elapsed_s is not None:
        snapshot["elapsed_s"] = round(float(elapsed_s), 3)
    try:
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if not torch.cuda.is_available():
            return snapshot
        device_index = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        scale = 1024 * 1024
        snapshot.update(
            {
                "device_index": int(device_index),
                "device_name": torch.cuda.get_device_name(device_index),
                "allocated_mb": round(torch.cuda.memory_allocated(device_index) / scale, 1),
                "reserved_mb": round(torch.cuda.memory_reserved(device_index) / scale, 1),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated(device_index) / scale, 1),
                "max_reserved_mb": round(torch.cuda.max_memory_reserved(device_index) / scale, 1),
                "free_mb": round(free_bytes / scale, 1),
                "total_mb": round(total_bytes / scale, 1),
            }
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must not break ASR
        snapshot["error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def _record_cuda_memory(
    log: list[str],
    snapshots: list[dict],
    stage: str,
    *,
    elapsed_s: float | None = None,
) -> None:
    snapshot = _cuda_memory_snapshot(stage, elapsed_s=elapsed_s)
    snapshots.append(snapshot)
    if not snapshot.get("cuda_available"):
        return
    log.append(
        "CUDA memory {stage}: allocated={allocated}MB reserved={reserved}MB "
        "max_reserved={max_reserved}MB free={free}MB total={total}MB".format(
            stage=stage,
            allocated=snapshot.get("allocated_mb"),
            reserved=snapshot.get("reserved_mb"),
            max_reserved=snapshot.get("max_reserved_mb"),
            free=snapshot.get("free_mb"),
            total=snapshot.get("total_mb"),
        )
    )


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


def _candidate_capture_window(
    candidate: dict,
    fallback_audio_path: str,
) -> tuple[str, float, float]:
    """Resolve the wav path and local capture window for a candidate.

    Candidate ``audio.path`` normally points at the extracted chunk wav. In that
    case the ASR-internals capture must use the chunk-local window
    ``0..duration``. Passing the global movie timestamp into a chunk wav slices
    past EOF and forces every v4 sample into fallback-keep.
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
        if item.get("mode") != "cueqc_mamba_v4_binary"
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


def _apply_cueqc_v4_model(
    *,
    refiner,
    candidates: list[dict],
    backend,
    audio_path: str,
    log: list[str],
) -> list[dict] | None:
    """Run the v4 binary refiner over candidates, reusing the shared ASR model.

    Captures ASR internals via ``backend.capture_asr_internals()`` which works in
    both inline and subprocess worker modes (the capture runs where the model is
    loaded). Returns one decision dict per candidate. Per-candidate capture or
    inference failures are handled inside the refiner, but model-level capture
    unavailability is fatal to the job.
    """
    if not candidates:
        return []
    if backend is None or not hasattr(backend, "capture_asr_internals"):
        log.append("CueQC v4 binary: backend has no capture_asr_internals; cannot continue")
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
        log.append(f"CueQC v4 binary: capture failed ({exc!r}); cannot continue")
        return None
    if not isinstance(asr_internals, list) or len(asr_internals) != len(candidates):
        log.append("CueQC v4 binary: capture count mismatch; cannot continue")
        return None
    capture_failed = [
        str(item.get("error") or item.get("detail") or "")
        for item in asr_internals
        if not (isinstance(item, dict) and item.get("ok"))
    ]
    if capture_failed:
        log.append(
            "CueQC v4 binary: capture fallback candidates={failed}/{total} top_errors={errors}".format(
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
        "CueQC v4 binary: model decisions={decisions} drops={drops} fallback={fallback} "
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


def _merge_cueqc_v4_decisions(
    report: dict,
    candidates: list[dict],
    model_decisions: list[dict],
) -> dict:
    """Replace report decisions with v4 model decisions, keyed by chunk_index."""
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
    report["decision_source"] = "cueqc_mamba_v4_binary"
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

        # v4 binary is required when CueQC is enabled; checkpoint mapping/load
        # failures should stop the job instead of falling back to old rules.
        model_path = checkpoint_path_for_repo_env(
            repo_id=ASR_BACKEND,
            mapping_env="CUEQC_MODEL_PATH_BY_REPO",
            default_mapping=DEFAULT_CUEQC_CHECKPOINT_BY_REPO,
        )
        refiner = _cueqc_refiner_for(model_path, expected_asr_repo_id=ASR_BACKEND) if model_path else None
        if refiner is not None:
            model_decisions = _apply_cueqc_v4_model(
                refiner=refiner,
                candidates=candidates,
                backend=backend,
                audio_path=audio_path,
                log=log,
            )
            if model_decisions is None:
                raise RuntimeError("CueQC v4 binary produced no model decisions")
            report = _merge_cueqc_v4_decisions(report, candidates, model_decisions)

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
        message = f"CueQC v4 binary failed; job cannot continue ({exc!r})"
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
        "alignment_issue_type": selected.get("alignment_issue_type", ""),
        "alignment_issue_subtype": selected.get("alignment_issue_subtype", ""),
        "alignment_quality_reasons": list(selected.get("alignment_quality_reasons") or []),
        "alignment_issue_active": bool(selected.get("alignment_issue_active")),
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
    cuda_memory: list[dict] = []
    transcript_chunks: list[dict] = []
    pre_asr_cueqc_report: dict = _pre_asr_cueqc_module.runtime_signature()
    chunk_dir: Path | None = None
    total_started = time.perf_counter()
    _record_cuda_memory(log, cuda_memory, "asr_start", elapsed_s=0.0)

    try:
        _notify("分析静音并切分音频...")
        split_started = time.perf_counter()
        chunk_spans = _build_processing_spans(audio_path)
        cache_log_entry = _boundary_cache_log_entry(_LAST_BOUNDARY_CACHE_EVENT)
        if cache_log_entry:
            log.append(cache_log_entry)
        pre_asr_candidates = _pre_asr_candidates_for_spans(audio_path, chunk_spans)
        chunk_spans, pre_asr_cueqc_report = _apply_pre_asr_cueqc(
            chunk_spans,
            log=log,
            candidates=pre_asr_candidates,
            on_stage=on_stage,
        )
        pre_asr_candidates = _pre_asr_candidates_with_decisions(
            pre_asr_candidates,
            pre_asr_cueqc_report,
        )
        chunk_dir, chunk_infos = _extract_wav_chunks(
            audio_path, _span_boundaries(chunk_spans), on_stage=on_stage
        )
        _annotate_packed_chunks(chunk_infos, chunk_spans, log)
        split_elapsed = time.perf_counter() - split_started
        log.append(f"切分完成：共 {len(chunk_infos)} 个处理块")
        _record_stage_timing(log, timings, "split_s", "静音分析与切块", split_elapsed)
        _record_cuda_memory(
            log,
            cuda_memory,
            "split_done",
            elapsed_s=time.perf_counter() - total_started,
        )

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
                "cuda_memory": cuda_memory,
                "word_count": 0,
                "segment_count": 0,
                "boundary_no_speech": True,
                "boundary_signature": dict(_LAST_BOUNDARY_SIGNATURE),
                "pre_asr_cueqc": pre_asr_cueqc_report,
                "pre_asr_candidates": pre_asr_candidates,
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
            _record_cuda_memory(
                log,
                cuda_memory,
                "asr_model_load_done",
                elapsed_s=time.perf_counter() - total_started,
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
            _record_cuda_memory(
                log,
                cuda_memory,
                "asr_text_transcribe_done",
                elapsed_s=time.perf_counter() - total_started,
            )

            text_results = [
                _with_alignment_window(chunk, text_result)
                for chunk, text_result in zip(chunk_infos, text_results)
            ]
            cueqc_shadow_report, cueqc_shadow_by_chunk = _run_cueqc_shadow(
                audio_path=audio_path,
                chunk_infos=chunk_infos,
                text_results=text_results,
                log=log,
                backend=backend,
            )
            _record_cuda_memory(
                log,
                cuda_memory,
                "cueqc_done",
                elapsed_s=time.perf_counter() - total_started,
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
            _record_cuda_memory(
                log,
                cuda_memory,
                "asr_model_unload_done",
                elapsed_s=time.perf_counter() - total_started,
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
            _record_cuda_memory(
                log,
                cuda_memory,
                "alignment_done",
                elapsed_s=time.perf_counter() - total_started,
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
                            "alignment_issue_type": outcome.get("alignment_issue_type", ""),
                            "alignment_issue_subtype": outcome.get("alignment_issue_subtype", ""),
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
        chunks_by_index = {
            int(chunk.get("index", index)): chunk
            for index, chunk in enumerate(chunk_infos)
        }
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
            primary_cut_candidates = _cut_candidates_for_segment(
                segment,
                chunks_by_index,
                key="primary_cut_candidates",
            )
            weak_cut_candidates = _cut_candidates_for_segment(
                segment,
                chunks_by_index,
                key="weak_cut_candidates",
            )
            if primary_cut_candidates:
                segment["primary_cut_candidates"] = primary_cut_candidates
            if weak_cut_candidates:
                segment["weak_cut_candidates"] = weak_cut_candidates
        segment_elapsed = time.perf_counter() - segment_started
        _record_stage_timing(
            log, timings, "subtitle_segment_s", "字幕分段", segment_elapsed
        )
        _record_cuda_memory(
            log,
            cuda_memory,
            "subtitle_segment_done",
            elapsed_s=time.perf_counter() - total_started,
        )

        total_elapsed = time.perf_counter() - total_started
        _record_stage_timing(
            log,
            timings,
            "asr_alignment_total_s",
            "ASR与字幕时间轴总计",
            total_elapsed,
        )
        _record_cuda_memory(
            log,
            cuda_memory,
            "asr_alignment_total_done",
            elapsed_s=total_elapsed,
        )
        log.append(f"过滤后保留字幕: {len(segments)}")

        details = {
            "backend": get_backend_label(),
            "audio_path": audio_path,
            "device": device,
            "chunk_count": len(chunk_infos),
            "transcript_chunks": transcript_chunks,
            "pre_asr_cueqc": pre_asr_cueqc_report,
            "pre_asr_candidates": pre_asr_candidates,
            "cueqc_shadow": cueqc_shadow_report,
            "stage_timings": timings,
            "cuda_memory": cuda_memory,
            "word_count": len(word_dicts),
            "segment_count": len(segments),
            "boundary_signature": dict(_LAST_BOUNDARY_SIGNATURE),
            "alignment_issue_count": sum(
                1
                for outcome in alignment_outcomes.values()
                if bool(outcome.get("alignment_issue_active"))
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
            "alignment_issue_subtype_counts": {
                key: sum(
                    1
                    for outcome in alignment_outcomes.values()
                    if str(outcome.get("alignment_issue_subtype") or "") == key
                )
                for key in sorted(
                    {
                        str(outcome.get("alignment_issue_subtype") or "")
                        for outcome in alignment_outcomes.values()
                        if outcome.get("alignment_issue_subtype")
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
