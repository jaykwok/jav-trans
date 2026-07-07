import importlib
import gc
import hashlib
import json
import os
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from audio.chunk_packer import PackedChunk
from boundary import cache as _boundary_cache_module
from boundary.sequence_features import (
    CHUNK_POOLED_PTM_SCHEMA,
    DEFAULT_CHUNK_POOLED_PTM_BINS,
    FRAME_SEQUENCE_FRAMES_SCHEMA,
    PTM_PROJECTION_SCHEMA,
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
    ptm_projection_digest,
)
from boundary.cut_refiner import load_cut_edge_refiner
from boundary.outer_refiner import load_outer_edge_refiner
from boundary.runtime_pipeline import build_semantic_boundary_chunks
from boundary.split_model import (
    load_semantic_split_feature_config,
    load_semantic_split_verifier,
)
from asr import checkpoint as _checkpoint_module
from asr import chunking as _chunking_module
from asr import pre_asr_cueqc as _pre_asr_cueqc_module
from asr import transcribe as _transcribe_module
from asr.backends.qwen import (
    DEFAULT_OUTER_EDGE_REFINER_CHECKPOINT_BY_REPO,
    DEFAULT_CUT_EDGE_REFINER_CHECKPOINT_BY_REPO,
    DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO,
    checkpoint_path_for_repo_env,
)
from asr.backends import registry as _registry_module

warnings.filterwarnings("ignore")

_registry_module = importlib.reload(_registry_module)
_chunking_module = importlib.reload(_chunking_module)
_checkpoint_module = importlib.reload(_checkpoint_module)
_pre_asr_cueqc_module = importlib.reload(_pre_asr_cueqc_module)
_transcribe_module = importlib.reload(_transcribe_module)
_boundary_cache_module = importlib.reload(_boundary_cache_module)

# Call-time backend resolution: reads ASR_BACKEND env at each call so a
# persistent worker serves jobs with different backends without reloading.
_current_asr_backend = _registry_module.current_asr_backend
_QWEN_BACKENDS = _registry_module._QWEN_BACKENDS
_VALID_ASR_BACKENDS = _registry_module._VALID_ASR_BACKENDS

_ASR_CHUNK_ROOT = _chunking_module._ASR_CHUNK_ROOT
_KEEP_ASR_CHUNKS = _chunking_module._KEEP_ASR_CHUNKS
_LAST_BOUNDARY_SIGNATURE: dict = _chunking_module._LAST_BOUNDARY_SIGNATURE
_LAST_BOUNDARY_CACHE_EVENT: dict | None = None
# APPEND=0 overwrites once per export path in this process; later same-path writes append for multi-video workflows.
_PRE_ASR_EXPORT_OVERWRITTEN_PATHS: set[str] = set()
_ASR_CHECKPOINT_ENABLED = _transcribe_module._ASR_CHECKPOINT_ENABLED
_JSON_PAYLOAD_INLINE_ARRAY_LIMIT = 4096


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: str) -> int:
    return int(float(os.getenv(name, default)))


def _boundary_config() -> dict:
    outer_refiner_path = checkpoint_path_for_repo_env(
        repo_id=_current_asr_backend(),
        mapping_env="OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        default_mapping=DEFAULT_OUTER_EDGE_REFINER_CHECKPOINT_BY_REPO,
    )
    split_model_path = checkpoint_path_for_repo_env(
        repo_id=_current_asr_backend(),
        mapping_env="SEMANTIC_SPLIT_MODEL_PATH_BY_REPO",
        default_mapping=DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO,
    )
    cut_refiner_path = checkpoint_path_for_repo_env(
        repo_id=_current_asr_backend(),
        mapping_env="CUT_EDGE_REFINER_MODEL_PATH_BY_REPO",
        default_mapping=DEFAULT_CUT_EDGE_REFINER_CHECKPOINT_BY_REPO,
    )
    return {
        "feature_frame_hop_s": _env_float("BOUNDARY_FEATURE_FRAME_HOP_S", "0.02"),
        "outer_edge_refiner_model_path": outer_refiner_path,
        "semantic_split_model_path": split_model_path,
        "cut_edge_refiner_model_path": cut_refiner_path,
        "outer_edge_refiner_device": os.getenv("OUTER_EDGE_REFINER_DEVICE", "auto").strip()
        or "auto",
        "semantic_split_device": os.getenv("SEMANTIC_SPLIT_DEVICE", "auto").strip() or "auto",
        "cut_edge_refiner_device": os.getenv("CUT_EDGE_REFINER_DEVICE", "auto").strip() or "auto",
    }


def _require_learned_candidates_for_island_split(
    split_verifier,
    boundary_parameters,
) -> None:
    """Refuse to run a v2 (island-sequence) split model over bootstrap
    candidates. v2 checkpoints are trained on learned-proposer candidates;
    the energy-valley bootstrap is only a dataset-bootstrapping tool and a
    legitimate runtime source solely for split-v1 chains."""

    if not hasattr(split_verifier, "decide_islands"):
        return
    if str((boundary_parameters or {}).get("proposal_checkpoint") or "").strip():
        return
    raise RuntimeError(
        "Semantic Split v2 requires learned boundary-proposal candidates, but "
        "the boundary backend produced bootstrap energy-valley candidates (no "
        "proposal checkpoint resolved for this ASR repo). Promote the "
        "BoundaryProposalScorer checkpoint or set "
        "SPEECH_BOUNDARY_JA_PROPOSAL_CHECKPOINT_BY_REPO; bootstrap candidates "
        "are not a valid input distribution for v2 split checkpoints."
    )


_split_projection_npz_cache: dict[tuple[str, int, int], str] = {}


def _split_checkpoint_projection_npz(checkpoint_path: Path) -> str:
    """Materialize the split checkpoint's embedded PTM projection as an npz
    next to the checkpoint and return its path ('' when the checkpoint has no
    projection). The checkpoint is the single source of truth: the boundary
    backend pre-projects sequence features with exactly this basis."""

    try:
        stat = checkpoint_path.stat()
    except OSError:
        return ""
    cache_key = (str(checkpoint_path), stat.st_mtime_ns, stat.st_size)
    cached = _split_projection_npz_cache.get(cache_key)
    if cached is not None and (not cached or Path(cached).exists()):
        return cached
    try:
        feature_config = load_semantic_split_feature_config(checkpoint_path)
    except Exception:
        # The authoritative error surfaces in load_semantic_split_verifier;
        # a projection-trained checkpoint that slips through here still fails
        # loudly downstream (provider refuses to project capped-dim frames).
        _pipeline_logger.debug(
            "[boundary] split checkpoint feature_config unreadable: %s",
            checkpoint_path,
            exc_info=True,
        )
        _split_projection_npz_cache[cache_key] = ""
        return ""
    projection = feature_config.get("ptm_projection") or None
    if projection is None:
        _split_projection_npz_cache[cache_key] = ""
        return ""
    mean = np.asarray(projection["mean"], dtype=np.float32)
    components = np.asarray(projection["components"], dtype=np.float32)
    digest = str(projection.get("digest") or "") or ptm_projection_digest(
        mean, components
    )
    npz_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}.ptm_projection.{digest[:16]}.npz"
    )
    if not npz_path.exists():
        tmp_path = npz_path.with_name(npz_path.name + f".tmp{os.getpid()}")
        np.savez(
            tmp_path,
            schema=np.asarray(PTM_PROJECTION_SCHEMA),
            mean=mean,
            components=components,
        )
        # np.savez appends .npz to paths without the suffix.
        os.replace(f"{tmp_path}.npz", npz_path)
    _split_projection_npz_cache[cache_key] = str(npz_path)
    return str(npz_path)


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
    if not isinstance(ptm, (list, np.ndarray)) or not isinstance(mfcc, (list, np.ndarray)):
        return None
    try:
        hop = float(frame_hop_s)
    except (TypeError, ValueError):
        return None
    if hop <= 0.0:
        return None
    ptm_projected = payload.get("ptm_projected")
    if not isinstance(ptm_projected, (list, np.ndarray)):
        ptm_projected = None
    return FrameSequenceFeatureProvider(
        duration_s=float(duration_s),
        frame_hop_s=hop,
        ptm=ptm,
        mfcc=mfcc,
        ptm_projected=ptm_projected,
        ptm_projected_digest=str(payload.get("ptm_projection_digest") or ""),
        config=FrameSequenceFeatureConfig(
            left_context_s=_env_float("BOUNDARY_FRAME_SEQUENCE_LEFT_CONTEXT_S", "0.60"),
            right_context_s=_env_float("BOUNDARY_FRAME_SEQUENCE_RIGHT_CONTEXT_S", "0.60"),
            max_ptm_dims=_env_int("BOUNDARY_FRAME_SEQUENCE_MAX_PTM_DIMS", "128"),
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
            "edge_sequence_v2 Boundary Refiner requires "
            f"{FRAME_SEQUENCE_FRAMES_SCHEMA} in SpeechBoundary-JA output"
        )
    return provider


get_backend_label = _registry_module.get_backend_label
_resolve_asr_backend = _registry_module._resolve_asr_backend
_create_asr_backend = _registry_module._create_asr_backend
_is_timed_out_result = _checkpoint_module._is_timed_out_result
_checkpointable_text_results = _checkpoint_module._checkpointable_text_results
_delete_path_for_cleanup = _checkpoint_module._delete_path_for_cleanup
_get_asr_checkpoint_source = _checkpoint_module._get_asr_checkpoint_source

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
_postprocess_segments = _transcribe_module._postprocess_segments
_repair_postprocessed_segment_windows = _transcribe_module._repair_postprocessed_segment_windows
_group_words_to_segments = _transcribe_module._group_words_to_segments


def _sync_checkpoint_state() -> None:
    _checkpoint_module._ASR_CHUNK_ROOT = _ASR_CHUNK_ROOT
    _checkpoint_module._LAST_BOUNDARY_SIGNATURE = _LAST_BOUNDARY_SIGNATURE


def _get_asr_generation_checkpoint_signature() -> dict:
    _sync_checkpoint_state()
    return _checkpoint_module._get_asr_runtime_signature(
        last_boundary_signature=_LAST_BOUNDARY_SIGNATURE,
    )


def _get_asr_runtime_signature(last_boundary_signature: dict | None = None) -> dict:
    _sync_checkpoint_state()
    return _checkpoint_module._get_asr_runtime_signature(
        last_boundary_signature=_LAST_BOUNDARY_SIGNATURE if last_boundary_signature is None else last_boundary_signature,
    )


def _get_asr_checkpoint_path(audio_path: str) -> Path:
    _sync_checkpoint_state()
    return _checkpoint_module._get_asr_checkpoint_path(
        audio_path,
        last_boundary_signature=_LAST_BOUNDARY_SIGNATURE,
        chunk_root=_ASR_CHUNK_ROOT,
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


def _json_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and value.size > _JSON_PAYLOAD_INLINE_ARRAY_LIMIT:
            array = np.ascontiguousarray(value)
            return {
                "array_type": "ndarray",
                "dtype": str(array.dtype),
                "shape": [int(item) for item in array.shape],
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        return _json_payload(value.tolist())
    if isinstance(value, np.generic):
        return _json_payload(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return str(value)


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
    *,
    on_stage: Callable[[str], None] | None = None,
) -> list[tuple[float, float]] | list[PackedChunk]:
    def progress(label: str, current: int, total: int) -> None:
        if on_stage is not None:
            on_stage(f"{label} {current}/{total}")

    cfg = _boundary_config()
    _set_last_boundary_cache_event(None)

    restore_sequence_export = os.environ.get("SPEECH_BOUNDARY_JA_EXPORT_SEQUENCE_FEATURES")
    os.environ["SPEECH_BOUNDARY_JA_EXPORT_SEQUENCE_FEATURES"] = "1"
    restore_sequence_projection = os.environ.get(
        "SPEECH_BOUNDARY_JA_SEQUENCE_PTM_PROJECTION"
    )
    split_projection_npz = _split_checkpoint_projection_npz(
        Path(cfg["semantic_split_model_path"])
    )
    if split_projection_npz:
        os.environ["SPEECH_BOUNDARY_JA_SEQUENCE_PTM_PROJECTION"] = split_projection_npz
    else:
        os.environ.pop("SPEECH_BOUNDARY_JA_SEQUENCE_PTM_PROJECTION", None)
    try:
        from boundary import get_boundary_backend

        boundary_backend = get_boundary_backend()
        boundary_signature = boundary_backend.signature()
        progress("边界缓存", 0, 1)
        cached = _boundary_cache_module.load_processing_spans(
            audio_path,
            boundary_signature=boundary_signature,
            boundary_config=cfg,
        )
        if cached is not None:
            progress("边界缓存", 1, 1)
            spans, runtime_boundary_signature, event = cached
            if on_stage is not None:
                on_stage(
                    "边界缓存命中：已复用 SpeechIsland/Outer/Split/Cut 结果，"
                    "本次未执行四个边界模型"
                )
            _set_last_boundary_signature(runtime_boundary_signature)
            _pipeline_logger.info(
                "[boundary-cache] hit path=%s digest=%s",
                event["path"],
                event["digest"],
            )
            _set_last_boundary_cache_event(event)
            return spans

        progress("边界缓存", 1, 1)
        progress("语音岛检测", 0, 1)
        result = boundary_backend.segment(audio_path)
        progress("语音岛检测", 1, 1)
    finally:
        if restore_sequence_export is not None:
            os.environ["SPEECH_BOUNDARY_JA_EXPORT_SEQUENCE_FEATURES"] = restore_sequence_export
        else:
            os.environ.pop("SPEECH_BOUNDARY_JA_EXPORT_SEQUENCE_FEATURES", None)
        if restore_sequence_projection is not None:
            os.environ["SPEECH_BOUNDARY_JA_SEQUENCE_PTM_PROJECTION"] = (
                restore_sequence_projection
            )
        else:
            os.environ.pop("SPEECH_BOUNDARY_JA_SEQUENCE_PTM_PROJECTION", None)
    frame_scores = result.parameters.get("frame_scores")
    candidate_frame_scores = result.parameters.get("candidate_frame_scores")
    score_frame_hop_s = result.parameters.get("frame_hop_s")
    sequence_feature_frames = result.parameters.get("sequence_feature_frames")
    outer_refiner = load_outer_edge_refiner(
        Path(cfg["outer_edge_refiner_model_path"]),
        device=cfg["outer_edge_refiner_device"],
        expected_ptm_repo_id=_current_asr_backend(),
    )
    split_verifier = load_semantic_split_verifier(
        Path(cfg["semantic_split_model_path"]),
        device=cfg["semantic_split_device"],
        expected_ptm_repo_id=_current_asr_backend(),
    )
    _require_learned_candidates_for_island_split(split_verifier, result.parameters)
    cut_refiner = load_cut_edge_refiner(
        Path(cfg["cut_edge_refiner_model_path"]),
        device=cfg["cut_edge_refiner_device"],
        expected_ptm_repo_id=_current_asr_backend(),
    )
    _pipeline_logger.info(
        "[boundary] semantic model devices "
        "outer_requested=%s outer_actual=%s "
        "split_requested=%s split_actual=%s "
        "cut_requested=%s cut_actual=%s",
        cfg["outer_edge_refiner_device"],
        getattr(outer_refiner, "device", "unknown"),
        cfg["semantic_split_device"],
        getattr(split_verifier, "device", "unknown"),
        cfg["cut_edge_refiner_device"],
        getattr(cut_refiner, "device", "unknown"),
    )
    sequence_feature_provider = _required_sequence_feature_provider_from_result(
        sequence_feature_frames,
        duration_s=result.audio_duration_sec,
    )
    speech_feature_export_path = os.getenv(
        "SPEECH_ISLAND_FEATURE_EXPORT_PATH", ""
    ).strip()
    if speech_feature_export_path:
        speech_feature_path = Path(speech_feature_export_path)
        speech_feature_path.parent.mkdir(parents=True, exist_ok=True)
        speech_feature_arrays = {
            "ptm": np.asarray(sequence_feature_frames["ptm"], dtype=np.float32),
            "mfcc": np.asarray(sequence_feature_frames["mfcc"], dtype=np.float32),
            "frame_hop_s": np.asarray(
                [sequence_feature_frames["frame_hop_s"]], dtype=np.float32
            ),
        }
        exported_ptm_projected = sequence_feature_frames.get("ptm_projected")
        if exported_ptm_projected is not None:
            speech_feature_arrays["ptm_projected"] = np.asarray(
                exported_ptm_projected, dtype=np.float32
            )
            speech_feature_arrays["ptm_projection_digest"] = np.asarray(
                [str(sequence_feature_frames.get("ptm_projection_digest") or "")]
            )
        np.savez_compressed(speech_feature_path, **speech_feature_arrays)
    result_parameters = {
        key: value
        for key, value in result.parameters.items()
        if key not in {"frame_scores", "candidate_frame_scores", "sequence_feature_frames"}
    }
    runtime_boundary_signature = {
        **result_parameters,
        "boundary_pipeline": {
            "version": 9,
            "order": [
                "speech_island_scorer",
                "outer_edge_refiner",
                "semantic_split_model",
                "cut_edge_refiner",
            ],
            "feature_frame_hop_s": cfg["feature_frame_hop_s"],
            "score_frame_hop_s": score_frame_hop_s,
            "feature_sources": {
                "speech_scores": frame_scores is not None,
                "acoustic_candidate_scores": candidate_frame_scores is not None,
            },
            "outer_edge_refiner": outer_refiner.signature(),
            "semantic_split_model": split_verifier.signature(),
            "cut_edge_refiner": cut_refiner.signature(),
            "sequence_feature_provider": sequence_feature_provider.signature(),
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
    if frame_scores is None:
        raise ValueError("semantic boundary pipeline requires speech frame scores")
    split_audit_records: list[dict] | None = (
        [] if os.getenv("SEMANTIC_SPLIT_FEATURE_EXPORT_PATH", "").strip() else None
    )
    packed = build_semantic_boundary_chunks(
        segments,
        duration_s=result.audio_duration_sec,
        speech_probabilities=frame_scores,
        feature_provider=sequence_feature_provider,
        outer_refiner=outer_refiner,
        split_verifier=split_verifier,
        cut_refiner=cut_refiner,
        split_audit_records=split_audit_records,
        on_stage=on_stage,
    )
    if split_audit_records is not None:
        _write_semantic_split_feature_export(
            Path(os.environ["SEMANTIC_SPLIT_FEATURE_EXPORT_PATH"]),
            audio_path=audio_path,
            records=split_audit_records,
        )
    packed = _annotate_scorer_stats_on_packed_chunks(
        packed,
        frame_scores=frame_scores,
        split_scores=candidate_frame_scores,
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


def _write_semantic_split_feature_export(
    path: Path,
    *,
    audio_path: str,
    records: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        np.savez_compressed(
            path,
            frame_features=np.stack([row["frame_features"] for row in records]),
            scalar_features=np.stack([row["scalar_features"] for row in records]),
            proposal_times_s=np.asarray(
                [row["candidate"]["time_s"] for row in records], dtype=np.float32
            ),
            core_starts_s=np.asarray([row["core_start"] for row in records], dtype=np.float32),
            core_ends_s=np.asarray([row["core_end"] for row in records], dtype=np.float32),
            accepted=np.asarray([row["accepted"] for row in records], dtype=np.bool_),
            p_cut=np.asarray([row["p_cut"] for row in records], dtype=np.float32),
            p_continue=np.asarray([row["p_continue"] for row in records], dtype=np.float32),
            p_unsure=np.asarray([row["p_unsure"] for row in records], dtype=np.float32),
        )
    else:
        # Silence/music-only windows can legitimately produce zero candidates.
        np.savez_compressed(
            path,
            frame_features=np.zeros((0, 0, 0), dtype=np.float32),
            scalar_features=np.zeros((0, 0), dtype=np.float32),
            proposal_times_s=np.zeros(0, dtype=np.float32),
            core_starts_s=np.zeros(0, dtype=np.float32),
            core_ends_s=np.zeros(0, dtype=np.float32),
            accepted=np.zeros(0, dtype=np.bool_),
            p_cut=np.zeros(0, dtype=np.float32),
            p_continue=np.zeros(0, dtype=np.float32),
            p_unsure=np.zeros(0, dtype=np.float32),
        )
    metadata_path = path.with_suffix(".jsonl")
    with metadata_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(records):
            handle.write(
                json.dumps(
                    _json_payload(
                        {
                            "index": index,
                            "audio": audio_path,
                            "core_start": row["core_start"],
                            "core_end": row["core_end"],
                            "accepted": row["accepted"],
                            "label": row["label"],
                            "p_cut": row["p_cut"],
                            "p_continue": row["p_continue"],
                            "p_unsure": row["p_unsure"],
                            **row["candidate"],
                        }
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )


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
            _json_payload(
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
        annotated.append(_json_payload(item))
    return annotated


def _write_pre_asr_candidates_if_requested(
    candidates: list[dict],
    *,
    log: list[str],
) -> None:
    if not candidates:
        return
    output_path_raw = os.getenv("PRE_ASR_CUEQC_EXPORT_CANDIDATES_PATH", "").strip()
    if not output_path_raw:
        return
    output_path = Path(output_path_raw).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    append_requested = os.getenv("PRE_ASR_CUEQC_EXPORT_CANDIDATES_APPEND", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    output_key = str(output_path.resolve())
    mode = "a" if append_requested or output_key in _PRE_ASR_EXPORT_OVERWRITTEN_PATHS else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for row in candidates:
            handle.write(
                json.dumps(_json_payload(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
    if not append_requested:
        _PRE_ASR_EXPORT_OVERWRITTEN_PATHS.add(output_key)
    log.append(
        "Pre-ASR CueQC: exported candidates path={path} count={count} mode={mode}".format(
            path=_display_cache_path(str(output_path)),
            count=len(candidates),
            mode=mode,
        )
    )


def _score_stats_for_span(
    values: list[float] | tuple[float, ...] | None,
    *,
    start_s: float,
    end_s: float,
    frame_hop_s: float,
) -> dict[str, float | None]:
    empty = {
        "mean": None,
        "max": None,
        "p10": None,
        "p50": None,
        "p90": None,
        "std": None,
        "active_ratio_05": None,
        "active_ratio_07": None,
        "active_ratio_09": None,
    }
    if not values:
        return empty
    hop = max(1e-6, float(frame_hop_s))
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    start = max(0, min(data.size, int(float(start_s) / hop)))
    end = max(start, min(data.size, int(np.ceil(float(end_s) / hop))))
    if end <= start:
        return empty
    window = data[start:end]
    if window.size == 0:
        return empty
    return {
        "mean": float(window.mean()),
        "max": float(window.max()),
        "p10": float(np.quantile(window, 0.10)),
        "p50": float(np.quantile(window, 0.50)),
        "p90": float(np.quantile(window, 0.90)),
        "std": float(window.std()),
        "active_ratio_05": float(np.mean(window >= 0.50)),
        "active_ratio_07": float(np.mean(window >= 0.70)),
        "active_ratio_09": float(np.mean(window >= 0.90)),
    }


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
        speech_stats = _score_stats_for_span(
            frame_scores,
            start_s=span.start,
            end_s=span.end,
            frame_hop_s=frame_hop_s,
        )
        split_stats = _score_stats_for_span(
            split_scores,
            start_s=span.start,
            end_s=span.end,
            frame_hop_s=frame_hop_s,
        )
        annotated.append(
            replace(
                span,
                scorer_speech_mean=speech_stats["mean"],
                scorer_speech_max=speech_stats["max"],
                scorer_speech_p90=speech_stats["p90"],
                scorer_speech_p10=speech_stats["p10"],
                scorer_speech_p50=speech_stats["p50"],
                scorer_speech_std=speech_stats["std"],
                scorer_speech_active_ratio_05=speech_stats["active_ratio_05"],
                scorer_speech_active_ratio_07=speech_stats["active_ratio_07"],
                scorer_speech_active_ratio_09=speech_stats["active_ratio_09"],
                scorer_split_mean=split_stats["mean"],
                scorer_split_max=split_stats["max"],
                scorer_split_p90=split_stats["p90"],
                scorer_split_std=split_stats["std"],
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
        bins=_pre_asr_cueqc_module.PRE_ASR_CUEQC_PTM_BINS
    )
    annotated: list[PackedChunk] = []
    for span in spans:
        values = sequence_feature_provider.chunk_pooled_ptm_features(
            start_s=span.start,
            end_s=span.end,
            bins=_pre_asr_cueqc_module.PRE_ASR_CUEQC_PTM_BINS,
        )
        annotated.append(
            replace(
                span,
                pre_asr_ptm_pooling_schema=CHUNK_POOLED_PTM_SCHEMA,
                pre_asr_ptm_pooling_bins=_pre_asr_cueqc_module.PRE_ASR_CUEQC_PTM_BINS,
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
        "schema": "pre_asr_cueqc_report_v2",
        "enabled": _pre_asr_cueqc_module.enabled(),
        "candidate_count": len(spans),
        "drop_count": 0,
        "decisions": [],
    }
    _progress("Pre-ASR CueQC 0/1")
    if not spans:
        _progress("Pre-ASR CueQC 1/1")
        return spans, report
    if not _pre_asr_cueqc_module.enabled():
        _progress(
            f"Pre-ASR CueQC disabled candidates={len(spans)} pass_to_asr={len(spans)}"
        )
        _progress("Pre-ASR CueQC 1/1")
        return spans, report
    started = time.perf_counter()
    model = _pre_asr_cueqc_module.load_active(expected_asr_repo_id=_current_asr_backend())
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
    _progress("Pre-ASR CueQC 1/1")
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
        for key in ("proposal_time_s", "p_cut", "p_continue", "p_unsure"):
            if item.get(key) is not None:
                candidate[key] = float(item[key])
        if item.get("label") is not None:
            candidate["label"] = str(item.get("label") or "")
        if item.get("shared_absolute_timestamp") is not None:
            candidate["shared_absolute_timestamp"] = bool(item["shared_absolute_timestamp"])
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
        chunk["source_abs_start"] = packed.source_abs_start
        chunk["source_abs_end"] = packed.source_abs_end
        chunk["speech_segment_count"] = len(packed.speech_segments)
        chunk["boundary_split_reason"] = packed.split_reason
        chunk["boundary_parent_chunk_id"] = packed.parent_chunk_id
        chunk["speech_island_id"] = packed.island_id
        chunk["speech_island_count"] = packed.island_count
        chunk["speech_internal_gap_count"] = packed.internal_gap_count
        chunk["speech_internal_gap_max_s"] = packed.internal_gap_max_s
        chunk["raw_start"] = packed.raw_start
        chunk["raw_end"] = packed.raw_end
        chunk["raw_duration"] = packed.raw_duration
        chunk["acoustic_start"] = packed.acoustic_start
        chunk["acoustic_end"] = packed.acoustic_end
        chunk["acoustic_duration"] = packed.acoustic_duration
        chunk["boundary_score"] = packed.boundary_score
        chunk["boundary_reason"] = packed.boundary_reason
        chunk["boundary_source"] = packed.boundary_source
        chunk["boundary_start_refine_delta_s"] = packed.boundary_start_refine_delta_s
        chunk["boundary_end_refine_delta_s"] = packed.boundary_end_refine_delta_s
        chunk["boundary_decision_source"] = packed.boundary_decision_source
        chunk["refiner_pred_start_delta_s"] = packed.refiner_pred_start_delta_s
        chunk["refiner_pred_end_delta_s"] = packed.refiner_pred_end_delta_s
        chunk["refiner_applied_start_delta_s"] = packed.refiner_applied_start_delta_s
        chunk["refiner_applied_end_delta_s"] = packed.refiner_applied_end_delta_s
        chunk["refiner_start_confidence"] = packed.refiner_start_confidence
        chunk["refiner_end_confidence"] = packed.refiner_end_confidence
        chunk["refiner_start_source"] = packed.refiner_start_source
        chunk["refiner_end_source"] = packed.refiner_end_source
        chunk["refiner_safety_action"] = packed.refiner_safety_action
        chunk["refiner_safety_reason"] = packed.refiner_safety_reason
        chunk["refiner_effective_start_delta_max_s"] = packed.refiner_effective_start_delta_max_s
        chunk["refiner_effective_end_delta_max_s"] = packed.refiner_effective_end_delta_max_s
        chunk["refiner_fallback_used"] = packed.refiner_fallback_used
        chunk["refiner_shared_boundary_adjusted"] = packed.refiner_shared_boundary_adjusted
        chunk["scorer_speech_mean"] = packed.scorer_speech_mean
        chunk["scorer_speech_max"] = packed.scorer_speech_max
        chunk["scorer_speech_p90"] = packed.scorer_speech_p90
        chunk["scorer_speech_p10"] = packed.scorer_speech_p10
        chunk["scorer_speech_p50"] = packed.scorer_speech_p50
        chunk["scorer_speech_std"] = packed.scorer_speech_std
        chunk["scorer_speech_active_ratio_05"] = packed.scorer_speech_active_ratio_05
        chunk["scorer_speech_active_ratio_07"] = packed.scorer_speech_active_ratio_07
        chunk["scorer_speech_active_ratio_09"] = packed.scorer_speech_active_ratio_09
        chunk["scorer_split_mean"] = packed.scorer_split_mean
        chunk["scorer_split_max"] = packed.scorer_split_max
        chunk["scorer_split_p90"] = packed.scorer_split_p90
        chunk["scorer_split_std"] = packed.scorer_split_std
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
            "decision_source={decision_source} "
            "conf=({start_conf},{end_conf}) safety={safety}".format(
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
                start_conf=packed.refiner_start_confidence,
                end_conf=packed.refiner_end_confidence,
                safety=packed.refiner_safety_action,
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
    _enforce_vram_budget_from_snapshot(snapshot)


def _vram_budget_mb() -> float:
    raw = os.getenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "0").strip().lower()
    if raw in {"", "0", "false", "no", "off", "none"}:
        return 0.0
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _enforce_vram_budget_from_snapshot(snapshot: dict) -> None:
    budget_mb = _vram_budget_mb()
    if budget_mb <= 0.0:
        return
    # Enforce on peak *allocated* VRAM (the real working set that responds to
    # batch size), not reserved. The caching allocator's reserved pool routinely
    # fills dedicated VRAM on a 6GB card without spilling to shared memory, and
    # it does not shrink when batch size is lowered, so a reserved-based budget
    # false-positives and never converges under OOM-retry downshift. Allocated
    # is reported in the message for diagnostics.
    allocated_values: list[float] = []
    reserved_values: list[float] = []
    for key in ("max_allocated_mb", "allocated_mb"):
        try:
            allocated_values.append(float(snapshot.get(key)))
        except (TypeError, ValueError):
            continue
    for key in ("max_reserved_mb", "reserved_mb"):
        try:
            reserved_values.append(float(snapshot.get(key)))
        except (TypeError, ValueError):
            continue
    if not allocated_values:
        return
    peak_allocated = max(allocated_values)
    if peak_allocated <= budget_mb:
        return
    peak_reserved = max(reserved_values) if reserved_values else peak_allocated
    total_mb = snapshot.get("total_mb", "")
    stage = snapshot.get("stage", "")
    raise RuntimeError(
        "GPU VRAM budget exceeded: "
        f"stage={stage} allocated_mb={peak_allocated:.1f} "
        f"reserved_mb={peak_reserved:.1f} "
        f"budget_mb={budget_mb:.1f} total_mb={total_mb}"
    )


def _release_stage_gpu_cache(
    log: list[str],
    snapshots: list[dict],
    stage: str,
    *,
    elapsed_s: float | None = None,
) -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _record_cuda_memory(log, snapshots, stage, elapsed_s=elapsed_s)


def _model_lifecycle_event(
    model_manager: Any | None,
    *,
    stage: str,
    action: str,
    on_stage: Callable[[str], None] | None = None,
) -> None:
    if model_manager is not None:
        hook = getattr(model_manager, "lifecycle_event", None)
        if callable(hook):
            hook(stage=stage, action=action)
    if on_stage is not None:
        on_stage(f"GPU model manager {stage} {action}")


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
    model_manager: Any | None = None,
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
        _model_lifecycle_event(
            model_manager,
            stage="boundary",
            action="load",
            on_stage=on_stage,
        )
        chunk_spans = _build_processing_spans(audio_path, on_stage=on_stage)
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
        _write_pre_asr_candidates_if_requested(pre_asr_candidates, log=log)
        _model_lifecycle_event(
            model_manager,
            stage="boundary_pre_asr",
            action="unload",
            on_stage=on_stage,
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
        _release_stage_gpu_cache(
            log,
            cuda_memory,
            "pre_asr_boundary_models_released",
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
            details = _json_payload({
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
            })
            return [], log, details

        backend = _resolve_asr_backend(device)
        word_dicts: list[dict] = []
        try:
            load_started = time.perf_counter()
            _model_lifecycle_event(
                model_manager,
                stage="asr",
                action="load_exclusive",
                on_stage=on_stage,
            )
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

            unload_started = time.perf_counter()
            backend.unload_model(on_stage=on_stage)
            _model_lifecycle_event(
                model_manager,
                stage="asr",
                action="unload",
                on_stage=on_stage,
            )
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
            try:
                acoustic_start = float(segment.get("start", 0.0))
                acoustic_end = max(acoustic_start, float(segment.get("end", acoustic_start)))
            except (TypeError, ValueError):
                acoustic_start = 0.0
                acoustic_end = 0.0
            segment["acoustic_start"] = acoustic_start
            segment["acoustic_end"] = acoustic_end
            segment["acoustic_duration"] = max(0.0, acoustic_end - acoustic_start)
            source_chunks = [
                chunks_by_index[index]
                for index in list(dict.fromkeys(chunk_indices))
                if index in chunks_by_index
            ]
            if source_chunks:
                chunk_start_values = [
                    float(chunk["acoustic_start"])
                    for chunk in source_chunks
                    if chunk.get("acoustic_start") is not None
                ]
                chunk_end_values = [
                    float(chunk["acoustic_end"])
                    for chunk in source_chunks
                    if chunk.get("acoustic_end") is not None
                ]
                if chunk_start_values and chunk_end_values:
                    segment["chunk_acoustic_start"] = min(chunk_start_values)
                    segment["chunk_acoustic_end"] = max(chunk_end_values)
                    segment["chunk_acoustic_duration"] = max(
                        0.0,
                        segment["chunk_acoustic_end"] - segment["chunk_acoustic_start"],
                    )
                for key in (
                    "raw_start",
                    "raw_end",
                    "raw_duration",
                    "refiner_start_confidence",
                    "refiner_end_confidence",
                    "refiner_start_source",
                    "refiner_end_source",
                    "refiner_safety_action",
                    "refiner_safety_reason",
                    "refiner_fallback_used",
                    "refiner_shared_boundary_adjusted",
                ):
                    values = [chunk.get(key) for chunk in source_chunks if chunk.get(key) is not None]
                    if values:
                        segment[key] = values[0]
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

        details = _json_payload({
            "backend": get_backend_label(),
            "audio_path": audio_path,
            "device": device,
            "chunk_count": len(chunk_infos),
            "transcript_chunks": transcript_chunks,
            "pre_asr_cueqc": pre_asr_cueqc_report,
            "pre_asr_candidates": pre_asr_candidates,
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
        })
        return segments, log, details
    finally:
        if chunk_dir is not None and chunk_dir.exists() and not _KEEP_ASR_CHUNKS:
            _delete_path_for_cleanup(chunk_dir)


def transcribe_and_align(
    audio_path: str,
    device: str,
    on_stage: Callable[[str], None] | None = None,
    include_details: bool = False,
    model_manager: Any | None = None,
) -> tuple[list[dict], list[str]] | tuple[list[dict], list[str], dict]:
    segments, log, details = _transcribe_and_align_local(
        audio_path,
        device,
        on_stage=on_stage,
        model_manager=model_manager,
    )
    if include_details:
        return segments, log, details
    return segments, log
