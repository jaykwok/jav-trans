import importlib
from collections import Counter
import gc
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from asr import chunking as _chunking_module
from asr import result_cache as _result_cache_module
from asr import transcribe as _transcribe_module
from asr.alignment import AlignmentHead, blank_runs
from asr.cueqc import build_candidate as build_cueqc_candidate
from asr.postgate import POSTGATE_SCHEMA, review_all as postgate_review_all
from asr.pregate import PREGATE_SCHEMA, cut_at_pauses
from asr.backends import registry as _registry_module
from pipeline import memory_safety as _memory_safety_module

_registry_module = importlib.reload(_registry_module)
_chunking_module = importlib.reload(_chunking_module)
_result_cache_module = importlib.reload(_result_cache_module)
_transcribe_module = importlib.reload(_transcribe_module)

# Call-time backend resolution: reads ASR_BACKEND env at each call so a
# persistent worker serves jobs with different backends without reloading.
_current_asr_backend = _registry_module.current_asr_backend
_QWEN_BACKENDS = _registry_module._QWEN_BACKENDS
_VALID_ASR_BACKENDS = _registry_module._VALID_ASR_BACKENDS

_LAST_BOUNDARY_SIGNATURE: dict = _chunking_module._LAST_BOUNDARY_SIGNATURE
_JSON_PAYLOAD_INLINE_ARRAY_LIMIT = 4096

# Audio is encoded in pieces of this length when looking for pauses; it is the
# window the ASR encoder is happiest with, not a property of the chunking.
_FEATURE_CHUNK_S = 30.0
ASR_LANGUAGE_FOR_CHUNKING = os.getenv("ASR_LANGUAGE", "Japanese").strip() or "Japanese"
# Loaded at most once per process, and only when an alignment head is actually
# configured. With no head the chunker never touches a model at all.
_FEATURE_MODEL: tuple[Any, Any] | None = None


def _load_asr_model_for_features() -> tuple[Any, Any]:
    global _FEATURE_MODEL
    if _FEATURE_MODEL is not None:
        return _FEATURE_MODEL
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from asr.backends.qwen import active_qwen_asr_model_id, active_qwen_asr_model_path
    from utils.model_paths import resolve_model_spec

    spec = resolve_model_spec(
        active_qwen_asr_model_path() or None, active_qwen_asr_model_id(), download=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(spec)
    model = AutoModelForMultimodalLM.from_pretrained(
        spec, dtype=dtype, device_map=str(device)
    )
    model.eval()
    _FEATURE_MODEL = (model, processor)
    return _FEATURE_MODEL




def current_asr_chunk_root() -> Path:
    return _chunking_module.current_asr_chunk_root()


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: str) -> int:
    return int(float(os.getenv(name, default)))










get_backend_label = _registry_module.get_backend_label
_resolve_asr_backend = _registry_module._resolve_asr_backend
_create_asr_backend = _registry_module._create_asr_backend
_is_timed_out_result = _result_cache_module._is_timed_out_result
_checkpointable_text_results = _result_cache_module._cacheable_text_results
_delete_path_for_cleanup = _transcribe_module._delete_path_for_cleanup

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


def _get_asr_runtime_signature(last_boundary_signature: dict | None = None) -> dict:
    """Model+decode identity plus chunking provenance, for job-level caches.

    The cross-job result cache deliberately keys on less than this (audio
    content replaces the boundary signature); this fuller signature remains the
    key for caches whose values depend on chunk *placement*, e.g. the aligned
    segments cache in main.py.
    """
    boundary_signature = (
        _LAST_BOUNDARY_SIGNATURE
        if last_boundary_signature is None
        else last_boundary_signature
    )
    model_signature = _result_cache_module.model_signature()
    return {
        "version": 9,
        "backend": model_signature["backend"],
        "worker_mode": _registry_module.current_asr_worker_mode(),
        "timestamp": {
            "source": "boundary_chunk_timeline",
        },
        "model": model_signature["model"],
        "language": model_signature["language"],
        "generation": model_signature["generation"],
        "boundary": _result_cache_module._jsonable(boundary_signature)
        if isinstance(boundary_signature, dict)
        else {},
    }


def aggregate_timeout_fragments(job_id: str) -> Path | None:
    return _transcribe_module.aggregate_timeout_fragments(job_id)


import logging as _logging
_pipeline_logger = _logging.getLogger(__name__)


def _set_last_boundary_signature(signature: dict) -> None:
    global _LAST_BOUNDARY_SIGNATURE
    _LAST_BOUNDARY_SIGNATURE = dict(signature)
    _chunking_module._LAST_BOUNDARY_SIGNATURE = _LAST_BOUNDARY_SIGNATURE


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


def _chunking_config() -> dict:
    return {
        "target_chunk_s": _env_float("ASR_CHUNK_TARGET_S", "20.0"),
        "max_chunk_s": _env_float("ASR_CHUNK_MAX_S", "30.0"),
        "min_chunk_s": _env_float("ASR_CHUNK_MIN_S", "2.0"),
        "min_blank_s": _env_float("ASR_CHUNK_MIN_PAUSE_S", "0.6"),
    }


def _blank_runs_for_audio(audio_path: str) -> tuple[list[tuple[float, float]], float, str]:
    """Pauses found by the alignment head, or nothing if it is not configured.

    Returns `(runs, duration_s, source)`. `source` records which path produced
    the cuts, because "the head was not loaded" and "the head found no pauses"
    lead to the same chunking and must not be indistinguishable in the log.
    """
    duration_s = float(_get_wav_duration(audio_path))
    head = None
    try:
        head = AlignmentHead.from_env()
    except Exception as error:  # noqa: BLE001
        _pipeline_logger.warning("[chunking] alignment head unavailable: %s", error)
        return [], duration_s, "fixed_length_head_unavailable"
    if head is None:
        return [], duration_s, "fixed_length_no_head_configured"

    try:
        import numpy as _np
        import torch

        from asr.qwen_native import move_processor_inputs, prepare_transcription_inputs
        from audio.loading import load_audio_16k_mono
        from asr.encoder_features import qwen3_asr_audio_output_lengths

        model, processor = _load_asr_model_for_features()
        audio, rate = load_audio_16k_mono(audio_path)
        if rate != 16000:
            return [], duration_s, "fixed_length_unexpected_sample_rate"
        clip = _np.asarray(audio, dtype=_np.float32)
        duration_s = len(clip) / 16000.0

        pieces = []
        width = int(_FEATURE_CHUNK_S * 16000)
        for offset in range(0, len(clip), width):
            piece = _np.ascontiguousarray(clip[offset : offset + width])
            if len(piece) < 8000:
                continue
            inputs = prepare_transcription_inputs(
                processor, audio=[piece], language=ASR_LANGUAGE_FOR_CHUNKING
            )
            moved = move_processor_inputs(
                inputs, device=model.device, dtype=model.dtype
            )
            with torch.inference_mode():
                features = model.get_audio_features(
                    input_features=moved["input_features"],
                    input_features_mask=moved["input_features_mask"],
                ).pooler_output
            frames = int(
                qwen3_asr_audio_output_lengths(
                    moved["input_features_mask"].sum(dim=1)
                )[0]
            )
            pieces.append(head.log_probs(features[:frames].detach().float().cpu().numpy()))
        if not pieces:
            return [], duration_s, "fixed_length_no_features"
        # Concatenated before deriving runs: a pause straddling a feature-chunk
        # seam would otherwise arrive as two shorter runs, fail the minimum
        # length, and stop being a legal cut point exactly at the seams.
        log_probs = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        runs = blank_runs(
            log_probs,
            upsample=head.upsample,
            min_seconds=_chunking_config()["min_blank_s"],
        )
        return runs, duration_s, "alignment_head_blank_runs"
    except Exception as error:  # noqa: BLE001
        # Chunking must never take transcription down with it. Fixed-length
        # cuts are worse placed, not wrong, because nothing is dropped either way.
        _pipeline_logger.warning("[chunking] blank runs unavailable: %s", error)
        return [], duration_s, "fixed_length_blank_runs_failed"


def _build_processing_spans(
    audio_path: str,
    *,
    on_stage: Callable[[str], None] | None = None,
) -> list[tuple[float, float]]:
    """Contiguous chunks covering the whole file, cut inside pauses.

    This replaced the Scorer/Outer/Split/CueQC/Inner chain on 2026-07-31. That
    chain decided *which audio to keep* from acoustics alone, before any text
    existed, and its errors were unrecoverable - it had been unrunnable since
    2026-07-18 because two of its stages were never trained.

    What is here instead does not decide anything irreversible. The alignment
    head's blank runs are used only to choose where the boundaries fall; the
    output tiles the file exactly, so every second still reaches the decoder and
    a badly placed cut costs a worse boundary rather than a lost line. The
    stronger reading of the same signal - skipping the blank stretches entirely
    - was measured on 2026-07-31 and falsified: it dropped real lines embedded
    in non-semantic vocalisation. See `asr.pregate`.

    With no head configured this degrades to fixed-length chunks, which is a
    placement change and not a correctness one.
    """
    if on_stage is not None:
        on_stage("切分 0/1")
    cfg = _chunking_config()
    runs, duration_s, source = _blank_runs_for_audio(audio_path)
    spans = cut_at_pauses(
        runs,
        duration_s,
        target_s=cfg["target_chunk_s"],
        max_s=cfg["max_chunk_s"],
        min_s=cfg["min_chunk_s"],
    )
    _set_last_boundary_signature(
        {
            "chunking": {
                "schema": PREGATE_SCHEMA,
                "source": source,
                "pause_count": len(runs),
                "chunk_count": len(spans),
                "duration_s": round(duration_s, 3),
                **{key: cfg[key] for key in sorted(cfg)},
            }
        }
    )
    _pipeline_logger.info(
        "[chunking] source=%s pauses=%d chunks=%d duration=%.2fs",
        source,
        len(runs),
        len(spans),
        duration_s,
    )
    if on_stage is not None:
        on_stage("切分 1/1")
    return spans




def _span_boundaries(
    spans: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    return [(float(start), float(end)) for start, end in spans]


























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
            snapshot.update(
                _memory_safety_module.runtime_memory_snapshot(
                    require_shared_vram=False,
                )
            )
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
    snapshot.update(
        _memory_safety_module.runtime_memory_snapshot(
            require_shared_vram=bool(snapshot.get("cuda_available")),
        )
    )
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
    _enforce_vram_budget_from_snapshot(snapshot)
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


def _vram_budget_mb() -> float:
    raw = os.getenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "0").strip().lower()
    if raw in {"", "0", "false", "no", "off", "none"}:
        return 0.0
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _shared_vram_spill_tolerance_mb() -> float:
    # The Windows PDH "GPU Process Memory shared usage" counter jitters by a
    # few MB from driver-side staging buffers even when the CUDA allocator is
    # hard-capped by set_per_process_memory_fraction and cannot spill. A real
    # WDDM spill moves hundreds of MB of tensor pages, so a small tolerance
    # separates counter noise from actual spill. A 4MB jitter once killed a
    # fully finished transcription pass three times in a row.
    raw = os.getenv("ASR_SHARED_VRAM_SPILL_TOLERANCE_MB", "64").strip()
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 64.0


def _enforce_vram_budget_from_snapshot(snapshot: dict) -> None:
    try:
        shared_vram_mb = float(snapshot.get("shared_vram_mb") or 0.0)
    except (TypeError, ValueError):
        shared_vram_mb = 0.0
    if shared_vram_mb > _shared_vram_spill_tolerance_mb():
        raise RuntimeError(
            "GPU shared VRAM spill detected: "
            f"stage={snapshot.get('stage', '')} shared_vram_mb={shared_vram_mb:.3f} "
            f"raw_mb={snapshot.get('shared_vram_raw_mb', '')} "
            f"baseline_mb={snapshot.get('shared_vram_baseline_mb', '')} "
            f"tolerance_mb={_shared_vram_spill_tolerance_mb():.1f}"
        )
    try:
        physical_ram_used_mb = float(snapshot.get("physical_ram_used_mb"))
        physical_ram_budget_mb = float(snapshot.get("physical_ram_budget_mb"))
    except (TypeError, ValueError):
        physical_ram_used_mb = 0.0
        physical_ram_budget_mb = 0.0
    if physical_ram_budget_mb > 0.0 and physical_ram_used_mb > physical_ram_budget_mb:
        raise RuntimeError(
            "Physical RAM budget exceeded: "
            f"stage={snapshot.get('stage', '')} used_mb={physical_ram_used_mb:.1f} "
            f"budget_mb={physical_ram_budget_mb:.1f}"
        )
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


def _empty_postgate_report() -> dict:
    return {
        "schema": POSTGATE_SCHEMA,
        "reviewed": 0,
        "flagged": 0,
        "flags": {},
        "alignment_score_checked": 0,
    }


def _apply_postgate(
    chunk_infos: list[dict],
    prepared_results: list,
    transcript_chunks: list[dict],
    *,
    log: list[str],
) -> dict:
    """Flag cues the audio does not support. Nothing is removed here.

    This is the reversible half of the redesign. The chain that used to sit
    *before* the decoder decided what to keep from acoustics alone and could not
    be second-guessed; every finding here is a mark on a cue that still exists,
    and the subtitle layer and any later audit can read or ignore it.

    `alignment_score` is the signal the old design never had: a runaway loop is
    caught by `unique_ratio`, but fluent invented text is not, and text the
    acoustics do not support aligns badly. The threshold for it is not
    calibrated yet, so `asr.postgate` leaves that check off - see its docstring.
    """
    text_results = [
        {
            "text": str(result.get("text") or ""),
            "raw_text": str(result.get("raw_text") or result.get("text") or ""),
        }
        for result, _ in prepared_results
    ]
    if not text_results or len(text_results) != len(chunk_infos):
        return _empty_postgate_report()

    scores: list[float | None] = []
    for result, _ in prepared_results:
        meta = result.get("timing_meta")
        value = meta.get("alignment_score") if isinstance(meta, dict) else None
        scores.append(float(value) if isinstance(value, (int, float)) else None)

    candidates = [
        build_cueqc_candidate(
            chunk=chunk,
            text_result=text_results[position],
            position=position,
            chunks=chunk_infos,
            text_results=text_results,
        )
        for position, chunk in enumerate(chunk_infos)
    ]
    verdicts = postgate_review_all(candidates, alignment_scores=scores)

    counts: Counter[str] = Counter()
    by_index = {
        int(chunk.get("index", position)): position
        for position, chunk in enumerate(chunk_infos)
    }
    for chunk_entry in transcript_chunks:
        position = by_index.get(int(chunk_entry.get("index", -1)))
        if position is None:
            continue
        verdict = verdicts[position]
        chunk_entry["postgate_flags"] = list(verdict["flags"])
        chunk_entry["alignment_score"] = verdict["alignment_score"]
    for verdict in verdicts:
        for flag in verdict["flags"]:
            counts[flag] += 1

    flagged = sum(1 for verdict in verdicts if verdict["flags"])
    if flagged:
        log.append(
            f"后置闸标记 {flagged}/{len(verdicts)} 块（不删除）: "
            + ", ".join(f"{flag}×{count}" for flag, count in counts.most_common())
        )
    return {
        "schema": POSTGATE_SCHEMA,
        "reviewed": len(verdicts),
        "flagged": flagged,
        "flags": dict(counts),
        "alignment_score_checked": sum(
            1 for verdict in verdicts if verdict["alignment_score_checked"]
        ),
    }


def _segment_alignment_outcome(segment: dict, outcomes: dict[int, dict]) -> dict:
    unique_indices = _segment_chunk_indices(segment)
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


def _segment_chunk_indices(segment: dict) -> list[int]:
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
    return list(dict.fromkeys(chunk_indices))


def _annotate_segments_with_postgate(
    segments: list[dict],
    transcript_chunks: list[dict],
) -> list[dict]:
    """Carry the chunk-level flags down to the cues they were raised against.

    The gate reviews a chunk, but the subtitle layer filters segments, so a flag
    that stopped at the chunk would be a verdict nobody downstream can act on. A
    segment can straddle two chunks; it takes the union, because the point of
    flagging is to be readable by whoever wants to skip it.
    """
    flags_by_chunk = {
        int(entry.get("index", -1)): list(entry.get("postgate_flags") or [])
        for entry in transcript_chunks
        if entry.get("postgate_flags")
    }
    if not flags_by_chunk:
        return segments
    for segment in segments:
        flags: list[str] = []
        for index in _segment_chunk_indices(segment):
            for flag in flags_by_chunk.get(index, ()):
                if flag not in flags:
                    flags.append(flag)
        if flags:
            segment["postgate_flags"] = flags
    return segments


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
    chunk_dir: Path | None = None
    total_started = time.perf_counter()
    _record_cuda_memory(log, cuda_memory, "asr_start", elapsed_s=0.0)

    try:
        _notify("分析静音并切分音频...")
        split_started = time.perf_counter()
        _model_lifecycle_event(
            model_manager,
            stage="chunking",
            action="load",
            on_stage=on_stage,
        )
        chunk_spans = _build_processing_spans(audio_path, on_stage=on_stage)
        _release_stage_gpu_cache(
            log,
            cuda_memory,
            "chunking_released",
            elapsed_s=time.perf_counter() - total_started,
        )
        _model_lifecycle_event(
            model_manager,
            stage="chunking",
            action="unload",
            on_stage=on_stage,
        )
        chunk_dir, chunk_infos = _extract_wav_chunks(
            audio_path, _span_boundaries(chunk_spans), on_stage=on_stage
        )
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
                "postgate": _empty_postgate_report(),
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
            postgate_report = _apply_postgate(
                chunk_infos,
                prepared_results,
                transcript_chunks,
                log=log,
            )
        finally:
            backend.close()

        segment_started = time.perf_counter()
        word_dicts.sort(key=lambda item: (item["start"], item["end"]))
        log.append(f"字幕时间轴汇总词数: {len(word_dicts)}")
        segments = _group_words_to_segments(word_dicts)
        segments = _postprocess_segments(segments)
        segments = _annotate_segments_with_alignment_outcomes(segments, alignment_outcomes)
        segments = _annotate_segments_with_postgate(segments, transcript_chunks)
        chunks_by_index = {
            int(chunk.get("index", index)): chunk
            for index, chunk in enumerate(chunk_infos)
        }
        for segment in segments:
            chunk_indices = _segment_chunk_indices(segment)
            try:
                acoustic_start = float(segment.get("start", 0.0))
                acoustic_end = max(acoustic_start, float(segment.get("end", acoustic_start)))
            except (TypeError, ValueError):
                acoustic_start = 0.0
                acoustic_end = 0.0
            segment["acoustic_start"] = acoustic_start
            segment["acoustic_end"] = acoustic_end
            segment["acoustic_duration"] = max(0.0, acoustic_end - acoustic_start)
            # A segment's source chunks used to carry the edge refiner's own
            # boundaries and confidences, which were copied up to here. Chunks
            # produced by `_extract_wav_chunks` carry only index/start/end/path,
            # so that copying moved nothing after the refiners were retired.
            source_chunks = [
                chunks_by_index[index]
                for index in chunk_indices
                if index in chunks_by_index
            ]
            # The `chunk_acoustic_*` names are kept: downstream readers such as
            # `tools/datasets/prepare_timeline_teacher_dataset.py` fall back to
            # the segment's own bounds when they are absent, so dropping them
            # would silently change what that dataset is built from.
            if source_chunks:
                chunk_start = min(float(chunk["start"]) for chunk in source_chunks)
                chunk_end = max(float(chunk["end"]) for chunk in source_chunks)
                segment["chunk_acoustic_start"] = chunk_start
                segment["chunk_acoustic_end"] = chunk_end
                segment["chunk_acoustic_duration"] = max(0.0, chunk_end - chunk_start)
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
        # No stage-success cleanup here anymore: per-chunk results live in the
        # cross-job content-addressed cache (asr.result_cache) and are meant to
        # outlive the job so identical audio never pays the decode twice.

        details = _json_payload({
            "backend": get_backend_label(),
            "audio_path": audio_path,
            "device": device,
            "chunk_count": len(chunk_infos),
            "transcript_chunks": transcript_chunks,
            "stage_timings": timings,
            "cuda_memory": cuda_memory,
            "word_count": len(word_dicts),
            "segment_count": len(segments),
            "boundary_signature": dict(_LAST_BOUNDARY_SIGNATURE),
            "postgate": postgate_report,
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
        if (
            chunk_dir is not None
            and chunk_dir.exists()
            and not _chunking_module.keep_asr_chunks()
        ):
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
