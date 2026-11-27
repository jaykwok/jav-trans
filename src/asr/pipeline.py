from collections import Counter
from dataclasses import dataclass
import gc
import hashlib
import logging
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np

from asr import audio_encoder, chunking, result_cache, transcribe
from asr.alignment import (
    ENCODER_FRAME_S,
    FRAME_CLASSES,
    AlignmentHead,
    alignment_head_configured,
    blank_runs,
    non_speech_runs,
    overlap_save_slices,
    plan_head_windows,
)
from asr.backends import registry
from asr.chunking import (
    CHUNK_CUT_SCHEMA,
    _extract_wav_chunks,
    _get_wav_duration,
    plan_chunk_cuts,
)
from asr.cue_features import (
    build_candidate as build_cue_candidate,
    precompute_compact_texts as precompute_cue_compact_texts,
)
from asr.postgate import POSTGATE_SCHEMA, review_all as postgate_review_all
from asr.transcribe import (
    _align_TRANSCRIPTION_results,
    _alignment_outcome_for_chunk,
    _build_transcript_chunks,
    _check_allocated_vram_budget,
    _delete_path_for_cleanup,
    _group_words_to_segments,
    _postprocess_segments,
    _transcribe_asr_chunks_text_only,
    _with_alignment_window,
)
from asr.backends.registry import get_backend_label, _resolve_asr_backend
from core.diagnostics import is_fatal_cuda_error
from core.typed_config import env_float, env_int, env_text
from pipeline import memory_safety

_JSON_PAYLOAD_INLINE_ARRAY_LIMIT = 4096

# Audio is encoded in pieces of this length when looking for pauses; it is the
# window the ASR encoder is happiest with, not a property of the chunking.
_FEATURE_CHUNK_S = 30.0


@dataclass(frozen=True)
class ChunkPlan:
    """Immutable chunking output returned by ``_build_processing_spans``.

    Implements the read-only sequence protocol over ``spans`` so callers and
    tests expecting a sequence of ``(start, end)`` tuples work unchanged while
    pipeline orchestration reads provenance and frame tracks without relying on
    mutable module globals.
    """

    spans: tuple[tuple[float, float], ...]
    boundary_signature: dict[str, Any]
    cut_provenance: dict[str, Any]
    frame_class_track: dict[str, Any] | None = None

    def __iter__(self):
        return iter(self.spans)

    def __len__(self) -> int:
        return len(self.spans)

    def __getitem__(self, index):
        return self.spans[index]

    def __bool__(self) -> bool:
        return bool(self.spans)


def _load_asr_model_for_features(device: str = "cuda") -> tuple[Any, Any]:
    from utils.gpu_safety import resolve_inference_device
    from utils.model_paths import resolve_model_spec
    from asr.backends.qwen import active_qwen_asr_model_id, active_qwen_asr_model_path

    resolved_device = resolve_inference_device(
        device, stage="Qwen3-ASR feature extraction"
    )
    device_str = str(resolved_device)
    dtype = audio_encoder.detect_asr_dtype(device_str)
    attention = audio_encoder.detect_asr_attention(device_str)
    spec = resolve_model_spec(
        active_qwen_asr_model_path() or None, active_qwen_asr_model_id(), download=True
    )
    return audio_encoder.load_qwen_asr_model(
        model_spec=spec,
        device=device_str,
        dtype=dtype,
        attention=attention,
    )


def _release_feature_model_vram() -> None:
    # The decode stage loads its own copy of the same weights right after
    # chunking. On an 8GB card the two cannot coexist (3.8GiB each), so the
    # feature model is strictly scoped to the blank-run pass — holding it as a
    # per-process cache OOM'd the very first head-enabled run at batch 1. The
    # caller drops every reference first; this only returns the freed blocks.
    outer_exc_active = sys.exc_info()[0] is not None
    try:
        import gc as _gc

        import torch

        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        if not outer_exc_active and is_fatal_cuda_error(exc):
            raise
        _pipeline_logger.warning("Failed to release feature-model VRAM cache: %s", exc)




def current_asr_chunk_root() -> Path:
    """Where chunk WAVs go. Forwarded because `main.py` asks the ASR stage for
    it, and the stage - not the chunker - is what main.py depends on."""
    return chunking.current_asr_chunk_root()


def _asr_language_for_chunking() -> str:
    """Read at call time like every other setting in this stage.

    It was a module-level constant, which made it the one value a persistent
    worker could not pick up between jobs - and the only remaining excuse for
    reloading this module to refresh the environment.
    """
    return env_text("ASR_LANGUAGE", "Japanese")


# Where the last chunking pass recorded how it cut the audio. Owned here because
# this is the only module that both writes and reads it; the chunker itself is
# stateless and takes its geometry as arguments.
_LAST_BOUNDARY_SIGNATURE: dict = {}

# How that pass chose each boundary. Kept apart from the signature above on
# purpose: that one is a cache key, and provenance is derivable from the cuts it
# already digests, so folding it in would invalidate every cached job to record
# something the key does not depend on.
_LAST_CHUNK_CUT_PROVENANCE: dict = {}


def _set_last_boundary_signature(signature: dict) -> None:
    global _LAST_BOUNDARY_SIGNATURE
    _LAST_BOUNDARY_SIGNATURE = dict(signature)


def _set_last_chunk_cut_provenance(provenance: dict) -> None:
    global _LAST_CHUNK_CUT_PROVENANCE
    _LAST_CHUNK_CUT_PROVENANCE = dict(provenance)


# The frame-class posteriors from the same pass that found the pauses. Module
# state for the same reason the boundary signature is: the chunker is stateless
# and the value is produced two calls below the stage that has to report it.
_LAST_FRAME_CLASS_TRACK: dict | None = None


def _set_last_frame_class_track(track: dict | None) -> None:
    global _LAST_FRAME_CLASS_TRACK
    _LAST_FRAME_CLASS_TRACK = track


def _spans_digest(spans: list[tuple[float, float]]) -> str:
    """Identity of a set of cuts, to millisecond resolution.

    Rounded rather than exact because float noise below a millisecond cannot
    move a subtitle, and an exact digest would invalidate caches over it.
    """
    joined = ";".join(f"{begin:.3f}-{end:.3f}" for begin, end in spans)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


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
    model_signature = result_cache.model_signature()
    return {
        # 10: the chunking signature gained `spans_sha256`, so caches keyed on
        # this can no longer be served across a change in cut geometry.
        "version": 10,
        "backend": model_signature["backend"],
        "worker_mode": registry.current_asr_worker_mode(),
        "timestamp": {
            "source": "boundary_chunk_timeline",
        },
        "model": model_signature["model"],
        "language": model_signature["language"],
        "generation": model_signature["generation"],
        "boundary": result_cache._jsonable(boundary_signature)
        if isinstance(boundary_signature, dict)
        else {},
    }


def aggregate_timeout_fragments(job_id: str) -> Path | None:
    """Forwarded because the web layer collects timeout diagnostics from the ASR
    stage, and should not have to know which file inside it does the writing."""
    return transcribe.aggregate_timeout_fragments(job_id)


_pipeline_logger = logging.getLogger(__name__)


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


PAUSE_READINGS = ("blank", "speech")


def _pause_reading() -> str:
    """Which head output supplies the cut points, refusing anything else.

    An unrecognised value must not fall back. `speach` would then run the
    shipped reading while the operator believed they were testing the other
    one, and the A/B would compare an arm against itself - the same silent
    no-op as `--frame-teacher-domain` and the `medium` reasoning tier, both of
    which cost this project days before anyone noticed the setting had never
    applied. Empty is still the default, because a forwarded `KEY=` line is how
    the tuning box clears a setting.
    """
    value = (os.getenv("ASR_CHUNK_PAUSE_READING", "") or "").strip().lower()
    if not value:
        return "blank"
    if value not in PAUSE_READINGS:
        raise ValueError(
            f"ASR_CHUNK_PAUSE_READING={value!r} is not one of {list(PAUSE_READINGS)}"
        )
    return value


def _chunking_config() -> dict:
    return {
        "max_chunk_s": env_float("ASR_CHUNK_MAX_S", 30.0, minimum=1.0),
        "min_chunk_s": env_float("ASR_CHUNK_MIN_S", 2.0, minimum=0.0),
        "min_blank_s": env_float("ASR_CHUNK_MIN_PAUSE_S", 0.6, minimum=0.0),
        # Which reading of the head supplies the cut points. `blank` is the
        # shipped one: stretches the CTC argmax covers entirely with blank.
        # `speech` is the v2 alternative - stretches the frame head says are not
        # speech - which means the same thing without inferring it from the
        # absence of a character. Off by default until the A/B says otherwise,
        # because the pause reading decides chunk placement for every job and a
        # worse placement is paid on every second of audio.
        "pause_reading": _pause_reading(),
        "speech_threshold": env_float("ASR_CHUNK_SPEECH_THRESHOLD", 0.5, minimum=0.0, maximum=1.0),
    }


FRAME_CLASS_TRACK_SCHEMA = "asr_frame_class_track_v1"


def _encode_frame_class_track(posteriors, *, frame_s: float) -> dict:
    """Whole-film frame-class posteriors, small enough to sit in a JSON cache.

    Quantised to a byte per class. The consumers are a mean over a cue's span and
    a comparison against a threshold, and 1/255 is far finer than any threshold
    here - while float32 JSON would be roughly twenty times the size for
    precision nothing reads.

    Carried rather than recomputed because the cue filter needs the classes over
    spans that do not exist yet when this runs: cues are cut by the layout DP
    long after chunking, and re-encoding the film to ask again would be a second
    pass that could disagree with this one.
    """
    import base64

    import numpy as _np

    array = _np.asarray(posteriors, dtype=_np.float32)
    if array.ndim != 2 or array.shape[1] != len(FRAME_CLASSES):
        raise ValueError(f"frame posteriors must be (T, {len(FRAME_CLASSES)})")
    quantised = _np.clip(_np.rint(array * 255.0), 0, 255).astype(_np.uint8)
    return {
        "schema": FRAME_CLASS_TRACK_SCHEMA,
        "classes": list(FRAME_CLASSES),
        "frame_s": round(float(frame_s), 8),
        "frames": int(quantised.shape[0]),
        "scale": 255.0,
        "data": base64.b64encode(quantised.tobytes()).decode("ascii"),
    }


def decode_frame_class_track(track: dict | None):
    """`(posteriors, frame_s)` or None. The inverse of the encoder above."""
    import base64

    import numpy as _np

    if not isinstance(track, dict) or track.get("schema") != FRAME_CLASS_TRACK_SCHEMA:
        return None
    classes = list(track.get("classes") or [])
    if tuple(classes) != FRAME_CLASSES:
        # Positional columns; a different order is a different meaning.
        return None
    try:
        raw = base64.b64decode(str(track["data"]))
        frames = int(track["frames"])
        scale = float(track.get("scale") or 255.0)
        frame_s = float(track["frame_s"])
    except (KeyError, TypeError, ValueError):
        return None
    if frames <= 0 or len(raw) != frames * len(FRAME_CLASSES):
        return None
    array = _np.frombuffer(raw, dtype=_np.uint8).reshape(frames, len(FRAME_CLASSES))
    return array.astype(_np.float32) / scale, frame_s


def _blank_runs_for_audio(
    audio_path: str,
    *,
    on_stage: Callable[[str], None] | None = None,
    backend: Any | None = None,
) -> tuple[list[tuple[float, float]], float, str, dict | None]:
    """Pauses found by the alignment head, or nothing if it is not configured.

    Returns ``(runs, duration_s, source, frame_track)``. When ``backend`` is
    supplied, shares its resident ASR model and ``AlignmentHead`` instead of
    loading a duplicate copy, and uses GPU-batched argmax / frame posteriors to
    avoid multi-gigabyte CPU CTC buffers.
    """
    duration_s = float(_get_wav_duration(audio_path))
    head = None
    if backend is not None and getattr(backend, "_alignment_head", False) not in (
        False,
        None,
    ):
        head = backend._alignment_head
    else:
        try:
            head = AlignmentHead.from_env()
        except Exception as error:  # noqa: BLE001
            if is_fatal_cuda_error(error):
                raise
            _pipeline_logger.warning("[chunking] alignment head unavailable: %s", error)
            return [], duration_s, "fixed_length_head_unavailable", None
        if head is not None and backend is not None and getattr(
            backend, "_alignment_head", False
        ) is False:
            backend._alignment_head = head
    if head is None:
        return [], duration_s, "fixed_length_no_head_configured", None

    model = None
    processor = None
    owns_model = False
    try:
        import numpy as _np
        import torch

        from audio.loading import load_audio_16k_mono

        if backend is not None:
            if getattr(backend, "model", None) is None:
                loader = getattr(backend, "load", None)
                if callable(loader):
                    loader(on_stage=on_stage)
            model = getattr(backend, "model", None)
            processor = getattr(backend, "processor", None)
            if model is None or processor is None:
                return [], duration_s, "fixed_length_model_unavailable", None
        else:
            model, processor = _load_asr_model_for_features()
            owns_model = True

        audio, rate = load_audio_16k_mono(audio_path)
        if rate != 16000:
            return [], duration_s, "fixed_length_unexpected_sample_rate", None
        clip = _np.asarray(audio, dtype=_np.float32)
        duration_s = len(clip) / 16000.0

        width = int(_FEATURE_CHUNK_S * 16000)
        context_frames = head.context_frames
        plan = plan_head_windows(
            len(clip), window_samples=width, context_frames=context_frames
        )
        windows = [
            _np.ascontiguousarray(clip[start:end]) for start, end, _ in plan
        ]
        bases = [base for _, _, base in plan]

        cfg = _chunking_config()
        has_frame_head = bool(getattr(head, "frame_head_available", False))
        use_gpu_head_batch = hasattr(head, "forward_batch_from_gpu_features")
        skip_ctc = (
            use_gpu_head_batch
            and cfg["pause_reading"] == "speech"
            and has_frame_head
        )

        batch_size = env_int("ASR_FEATURE_BATCH_SIZE", 4, minimum=1)
        pieces: list = []
        frame_pieces: list = []
        returned_frames: list[int] = []
        language = _asr_language_for_chunking()
        model_device = getattr(model, "device", getattr(backend, "device", "cuda"))
        model_dtype = getattr(model, "dtype", getattr(backend, "dtype", None))

        for start in range(0, len(windows), batch_size):
            group = windows[start : start + batch_size]
            features, lengths = audio_encoder.encode_audio_features_batch(
                model,
                processor,
                group,
                device=model_device,
                dtype=model_dtype,
                language=language,
                stage_label="chunking",
            )
            if use_gpu_head_batch:
                ctc_batch, frame_batch = head.forward_batch_from_gpu_features(
                    features,
                    lengths,
                    need_ctc_log_probs=not skip_ctc,
                    argmax_only=True,
                    need_frames=has_frame_head,
                )
                pieces.extend(ctc_batch)
                frame_pieces.extend(frame_batch)
                returned_frames.extend(lengths)
            else:
                hidden_list = audio_encoder.split_encoded_features_numpy(
                    features, lengths
                )
                for hidden_slice, length in zip(hidden_list, lengths, strict=True):
                    ctc, frames = head.log_probs_with_frames(hidden_slice)
                    pieces.append(ctc)
                    frame_pieces.append(frames)
                    returned_frames.append(int(length))
            if on_stage is not None and windows:
                processed = min(len(windows), start + len(group))
                on_stage(f"切分 {processed}/{len(windows)}")

        if not returned_frames:
            return [], duration_s, "fixed_length_no_features", None

        keep = overlap_save_slices(
            list(zip(bases[: len(returned_frames)], returned_frames)),
            context_frames=context_frames,
        )

        frame_track = None
        posteriors = None
        if frame_pieces and all(piece is not None for piece in frame_pieces):
            kept_frames = [
                piece[begin * head.upsample : finish * head.upsample]
                for piece, (begin, finish) in zip(frame_pieces, keep)
                if finish > begin
            ]
            if kept_frames:
                if isinstance(kept_frames[0], _np.ndarray):
                    posteriors = (
                        _np.concatenate(kept_frames, axis=0)
                        if len(kept_frames) > 1
                        else kept_frames[0]
                    )
                else:
                    stacked = (
                        torch.cat(kept_frames, dim=0)
                        if len(kept_frames) > 1
                        else kept_frames[0]
                    )
                    posteriors = stacked.exp().detach().cpu().numpy()
                frame_track = _encode_frame_class_track(
                    posteriors,
                    frame_s=ENCODER_FRAME_S / float(head.upsample),
                )
                if cfg["pause_reading"] == "speech":
                    runs = non_speech_runs(
                        posteriors,
                        upsample=head.upsample,
                        min_seconds=cfg["min_blank_s"],
                        speech_threshold=cfg["speech_threshold"],
                    )
                    return runs, duration_s, "alignment_head_speech_runs", frame_track

        kept = [
            piece[begin * head.upsample : finish * head.upsample]
            for piece, (begin, finish) in zip(pieces, keep)
            if piece is not None and finish > begin
        ]
        if not kept:
            return [], duration_s, "fixed_length_no_features", frame_track
        if isinstance(kept[0], _np.ndarray):
            ctc_sequence = (
                _np.concatenate(kept, axis=0) if len(kept) > 1 else kept[0]
            )
        else:
            ctc_sequence = torch.cat(kept, dim=0) if len(kept) > 1 else kept[0]
        runs = blank_runs(
            ctc_sequence,
            upsample=head.upsample,
            min_seconds=cfg["min_blank_s"],
            silent_classes=head.silent_classes,
        )
        return runs, duration_s, "alignment_head_blank_runs", frame_track
    except Exception as error:  # noqa: BLE001
        if is_fatal_cuda_error(error):
            raise
        # Chunking must never take transcription down with it. Fixed-length
        # cuts are worse placed, not wrong, because nothing is dropped either way.
        _pipeline_logger.warning(
            "[chunking] blank runs unavailable: %s", error, exc_info=True
        )
        return [], duration_s, "fixed_length_blank_runs_failed", None
    finally:
        if owns_model and (model is not None or processor is not None):
            model = None
            processor = None
            _release_feature_model_vram()


def _build_processing_spans(
    audio_path: str,
    *,
    on_stage: Callable[[str], None] | None = None,
    backend: Any | None = None,
) -> ChunkPlan:
    """Contiguous chunks covering the whole file, cut inside pauses.

    Returns an immutable ``ChunkPlan`` (which also acts as a sequence of
    ``(start, end)`` tuples) and atomically mirrors the module-level signature
    cache for runtime-signature readers.
    """
    if on_stage is not None:
        on_stage("切分 0/1")
    cfg = _chunking_config()
    if backend is not None:
        runs, duration_s, source, frame_track = _blank_runs_for_audio(
            audio_path, on_stage=on_stage, backend=backend
        )
    elif on_stage is not None:
        runs, duration_s, source, frame_track = _blank_runs_for_audio(
            audio_path, on_stage=on_stage
        )
    else:
        runs, duration_s, source, frame_track = _blank_runs_for_audio(audio_path)

    spans, cut_provenance = plan_chunk_cuts(
        runs,
        duration_s,
        max_s=cfg["max_chunk_s"],
        min_s=cfg["min_chunk_s"],
    )
    full_provenance = {**cut_provenance, "source": source}
    boundary_signature = {
        "chunking": {
            "schema": CHUNK_CUT_SCHEMA,
            "source": source,
            "pause_count": len(runs),
            "chunk_count": len(spans),
            "duration_s": round(duration_s, 3),
            "spans_sha256": _spans_digest(spans),
            **{key: cfg[key] for key in sorted(cfg)},
        }
    }
    _set_last_frame_class_track(frame_track)
    _set_last_chunk_cut_provenance(full_provenance)
    _set_last_boundary_signature(boundary_signature)
    _pipeline_logger.info(
        "[chunking] source=%s pauses=%d chunks=%d hard_cuts=%d duration=%.2fs",
        source,
        len(runs),
        len(spans),
        int(cut_provenance.get("max_chunk_fallback_count") or 0),
        duration_s,
    )
    if on_stage is not None:
        on_stage("切分 1/1")
    return ChunkPlan(
        spans=tuple((float(begin), float(end)) for begin, end in spans),
        boundary_signature=boundary_signature,
        cut_provenance=full_provenance,
        frame_class_track=frame_track,
    )


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
    outer_exc_active = sys.exc_info()[0] is not None
    try:
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if not torch.cuda.is_available():
            snapshot.update(
                memory_safety.runtime_memory_snapshot(
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
        if not outer_exc_active and is_fatal_cuda_error(exc):
            raise
        snapshot["error"] = f"{type(exc).__name__}: {exc}"
    snapshot.update(
        memory_safety.runtime_memory_snapshot(
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
    return transcribe._vram_budget_mb()


def _shared_vram_spill_tolerance_mb() -> float:
    return transcribe._shared_vram_spill_tolerance_mb()


def _enforce_vram_budget_from_snapshot(snapshot: dict) -> None:
    spill_detail = transcribe._check_shared_vram_spill(
        str(snapshot.get("stage", "")), snapshot
    )
    if spill_detail is not None:
        raise RuntimeError(spill_detail)
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
    peak_reserved = max(reserved_values) if reserved_values else None
    detail = _check_allocated_vram_budget(
        stage=str(snapshot.get("stage", "")),
        allocated_mb=peak_allocated,
        reserved_mb=peak_reserved,
        budget_mb=budget_mb,
        total_mb=snapshot.get("total_mb", ""),
    )
    if detail is not None:
        raise RuntimeError(detail)


def _release_stage_gpu_cache(
    log: list[str],
    snapshots: list[dict],
    stage: str,
    *,
    elapsed_s: float | None = None,
) -> None:
    outer_exc_active = sys.exc_info()[0] is not None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        if not outer_exc_active and is_fatal_cuda_error(exc):
            raise
        _pipeline_logger.warning("Failed to release stage GPU cache at %s: %s", stage, exc)
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

    compact_values = precompute_cue_compact_texts(text_results)
    candidates = [
        build_cue_candidate(
            chunk=chunk,
            text_result=text_results[position],
            position=position,
            chunks=chunk_infos,
            text_results=text_results,
            include_audio_ref=False,
            compact_values=compact_values,
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


def _count_decode_cap_truncations(text_results: list[dict], *, log: list[str]) -> int:
    """Chunks that emitted more tokens than their audio can physically hold.

    The budget is derived from each chunk's duration (`decode_guard`), so this is
    a runaway counter, not a "raise the cap" hint: the model generated faster than
    anyone can speak. Reported because such a decode has no other symptom - the
    text comes back well-formed and simply stops, so nothing downstream can tell
    it from a chunk that had less to say.
    """
    flagged = sum(
        1
        for result in text_results
        if bool((result.get("asr_generation") or {}).get("truncated_at_cap"))
    )
    if flagged:
        log.append(
            f"解码超出音频可容纳的 token 量 {flagged}/{len(text_results)} 块"
            "（超出发音速率上限，判为解码失控；文本尾部已截断）"
        )
    return flagged


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
    defer_cleanup: Callable[[Callable[[], None]], object] | None = None,
) -> tuple[list[dict], list[str], dict]:
    def _notify(message: str) -> None:
        if on_stage:
            on_stage(message)

    log: list[str] = [f"ASR backend: {get_backend_label()}"]
    timings: dict[str, float] = {}
    cuda_memory: list[dict] = []
    transcript_chunks: list[dict] = []
    chunk_dir: Path | None = None
    completed = False
    total_started = time.perf_counter()
    _record_cuda_memory(log, cuda_memory, "asr_start", elapsed_s=0.0)

    try:
        backend = _resolve_asr_backend(device)
        word_dicts: list[dict] = []
        try:
            _notify("分析静音并切分音频...")
            split_started = time.perf_counter()
            _model_lifecycle_event(
                model_manager,
                stage="asr",
                action="load_exclusive",
                on_stage=on_stage,
            )
            load_before_split_s = float(getattr(backend, "cumulative_load_s", 0.0))
            chunk_plan = _build_processing_spans(
                audio_path, on_stage=on_stage, backend=backend
            )
            split_load_s = max(
                0.0,
                float(getattr(backend, "cumulative_load_s", 0.0)) - load_before_split_s,
            )
            if isinstance(chunk_plan, ChunkPlan):
                chunk_spans = chunk_plan.spans
                boundary_signature = dict(chunk_plan.boundary_signature)
                chunk_cut_provenance = dict(chunk_plan.cut_provenance)
                frame_class_track = chunk_plan.frame_class_track
            else:
                chunk_spans = chunk_plan
                boundary_signature = dict(_LAST_BOUNDARY_SIGNATURE)
                chunk_cut_provenance = dict(_LAST_CHUNK_CUT_PROVENANCE)
                frame_class_track = _LAST_FRAME_CLASS_TRACK

            chunk_dir, chunk_infos = _extract_wav_chunks(
                audio_path, _span_boundaries(chunk_spans), on_stage=on_stage
            )
            split_elapsed = max(0.0, time.perf_counter() - split_started - split_load_s)
            log.append(f"切分完成：共 {len(chunk_infos)} 个处理块")
            _record_stage_timing(log, timings, "split_s", "静音分析与切块", split_elapsed)
            _release_stage_gpu_cache(
                log,
                cuda_memory,
                "split_done",
                elapsed_s=time.perf_counter() - total_started,
            )

            if not chunk_infos:
                backend.unload_model(on_stage=on_stage)
                _model_lifecycle_event(
                    model_manager,
                    stage="asr",
                    action="unload",
                    on_stage=on_stage,
                )
                timings.update(
                    {
                        "asr_model_load_s": float(
                            getattr(backend, "cumulative_load_s", 0.0)
                        ),
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
                    "boundary_signature": boundary_signature,
                    "chunk_cuts": chunk_cut_provenance,
                    "postgate": _empty_postgate_report(),
                })
                completed = True
                return [], log, details

            # No duplicate load: chunking and transcription share this backend,
            # and a fully cached rerun with no head configured never loads weights.
            load_before_text_s = float(getattr(backend, "cumulative_load_s", 0.0))
            text_results, text_timings = _transcribe_asr_chunks_text_only(
                backend,
                chunk_infos,
                "ASR 文本转写",
                on_stage=on_stage,
            )
            text_load_s = max(
                0.0,
                float(getattr(backend, "cumulative_load_s", 0.0)) - load_before_text_s,
            )
            _record_stage_timing(
                log,
                timings,
                "asr_text_transcribe_s",
                "ASR文本转写",
                max(0.0, text_timings["text_transcribe_s"] - text_load_s),
            )
            for timing_key, timing_value in text_timings.items():
                if timing_key == "text_transcribe_s":
                    continue
                timings[timing_key] = timing_value
            _release_stage_gpu_cache(
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
            head_owns_unload = alignment_head_configured()
            if not head_owns_unload:
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
            if head_owns_unload:
                _model_lifecycle_event(
                    model_manager,
                    stage="asr",
                    action="unload",
                    on_stage=on_stage,
                )
            total_load_s = float(getattr(backend, "cumulative_load_s", 0.0))
            align_load_s = max(0.0, total_load_s - load_before_text_s - text_load_s)
            _record_stage_timing(
                log,
                timings,
                "asr_model_load_s",
                "ASR模型加载",
                total_load_s,
            )
            _record_stage_timing(
                log,
                timings,
                "alignment_s",
                "字幕时间轴生成",
                max(0.0, align_timings["alignment_s"] - align_load_s),
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
                            "ASR 解码超出音频可容纳",
                            "speech-island",
                            "TIMEOUT:",
                            "QUARANTINED:",
                            "[WARN]",
                        )
                    ):
                        log.append(f"chunk {idx}: {entry}")

                for word in chunk_words:
                    word_dicts.append(
                        {
                            "start": chunk["start"] + float(word["start"]),
                            "end": chunk["start"] + float(word["end"]),
                            "word": word["word"],
                            "timestamp_kind": str(word.get("timestamp_kind") or ""),
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
            decode_cap_truncations = _count_decode_cap_truncations(
                text_results,
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
            "boundary_signature": boundary_signature,
            "chunk_cuts": chunk_cut_provenance,
            # Absent on a v1 head, and every reader has to treat it that way: a
            # promoted head outlives the code that trained it, and rolling back
            # must not disable the subtitle filter.
            **(
                {"frame_class_track": frame_class_track}
                if frame_class_track
                else {}
            ),
            "postgate": postgate_report,
            "decode_cap_truncations": decode_cap_truncations,
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
        completed = True
        return segments, log, details
    finally:
        if chunk_dir is not None and not chunking.keep_asr_chunks():
            if completed and defer_cleanup is not None:
                defer_cleanup(partial(_delete_path_for_cleanup, chunk_dir))
            else:
                _delete_path_for_cleanup(chunk_dir)


def transcribe_and_align(
    audio_path: str,
    device: str,
    on_stage: Callable[[str], None] | None = None,
    include_details: bool = False,
    model_manager: Any | None = None,
    *,
    defer_cleanup: Callable[[Callable[[], None]], object] | None = None,
) -> tuple[list[dict], list[str]] | tuple[list[dict], list[str], dict]:
    segments, log, details = _transcribe_and_align_local(
        audio_path,
        device,
        on_stage=on_stage,
        model_manager=model_manager,
        defer_cleanup=defer_cleanup,
    )
    if include_details:
        return segments, log, details
    return segments, log
