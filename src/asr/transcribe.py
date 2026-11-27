import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from asr.backends.base import BaseAsrBackend
from asr.alignment import alignment_head_configured
from asr.alignment_quality import classify_alignment_quality
from asr import chunking as _chunking_module
from asr import result_cache as _result_cache_module
from asr.result_cache import _is_timed_out_result
from asr.text_normalize import strip_text_punctuation
from core.diagnostics import is_fatal_cuda_error
from core.typed_config import env_float, env_text
from asr.local_backend import LocalAsrBackend


logger = logging.getLogger(__name__)


def _emit_progress(on_stage: Callable[[str], None] | None, message: str) -> None:
    if on_stage:
        on_stage(message)
    print(message, flush=True)


_TRIVIAL_SEGMENT = re.compile(
    r"^[。！？…、\s,.!?・「」（）【】；：\-—–]+$"
)


def _asr_invalid_segment_duration_s() -> float:
    return env_float("ASR_INVALID_SEGMENT_DURATION", 0.1, minimum=0.0)


def _asr_min_repaired_segment_duration_s() -> float:
    return env_float("ASR_MIN_REPAIRED_SEGMENT_DURATION", 0.6, minimum=0.0)


def _vram_budget_mb() -> float:
    raw = env_text("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "0", lower=True)
    if raw in {"0", "false", "no", "off", "none"}:
        return 0.0
    return env_float("ASR_STAGE_WORKER_VRAM_BUDGET_MB", 0.0, minimum=0.0)


def _check_allocated_vram_budget(
    *,
    stage: str,
    allocated_mb: float,
    reserved_mb: float | None,
    budget_mb: float,
    total_mb: object = "",
) -> str | None:
    """Return an error detail string when peak allocated VRAM exceeds ``budget_mb``."""
    if budget_mb <= 0.0 or allocated_mb <= budget_mb:
        return None
    peak_reserved = allocated_mb if reserved_mb is None else reserved_mb
    return (
        "GPU VRAM budget exceeded: "
        f"stage={stage} allocated_mb={allocated_mb:.1f} "
        f"reserved_mb={peak_reserved:.1f} "
        f"budget_mb={budget_mb:.1f} total_mb={total_mb}"
    )


def _shared_vram_spill_tolerance_mb() -> float:
    # The Windows PDH "GPU Process Memory shared usage" counter jitters by a
    # few MB from driver-side staging buffers even when the CUDA allocator is
    # hard-capped by set_per_process_memory_fraction and cannot spill. A real
    # WDDM spill moves tens/hundreds of MB of tensor pages, so a small tolerance
    # separates counter noise from actual spill.
    return env_float("ASR_SHARED_VRAM_SPILL_TOLERANCE_MB", 64.0, minimum=0.0)


def _check_shared_vram_spill(stage: str, snapshot: dict[str, Any]) -> str | None:
    try:
        shared_vram_mb = float(snapshot.get("shared_vram_mb") or 0.0)
    except (TypeError, ValueError):
        return None
    tolerance_mb = _shared_vram_spill_tolerance_mb()
    if shared_vram_mb <= tolerance_mb:
        return None
    return (
        "GPU shared VRAM spill detected: "
        f"stage={stage} shared_vram_mb={shared_vram_mb:.3f} "
        f"raw_mb={snapshot.get('shared_vram_raw_mb', '')} "
        f"baseline_mb={snapshot.get('shared_vram_baseline_mb', '')} "
        f"tolerance_mb={tolerance_mb:.1f}"
    )


def _enforce_vram_budget(stage: str, on_stage: Callable[[str], None] | None) -> None:
    budget_mb = _vram_budget_mb()
    if budget_mb <= 0.0:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        device_index = torch.cuda.current_device()
        scale = 1024 * 1024
        allocated_mb = (
            max(
                torch.cuda.max_memory_allocated(device_index),
                torch.cuda.memory_allocated(device_index),
            )
            / scale
        )
        # Probe reserved memory for diagnostics and fatal CUDA error detection,
        # but enforce the budget strictly on allocated memory (the working set
        # that responds to batch size; caching-allocator reserved memory does not).
        try:
            reserved_mb: float | None = (
                max(
                    torch.cuda.max_memory_reserved(device_index),
                    torch.cuda.memory_reserved(device_index),
                )
                / scale
            )
        except Exception as exc:
            if is_fatal_cuda_error(exc):
                raise
            logger.debug(
                "Reserved VRAM probe unavailable at stage=%s: %s", stage, exc
            )
            reserved_mb = None

        logger.debug(
            "VRAM budget check stage=%s allocated_mb=%.1f reserved_mb=%s budget_mb=%.1f",
            stage,
            allocated_mb,
            f"{reserved_mb:.1f}" if reserved_mb is not None else "n/a",
            budget_mb,
        )
        if allocated_mb <= budget_mb:
            # Also catch Windows WDDM shared-VRAM spills right after batch 1
            # (fail-fast) instead of burning minutes across all remaining batches.
            try:
                from pipeline import memory_safety

                mem_snapshot = memory_safety.runtime_memory_snapshot(
                    require_shared_vram=False
                )
            except Exception as exc:
                logger.debug(
                    "Shared VRAM probe unavailable at stage=%s: %s", stage, exc
                )
                return
            spill_detail = _check_shared_vram_spill(stage, mem_snapshot)
            if spill_detail is not None:
                _emit_progress(on_stage, f"[WARN] {spill_detail}")
                raise RuntimeError(spill_detail)
            return
        try:
            _free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            total_mb: object = round(total_bytes / scale, 1)
        except Exception as exc:
            if is_fatal_cuda_error(exc):
                raise
            total_mb = ""
    except Exception as exc:
        msg = str(exc)
        if (
            "GPU VRAM budget exceeded" in msg
            or "GPU shared VRAM spill detected" in msg
            or is_fatal_cuda_error(exc)
        ):
            raise
        logger.warning(
            "VRAM budget probe failed at stage=%s: %s", stage, exc, exc_info=True
        )
        return
    detail = _check_allocated_vram_budget(
        stage=stage,
        allocated_mb=allocated_mb,
        reserved_mb=reserved_mb,
        budget_mb=budget_mb,
        total_mb=total_mb,
    )
    if detail is None:
        return
    _emit_progress(on_stage, f"[WARN] {detail}")
    raise RuntimeError(detail)


def _strip_punctuation(text: str) -> str:
    # `asr.text_normalize` owns this pattern and `local_backend` already reads
    # it from there. The private copy that used to live here was byte-identical,
    # which is the problem: both are applied to ASR output on the same path, so
    # a fix to one silently left the other judging text by the old rule.
    return strip_text_punctuation(text)


def _collapse_repeated_noise(text: str) -> str:
    cleaned = re.sub(r"[ \t]+", " ", (text or "").strip())
    return cleaned.strip()


def _is_low_value_text(text: str) -> bool:
    normalized = _collapse_repeated_noise(text)
    compact = _strip_punctuation(normalized)
    if not compact:
        return True

    if _TRIVIAL_SEGMENT.match(normalized):
        return True

    return False


def _clean_segment_text(text: str) -> str:
    return _collapse_repeated_noise((text or "").replace("\r", " ").replace("\n", " "))




def _with_alignment_window(chunk: dict, text_result: dict) -> dict:
    result = dict(text_result)
    try:
        duration = float(result.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0.0:
        duration = max(
            0.0,
            float(chunk.get("end", 0.0)) - float(chunk.get("start", 0.0)),
        )
    start_s = 0.0
    end_s = max(0.0, float(duration))
    source = "chunk"
    result["alignment_window_start_s"] = round(start_s, 6)
    result["alignment_window_end_s"] = round(end_s, 6)
    result["alignment_window_source"] = source
    if chunk.get("audio_sha256") and not result.get("audio_sha256"):
        result["audio_sha256"] = str(chunk["audio_sha256"])
    return result


def _word_timing_stats(words: list[dict]) -> dict:
    if not words:
        return {
            "word_count": 0,
            "zero_or_negative_count": 0,
            "tiny_span_count": 0,
            "min_word_duration_s": None,
            "max_word_duration_s": None,
        }
    durations: list[float] = []
    zero_or_negative = 0
    tiny = 0
    for word in words:
        try:
            start = float(word.get("start", 0.0))
            end = float(word.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        duration = end - start
        durations.append(duration)
        if duration <= 0.0:
            zero_or_negative += 1
        if duration <= 0.08:
            tiny += 1
    if not durations:
        return {
            "word_count": len(words),
            "zero_or_negative_count": 0,
            "tiny_span_count": 0,
            "min_word_duration_s": None,
            "max_word_duration_s": None,
        }
    return {
        "word_count": len(words),
        "zero_or_negative_count": zero_or_negative,
        "tiny_span_count": tiny,
        "min_word_duration_s": round(min(durations), 4),
        "max_word_duration_s": round(max(durations), 4),
    }


def _alignment_outcome_for_chunk(
    *,
    chunk: dict,
    chunk_result: dict,
    chunk_words: list[dict],
) -> dict:
    text = str(chunk_result.get("text") or chunk_result.get("raw_text") or "").strip()
    alignment_mode = str(chunk_result.get("alignment_mode") or "").strip()
    align_error = str(chunk_result.get("align_error") or "").strip()
    try:
        duration = float(chunk_result.get("duration"))
    except (TypeError, ValueError):
        duration = max(0.0, float(chunk.get("end", 0.0)) - float(chunk.get("start", 0.0)))
    word_stats = _word_timing_stats(chunk_words)
    compact_text = _strip_punctuation(_clean_segment_text(text))
    nonlexical_text = bool(text and not compact_text)
    quality = classify_alignment_quality(
        text=text,
        duration_s=duration,
        nonlexical_text=nonlexical_text,
        alignment_mode=alignment_mode,
        align_error=align_error,
        aligned_segment_count=1 if chunk_words else 0,
        word_stats=word_stats,
    )
    return {
        "alignment_mode": alignment_mode,
        "alignment_quality": quality["alignment_quality"],
        "alignment_issue_type": quality["alignment_issue_type"],
        "alignment_issue_subtype": quality["alignment_issue_subtype"],
        "alignment_quality_reasons": quality["alignment_quality_reasons"],
        "alignment_word_count": word_stats["word_count"],
        "align_error": align_error,
        "word_timing": word_stats,
        "alignment_issue_active": quality["alignment_issue_type"] != "none",
    }


def _transcribe_asr_chunks_text_only(
    backend: LocalAsrBackend,
    chunks: list[dict],
    text_stage_label: str,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict[str, float]]:
    if not chunks:
        return [], {"text_transcribe_s": 0.0}

    request_batch_size = max(1, int(getattr(backend, "request_batch_size", 1)))
    cache_enabled = _result_cache_module.result_cache_enabled()
    text_signature = _result_cache_module.model_signature() if cache_enabled else None
    text_cache_dir = (
        _result_cache_module._cache_dir(text_signature)
        if text_signature is not None
        else None
    )
    # Content-addressed prefill: any chunk whose audio+model identity was
    # transcribed before — this run, a crashed run, or a different job over the
    # same source — comes straight from the cache.
    text_results_by_index: dict[int, dict] = {}
    if cache_enabled:
        for chunk in chunks:
            cached = _result_cache_module.lookup(
                chunk["path"],
                audio_sha256=chunk.get("audio_sha256"),
                signature=text_signature,
                cache_dir=text_cache_dir,
            )
            if cached is None:
                continue
            text_results_by_index[int(chunk["index"])] = _result_cache_module.restore_text_result(
                chunk,
                cached,
            )
    if text_results_by_index and on_stage:
        on_stage(f"ASR 缓存命中 {len(text_results_by_index)}/{len(chunks)} 个块")

    text_started = time.perf_counter()
    max_observed_batch_size = 0

    def _store_text_results(batch_chunks: list[dict], batch_text_results: list[dict]) -> None:
        if len(batch_text_results) != len(batch_chunks):
            raise RuntimeError(
                "ASR batch result count mismatch: "
                f"expected={len(batch_chunks)} actual={len(batch_text_results)}; "
                "refusing to continue because positional pairing would shift "
                "subsequent subtitle timestamps"
            )
        for chunk, text_result in zip(batch_chunks, batch_text_results, strict=True):
            text_results_by_index[int(chunk["index"])] = text_result
            # Per-chunk files land immediately, so a crash resumes for free.
            if cache_enabled:
                _result_cache_module.store(
                    chunk["path"],
                    text_result,
                    audio_sha256=chunk.get("audio_sha256"),
                    signature=text_signature,
                    cache_dir=text_cache_dir,
                )

    def _transcribe_batch(batch_chunks: list[dict]) -> list[dict]:
        audio_paths = [chunk["path"] for chunk in batch_chunks]
        return backend.transcribe_texts(audio_paths, on_stage=on_stage)

    # Filter all pending chunks first, sort longest-first (for dense length
    # bucketing, minimal padding waste, and OOM fail-fast on batch 1), then
    # pack full `request_batch_size` batches without emitting empty `size=0` steps.
    pending_chunks = [
        chunk
        for chunk in chunks
        if int(chunk["index"]) not in text_results_by_index
    ]
    pending_chunks.sort(
        key=lambda chunk: (
            -max(0.0, float(chunk.get("end", 0.0)) - float(chunk.get("start", 0.0))),
            int(chunk["index"]),
        )
    )

    for batch_number, batch_start in enumerate(
        range(0, len(pending_chunks), request_batch_size), 1
    ):
        batch_chunks = pending_chunks[batch_start : batch_start + request_batch_size]
        batch_started = time.perf_counter()
        _emit_progress(
            on_stage,
            (
                f"{text_stage_label} {len(text_results_by_index)}/{len(chunks)} "
                f"batch={batch_number} size={len(batch_chunks)} start"
            ),
        )

        max_observed_batch_size = max(max_observed_batch_size, len(batch_chunks))
        batch_text_results = _transcribe_batch(batch_chunks)
        _enforce_vram_budget(
            f"{text_stage_label}_batch_{batch_number}",
            on_stage,
        )
        _store_text_results(batch_chunks, batch_text_results)
        batch_elapsed = time.perf_counter() - batch_started
        _emit_progress(
            on_stage,
            (
                f"{text_stage_label} {len(text_results_by_index)}/{len(chunks)} "
                f"batch={batch_number} size={len(batch_text_results)} done "
                f"elapsed={batch_elapsed:.2f}s "
                f"sec_per_chunk={batch_elapsed / max(len(batch_text_results), 1):.3f}"
            ),
        )

    timeout_count = sum(
        1 for result in text_results_by_index.values() if _is_timed_out_result(result)
    )
    if timeout_count and on_stage:
        on_stage(
            f"[WARN] 本轮 {timeout_count} 个块超时跳过，未写入结果缓存，下次续跑将重试"
        )

    text_results = [
        text_results_by_index[int(chunk["index"])]
        for chunk in chunks
        if int(chunk["index"]) in text_results_by_index
    ]

    text_elapsed = time.perf_counter() - text_started
    return text_results, {
        "text_transcribe_s": text_elapsed,
        # The configured request batch is only a ceiling. A short job, or one
        # mostly served from the per-chunk result cache, may never exercise it.
        # GPU profile learning must distinguish "asked for 11" from "actually
        # allocated 11 rows" or a two-row smoke can promote an untested batch.
        "asr_text_max_observed_batch_size": max_observed_batch_size,
    }


def _warn_cleanup_error(_function, _path, exc: BaseException) -> None:
    if isinstance(exc, FileNotFoundError):
        return
    if isinstance(exc, SystemExit):
        logger.warning(
            "ASR temporary cleanup interrupted by SystemExit(%r) "
            "(possibly an external sandbox or safe-delete hook)",
            exc.code,
        )
        return
    logger.warning(
        "ASR temporary cleanup failed: %s errno=%s winerror=%s",
        type(exc).__name__,
        getattr(exc, "errno", None),
        getattr(exc, "winerror", None),
    )


def _delete_path_for_cleanup(path: Path) -> None:
    try:
        if path.is_dir():
            import shutil

            shutil.rmtree(path, onexc=_warn_cleanup_error)
        else:
            path.unlink(missing_ok=True)
    except (Exception, SystemExit) as exc:
        _warn_cleanup_error(None, path, exc)


def aggregate_timeout_fragments(job_id: str) -> Path | None:
    normalized_job_id = re.sub(r"[^0-9A-Za-z._-]+", "_", (job_id or "").strip())
    normalized_job_id = normalized_job_id.strip("._-")
    if not normalized_job_id:
        return None

    out_dir = _chunking_module.current_asr_chunk_root().parent / "asr_timeouts"
    if not out_dir.exists() or not out_dir.is_dir():
        return None

    fragments: list[Path] = []
    for path in out_dir.glob("timeouts_*.json"):
        if path.name.startswith("timeouts_summary_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates = {
            str(payload.get("job_id") or ""),
            str(payload.get("video_name") or ""),
            str(payload.get("video_stem") or ""),
            str(payload.get("source_job_id") or ""),
        }
        audio_path = str(payload.get("audio_path") or "")
        if normalized_job_id in candidates or f"/{normalized_job_id}/" in audio_path.replace("\\", "/"):
            fragments.append(path)

    if not fragments:
        return None

    records: list[dict] = []
    for path in sorted(fragments, key=lambda item: item.name):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            records.append({"source_file": path.name, "parse_error": repr(exc)})

    summary_path = out_dir / f"timeouts_summary_{normalized_job_id}.json"
    tmp_path = summary_path.with_name(f"{summary_path.name}.{uuid.uuid4().hex[:8]}.tmp")
    payload = {
        "job_id": normalized_job_id,
        "count": len(records),
        "fragments": [path.name for path in sorted(fragments, key=lambda item: item.name)],
        "records": records,
    }
    with open(tmp_path, "w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False, indent=2)
    tmp_path.replace(summary_path)

    for path in fragments:
        _delete_path_for_cleanup(path)

    return summary_path


def _is_empty_segment_text_result(text_result: dict) -> bool:
    if "segments" not in text_result:
        return False
    if text_result.get("segments"):
        return False
    text = str(text_result.get("text") or text_result.get("raw_text") or "").strip()
    return not text


def _empty_alignment_placeholder(
    text_result: dict,
    *,
    reason: str = "Subtitle timing skipped: empty segments placeholder",
) -> tuple[dict, list[str]]:
    log = list(text_result.get("log", []))
    if reason and reason not in log:
        log.append(reason)
    try:
        duration = float(text_result.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return (
        {
            "words": [],
            "text": str(text_result.get("text") or "").strip(),
            "raw_text": str(text_result.get("raw_text") or "").strip(),
            "alignment_mode": "empty",
            "duration": duration,
            "language": str(text_result.get("language") or "Japanese").strip()
            or "Japanese",
        },
        log,
    )


def _empty_segments_quarantine_placeholder() -> tuple[dict, list[str]]:
    return _empty_alignment_placeholder(
        {
            "text": "",
            "raw_text": "",
            "duration": 0.0,
            "language": "Japanese",
            "normalized_path": "",
            "segments": [],
            "log": [
                (
                    "QUARANTINED: "
                    "kind=empty_segment, respawn_count=0, "
                    "run_id=, detail=empty subtitle timing input"
                )
            ],
        }
    )


def _align_TRANSCRIPTION_results(
    backend: BaseAsrBackend,
    text_results: list[dict],
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[tuple[dict, list[str]]], dict[str, float]]:
    if not text_results:
        return [_empty_segments_quarantine_placeholder()], {"alignment_s": 0.0}

    align_started = time.perf_counter()
    prepared: list[tuple[dict, list[str]] | None] = [None] * len(text_results)
    pending_results: list[dict] = []
    pending_indices: list[int] = []
    for idx, text_result in enumerate(text_results):
        if _is_empty_segment_text_result(text_result):
            prepared[idx] = _empty_alignment_placeholder(text_result)
        else:
            pending_results.append(text_result)
            pending_indices.append(idx)

    if not pending_results:
        return [item for item in prepared if item is not None], {
            "alignment_s": time.perf_counter() - align_started
        }

    head_configured = alignment_head_configured()
    final_cache_enabled = head_configured and _result_cache_module.result_cache_enabled()
    final_signature = (
        _result_cache_module.finalize_signature() if final_cache_enabled else None
    )
    final_cache_dir = (
        _result_cache_module._cache_dir(final_signature)
        if final_signature is not None
        else None
    )

    if final_signature is not None:
        # Word timing is worth caching only on this path: with a head each
        # chunk pays an encoder forward + a Viterbi walk, and the output pins
        # one more artifact (the head checkpoint, hashed into the key).
        cache_hits = 0
        uncached_results: list[dict] = []
        uncached_indices: list[int] = []
        for text_result, result_index in zip(pending_results, pending_indices):
            cached = _result_cache_module.finalize_lookup(
                str(text_result.get("normalized_path") or ""),
                text=str(text_result.get("text") or "").strip(),
                audio_sha256=text_result.get("audio_sha256"),
                signature=final_signature,
                cache_dir=final_cache_dir,
            )
            if cached is None:
                uncached_results.append(text_result)
                uncached_indices.append(result_index)
            else:
                prepared[result_index] = cached
                cache_hits += 1
        pending_results = uncached_results
        pending_indices = uncached_indices
        if cache_hits:
            _emit_progress(on_stage, f"字幕时间轴缓存命中 {cache_hits} 个块")

    def _backend_has_gpu_state() -> bool:
        return getattr(backend, "model", None) is not None or (
            getattr(backend, "_alignment_head", False) not in (False, None)
        )

    if not pending_results:
        if _backend_has_gpu_state():
            backend.unload_model(on_stage=on_stage)
        # Still report the stage as run-and-finished. The web progress bar keys
        # off this label, and a fully cached run that skipped it silently made
        # the bar jump straight from text transcription to writing.
        if on_stage:
            on_stage(f"字幕时间轴 {len(text_results)}/{len(text_results)}...")
        return [
            item
            if item is not None
            else _empty_alignment_placeholder(
                text_results[idx],
                reason="Subtitle timing skipped: missing finalize result",
            )
            for idx, item in enumerate(prepared)
        ], {"alignment_s": time.perf_counter() - align_started}

    if head_configured:
        # The aligned pass reads encoder features off the very model that just
        # decoded. Unloading it here — the old unconditional behavior — made
        # _align_characters bail on `self.model is None` and silently degraded
        # every chunk to proportional timing, so a configured head never
        # actually ran in this flow. A fully cache-hit text stage arrives with
        # the model never loaded at all; load it now.
        if getattr(backend, "model", None) is None:
            loader = getattr(backend, "load", None)
            if callable(loader):
                try:
                    loader(on_stage=on_stage)
                except Exception as error:  # noqa: BLE001
                    if is_fatal_cuda_error(error):
                        raise
                    logger.warning(
                        "ASR model load for alignment failed: %s", error, exc_info=True
                    )
                    _emit_progress(
                        on_stage,
                        f"[WARN] 对齐用 ASR 模型加载失败，本轮退化为比例时间轴: {error}",
                    )
    elif _backend_has_gpu_state():
        backend.unload_model(on_stage=on_stage)

    if on_stage:
        on_stage(f"字幕时间轴 {len(pending_results)}/{len(pending_results)}...")
    try:
        finalized_batch = backend.finalize_text_results(
            pending_results,
            on_stage=on_stage,
        )
    finally:
        if head_configured and _backend_has_gpu_state():
            backend.unload_model(on_stage=on_stage)
    if head_configured and final_cache_enabled:
        # Head may have just been downloaded during finalize if it was absent on lookup.
        store_signature = (
            final_signature
            if final_signature is not None
            else _result_cache_module.finalize_signature()
        )
        store_cache_dir = (
            final_cache_dir
            if final_cache_dir is not None
            else (
                _result_cache_module._cache_dir(store_signature)
                if store_signature is not None
                else None
            )
        )
        if store_signature is not None:
            for text_result, finalized in zip(pending_results, finalized_batch):
                finalized_result, finalized_log = finalized
                _result_cache_module.finalize_store(
                    str(text_result.get("normalized_path") or ""),
                    text=str(text_result.get("text") or "").strip(),
                    result=finalized_result,
                    log=finalized_log,
                    audio_sha256=text_result.get("audio_sha256"),
                    signature=store_signature,
                    cache_dir=store_cache_dir,
                )
    for result_index, finalized in zip(pending_indices, finalized_batch):
        prepared[result_index] = finalized
    if len(finalized_batch) < len(pending_results):
        for missing_index in pending_indices[len(finalized_batch) :]:
            prepared[missing_index] = _empty_alignment_placeholder(
                text_results[missing_index],
                reason="Subtitle timing skipped: backend returned no finalize result",
            )

    align_elapsed = time.perf_counter() - align_started
    return [
        item
        if item is not None
        else _empty_alignment_placeholder(
            text_results[idx],
            reason="Subtitle timing skipped: missing finalize result",
        )
        for idx, item in enumerate(prepared)
    ], {"alignment_s": align_elapsed}


def _build_transcript_chunks(
    chunks: list[dict],
    text_results: list[dict],
    alignment_outcomes: dict[int, dict] | None = None,
) -> list[dict]:
    transcript_chunks: list[dict] = []
    alignment_outcomes = alignment_outcomes or {}
    for chunk, text_result in zip(chunks, text_results):
        chunk_index = int(chunk["index"])
        item = {
            "index": chunk_index,
            "start": float(chunk["start"]),
            "end": float(chunk["end"]),
            "duration": float(text_result.get("duration", 0.0)),
            "language": text_result.get("language", ""),
            "text": text_result.get("text", ""),
            "raw_text": text_result.get("raw_text", ""),
        }
        if "alignment_window_start_s" in text_result and "alignment_window_end_s" in text_result:
            window_start = float(text_result.get("alignment_window_start_s") or 0.0)
            window_end = float(text_result.get("alignment_window_end_s") or window_start)
            window_end = max(window_start, window_end)
            chunk_start = float(chunk["start"])
            item["alignment_window_start_s"] = window_start
            item["alignment_window_end_s"] = window_end
            item["alignment_window_duration_s"] = max(0.0, window_end - window_start)
            item["alignment_window_abs_start_s"] = chunk_start + window_start
            item["alignment_window_abs_end_s"] = chunk_start + window_end
            item["alignment_window_source"] = text_result.get("alignment_window_source", "chunk")
        outcome = alignment_outcomes.get(chunk_index)
        if outcome:
            item.update(
                {
                    key: value
                    for key, value in outcome.items()
                    if key
                    in {
                        "alignment_mode",
                        "alignment_quality",
                        "alignment_issue_type",
                        "alignment_issue_subtype",
                        "alignment_quality_reasons",
                        "alignment_word_count",
                        "align_error",
                        "word_timing",
                        "alignment_issue_active",
                    }
                }
            )
        transcript_chunks.append(item)
    return transcript_chunks


def _postprocess_segments(segments: list[dict]) -> list[dict]:
    cleaned_segments: list[dict] = []

    for segment in segments:
        segment_words = list(segment.get("words") or [])
        text = _clean_segment_text(segment.get("text", ""))
        if not text:
            continue
        if _is_low_value_text(text):
            continue
        if _TRIVIAL_SEGMENT.match(text):
            continue

        cleaned_segments.append(
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": text,
                "source_chunk_index": segment.get("source_chunk_index"),
                "words": segment_words,
            }
        )

    cleaned_segments.sort(key=lambda item: (item["start"], item["end"]))
    return _repair_postprocessed_segment_windows(cleaned_segments)


def _repair_postprocessed_segment_windows(segments: list[dict]) -> list[dict]:
    if not segments:
        return []

    invalid_duration_s = _asr_invalid_segment_duration_s()
    min_repaired_duration_s = _asr_min_repaired_segment_duration_s()

    normalized: list[dict] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = max(0.0, float(segment.get("start", 0.0)))
        end = max(start, float(segment.get("end", start)))
        normalized.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "source_chunk_index": segment.get("source_chunk_index"),
                "words": list(segment.get("words") or []),
            }
        )

    normalized.sort(key=lambda item: (item["start"], item["end"]))
    repaired: list[dict] = []
    count = len(normalized)
    for idx, segment in enumerate(normalized):
        start = segment["start"]
        end = segment["end"]
        if end - start <= invalid_duration_s:
            end = max(end, start + min_repaired_duration_s)
        if idx + 1 < count:
            next_start = normalized[idx + 1]["start"]
            end = max(start, min(end, next_start))
        segment["end"] = end
        repaired.append(segment)

    return repaired


def _join_word_tokens(word_list: list[dict]) -> str:
    parts: list[str] = []
    for item in word_list:
        token = str(item.get("word") or "")
        if not token:
            continue
        if (
            parts
            and parts[-1][-1].isascii()
            and parts[-1][-1].isalnum()
            and token[0].isascii()
            and token[0].isalnum()
        ):
            parts.append(" ")
        parts.append(token)
    return "".join(parts).strip()


def _group_words_to_segments(words: list[dict]) -> list[dict]:
    if not words:
        return []

    segments: list[dict] = []
    current_words: list[dict] = []

    def flush(word_list: list[dict]) -> None:
        if not word_list:
            return
        text = _join_word_tokens(word_list)
        if not text:
            return
        if _TRIVIAL_SEGMENT.match(text):
            return
        segments.append(
            {
                "start": word_list[0]["start"],
                "end": word_list[-1]["end"],
                "text": text,
                "source_chunk_index": word_list[0].get("source_chunk_index"),
                "words": list(word_list),
            }
        )

    for word in words:
        if not current_words:
            current_words.append(word)
            continue

        current_chunk = current_words[-1].get("source_chunk_index")
        next_chunk = word.get("source_chunk_index")
        crosses_chunk_boundary = (
            current_chunk is not None
            and next_chunk is not None
            and current_chunk != next_chunk
        )

        if crosses_chunk_boundary:
            flush(current_words)
            current_words = [word]
        else:
            current_words.append(word)

    flush(current_words)
    return segments
