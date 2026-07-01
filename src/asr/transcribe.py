import logging
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable

from asr.backends.base import BaseAsrBackend
from asr.backends.registry import current_asr_worker_mode, _is_subprocess_backend
from asr.alignment_quality import classify_alignment_quality
from asr.checkpoint import (
    _build_quarantined_text_result,
    _checkpointable_text_results,
    _delete_path_for_cleanup,
    _get_asr_checkpoint_path,
    _get_asr_checkpoint_source,
    _is_timed_out_result,
    _load_asr_checkpoint,
    _quarantine_failed_chunks,
    _save_asr_checkpoint,
)
from asr.local_backend import (
    LocalAsrBackend,
    SubprocessAsrBackend,
    WorkerError,
    WorkerTimeoutError,
)


logger = logging.getLogger(__name__)


def _emit_progress(on_stage: Callable[[str], None] | None, message: str) -> None:
    if on_stage:
        on_stage(message)
    print(message, flush=True)

_ASR_INVALID_SEGMENT_DURATION_S = float(
    os.getenv("ASR_INVALID_SEGMENT_DURATION", "0.1")
)
_ASR_MIN_REPAIRED_SEGMENT_DURATION_S = float(
    os.getenv("ASR_MIN_REPAIRED_SEGMENT_DURATION", "0.6")
)
_ASR_CHECKPOINT_INTERVAL = max(1, int(os.getenv("ASR_CHECKPOINT_INTERVAL", "50")))
_ASR_SUBPROCESS_RESPAWN_MAX = max(
    0,
    int(os.getenv("ASR_SUBPROCESS_RESPAWN_MAX", "2")),
)
_ASR_SUBPROCESS_CONSECUTIVE_TIMEOUT_LIMIT = max(
    1,
    int(os.getenv("ASR_SUBPROCESS_CONSECUTIVE_TIMEOUT_LIMIT", "3")),
)
_ASR_CHECKPOINT_ENABLED = os.getenv("ASR_CHECKPOINT_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_TRIVIAL_SEGMENT = re.compile(
    r"^[。！？…、\s,.!?・「」（）【】；：\-—–]+$"
)
_STRIP_PUNCT_RE = re.compile(r"[。！？…、,.!?・「」『』（）()【】\[\]\s~〜ー-]+")


def _current_asr_worker_mode() -> str:
    return current_asr_worker_mode()


class ASRWorkerSystemError(RuntimeError):
    pass


def _strip_punctuation(text: str) -> str:
    return _STRIP_PUNCT_RE.sub("", text or "")


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


def _word_backed_segment_text(words: list[dict]) -> str:
    text = _clean_segment_text("".join(str(word.get("word", "")) for word in words))
    if not text or not _strip_punctuation(text):
        return ""
    return text


def _alignment_window_from_chunk(
    *,
    duration: float,
) -> tuple[float, float, str]:
    return 0.0, max(0.0, float(duration)), "chunk"


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
    start_s, end_s, source = _alignment_window_from_chunk(duration=duration)
    result["alignment_window_start_s"] = round(start_s, 6)
    result["alignment_window_end_s"] = round(end_s, 6)
    result["alignment_window_source"] = source
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
    backend: LocalAsrBackend | SubprocessAsrBackend,
    chunks: list[dict],
    text_stage_label: str,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict[str, float]]:
    if not chunks:
        return [], {"text_transcribe_s": 0.0}

    request_batch_size = max(1, int(getattr(backend, "request_batch_size", 1)))
    checkpoint_source = _get_asr_checkpoint_source(chunks, text_stage_label)
    checkpoint_path = _get_asr_checkpoint_path(checkpoint_source)
    run_id = uuid.uuid4().hex[:8]
    text_results_by_index = _load_asr_checkpoint(
        checkpoint_path,
        checkpoint_source,
        chunks,
        run_id=run_id,
    )
    if text_results_by_index and on_stage:
        on_stage(f"ASR checkpoint 恢复 {len(text_results_by_index)}/{len(chunks)} 个块")

    processed_since_checkpoint = 0
    completed = False
    final_checkpoint_saved = False
    is_subprocess_backend = _is_subprocess_backend(backend)
    respawn_count: defaultdict[int, int] = defaultdict(int)
    consecutive_failures = 0
    failure_records: list[dict] = []
    text_started = time.perf_counter()

    def _save_progress_checkpoint() -> None:
        nonlocal processed_since_checkpoint
        if processed_since_checkpoint < _ASR_CHECKPOINT_INTERVAL:
            return
        _save_asr_checkpoint(
            checkpoint_path,
            checkpoint_source,
            chunks,
            _checkpointable_text_results(text_results_by_index),
            run_id=run_id,
        )
        processed_since_checkpoint = 0
        if on_stage:
            on_stage(
                f"ASR checkpoint 已保存 {len(text_results_by_index)}/{len(chunks)} 个块"
            )

    def _store_text_results(batch_chunks: list[dict], batch_text_results: list[dict]) -> None:
        nonlocal processed_since_checkpoint
        for chunk, text_result in zip(batch_chunks, batch_text_results):
            text_results_by_index[int(chunk["index"])] = text_result
        processed_since_checkpoint += len(batch_text_results)
        _save_progress_checkpoint()

    def _quarantine_chunk(chunk: dict, *, kind: str, detail: str) -> None:
        nonlocal processed_since_checkpoint
        chunk_index = int(chunk["index"])
        if chunk_index in text_results_by_index:
            return
        count = int(respawn_count[chunk_index])
        text_results_by_index[chunk_index] = _build_quarantined_text_result(
            chunk,
            kind=kind,
            detail=detail,
            respawn_count=count,
            run_id=run_id,
        )
        failure_records.append(
            {
                "index": chunk_index,
                "kind": kind,
                "detail": detail,
                "respawn_count": count,
                "run_id": run_id,
                "worker_mode": _current_asr_worker_mode(),
            }
        )
        processed_since_checkpoint += 1

    def _schedule_retries(
        failed_chunks: list[dict],
        *,
        kind: str,
        detail: str,
    ) -> list[dict]:
        retry_chunks: list[dict] = []
        for chunk in failed_chunks:
            chunk_index = int(chunk["index"])
            if chunk_index in text_results_by_index:
                continue
            respawn_count[chunk_index] += 1
            if respawn_count[chunk_index] > _ASR_SUBPROCESS_RESPAWN_MAX:
                _quarantine_chunk(chunk, kind=kind, detail=detail)
            else:
                retry_chunks.append(chunk)
        _save_progress_checkpoint()
        return retry_chunks

    def _quarantine_many(
        failed_chunks: list[dict],
        *,
        kind: str,
        detail: str,
    ) -> None:
        for chunk in failed_chunks:
            _quarantine_chunk(chunk, kind=kind, detail=detail)
        _save_progress_checkpoint()

    def _transcribe_batch(batch_chunks: list[dict]) -> list[dict]:
        audio_paths = [chunk["path"] for chunk in batch_chunks]
        return backend.transcribe_texts(audio_paths, on_stage=on_stage)

    try:
        if not is_subprocess_backend:
            for batch_start in range(0, len(chunks), request_batch_size):
                batch_chunks = chunks[batch_start : batch_start + request_batch_size]
                pending_chunks = [
                    chunk
                    for chunk in batch_chunks
                    if int(chunk["index"]) not in text_results_by_index
                ]
                batch_end = batch_start + len(batch_chunks)
                batch_number = batch_start // request_batch_size + 1
                batch_started = time.perf_counter()
                _emit_progress(
                    on_stage,
                    (
                        f"{text_stage_label} {batch_end}/{len(chunks)} "
                        f"batch={batch_number} size={len(pending_chunks)} start"
                    ),
                )
                if not pending_chunks:
                    continue

                batch_text_results = _transcribe_batch(pending_chunks)
                _store_text_results(pending_chunks, batch_text_results)
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
        else:
            pending_chunks = [
                chunk for chunk in chunks if int(chunk["index"]) not in text_results_by_index
            ]
            total_batches = max(
                1,
                (len(pending_chunks) + request_batch_size - 1) // request_batch_size,
            )
            batch_number = 0
            while pending_chunks:
                batch_chunks = pending_chunks[:request_batch_size]
                pending_chunks = pending_chunks[request_batch_size:]
                batch_number += 1
                batch_target = min(len(text_results_by_index) + len(batch_chunks), len(chunks))
                batch_started = time.perf_counter()
                _emit_progress(
                    on_stage,
                    (
                        f"{text_stage_label} {batch_target}/{len(chunks)} "
                        f"batch={batch_number}/{total_batches} "
                        f"size={len(batch_chunks)} start"
                    ),
                )

                try:
                    batch_text_results = _transcribe_batch(batch_chunks)
                except WorkerTimeoutError as exc:
                    consecutive_failures += 1
                    retry_chunks = _schedule_retries(
                        batch_chunks,
                        kind="timeout",
                        detail=str(getattr(exc, "detail", str(exc))),
                    )
                    if (
                        consecutive_failures
                        >= _ASR_SUBPROCESS_CONSECUTIVE_TIMEOUT_LIMIT
                    ):
                        _quarantine_many(
                            retry_chunks + pending_chunks,
                            kind="timeout",
                            detail="circuit breaker",
                        )
                        pending_chunks = []
                        break
                    pending_chunks = retry_chunks + pending_chunks
                    continue
                except WorkerError as exc:
                    failure_kind = str(getattr(exc, "kind", "crash") or "crash")
                    failure_detail = str(getattr(exc, "detail", str(exc)) or str(exc))
                    if failure_kind == "oom":
                        retry_chunks: list[dict] = []
                        for chunk in batch_chunks:
                            chunk_index = int(chunk["index"])
                            if chunk_index in text_results_by_index:
                                continue
                            respawn_count[chunk_index] += 1
                            try:
                                single_result = _transcribe_batch([chunk])
                            except WorkerTimeoutError as single_exc:
                                consecutive_failures += 1
                                retry_chunks.extend(
                                    _schedule_retries(
                                        [chunk],
                                        kind="timeout",
                                        detail=str(
                                            getattr(
                                                single_exc,
                                                "detail",
                                                str(single_exc),
                                            )
                                        ),
                                    )
                                )
                            except WorkerError as single_exc:
                                single_kind = str(
                                    getattr(single_exc, "kind", "crash") or "crash"
                                )
                                single_detail = str(
                                    getattr(single_exc, "detail", str(single_exc))
                                    or str(single_exc)
                                )
                                if single_kind == "oom":
                                    respawn_count[chunk_index] += 1
                                    _quarantine_chunk(
                                        chunk,
                                        kind="oom",
                                        detail=single_detail,
                                    )
                                    _save_progress_checkpoint()
                                elif single_kind in {"crash", "protocol_error"}:
                                    consecutive_failures += 1
                                    retry_chunks.extend(
                                        _schedule_retries(
                                            [chunk],
                                            kind=single_kind,
                                            detail=single_detail,
                                        )
                                    )
                                else:
                                    raise
                            else:
                                _store_text_results([chunk], single_result)
                                consecutive_failures = 0

                        if (
                            consecutive_failures
                            >= _ASR_SUBPROCESS_CONSECUTIVE_TIMEOUT_LIMIT
                        ):
                            _quarantine_many(
                                retry_chunks + pending_chunks,
                                kind="timeout",
                                detail="circuit breaker",
                            )
                            pending_chunks = []
                            break
                        pending_chunks = retry_chunks + pending_chunks
                        continue

                    if failure_kind in {"crash", "protocol_error"}:
                        consecutive_failures += 1
                        retry_chunks = _schedule_retries(
                            batch_chunks,
                            kind=failure_kind,
                            detail=failure_detail,
                        )
                        if (
                            consecutive_failures
                            >= _ASR_SUBPROCESS_CONSECUTIVE_TIMEOUT_LIMIT
                        ):
                            _quarantine_many(
                                retry_chunks + pending_chunks,
                                kind=failure_kind,
                                detail="circuit breaker",
                            )
                            pending_chunks = []
                            break
                        pending_chunks = retry_chunks + pending_chunks
                        continue
                    raise
                else:
                    _store_text_results(batch_chunks, batch_text_results)
                    batch_elapsed = time.perf_counter() - batch_started
                    _emit_progress(
                        on_stage,
                        (
                            f"{text_stage_label} {len(text_results_by_index)}/{len(chunks)} "
                            f"batch={batch_number}/{total_batches} "
                            f"size={len(batch_text_results)} done "
                            f"elapsed={batch_elapsed:.2f}s "
                            f"sec_per_chunk={batch_elapsed / max(len(batch_text_results), 1):.3f}"
                        ),
                    )
                    consecutive_failures = 0

        timeout_count = sum(
            1 for result in text_results_by_index.values() if _is_timed_out_result(result)
        )
        if failure_records:
            quarantine_paths = _quarantine_failed_chunks(
                checkpoint_source,
                chunks,
                failure_records,
                run_id=run_id,
                worker_mode=_current_asr_worker_mode(),
            )
            if quarantine_paths and on_stage:
                on_stage(f"ASR quarantine 已写入 {len(quarantine_paths)} 个记录")
        checkpointable_results = _checkpointable_text_results(text_results_by_index)
        completed = len(checkpointable_results) >= len(chunks)
        if timeout_count:
            if on_stage:
                on_stage(
                    f"[WARN] 本轮 {timeout_count} 个块超时跳过，已从 checkpoint 中排除，下次续跑将重试"
                )
            _save_asr_checkpoint(
                checkpoint_path,
                checkpoint_source,
                chunks,
                checkpointable_results,
                run_id=run_id,
            )
            final_checkpoint_saved = True
        if completed:
            _delete_path_for_cleanup(checkpoint_path)
    finally:
        if (
            _ASR_CHECKPOINT_ENABLED
            and not completed
            and not final_checkpoint_saved
            and processed_since_checkpoint > 0
        ):
            _save_asr_checkpoint(
                checkpoint_path,
                checkpoint_source,
                chunks,
                _checkpointable_text_results(text_results_by_index),
                run_id=run_id,
            )

    if is_subprocess_backend:
        missing_indices = [
            int(chunk["index"])
            for chunk in chunks
            if int(chunk["index"]) not in text_results_by_index
        ]
        if missing_indices:
            raise ASRWorkerSystemError(
                f"subprocess ASR missing text results for chunks: {missing_indices[:10]}"
            )
        text_results = [text_results_by_index[int(chunk["index"])] for chunk in chunks]
    else:
        text_results = [
            text_results_by_index[int(chunk["index"])]
            for chunk in chunks
            if int(chunk["index"]) in text_results_by_index
        ]

    text_elapsed = time.perf_counter() - text_started
    return text_results, {"text_transcribe_s": text_elapsed}


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

    if _is_subprocess_backend(backend) or getattr(backend, "model", None) is not None:
        backend.unload_model(on_stage=on_stage)

    if on_stage:
        on_stage(f"字幕时间轴 {len(pending_results)}/{len(pending_results)}...")
    finalized_batch = backend.finalize_text_results(
        pending_results,
        on_stage=on_stage,
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
        word_backed_text = _word_backed_segment_text(segment_words)
        text = word_backed_text or _clean_segment_text(segment.get("text", ""))
        if not text:
            continue
        if not word_backed_text:
            if _is_low_value_text(text):
                continue
        elif _TRIVIAL_SEGMENT.match(text):
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

    repaired: list[dict] = []
    for idx, segment in enumerate(segments):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start = max(0.0, float(segment.get("start", 0.0)))
        end = max(start, float(segment.get("end", start)))
        if repaired and start < float(repaired[-1]["end"]):
            repaired[-1]["end"] = max(float(repaired[-1]["start"]), start)

        duration = end - start
        if duration <= _ASR_INVALID_SEGMENT_DURATION_S:
            target_end = start + _ASR_MIN_REPAIRED_SEGMENT_DURATION_S
            end = max(end, target_end)

        repaired.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "source_chunk_index": segment.get("source_chunk_index"),
                "words": list(segment.get("words") or []),
            }
        )

    repaired.sort(key=lambda item: (item["start"], item["end"]))
    non_overlapping: list[dict] = []
    for segment in repaired:
        start = float(segment["start"])
        end = float(segment["end"])
        if non_overlapping and start < float(non_overlapping[-1]["end"]):
            non_overlapping[-1]["end"] = max(float(non_overlapping[-1]["start"]), start)
        if end - start <= _ASR_INVALID_SEGMENT_DURATION_S:
            end = start + _ASR_MIN_REPAIRED_SEGMENT_DURATION_S
        non_overlapping.append(
            {
                "start": start,
                "end": end,
                "text": str(segment["text"]).strip(),
                "source_chunk_index": segment.get("source_chunk_index"),
                "words": list(segment.get("words") or []),
            }
        )

    return non_overlapping


def _group_words_to_segments(words: list[dict]) -> list[dict]:
    if not words:
        return []

    segments: list[dict] = []
    current_words: list[dict] = []

    def flush(word_list: list[dict]) -> None:
        if not word_list:
            return
        text = "".join(w["word"] for w in word_list).strip()
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
