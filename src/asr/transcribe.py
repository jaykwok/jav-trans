import inspect
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
from asr.chunking import _KEEP_ASR_CHUNKS, _extract_wav_chunks
from asr.local_backend import (
    LocalAsrBackend,
    SubprocessAsrBackend,
    WorkerError,
    WorkerTimeoutError,
    looks_like_word_timing_failure,
    word_timing_failure_reasons,
)
from asr.timestamp_fallback import build_word_timestamps_fallback


logger = logging.getLogger(__name__)

_ASR_CONTEXT_RESET_GAP_S = float(os.getenv("ASR_CONTEXT_RESET_GAP_S", "0.5"))
_ASR_SLIDING_CONTEXT_SEGS = max(0, int(os.getenv("ASR_SLIDING_CONTEXT_SEGS", "2")))
_ASR_INITIAL_PROMPT_MAX_CHARS = int(os.getenv("ASR_INITIAL_PROMPT_MAX_CHARS", "240"))
_ALIGNMENT_STEP_DOWN_CHUNK_S = float(os.getenv("ALIGNMENT_STEP_DOWN_CHUNK", "6.0"))
_ALIGNMENT_COARSE_REFINE_CHUNK_S = float(os.getenv("ALIGNMENT_COARSE_REFINE_CHUNK", "18.0"))
_ALIGNMENT_MAX_REFINE_DEPTH = max(0, int(os.getenv("ALIGNMENT_MAX_REFINE_DEPTH", "2")))
_ASR_HEAD_CONTEXT_MAX_START_S = float(os.getenv("ASR_HEAD_CONTEXT_MAX_START_S", "16"))
_ALIGNMENT_MIN_SPAN_MS = float(os.getenv("ALIGNMENT_MIN_SPAN_MS", "120"))
_ALIGNMENT_MAX_ZERO_RATIO = float(os.getenv("ALIGNMENT_MAX_ZERO_RATIO", "0.55"))
_ALIGNMENT_MAX_REPEAT_RATIO = float(os.getenv("ALIGNMENT_MAX_REPEAT_RATIO", "0.65"))
_ALIGNMENT_MAX_COVERAGE_RATIO = float(os.getenv("ALIGNMENT_MAX_COVERAGE_RATIO", "0.05"))
_ALIGNMENT_MAX_CPS = float(os.getenv("ALIGNMENT_MAX_CPS", "50.0"))
_ALIGNMENT_RETRY_SKIP_MAX_TEXT_LEN = max(
    1,
    int(os.getenv("ALIGNMENT_RETRY_SKIP_MAX_TEXT_LEN", "10")),
)
_ALIGNMENT_SENTINEL_ISLAND_MIN_S = float(
    os.getenv("ALIGNMENT_SENTINEL_ISLAND_MIN_S", "0.25")
)
_ALIGNMENT_SENTINEL_ISLAND_PAD_FRAMES = max(
    0,
    int(os.getenv("ALIGNMENT_SENTINEL_ISLAND_PAD_FRAMES", "6")),
)
_ALIGNMENT_SENTINEL_ISLAND_MERGE_GAP_FRAMES = max(
    0,
    int(os.getenv("ALIGNMENT_SENTINEL_ISLAND_MERGE_GAP_FRAMES", "6")),
)
_ALIGNMENT_SENTINEL_ISLAND_MAX_SPLITS = max(
    1,
    int(os.getenv("ALIGNMENT_SENTINEL_ISLAND_MAX_SPLITS", "8")),
)
_ASR_FRAGMENT_MERGE_MAX_GAP_S = float(os.getenv("ASR_FRAGMENT_MERGE_MAX_GAP", "1.0"))
_ASR_FRAGMENT_MERGE_MAX_CHARS = max(
    1,
    int(os.getenv("ASR_FRAGMENT_MERGE_MAX_CHARS", "72")),
)
_ASR_FRAGMENT_MERGE_MAX_DURATION_S = float(
    os.getenv("ASR_FRAGMENT_MERGE_MAX_DURATION", "12.5")
)
_ASR_INVALID_SEGMENT_DURATION_S = float(
    os.getenv("ASR_INVALID_SEGMENT_DURATION", "0.1")
)
_ASR_MIN_REPAIRED_SEGMENT_DURATION_S = float(
    os.getenv("ASR_MIN_REPAIRED_SEGMENT_DURATION", "0.6")
)
_ASR_POSTPROCESS_MAX_CHARS = max(
    8,
    int(os.getenv("ASR_POSTPROCESS_MAX_CHARS", "60")),
)
_ASR_POSTPROCESS_MAX_DURATION_S = float(
    os.getenv("ASR_POSTPROCESS_MAX_DURATION", "12.5")
)
_ASR_POSTPROCESS_SPLIT_MIN_CHARS = max(
    4,
    int(os.getenv("ASR_POSTPROCESS_SPLIT_MIN_CHARS", "4")),
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
_TEXT_UNIT_COMPACT_RE = re.compile(r"[^0-9A-Za-zぁ-ゖァ-ヺ一-龯々〆ヵヶ]+")
_SENTENCE_TERMINAL_RE = re.compile(r"[。！？!?…」』）)\]]$")


def _asr_context() -> str:
    return os.getenv("ASR_CONTEXT", "").strip()


def _asr_head_context() -> str:
    return os.getenv("ASR_HEAD_CONTEXT", "").strip()
_FRAGMENT_CONTINUATION_START_RE = re.compile(
    r"^(?:ます|ました|ません|です|でした|でしょう|ながら|ので|けど|から|たり|"
    r"程度|ところ|いる|いき|して|され|なり|効果|で[、,]?|に|を|が|は|も)"
)


def _current_asr_worker_mode() -> str:
    return current_asr_worker_mode()


class ASRWorkerSystemError(RuntimeError):
    pass


def _strip_punctuation(text: str) -> str:
    return _STRIP_PUNCT_RE.sub("", text or "")


def _compact_text_units(text: str) -> str:
    return _TEXT_UNIT_COMPACT_RE.sub("", text or "")


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


def _build_timestamp_fallback(
    text: str,
    start: float,
    end: float,
    audio_path: str | None = None,
) -> tuple[list[dict], str, dict]:
    return build_word_timestamps_fallback(
        _clean_segment_text(text),
        start,
        end,
        audio_path=audio_path,
    )


def _alignment_fallback_window_from_chunk(
    chunk: dict,
    *,
    duration: float,
) -> tuple[float, float, str]:
    full_start = 0.0
    full_end = max(0.0, float(duration))
    try:
        left_padding_s = max(0.0, float(chunk.get("speech_left_padding_s") or 0.0))
    except (TypeError, ValueError):
        left_padding_s = 0.0
    try:
        right_padding_s = max(0.0, float(chunk.get("speech_right_padding_s") or 0.0))
    except (TypeError, ValueError):
        right_padding_s = 0.0

    if left_padding_s <= 0.0 and right_padding_s <= 0.0:
        return full_start, full_end, "chunk"

    core_start = min(full_end, left_padding_s)
    core_end = max(core_start, full_end - right_padding_s)
    if core_end - core_start < 0.05:
        return full_start, full_end, "chunk"
    return core_start, core_end, "speech_core"


def _with_alignment_fallback_window(chunk: dict, text_result: dict) -> dict:
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
    start_s, end_s, source = _alignment_fallback_window_from_chunk(
        chunk,
        duration=duration,
    )
    result["alignment_fallback_start_s"] = round(start_s, 6)
    result["alignment_fallback_end_s"] = round(end_s, 6)
    result["alignment_fallback_source"] = source
    return result


def _fallback_window_for_chunk_log(chunk: dict, duration: float) -> tuple[float, float, str]:
    start_s, end_s, source = _alignment_fallback_window_from_chunk(
        chunk,
        duration=duration,
    )
    return start_s, end_s, source


def _append_fallback_window_log(chunk_log: list[str], source: str) -> None:
    if source == "speech_core":
        chunk_log.append("Alignment 回退窗口: speech_core")


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _sentinel_island_split_enabled() -> bool:
    return _env_bool("ALIGNMENT_SENTINEL_ISLAND_SPLIT", "0")


def _alignment_pack_frame_hop_s() -> float:
    try:
        value = float(os.getenv("BOUNDARY_FEATURE_FRAME_HOP_S", "0.02"))
    except (TypeError, ValueError):
        value = 0.02
    return value if value > 0 else 0.02


def _alignment_sentinel_island_pad_s() -> float:
    return _ALIGNMENT_SENTINEL_ISLAND_PAD_FRAMES * _alignment_pack_frame_hop_s()


def _alignment_sentinel_island_merge_gap_s() -> float:
    return _ALIGNMENT_SENTINEL_ISLAND_MERGE_GAP_FRAMES * _alignment_pack_frame_hop_s()


def _clamp_alignment_islands(
    spans: list[tuple[float, float]],
    *,
    duration: float,
    pad_s: float,
    merge_gap_s: float,
    min_s: float,
) -> list[tuple[float, float]]:
    prepared: list[tuple[float, float]] = []
    for raw_start, raw_end in spans:
        try:
            start = float(raw_start)
            end = float(raw_end)
        except (TypeError, ValueError):
            continue
        start = max(0.0, min(duration, start - pad_s))
        end = max(start, min(duration, end + pad_s))
        if end - start >= min_s:
            prepared.append((start, end))

    if not prepared:
        return []
    prepared.sort(key=lambda item: (item[0], item[1]))

    merged: list[tuple[float, float]] = []
    for start, end in prepared:
        if not merged or start - merged[-1][1] > merge_gap_s:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        merged[-1] = (prev_start, max(prev_end, end))
    return merged


def _looks_like_alignment_failure(
    words: list[dict],
    scene_duration_sec: float | None = None,
) -> bool:
    return looks_like_word_timing_failure(
        words,
        min_span_ms=_ALIGNMENT_MIN_SPAN_MS,
        max_zero_ratio=_ALIGNMENT_MAX_ZERO_RATIO,
        max_repeat_ratio=_ALIGNMENT_MAX_REPEAT_RATIO,
        scene_duration_sec=scene_duration_sec,
        max_coverage_ratio=_ALIGNMENT_MAX_COVERAGE_RATIO,
        max_cps=_ALIGNMENT_MAX_CPS,
    )


def _alignment_failure_reasons(
    words: list[dict],
    scene_duration_sec: float | None = None,
) -> list[str]:
    return word_timing_failure_reasons(
        words,
        min_span_ms=_ALIGNMENT_MIN_SPAN_MS,
        max_zero_ratio=_ALIGNMENT_MAX_ZERO_RATIO,
        max_repeat_ratio=_ALIGNMENT_MAX_REPEAT_RATIO,
        scene_duration_sec=scene_duration_sec,
        max_coverage_ratio=_ALIGNMENT_MAX_COVERAGE_RATIO,
        max_cps=_ALIGNMENT_MAX_CPS,
    )


def _split_span_evenly(
    start: float, end: float, chunk_size: float
) -> list[tuple[float, float]]:
    if end - start <= chunk_size:
        return [(start, end)]

    spans: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + chunk_size)
        spans.append((cursor, chunk_end))
        cursor = chunk_end
    return spans


def _prepare_asr_chunk_results(
    backend: LocalAsrBackend,
    chunks: list[dict],
    text_stage_label: str,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[tuple[dict, list[str]]], dict[str, float]]:
    stage_started = time.perf_counter()
    text_results, text_timings = _transcribe_asr_chunks_text_only(
        backend,
        chunks,
        text_stage_label,
        on_stage=on_stage,
    )
    text_results = [
        _with_alignment_fallback_window(chunk, text_result)
        for chunk, text_result in zip(chunks, text_results)
    ]
    unload_started = time.perf_counter()
    backend.unload_model(on_stage=on_stage)
    unload_elapsed = time.perf_counter() - unload_started
    prepared, align_timings = _align_TRANSCRIPTION_results(
        backend,
        text_results,
        on_stage=on_stage,
    )
    total_elapsed = time.perf_counter() - stage_started

    return prepared, {
        "text_transcribe_s": text_timings["text_transcribe_s"],
        "asr_unload_s": unload_elapsed,
        "alignment_s": align_timings["alignment_s"],
        "total_s": total_elapsed,
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
    supports_initial_prompts = _backend_accepts_initial_prompts(backend)
    chunk_positions = {int(chunk["index"]): position for position, chunk in enumerate(chunks)}
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
        batch_contexts = [_build_ASR_CONTEXT_for_chunk(chunk) for chunk in batch_chunks]
        audio_paths = [chunk["path"] for chunk in batch_chunks]
        kwargs = {
            "contexts": batch_contexts,
            "on_stage": on_stage,
        }
        if supports_initial_prompts:
            kwargs["initial_prompts"] = [
                _build_initial_prompt_for_chunk(
                    chunk,
                    chunks,
                    text_results_by_index,
                    chunk_positions,
                )
                for chunk in batch_chunks
            ]
        return backend.transcribe_texts(audio_paths, **kwargs)

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
                if on_stage:
                    on_stage(f"{text_stage_label} {batch_end}/{len(chunks)}...")
                if not pending_chunks:
                    continue

                _store_text_results(pending_chunks, _transcribe_batch(pending_chunks))
        else:
            pending_chunks = [
                chunk for chunk in chunks if int(chunk["index"]) not in text_results_by_index
            ]
            while pending_chunks:
                batch_chunks = pending_chunks[:request_batch_size]
                pending_chunks = pending_chunks[request_batch_size:]
                if on_stage:
                    on_stage(
                        f"{text_stage_label} {min(len(text_results_by_index) + len(batch_chunks), len(chunks))}/{len(chunks)}..."
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
    reason: str = "Alignment skipped: empty segments placeholder",
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
                    "run_id=, detail=empty alignment input"
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

    finalize_batch_size = max(1, int(getattr(backend, "align_batch_size", 1)))

    for batch_start in range(0, len(pending_results), finalize_batch_size):
        batch_results = pending_results[batch_start : batch_start + finalize_batch_size]
        batch_indices = pending_indices[batch_start : batch_start + finalize_batch_size]
        batch_end = batch_start + len(batch_results)
        if on_stage:
            on_stage(f"Alignment 对齐 {batch_end}/{len(pending_results)}...")
        finalized_batch = backend.finalize_text_results(
            batch_results,
            on_stage=on_stage,
        )
        for result_index, finalized in zip(batch_indices, finalized_batch):
            prepared[result_index] = finalized
        if len(finalized_batch) < len(batch_results):
            for missing_index in batch_indices[len(finalized_batch) :]:
                prepared[missing_index] = _empty_alignment_placeholder(
                    text_results[missing_index],
                    reason="Alignment skipped: backend returned no finalize result",
                )

    align_elapsed = time.perf_counter() - align_started
    return [
        item
        if item is not None
        else _empty_alignment_placeholder(
            text_results[idx],
            reason="Alignment skipped: missing finalize result",
        )
        for idx, item in enumerate(prepared)
    ], {"alignment_s": align_elapsed}


def _build_transcript_chunks(
    chunks: list[dict], text_results: list[dict]
) -> list[dict]:
    transcript_chunks: list[dict] = []
    for chunk, text_result in zip(chunks, text_results):
        item = {
            "index": chunk["index"],
            "start": float(chunk["start"]),
            "end": float(chunk["end"]),
            "duration": float(text_result.get("duration", 0.0)),
            "language": text_result.get("language", ""),
            "text": text_result.get("text", ""),
            "raw_text": text_result.get("raw_text", ""),
        }
        if "alignment_fallback_start_s" in text_result and "alignment_fallback_end_s" in text_result:
            fallback_start = float(text_result.get("alignment_fallback_start_s") or 0.0)
            fallback_end = float(text_result.get("alignment_fallback_end_s") or fallback_start)
            fallback_end = max(fallback_start, fallback_end)
            chunk_start = float(chunk["start"])
            item["alignment_fallback_start_s"] = fallback_start
            item["alignment_fallback_end_s"] = fallback_end
            item["alignment_fallback_duration_s"] = max(0.0, fallback_end - fallback_start)
            item["alignment_fallback_abs_start_s"] = chunk_start + fallback_start
            item["alignment_fallback_abs_end_s"] = chunk_start + fallback_end
            item["alignment_fallback_source"] = text_result.get("alignment_fallback_source", "chunk")
        if chunk.get("merged_from"):
            item["merged_from"] = list(chunk.get("merged_from") or [])
        transcript_chunks.append(item)
    return transcript_chunks


def _build_ASR_CONTEXT_for_chunk(chunk: dict) -> str:
    parts: list[str] = []
    chunk_start = float(chunk.get("start", 0.0))
    head_context = _asr_head_context()
    context = _asr_context()
    try:
        head_context_max_start_s = float(os.getenv("ASR_HEAD_CONTEXT_MAX_START_S", "16"))
    except (TypeError, ValueError):
        head_context_max_start_s = _ASR_HEAD_CONTEXT_MAX_START_S
    if head_context and chunk_start <= head_context_max_start_s:
        parts.append(head_context)
    if context:
        parts.append(context)
    return "\n".join(part for part in parts if part)


def _backend_accepts_initial_prompts(backend: BaseAsrBackend) -> bool:
    try:
        return "initial_prompts" in inspect.signature(backend.transcribe_texts).parameters
    except (TypeError, ValueError):
        return False


def _should_reset_sliding_context(previous: dict, current: dict) -> bool:
    gap = float(current.get("start", 0.0)) - float(previous.get("end", 0.0))
    return gap > _ASR_CONTEXT_RESET_GAP_S


def _sliding_context_result_text(text_result: dict) -> str:
    text = _clean_segment_text(str(text_result.get("text", "") or ""))
    if not text:
        return ""
    if _is_low_value_text(text):
        return ""
    return text


def _truncate_initial_prompt(prompt: str) -> str:
    try:
        max_chars = int(
            os.getenv(
                "ASR_INITIAL_PROMPT_MAX_CHARS",
                str(_ASR_INITIAL_PROMPT_MAX_CHARS),
            )
        )
    except (TypeError, ValueError):
        max_chars = _ASR_INITIAL_PROMPT_MAX_CHARS
    if max_chars <= 0:
        return ""
    if len(prompt) <= max_chars:
        return prompt

    cut_pos = len(prompt) - max_chars
    truncated = prompt[cut_pos:]
    # only advance to next word boundary when we cut mid-word
    if cut_pos > 0 and not prompt[cut_pos - 1].isspace():
        first_space = truncated.find(" ")
        if first_space > 0 and first_space + 1 < len(truncated):
            truncated = truncated[first_space + 1 :]
    return truncated


def _build_initial_prompt_for_chunk(
    chunk: dict,
    chunks: list[dict],
    text_results_by_index: dict[int, dict],
    chunk_positions: dict[int, int],
) -> str | None:
    if _ASR_SLIDING_CONTEXT_SEGS <= 0:
        return None

    chunk_index = int(chunk.get("index", 0))
    position = chunk_positions.get(chunk_index)
    if position is None or position <= 0:
        return None

    prompt_parts: list[str] = []
    cursor = position - 1
    while cursor >= 0 and len(prompt_parts) < _ASR_SLIDING_CONTEXT_SEGS:
        previous_chunk = chunks[cursor]
        next_chunk = chunks[cursor + 1]
        if _should_reset_sliding_context(previous_chunk, next_chunk):
            break

        previous_index = int(previous_chunk.get("index", 0))
        previous_result = text_results_by_index.get(previous_index)
        if previous_result is None:
            break

        previous_text = _sliding_context_result_text(previous_result)
        if previous_text:
            prompt_parts.append(previous_text)
        cursor -= 1

    if not prompt_parts:
        return None
    return _truncate_initial_prompt("\n".join(reversed(prompt_parts)))


def _should_skip_alignment_retry(text: str) -> bool:
    cleaned = _clean_segment_text(text)
    compact = _strip_punctuation(cleaned)
    if not compact:
        return True
    if len(compact) <= _ALIGNMENT_RETRY_SKIP_MAX_TEXT_LEN:
        return True
    return _is_low_value_text(cleaned)


def _needs_alignment_fallback(
    words: list[dict],
    text: str,
    scene_duration_sec: float | None = None,
) -> bool:
    compact_len = len(_strip_punctuation(text))
    if compact_len > 0 and not words:
        return True
    if compact_len >= 18 and len(words) <= 1:
        return True
    return _looks_like_alignment_failure(words, scene_duration_sec=scene_duration_sec)


def _finalize_aligned_chunk_without_asr_retry(
    chunk: dict,
    chunk_result: dict,
    chunk_log: list[str],
    backend: BaseAsrBackend | None = None,
    source_audio_path: str | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[str]]:
    chunk_words = list(chunk_result.get("words", []))
    text = chunk_result.get("text", "")
    start = float(chunk["start"])
    end = float(chunk["end"])
    duration = max(0.0, end - start)

    if not _needs_alignment_fallback(chunk_words, text, scene_duration_sec=duration):
        return chunk_words, chunk_log

    if backend is not None and source_audio_path:
        split_words, split_log = _split_alignment_sentinel_with_speech_islands(
            backend,
            source_audio_path,
            chunk,
            chunk_result,
            on_stage=on_stage,
        )
        chunk_log.extend(split_log)
        if split_words:
            return split_words, chunk_log

    chunk_log.append(
        "Alignment 哨兵触发: 时间轴异常，不重新调用 ASR，改用 VAD/比例回退"
    )
    fallback_start, fallback_end, fallback_source = _fallback_window_for_chunk_log(
        chunk,
        duration,
    )
    fallback_words, fallback_mode, fallback_meta = _build_timestamp_fallback(
        text,
        fallback_start,
        fallback_end,
        audio_path=chunk["path"],
    )
    _append_fallback_window_log(chunk_log, fallback_source)
    if fallback_mode == "aligner_vad_fallback":
        chunk_log.append("Alignment 回退: 使用 VAD 约束比例时间戳")
        chunk_log.append(
            f"Alignment VAD 回退语音区间: {fallback_meta.get('speech_span_count', 0)}"
        )
    else:
        chunk_log.append("Alignment 回退: 使用等比分配时间戳")
        if fallback_meta.get("vad_error"):
            chunk_log.append(f"Alignment VAD 回退异常: {fallback_meta['vad_error']}")
    chunk_log.append(f"Alignment 模式: {fallback_mode}")
    return fallback_words, chunk_log


def _refine_chunk_with_subchunks(
    backend: LocalAsrBackend,
    source_audio_path: str,
    chunk: dict,
    subchunk_size: float,
    retry_depth: int,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[str]]:
    start = float(chunk["start"])
    end = float(chunk["end"])
    refine_spans = _split_span_evenly(start, end, subchunk_size)
    if len(refine_spans) <= 1:
        return [], []

    refine_dir, refine_infos = _extract_wav_chunks(source_audio_path, refine_spans)
    refine_words: list[dict] = []
    refine_log: list[str] = []
    try:
        prepared_results, _stage_timings = _prepare_asr_chunk_results(
            backend,
            refine_infos,
            "Alignment 局部细化",
            on_stage=on_stage,
        )
        for refine_idx, refine_chunk, (initial_chunk_result, initial_chunk_log) in zip(
            range(1, len(refine_infos) + 1),
            refine_infos,
            prepared_results,
        ):
            sub_words, sub_log = _transcribe_asr_chunk_with_retry(
                backend,
                source_audio_path,
                refine_chunk,
                on_stage=on_stage,
                retry_depth=retry_depth + 1,
                initial_chunk_result=initial_chunk_result,
                initial_chunk_log=initial_chunk_log,
            )
            refine_log.extend(
                f"refine {refine_idx}: {line}"
                for line in sub_log
                if line.startswith(
                    (
                        "ASR 时间戳策略",
                        "ASR 原生时间戳词数",
                        "ASR 原生时间戳异常",
                        "ASR 原生时间戳需局部细化",
                        "Alignment 模式",
                        "Alignment 异常",
                        "Alignment 哨兵",
                        "Alignment 降级",
                        "Alignment VAD 回退",
                    )
                )
            )
            for word in sub_words:
                refine_words.append(
                    {
                        "start": float(word["start"]) + refine_chunk["start"] - start,
                        "end": float(word["end"]) + refine_chunk["start"] - start,
                        "word": word["word"],
                    }
                )
    finally:
        if refine_dir.exists() and not _KEEP_ASR_CHUNKS:
            _delete_path_for_cleanup(refine_dir)

    refine_words.sort(key=lambda item: (item["start"], item["end"]))
    return refine_words, refine_log


def _split_alignment_sentinel_with_speech_islands(
    backend: BaseAsrBackend,
    source_audio_path: str,
    chunk: dict,
    chunk_result: dict,
    *,
    retry_depth: int = 0,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[str]]:
    plan = _build_alignment_sentinel_island_plan(
        source_audio_path,
        chunk,
        chunk_result,
        retry_depth=retry_depth,
    )
    if plan is None:
        return [], []
    if not plan["island_spans"]:
        return [], list(plan["log"])

    source_path = str(plan["source_path"])
    chunk_start = float(plan["chunk_start"])
    duration = float(plan["duration"])
    split_log = list(plan["log"])
    split_dir, split_infos = _extract_wav_chunks(source_path, list(plan["island_spans"]))
    try:
        backend.unload_forced_aligner(on_stage=on_stage)
        prepared_results, _stage_timings = _prepare_asr_chunk_results(
            backend,
            split_infos,
            "Alignment speech-island split",
            on_stage=on_stage,
        )
        backend.unload_forced_aligner(on_stage=on_stage)
        return _collect_alignment_island_split_words(
            split_infos,
            prepared_results,
            split_log,
            parent_start=chunk_start,
            parent_duration=duration,
        )
    finally:
        backend.unload_forced_aligner(on_stage=on_stage)
        if split_dir.exists() and not _KEEP_ASR_CHUNKS:
            _delete_path_for_cleanup(split_dir)


def _build_alignment_sentinel_island_plan(
    source_audio_path: str | None,
    chunk: dict,
    chunk_result: dict,
    *,
    retry_depth: int = 0,
) -> dict | None:
    if not _sentinel_island_split_enabled() or retry_depth > 0:
        return None

    alignment_mode = str(chunk_result.get("alignment_mode") or "").strip()
    if alignment_mode in {"empty", "nonlexical", "align_text_empty"}:
        return None

    text = str(chunk_result.get("text") or chunk_result.get("raw_text") or "")
    if not _strip_punctuation(_clean_segment_text(text)):
        return None

    try:
        chunk_start = float(chunk["start"])
        chunk_end = float(chunk["end"])
        duration = max(0.0, chunk_end - chunk_start)
    except (KeyError, TypeError, ValueError):
        return None

    if duration <= 0:
        return None

    source_path = str(source_audio_path or chunk.get("source_audio_path") or "")
    if not source_path or not Path(source_path).exists():
        return {
            "source_path": source_path,
            "chunk_start": chunk_start,
            "duration": duration,
            "island_spans": [],
            "log": ["Alignment speech-island split 跳过: source_audio_path missing"],
        }

    return {
        "source_path": source_path,
        "chunk_start": chunk_start,
        "duration": duration,
        "island_spans": [],
        "log": ["Alignment speech-island split 跳过: local fallback VAD has been removed"],
    }


def _collect_alignment_island_split_words(
    split_infos: list[dict],
    prepared_results: list[tuple[dict, list[str]]],
    split_log: list[str],
    *,
    parent_start: float,
    parent_duration: float,
) -> tuple[list[dict], list[str]]:
    split_words: list[dict] = []
    forced_chunks = 0
    empty_chunks = 0
    failed_chunks = 0
    for split_idx, split_chunk, (split_result, split_result_log) in zip(
        range(1, len(split_infos) + 1),
        split_infos,
        prepared_results,
    ):
        split_mode = str(split_result.get("alignment_mode", "")).strip()
        split_text = str(split_result.get("text") or split_result.get("raw_text") or "")
        words = list(split_result.get("words") or [])
        if not split_text.strip():
            empty_chunks += 1
            continue
        if split_mode != "forced_aligner":
            failed_chunks += 1
            split_log.extend(
                f"speech-island {split_idx}: {line}"
                for line in split_result_log
                if line.startswith(("Alignment 模式", "Alignment 异常", "Alignment VAD 回退"))
            )
            continue
        if not words or _looks_like_alignment_failure(words):
            failed_chunks += 1
            reasons = _alignment_failure_reasons(words)
            split_log.append(
                "Alignment speech-island split 子片段异常: "
                f"idx={split_idx} reasons={','.join(reasons) or 'empty_or_unknown'}"
            )
            continue

        forced_chunks += 1
        offset = float(split_chunk["start"]) - parent_start
        for word in words:
            try:
                word_start = float(word["start"]) + offset
                word_end = float(word["end"]) + offset
            except (KeyError, TypeError, ValueError):
                continue
            word_start = max(0.0, min(parent_duration, word_start))
            word_end = max(word_start, min(parent_duration, word_end))
            split_words.append(
                {
                    "start": word_start,
                    "end": word_end,
                    "word": word.get("word", ""),
                }
            )

    split_words.sort(key=lambda item: (item["start"], item["end"]))
    if split_words and not _looks_like_alignment_failure(split_words):
        split_log.append(
            "Alignment speech-island split 成功: "
            f"islands={len(split_infos)} forced_chunks={forced_chunks} "
            f"empty_chunks={empty_chunks} failed_chunks={failed_chunks} words={len(split_words)}"
        )
        return split_words, split_log

    split_log.append(
        "Alignment speech-island split 失败: "
        f"forced_chunks={forced_chunks} empty_chunks={empty_chunks} "
        f"failed_chunks={failed_chunks} words={len(split_words)}"
    )
    return [], split_log


def _split_alignment_sentinels_with_speech_islands_batch(
    backend: BaseAsrBackend,
    source_audio_path: str,
    chunks: list[dict],
    prepared_results: list[tuple[dict, list[str]]],
    *,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[dict[int, list[dict]], dict[int, list[str]], bool]:
    if not _sentinel_island_split_enabled():
        return {}, {}, False

    words_by_index: dict[int, list[dict]] = {}
    logs_by_index: dict[int, list[str]] = {}
    split_dirs: list[Path] = []
    split_infos: list[dict] = []
    parent_groups: dict[int, list[dict]] = {}
    parent_meta: dict[int, tuple[float, float]] = {}

    for chunk, (chunk_result, _chunk_log) in zip(chunks, prepared_results):
        try:
            chunk_index = int(chunk["index"])
            chunk_start = float(chunk["start"])
            chunk_end = float(chunk["end"])
        except (KeyError, TypeError, ValueError):
            continue
        duration = max(0.0, chunk_end - chunk_start)
        if not _needs_alignment_fallback(
            list(chunk_result.get("words", [])),
            str(chunk_result.get("text") or ""),
            scene_duration_sec=duration,
        ):
            continue

        plan = _build_alignment_sentinel_island_plan(
            source_audio_path,
            chunk,
            chunk_result,
        )
        if plan is None:
            continue
        logs_by_index.setdefault(chunk_index, []).extend(list(plan["log"]))
        if not plan["island_spans"]:
            continue

        split_dir, parent_split_infos = _extract_wav_chunks(
            str(plan["source_path"]),
            list(plan["island_spans"]),
        )
        split_dirs.append(split_dir)
        parent_meta[chunk_index] = (float(plan["chunk_start"]), float(plan["duration"]))
        for split_info in parent_split_infos:
            copied_info = dict(split_info)
            copied_info["index"] = len(split_infos)
            copied_info["_parent_chunk_index"] = chunk_index
            parent_groups.setdefault(chunk_index, []).append(copied_info)
            split_infos.append(copied_info)

    if not split_infos:
        return words_by_index, logs_by_index, True

    try:
        backend.unload_forced_aligner(on_stage=on_stage)
        prepared_split_results, _stage_timings = _prepare_asr_chunk_results(
            backend,
            split_infos,
            "Alignment speech-island split batch",
            on_stage=on_stage,
        )
        backend.unload_forced_aligner(on_stage=on_stage)

        results_by_split_index = {
            int(split_info["index"]): prepared
            for split_info, prepared in zip(split_infos, prepared_split_results)
        }
        for chunk_index, group_infos in parent_groups.items():
            parent_start, parent_duration = parent_meta[chunk_index]
            group_results = [
                results_by_split_index[int(split_info["index"])]
                for split_info in group_infos
                if int(split_info["index"]) in results_by_split_index
            ]
            split_words, split_log = _collect_alignment_island_split_words(
                group_infos,
                group_results,
                list(logs_by_index.get(chunk_index, [])),
                parent_start=parent_start,
                parent_duration=parent_duration,
            )
            logs_by_index[chunk_index] = split_log
            if split_words:
                words_by_index[chunk_index] = split_words
    finally:
        backend.unload_forced_aligner(on_stage=on_stage)
        if not _KEEP_ASR_CHUNKS:
            for split_dir in split_dirs:
                if split_dir.exists():
                    _delete_path_for_cleanup(split_dir)

    return words_by_index, logs_by_index, True


def _transcribe_asr_chunk_with_retry(
    backend: LocalAsrBackend,
    source_audio_path: str,
    chunk: dict,
    on_stage: Callable[[str], None] | None = None,
    retry_depth: int = 0,
    initial_chunk_result: dict | None = None,
    initial_chunk_log: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    if initial_chunk_result is None:
        chunk_result, chunk_log = backend.transcribe_to_words(
            chunk["path"], on_stage=on_stage
        )
    else:
        chunk_result = initial_chunk_result
        chunk_log = list(initial_chunk_log or [])
    chunk_words = list(chunk_result.get("words", []))
    chunk_mode = str(chunk_result.get("alignment_mode", "")).strip()
    start = float(chunk["start"])
    end = float(chunk["end"])
    duration = max(0.0, end - start)

    if (
        chunk_mode == "ASR_NATIVE_refine_needed"
        and retry_depth < _ALIGNMENT_MAX_REFINE_DEPTH
        and duration > _ALIGNMENT_COARSE_REFINE_CHUNK_S
    ):
        chunk_log.append(
            "ASR 原生时间戳需局部细化: 长块不直接整块强制对齐，先拆为中等片段"
        )
        refined_words, refined_log = _refine_chunk_with_subchunks(
            backend,
            source_audio_path,
            chunk,
            _ALIGNMENT_COARSE_REFINE_CHUNK_S,
            retry_depth,
            on_stage=on_stage,
        )
        chunk_log.extend(refined_log)
        if refined_words:
            return refined_words, chunk_log

    if not _looks_like_alignment_failure(chunk_words, scene_duration_sec=duration):
        return chunk_words, chunk_log

    if chunk_mode == "forced_aligner":
        split_words, split_log = _split_alignment_sentinel_with_speech_islands(
            backend,
            source_audio_path,
            chunk,
            chunk_result,
            retry_depth=retry_depth,
            on_stage=on_stage,
        )
        chunk_log.extend(split_log)
        if split_words:
            return split_words, chunk_log
        chunk_log.append("Alignment 哨兵触发: forced 模式保留原文，不再对子片段重转写")
        return chunk_words, chunk_log

    if _should_skip_alignment_retry(chunk_result.get("text", "")):
        chunk_log.append("Alignment 哨兵触发: 低价值/短文本直接回退，不再做子片段重试")
        fallback_start, fallback_end, fallback_source = _fallback_window_for_chunk_log(
            chunk,
            duration,
        )
        fallback_words, fallback_mode, fallback_meta = _build_timestamp_fallback(
            chunk_result.get("text", ""),
            fallback_start,
            fallback_end,
            audio_path=chunk["path"],
        )
        _append_fallback_window_log(chunk_log, fallback_source)
        if fallback_mode == "aligner_vad_fallback":
            chunk_log.append("Alignment 快速回退: 使用 VAD 约束比例时间戳")
            chunk_log.append(
                f"Alignment VAD 回退语音区间: {fallback_meta.get('speech_span_count', 0)}"
            )
        else:
            chunk_log.append("Alignment 快速回退: 使用等比分配时间戳")
            if fallback_meta.get("vad_error"):
                chunk_log.append(f"Alignment VAD 回退异常: {fallback_meta['vad_error']}")
        chunk_log.append(f"Alignment 模式: {fallback_mode}")
        return fallback_words, chunk_log

    if retry_depth < _ALIGNMENT_MAX_REFINE_DEPTH and duration > _ALIGNMENT_COARSE_REFINE_CHUNK_S:
        chunk_log.append("Alignment 哨兵触发: 长块先细化为中等片段重试")
        refined_words, refined_log = _refine_chunk_with_subchunks(
            backend,
            source_audio_path,
            chunk,
            _ALIGNMENT_COARSE_REFINE_CHUNK_S,
            retry_depth,
            on_stage=on_stage,
        )
        chunk_log.extend(refined_log)
        if refined_words and not _looks_like_alignment_failure(
            refined_words,
            scene_duration_sec=duration,
        ):
            chunk_log.append(f"Alignment 长块细化成功: {len(refined_words)} 词")
            return refined_words, chunk_log

    chunk_log.append(
        "Alignment 哨兵触发: 检测到异常密集/重复时间戳，准备切为更小片段重试"
    )
    retry_spans = _split_span_evenly(start, end, _ALIGNMENT_STEP_DOWN_CHUNK_S)
    if len(retry_spans) <= 1:
        fallback_start, fallback_end, fallback_source = _fallback_window_for_chunk_log(
            chunk,
            duration,
        )
        fallback_words, fallback_mode, fallback_meta = _build_timestamp_fallback(
            chunk_result.get("text", ""),
            fallback_start,
            fallback_end,
            audio_path=chunk["path"],
        )
        _append_fallback_window_log(chunk_log, fallback_source)
        if fallback_mode == "aligner_vad_fallback":
            chunk_log.append("Alignment 降级失败: 子片段不足，改用 VAD 约束比例时间戳")
            chunk_log.append(
                f"Alignment VAD 回退语音区间: {fallback_meta.get('speech_span_count', 0)}"
            )
        else:
            chunk_log.append("Alignment 降级失败: 子片段不足，改用等比分配时间戳")
            if fallback_meta.get("vad_error"):
                chunk_log.append(f"Alignment VAD 回退异常: {fallback_meta['vad_error']}")
        chunk_log.append(f"Alignment 模式: {fallback_mode}")
        return fallback_words, chunk_log

    retry_dir, retry_infos = _extract_wav_chunks(source_audio_path, retry_spans)
    retry_words: list[dict] = []
    try:
        prepared_results, _stage_timings = _prepare_asr_chunk_results(
            backend,
            retry_infos,
            "Alignment 降级重试",
            on_stage=on_stage,
        )
        for retry_idx, retry_chunk, (retry_result, retry_log) in zip(
            range(1, len(retry_infos) + 1),
            retry_infos,
            prepared_results,
        ):
            chunk_log.extend(
                f"retry {retry_idx}: {line}"
                for line in retry_log
                if line.startswith(("Alignment 模式", "Alignment 异常"))
            )
            for word in retry_result.get("words", []):
                retry_words.append(
                    {
                        "start": float(word["start"]) + retry_chunk["start"] - start,
                        "end": float(word["end"]) + retry_chunk["start"] - start,
                        "word": word["word"],
                    }
                )
    finally:
        if retry_dir.exists() and not _KEEP_ASR_CHUNKS:
            _delete_path_for_cleanup(retry_dir)

    retry_words.sort(key=lambda item: (item["start"], item["end"]))
    if retry_words and not _looks_like_alignment_failure(
        retry_words,
        scene_duration_sec=duration,
    ):
        chunk_log.append(f"Alignment 降级成功: {len(retry_infos)} 个子片段")
        return retry_words, chunk_log

    fallback_start, fallback_end, fallback_source = _fallback_window_for_chunk_log(
        chunk,
        duration,
    )
    fallback_words, fallback_mode, fallback_meta = _build_timestamp_fallback(
        chunk_result.get("text", ""),
        fallback_start,
        fallback_end,
        audio_path=chunk["path"],
    )
    _append_fallback_window_log(chunk_log, fallback_source)
    if fallback_mode == "aligner_vad_fallback":
        chunk_log.append("Alignment 降级后仍异常: 改用 VAD 约束比例时间戳")
        chunk_log.append(
            f"Alignment VAD 回退语音区间: {fallback_meta.get('speech_span_count', 0)}"
        )
    else:
        chunk_log.append("Alignment 降级后仍异常: 改用等比分配时间戳")
        if fallback_meta.get("vad_error"):
            chunk_log.append(f"Alignment VAD 回退异常: {fallback_meta['vad_error']}")
    chunk_log.append(f"Alignment 模式: {fallback_mode}")
    return fallback_words, chunk_log


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
    merged_segments = _merge_fragment_segments(cleaned_segments)
    split_segments = _split_long_postprocessed_segments(merged_segments)
    return _repair_postprocessed_segment_windows(split_segments)


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_TERMINAL_RE.search((text or "").strip()))


def _same_source_chunk(current: dict, following: dict) -> bool:
    current_chunk = current.get("source_chunk_index")
    following_chunk = following.get("source_chunk_index")
    return (
        current_chunk is not None
        and following_chunk is not None
        and current_chunk == following_chunk
    )


def _should_merge_fragment(current: dict, following: dict) -> bool:
    current_text = str(current.get("text", "")).strip()
    following_text = str(following.get("text", "")).strip()
    if not current_text or not following_text:
        return False
    if not _same_source_chunk(current, following):
        return False
    if _ends_sentence(current_text):
        return False

    gap = float(following["start"]) - float(current["end"])
    if gap < -0.05 or gap > _ASR_FRAGMENT_MERGE_MAX_GAP_S:
        return False

    combined_chars = len(_compact_text_units(current_text + following_text))
    if combined_chars > _ASR_FRAGMENT_MERGE_MAX_CHARS:
        return False

    combined_duration = float(following["end"]) - float(current["start"])
    if combined_duration > _ASR_FRAGMENT_MERGE_MAX_DURATION_S:
        return False

    return True


def _join_segment_text(left: str, right: str) -> str:
    left = (left or "").strip()
    right = (right or "").strip()
    if not left:
        return right
    if not right:
        return left
    if _FRAGMENT_CONTINUATION_START_RE.match(right):
        return left + right
    return left + right


def _merge_fragment_segments(segments: list[dict]) -> list[dict]:
    if len(segments) < 2:
        return segments

    merged: list[dict] = []
    current = dict(segments[0])

    for following in segments[1:]:
        if _should_merge_fragment(current, following):
            current["end"] = float(following["end"])
            current["text"] = _join_segment_text(current.get("text", ""), following.get("text", ""))
            current["words"] = list(current.get("words") or []) + list(
                following.get("words") or []
            )
            continue

        merged.append(current)
        current = dict(following)

    merged.append(current)
    return merged


def _pick_postprocess_split_index(text: str) -> int:
    text = (text or "").strip()
    if len(text) < _ASR_POSTPROCESS_SPLIT_MIN_CHARS * 2:
        return 0

    target = len(text) // 2
    split_chars = "。！？!?…、,，・ "
    candidates = [
        idx + 1
        for idx, char in enumerate(text[:-1])
        if char in split_chars
        and _ASR_POSTPROCESS_SPLIT_MIN_CHARS <= idx + 1
        and len(text) - (idx + 1) >= _ASR_POSTPROCESS_SPLIT_MIN_CHARS
    ]
    if candidates:
        return min(candidates, key=lambda idx: abs(idx - target))

    return target


def _split_long_postprocessed_segment(segment: dict) -> list[dict]:
    text = str(segment.get("text", "")).strip()
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    duration = max(0.0, end - start)
    compact_len = len(_compact_text_units(text))
    original_words = list(segment.get("words") or [])

    if (
        compact_len <= _ASR_POSTPROCESS_MAX_CHARS
        and duration <= _ASR_POSTPROCESS_MAX_DURATION_S
    ):
        return [segment]

    split_idx = _pick_postprocess_split_index(text)
    if not split_idx:
        return [segment]

    left_text = text[:split_idx].strip()
    right_text = text[split_idx:].strip()
    if not left_text or not right_text:
        return [segment]

    left_weight = max(1, len(_compact_text_units(left_text)))
    right_weight = max(1, len(_compact_text_units(right_text)))
    split_time = start + duration * (left_weight / (left_weight + right_weight))
    if (
        split_time - start <= _ASR_INVALID_SEGMENT_DURATION_S
        or end - split_time <= _ASR_INVALID_SEGMENT_DURATION_S
    ):
        return [segment]

    left_words = [
        word
        for word in original_words
        if float(word.get("start", start)) < split_time
    ]
    right_words = [
        word
        for word in original_words
        if float(word.get("start", start)) >= split_time
    ]
    assert len(left_words) + len(right_words) == len(original_words)

    left_segment = {
        "start": start,
        "end": split_time,
        "text": left_text,
        "source_chunk_index": segment.get("source_chunk_index"),
        "words": left_words,
    }
    right_segment = {
        "start": split_time,
        "end": end,
        "text": right_text,
        "source_chunk_index": segment.get("source_chunk_index"),
        "words": right_words,
    }
    return (
        _split_long_postprocessed_segment(left_segment)
        + _split_long_postprocessed_segment(right_segment)
    )


def _split_long_postprocessed_segments(segments: list[dict]) -> list[dict]:
    split_segments: list[dict] = []
    for segment in segments:
        split_segments.extend(_split_long_postprocessed_segment(dict(segment)))
    return split_segments


def _repair_postprocessed_segment_windows(segments: list[dict]) -> list[dict]:
    if not segments:
        return []

    repaired: list[dict] = []
    epsilon = 0.01
    for idx, segment in enumerate(segments):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start = max(0.0, float(segment.get("start", 0.0)))
        end = max(start, float(segment.get("end", start)))
        if repaired and start < float(repaired[-1]["end"]):
            start = float(repaired[-1]["end"])
            end = max(end, start)

        duration = end - start
        if duration <= _ASR_INVALID_SEGMENT_DURATION_S:
            target_end = start + _ASR_MIN_REPAIRED_SEGMENT_DURATION_S
            if idx + 1 < len(segments):
                next_start = float(segments[idx + 1].get("start", target_end))
                if next_start > start:
                    target_end = min(target_end, max(start, next_start - epsilon))

            if target_end - start <= _ASR_INVALID_SEGMENT_DURATION_S:
                if (
                    repaired
                    and start - float(repaired[-1]["end"])
                    <= _ASR_FRAGMENT_MERGE_MAX_GAP_S
                    and _same_source_chunk(repaired[-1], segment)
                ):
                    repaired[-1]["text"] = _join_segment_text(
                        str(repaired[-1].get("text", "")),
                        text,
                    )
                    repaired[-1]["end"] = max(
                        float(repaired[-1]["end"]),
                        end,
                        start + _ASR_MIN_REPAIRED_SEGMENT_DURATION_S,
                    )
                    repaired[-1]["words"] = list(repaired[-1].get("words") or []) + list(
                        segment.get("words") or []
                    )
                    continue
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
            start = float(non_overlapping[-1]["end"])
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


def _merge_words_to_segments(words: list[dict]) -> list[dict]:
    if not words:
        return []

    japanese_sentence_end = {"。", "！", "？", "…", "!", "?"}
    soft_max_chars = 45
    hard_max_chars = 60
    max_duration = 8.5
    hard_max_duration = float(os.getenv("ASR_MERGE_HARD_MAX_DURATION", "9.0"))
    max_gap = 1.2

    segments: list[dict] = []
    current_words: list[dict] = []

    def _pick_sentence_split(word_list: list[dict]) -> int:
        if len(word_list) < 2:
            return 0

        total_text = "".join(w["word"] for w in word_list).strip()
        total_len = len(total_text)
        segment_start = word_list[0]["start"]
        running_len = 0
        candidates: list[tuple[float, int]] = []

        for idx, item in enumerate(word_list[:-1], 1):
            token = item["word"]
            running_len += len(token)
            if not token or token[-1] not in japanese_sentence_end:
                continue

            left_duration = item["end"] - segment_start
            right_words = word_list[idx:]
            right_text_len = total_len - running_len
            right_duration = (
                word_list[-1]["end"] - right_words[0]["start"] if right_words else 0.0
            )

            if running_len < 8 or left_duration < 1.0:
                continue

            score = abs(running_len - soft_max_chars)
            if right_text_len < 6:
                score += 100
            elif right_text_len < 10:
                score += 12
            if right_duration < 0.5:
                score += 40
            if running_len > hard_max_chars:
                score += 25

            candidates.append((score, idx))

        if not candidates:
            return 0

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def flush(word_list: list[dict]) -> None:
        if not word_list:
            return
        text = "".join(w["word"] for w in word_list).strip()
        if not text:
            return
        segment_duration = word_list[-1]["end"] - word_list[0]["start"]
        if len(text) > soft_max_chars or segment_duration > max_duration:
            split_idx = _pick_sentence_split(word_list)
            if split_idx:
                flush(word_list[:split_idx])
                flush(word_list[split_idx:])
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

        segment_duration = current_words[-1]["end"] - current_words[0]["start"]
        segment_text = "".join(w["word"] for w in current_words)
        gap = word["start"] - current_words[-1]["end"]
        ends_sentence = bool(segment_text) and segment_text[-1] in japanese_sentence_end
        projected_text = segment_text + word["word"]
        projected_duration = word["end"] - current_words[0]["start"]
        current_chunk = current_words[-1].get("source_chunk_index")
        next_chunk = word.get("source_chunk_index")
        crosses_chunk_boundary = (
            current_chunk is not None
            and next_chunk is not None
            and current_chunk != next_chunk
        )
        should_split_turn = (
            (
                ends_sentence
                and segment_duration >= 0.8
                and len(segment_text) >= 4
            )
            or (ends_sentence and gap > 0.05)
            or (
                gap > 0.35
                and segment_duration >= 0.8
                and len(segment_text) >= 4
            )
            or (
                gap > 0.2
                and segment_duration >= 1.2
                and len(segment_text) >= 8
            )
            or (
                segment_duration >= 4.0
                and len(segment_text) >= 20
            )
        )
        exceeds_segment_limits = (
            (
                ends_sentence
                and (
                    gap > max_gap
                    or segment_duration >= max_duration
                    or len(segment_text) >= soft_max_chars
                )
            )
            or len(projected_text) > hard_max_chars
            or projected_duration >= hard_max_duration
        )

        if (
            crosses_chunk_boundary
            or should_split_turn
            or exceeds_segment_limits
        ):
            flush(current_words)
            current_words = [word]
        else:
            current_words.append(word)

    flush(current_words)
    return segments
