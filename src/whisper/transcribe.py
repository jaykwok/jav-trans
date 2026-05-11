import inspect
import os
import re
import time
import uuid
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from whisper.backends.base import BaseAsrBackend
from whisper.backends.registry import current_asr_worker_mode, _is_subprocess_backend
from whisper.checkpoint import (
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
from whisper.chunking import _KEEP_ASR_CHUNKS, _extract_wav_chunks
from whisper.local_backend import (
    LocalAsrBackend,
    SubprocessAsrBackend,
    WorkerError,
    WorkerTimeoutError,
    looks_like_word_timing_failure,
)
from whisper.timestamp_fallback import build_word_timestamps_fallback


_GRAY_MAX_DURATION_S = float(os.getenv("ASR_GRAY_MAX_DURATION", "2.5"))
_SEGMENT_CUT_MIN_SILENCE_S = float(os.getenv("SEGMENT_CUT_MIN_SILENCE", "0.5"))
_ASR_SLIDING_CONTEXT_SEGS = max(0, int(os.getenv("ASR_SLIDING_CONTEXT_SEGS", "2")))
_ALIGNMENT_STEP_DOWN_CHUNK_S = float(os.getenv("ALIGNMENT_STEP_DOWN_CHUNK", "6.0"))
_ALIGNMENT_COARSE_REFINE_CHUNK_S = float(os.getenv("ALIGNMENT_COARSE_REFINE_CHUNK", "18.0"))
_ALIGNMENT_MAX_REFINE_DEPTH = max(0, int(os.getenv("ALIGNMENT_MAX_REFINE_DEPTH", "2")))
_ASR_CONTEXT = os.getenv("ASR_CONTEXT", "").strip()
_ASR_HEAD_CONTEXT = os.getenv("ASR_HEAD_CONTEXT", "").strip()
_ASR_HEAD_CONTEXT_MAX_START_S = float(os.getenv("ASR_HEAD_CONTEXT_MAX_START_S", "0"))
_ALIGNMENT_MIN_SPAN_MS = float(os.getenv("ALIGNMENT_MIN_SPAN_MS", "120"))
_ALIGNMENT_MAX_ZERO_RATIO = float(os.getenv("ALIGNMENT_MAX_ZERO_RATIO", "0.55"))
_ALIGNMENT_MAX_REPEAT_RATIO = float(os.getenv("ALIGNMENT_MAX_REPEAT_RATIO", "0.65"))
_ALIGNMENT_MAX_COVERAGE_RATIO = float(os.getenv("ALIGNMENT_MAX_COVERAGE_RATIO", "0.05"))
_ALIGNMENT_MAX_CPS = float(os.getenv("ALIGNMENT_MAX_CPS", "50.0"))
_ALIGNMENT_RETRY_SKIP_MAX_TEXT_LEN = max(
    1,
    int(os.getenv("ALIGNMENT_RETRY_SKIP_MAX_TEXT_LEN", "10")),
)
_ASR_CONTEXT_LEAK_SIMILARITY = float(os.getenv("ASR_CONTEXT_LEAK_SIMILARITY", "0.88"))
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
    r"^[あっんーっ。！？\s…、ぁぃぅぇぉアイウエオ]{1,5}$"
    r"|^[。！？…、\s,.!?・「」（）【】；：\-—–]+$"
    r"|^[あいうえおぁぃぅぇぉんっーはふへほ]+$"
)
_TOOL_SIGNATURE_RE = re.compile(r"whisperjav\s+\d", re.I)
_STRIP_PUNCT_RE = re.compile(r"[。！？…、,.!?・「」『』（）()【】\[\]\s~〜ー-]+")
_CONTEXT_COMPACT_RE = re.compile(r"[^0-9A-Za-zぁ-ゖァ-ヺ一-龯々〆ヵヶ]+")
_CONTEXT_TOKEN_SPLIT_RE = re.compile(r"[。！？…、,.!?・「」『』（）()【】\[\]\s~〜ー\-；;：:\n\r\t]+")
_SENTENCE_TERMINAL_RE = re.compile(r"[。！？!?…」』）)\]]$")
_SENTENCE_FRAGMENT_RE = re.compile(r"[^。！？!?…]+[。！？!?…]?")
_FRAGMENT_CONTINUATION_START_RE = re.compile(
    r"^(?:ます|ました|ません|です|でした|でしょう|ながら|ので|けど|から|たり|"
    r"程度|ところ|いる|いき|して|され|なり|効果|で[、,]?|に|を|が|は|も)"
)
_MOAN_CHAR_RE = re.compile(
    r"^[ぁ-ゖァ-ヺー〜～っッんンあいうえおアイウエオはひふへほハヒフヘホ]+$"
)
_RHYTHMIC_HALLUCINATION_RE = re.compile(
    r"^.+(?:ささ|まま|だだ|なな|たた|かか|やや|ばば|ぱぱ|ちゃちゃ|じゃじゃ|べべ|ぺぺ)$"
    r"|^(.{2,5})\1+$"
)

_DEFAULT_NOISE_WORDS = (
    "あ,い,う,え,お,ん,"
    "あっ,うん,えっ,おっ,"
    "あー,えー,うー,おー,んー,"
    "ねえ,ねー,はい,もう,ふぁ,はぁ,くっ,あぁ,うぅ,おぉ"
)
_NOISE_WORDS = frozenset(
    w.strip()
    for w in os.getenv("ASR_NOISE_WORDS", _DEFAULT_NOISE_WORDS).split(",")
    if w.strip()
)

_GRAY_WORDS = frozenset(
    {
        "気持ち",
        "好き",
        "すごい",
        "すご",
        "いい",
        "だめ",
        "やばい",
        "ちょっと",
        "あかん",
    }
)
_LOW_VALUE_KEEP_WORDS = frozenset(
    {
        "だめ",
        "ダメ",
        "いや",
        "イヤ",
        "やめ",
        "やめて",
        "もっと",
        "まだ",
        "好き",
        "気持ちいい",
        "すごい",
        "お久しぶり",
        "出てきた",
        "奥",
        "本当に",
    }
)


def _current_asr_worker_mode() -> str:
    return current_asr_worker_mode()


class ASRWorkerSystemError(RuntimeError):
    pass


def _strip_punctuation(text: str) -> str:
    return _STRIP_PUNCT_RE.sub("", text or "")


def _compact_context_text(text: str) -> str:
    return _CONTEXT_COMPACT_RE.sub("", text or "")


def _context_tokens(context: str) -> list[str]:
    tokens = []
    for token in _CONTEXT_TOKEN_SPLIT_RE.split(context or ""):
        compact = _compact_context_text(token)
        if len(compact) >= 2:
            tokens.append(compact)
    return tokens


def _is_context_leak(text: str) -> bool:
    compact = _compact_context_text(text)
    if len(compact) < 3:
        return False

    for context in (_ASR_CONTEXT, _ASR_HEAD_CONTEXT):
        context_compact = _compact_context_text(context)
        if len(context_compact) < 3:
            continue

        if compact == context_compact:
            return True
        if compact in context_compact:
            return True
        if context_compact in compact and len(compact) <= len(context_compact) * 1.25:
            return True
        if SequenceMatcher(None, compact, context_compact).ratio() >= _ASR_CONTEXT_LEAK_SIMILARITY:
            return True

        remainder = compact
        for token in sorted(_context_tokens(context), key=len, reverse=True):
            remainder = remainder.replace(token, "")
        if not remainder:
            return True

    return False


def _collapse_repeated_noise(text: str) -> str:
    cleaned = re.sub(r"[ \t]+", " ", (text or "").strip())
    cleaned = re.sub(r"(.)\1{3,}", r"\1\1", cleaned)

    for _ in range(2):
        updated = re.sub(
            r"([ぁ-ゖァ-ヺ一-龯]{1,8})(?:[、。！？…\s]*\1){2,}",
            r"\1、\1",
            cleaned,
        )
        if updated == cleaned:
            break
        cleaned = updated

    return cleaned.strip()


def _is_noise_token(text: str) -> bool:
    token = _strip_punctuation(text)
    if not token:
        return True
    if token in _LOW_VALUE_KEEP_WORDS:
        return False
    if token in _NOISE_WORDS:
        return True
    return len(token) <= 4 and bool(_MOAN_CHAR_RE.fullmatch(token))


def _is_low_value_text(text: str) -> bool:
    normalized = _collapse_repeated_noise(text)
    compact = _strip_punctuation(normalized)
    if not compact:
        return True
    if compact in _LOW_VALUE_KEEP_WORDS:
        return False

    if _ASR_CONTEXT:
        context_compact = _strip_punctuation(_ASR_CONTEXT)
        if context_compact and context_compact in compact and len(compact) <= len(context_compact) + 3:
            return True

    if _TRIVIAL_SEGMENT.match(normalized):
        return True
        
    if _RHYTHMIC_HALLUCINATION_RE.fullmatch(compact):
        return True

    parts = [part for part in re.split(r"[、。！？…,.!?・\s]+", normalized) if part]
    if parts and all(_is_noise_token(part) for part in parts):
        return True

    return _is_noise_token(normalized)


def _clean_segment_text(text: str) -> str:
    return _collapse_repeated_noise((text or "").replace("\r", " ").replace("\n", " "))


def _remove_context_leak_fragments(text: str) -> str:
    if not (_ASR_CONTEXT or _ASR_HEAD_CONTEXT):
        return text

    context_tokens = sorted(
        {
            token
            for context in (_ASR_CONTEXT, _ASR_HEAD_CONTEXT)
            for token in _context_tokens(context)
            if len(token) >= 2
        },
        key=len,
        reverse=True,
    )
    strong_tokens = [token for token in context_tokens if len(token) >= 3]

    cleaned_text = text or ""
    if len(context_tokens) >= 2 and strong_tokens:
        token_pattern = "|".join(re.escape(token) for token in context_tokens)
        sequence_re = re.compile(
            rf"(?:{token_pattern})(?:[；;、。,.!?！？…・\s]+(?:{token_pattern}))+[。！？!?…]?"
        )

        def _drop_context_sequence(match: re.Match) -> str:
            compact = _compact_context_text(match.group(0))
            if any(token in compact for token in strong_tokens):
                return ""
            return match.group(0)

        cleaned_text = sequence_re.sub(_drop_context_sequence, cleaned_text)

    fragments = _SENTENCE_FRAGMENT_RE.findall(cleaned_text)
    if not fragments:
        return "" if _is_context_leak(cleaned_text) else cleaned_text

    kept_fragments = []
    changed = cleaned_text != text
    for fragment in fragments:
        if _is_context_leak(fragment):
            changed = True
            continue

        cleaned_fragment = fragment
        for token in strong_tokens:
            if token not in cleaned_fragment:
                continue
            without_token = cleaned_fragment.replace(token, "")
            if _is_low_value_text(without_token):
                changed = True
                cleaned_fragment = without_token

        cleaned_fragment = re.sub(r"[、,\s]+([。！？!?…])", r"\1", cleaned_fragment)
        cleaned_fragment = re.sub(r"^[、。！？!?…,.・\s]+|[、,.・\s]+$", "", cleaned_fragment)
        cleaned_fragment = _collapse_repeated_noise(cleaned_fragment)

        if not cleaned_fragment or _is_low_value_text(cleaned_fragment):
            changed = True
            continue
        kept_fragments.append(cleaned_fragment)

    if not changed:
        return text
    return _collapse_repeated_noise("".join(kept_fragments))


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

                batch_text_results = _transcribe_batch(pending_chunks)
                _store_text_results(pending_chunks, batch_text_results)
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
        if chunk.get("merged_from"):
            item["merged_from"] = list(chunk.get("merged_from") or [])
        transcript_chunks.append(item)
    return transcript_chunks


def _build_ASR_CONTEXT_for_chunk(chunk: dict) -> str:
    parts: list[str] = []
    chunk_start = float(chunk.get("start", 0.0))
    if _ASR_HEAD_CONTEXT and chunk_start <= _ASR_HEAD_CONTEXT_MAX_START_S:
        parts.append(_ASR_HEAD_CONTEXT)
    if _ASR_CONTEXT:
        parts.append(_ASR_CONTEXT)
    return "\n".join(part for part in parts if part)


def _backend_accepts_initial_prompts(backend: BaseAsrBackend) -> bool:
    try:
        return "initial_prompts" in inspect.signature(backend.transcribe_texts).parameters
    except (TypeError, ValueError):
        return False


def _chunk_gender_label(chunk: dict) -> str:
    for key in ("gender", "speaker_gender", "dominant_gender"):
        value = str(chunk.get(key) or "").strip().lower()
        if value:
            return value
    return ""


def _should_reset_sliding_context(previous: dict, current: dict) -> bool:
    gap = float(current.get("start", 0.0)) - float(previous.get("end", 0.0))
    if gap > _SEGMENT_CUT_MIN_SILENCE_S:
        return True

    previous_gender = _chunk_gender_label(previous)
    current_gender = _chunk_gender_label(current)
    return bool(previous_gender and current_gender and previous_gender != current_gender)


def _sliding_context_result_text(text_result: dict) -> str:
    text = _clean_segment_text(str(text_result.get("text", "") or ""))
    if not text:
        return ""
    if _is_context_leak(text) or _is_low_value_text(text):
        return ""
    return text


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
    return "\n".join(reversed(prompt_parts))


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
) -> tuple[list[dict], list[str]]:
    chunk_words = list(chunk_result.get("words", []))
    text = chunk_result.get("text", "")
    start = float(chunk["start"])
    end = float(chunk["end"])
    duration = max(0.0, end - start)

    if not _needs_alignment_fallback(chunk_words, text, scene_duration_sec=duration):
        return chunk_words, chunk_log

    chunk_log.append(
        "Alignment 哨兵触发: 时间轴异常，不重新调用 ASR，改用 VAD/比例回退"
    )
    fallback_words, fallback_mode, fallback_meta = _build_timestamp_fallback(
        text,
        0.0,
        duration,
        audio_path=chunk["path"],
    )
    if fallback_mode == "aligner_vad_fallback":
        chunk_log.append("Alignment 回退: 使用 VAD 约束比例时间戳")
        chunk_log.append(
            f"Alignment VAD 回退语音区间: {fallback_meta.get('speech_span_count', 0)}"
        )
    else:
        chunk_log.append("Alignment 回退: 使用等比分配时间戳")
        if fallback_meta.get("vad_error"):
            chunk_log.append(f"Alignment VAD 回退异常: {fallback_meta['vad_error']}")
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
        chunk_log.append("Alignment 哨兵触发: forced 模式保留原文，不再对子片段重转写")
        return chunk_words, chunk_log

    if _should_skip_alignment_retry(chunk_result.get("text", "")):
        chunk_log.append("Alignment 哨兵触发: 低价值/短文本直接回退，不再做子片段重试")
        fallback_words, fallback_mode, fallback_meta = _build_timestamp_fallback(
            chunk_result.get("text", ""),
            0.0,
            duration,
            audio_path=chunk["path"],
        )
        if fallback_mode == "aligner_vad_fallback":
            chunk_log.append("Alignment 快速回退: 使用 VAD 约束比例时间戳")
            chunk_log.append(
                f"Alignment VAD 回退语音区间: {fallback_meta.get('speech_span_count', 0)}"
            )
        else:
            chunk_log.append("Alignment 快速回退: 使用等比分配时间戳")
            if fallback_meta.get("vad_error"):
                chunk_log.append(f"Alignment VAD 回退异常: {fallback_meta['vad_error']}")
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
        fallback_words, fallback_mode, fallback_meta = _build_timestamp_fallback(
            chunk_result.get("text", ""),
            0.0,
            duration,
            audio_path=chunk["path"],
        )
        if fallback_mode == "aligner_vad_fallback":
            chunk_log.append("Alignment 降级失败: 子片段不足，改用 VAD 约束比例时间戳")
            chunk_log.append(
                f"Alignment VAD 回退语音区间: {fallback_meta.get('speech_span_count', 0)}"
            )
        else:
            chunk_log.append("Alignment 降级失败: 子片段不足，改用等比分配时间戳")
            if fallback_meta.get("vad_error"):
                chunk_log.append(f"Alignment VAD 回退异常: {fallback_meta['vad_error']}")
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

    fallback_words, fallback_mode, fallback_meta = _build_timestamp_fallback(
        chunk_result.get("text", ""),
        0.0,
        duration,
        audio_path=chunk["path"],
    )
    if fallback_mode == "aligner_vad_fallback":
        chunk_log.append("Alignment 降级后仍异常: 改用 VAD 约束比例时间戳")
        chunk_log.append(
            f"Alignment VAD 回退语音区间: {fallback_meta.get('speech_span_count', 0)}"
        )
    else:
        chunk_log.append("Alignment 降级后仍异常: 改用等比分配时间戳")
        if fallback_meta.get("vad_error"):
            chunk_log.append(f"Alignment VAD 回退异常: {fallback_meta['vad_error']}")
    return fallback_words, chunk_log


def _postprocess_segments(segments: list[dict]) -> list[dict]:
    cleaned_segments: list[dict] = []

    for segment in segments:
        text = _clean_segment_text(segment.get("text", ""))
        if not text or _TOOL_SIGNATURE_RE.search(text):
            continue
        text = _remove_context_leak_fragments(text)
        if not text:
            continue
        if _is_context_leak(text):
            continue
        if _is_low_value_text(text):
            continue

        cleaned_segments.append(
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": text,
                "source_chunk_index": segment.get("source_chunk_index"),
                "words": list(segment.get("words") or []),
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

    combined_chars = len(_compact_context_text(current_text + following_text))
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
    compact_len = len(_compact_context_text(text))
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

    left_weight = max(1, len(_compact_context_text(left_text)))
    right_weight = max(1, len(_compact_context_text(right_text)))
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


def _should_split_on_gender(prev_word, next_word, gap, *, min_gap_s=0.15):
    pg, ng = prev_word.get('gender'), next_word.get('gender')
    return pg is not None and ng is not None and pg != ng and gap >= min_gap_s


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
        if text in _NOISE_WORDS:
            return
        if _TRIVIAL_SEGMENT.match(text):
            return
        if text in _GRAY_WORDS and segment_duration > _GRAY_MAX_DURATION_S:
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
        should_split_turn = should_split_turn or _should_split_on_gender(current_words[-1], word, gap)
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
