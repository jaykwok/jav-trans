import importlib
import time
import warnings
from pathlib import Path
from typing import Callable

from whisper import checkpoint as _checkpoint_module
from whisper import chunking as _chunking_module
from whisper import recovery as _recovery_module
from whisper import transcribe as _transcribe_module
from whisper.backends import registry as _registry_module

warnings.filterwarnings("ignore")

_registry_module = importlib.reload(_registry_module)
_chunking_module = importlib.reload(_chunking_module)
_checkpoint_module = importlib.reload(_checkpoint_module)
_transcribe_module = importlib.reload(_transcribe_module)
_recovery_module = importlib.reload(_recovery_module)

ASR_BACKEND = _registry_module.current_asr_backend()
WHISPER_TIMESTAMP_MODE = _checkpoint_module.os.getenv(
    "WHISPER_TIMESTAMP_MODE", "forced"
).strip().lower()
_ASR_WORKER_MODE = _registry_module.current_asr_worker_mode()
_WHISPER_BACKENDS = _registry_module._WHISPER_BACKENDS
_QWEN_BACKENDS = _registry_module._QWEN_BACKENDS
_VALID_ASR_BACKENDS = _registry_module._VALID_ASR_BACKENDS
_VALID_ASR_WORKER_MODES = _registry_module._VALID_ASR_WORKER_MODES

_ASR_CHUNK_ROOT = _chunking_module._ASR_CHUNK_ROOT
_KEEP_ASR_CHUNKS = _chunking_module._KEEP_ASR_CHUNKS
_LAST_VAD_SIGNATURE: dict = _chunking_module._LAST_VAD_SIGNATURE
_ASR_SLIDING_CONTEXT_SEGS = _transcribe_module._ASR_SLIDING_CONTEXT_SEGS
_ASR_CHECKPOINT_ENABLED = _transcribe_module._ASR_CHECKPOINT_ENABLED

get_backend_label = _registry_module.get_backend_label
_create_whisper_backend = _registry_module._create_whisper_backend
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
_can_merge_short_vad_chunks = _chunking_module._can_merge_short_vad_chunks
_write_merged_vad_chunk = _chunking_module._write_merged_vad_chunk
_merge_short_vad_chunks = _chunking_module._merge_short_vad_chunks

ASRWorkerSystemError = _transcribe_module.ASRWorkerSystemError
_strip_punctuation = _transcribe_module._strip_punctuation
_compact_context_text = _transcribe_module._compact_context_text
_context_tokens = _transcribe_module._context_tokens
_is_context_leak = _transcribe_module._is_context_leak
_collapse_repeated_noise = _transcribe_module._collapse_repeated_noise
_is_noise_token = _transcribe_module._is_noise_token
_is_low_value_text = _transcribe_module._is_low_value_text
_clean_segment_text = _transcribe_module._clean_segment_text
_remove_context_leak_fragments = _transcribe_module._remove_context_leak_fragments
_build_timestamp_fallback = _transcribe_module._build_timestamp_fallback
_looks_like_alignment_failure = _transcribe_module._looks_like_alignment_failure
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
_chunk_gender_label = _transcribe_module._chunk_gender_label
_should_reset_sliding_context = _transcribe_module._should_reset_sliding_context
_sliding_context_result_text = _transcribe_module._sliding_context_result_text
_build_initial_prompt_for_chunk = _transcribe_module._build_initial_prompt_for_chunk
_should_skip_alignment_retry = _transcribe_module._should_skip_alignment_retry
_needs_alignment_fallback = _transcribe_module._needs_alignment_fallback
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
_should_split_on_gender = _transcribe_module._should_split_on_gender
_merge_words_to_segments = _transcribe_module._merge_words_to_segments

_run_TRANSCRIPTION_qc = _recovery_module._run_TRANSCRIPTION_qc
_recover_TRANSCRIPTION_results_if_needed = _recovery_module._recover_TRANSCRIPTION_results_if_needed


def _sync_checkpoint_state() -> None:
    _checkpoint_module._ASR_CHUNK_ROOT = _ASR_CHUNK_ROOT
    _checkpoint_module._LAST_VAD_SIGNATURE = _LAST_VAD_SIGNATURE


def _get_whisper_generation_checkpoint_signature() -> str:
    _sync_checkpoint_state()
    return _checkpoint_module._get_whisper_generation_checkpoint_signature(
        sliding_context_segs=_ASR_SLIDING_CONTEXT_SEGS
    )


def _get_asr_checkpoint_path(audio_path: str) -> Path:
    _sync_checkpoint_state()
    return _checkpoint_module._get_asr_checkpoint_path(
        audio_path,
        last_vad_signature=_LAST_VAD_SIGNATURE,
        chunk_root=_ASR_CHUNK_ROOT,
        sliding_context_segs=_ASR_SLIDING_CONTEXT_SEGS,
    )


def _chunk_checkpoint_signature(chunks: list[dict]) -> dict[str, dict[str, float | str]]:
    _sync_checkpoint_state()
    return _checkpoint_module._chunk_checkpoint_signature(
        chunks,
        last_vad_signature=_LAST_VAD_SIGNATURE,
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
        last_vad_signature=_LAST_VAD_SIGNATURE,
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
        last_vad_signature=_LAST_VAD_SIGNATURE,
        checkpoint_enabled=_ASR_CHECKPOINT_ENABLED,
    )


def aggregate_timeout_fragments(job_id: str) -> Path | None:
    _sync_checkpoint_state()
    return _checkpoint_module.aggregate_timeout_fragments(job_id)


def _build_processing_spans(audio_path: str) -> list[tuple[float, float]]:
    global _LAST_VAD_SIGNATURE
    spans = _chunking_module._build_processing_spans(audio_path)
    _LAST_VAD_SIGNATURE = _chunking_module._LAST_VAD_SIGNATURE
    _sync_checkpoint_state()
    return spans


def _record_stage_timing(
    log: list[str],
    timings: dict[str, float],
    key: str,
    label: str,
    elapsed_s: float,
) -> None:
    timings[key] = elapsed_s
    log.append(f"ASR 阶段耗时: {label}={elapsed_s:.2f}s")


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
        chunk_dir, chunk_infos = _extract_wav_chunks(
            audio_path, chunk_spans, on_stage=on_stage
        )
        chunk_infos = _merge_short_vad_chunks(
            chunk_dir,
            chunk_infos,
            on_stage=on_stage,
        )
        split_elapsed = time.perf_counter() - split_started
        log.append(f"切分完成：共 {len(chunk_infos)} 个处理块")
        _record_stage_timing(log, timings, "split_s", "静音分析与切块", split_elapsed)

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
                backend,
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

            text_results, recovery_timings = _recover_TRANSCRIPTION_results_if_needed(
                backend,
                chunk_infos,
                text_results,
                qc_report,
                log,
                on_stage=on_stage,
            )
            if recovery_timings["asr_recovery_s"] > 0:
                _record_stage_timing(
                    log,
                    timings,
                    "asr_recovery_s",
                    "ASR局部修复",
                    recovery_timings["asr_recovery_s"],
                )
                timings["asr_recovered_chunks"] = recovery_timings[
                    "asr_recovered_chunks"
                ]
                for timing_key, timing_label in (
                    ("asr_recovery_separator_s", "ASR人声分离"),
                    ("asr_recovery_retranscribe_s", "ASR修复重转写"),
                    ("asr_recovery_model_unload_s", "ASR修复模型卸载"),
                ):
                    if recovery_timings.get(timing_key, 0.0) > 0:
                        _record_stage_timing(
                            log,
                            timings,
                            timing_key,
                            timing_label,
                            recovery_timings[timing_key],
                        )

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

            for idx, chunk, (chunk_result, chunk_log) in zip(
                range(1, len(chunk_infos) + 1),
                chunk_infos,
                prepared_results,
            ):
                chunk_words, chunk_log = _finalize_aligned_chunk_without_asr_retry(
                    chunk,
                    chunk_result,
                    list(chunk_log),
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
                            "Alignment VAD 回退",
                            "AnimeWhisper 对齐模式",
                            "AnimeWhisper VAD 回退",
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
