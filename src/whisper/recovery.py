import os
import time
from pathlib import Path
from typing import Callable

from whisper.backends.base import BaseAsrBackend
from whisper.local_backend import LocalAsrBackend
from whisper.qc import (
    ASR_QC_ENABLED,
    ASR_RECOVERY_ENABLED,
    evaluate_asr_text_results_qc,
    format_qc_log_items,
)
from whisper.transcribe import (
    _is_context_leak,
    _is_low_value_text,
    _transcribe_asr_chunks_text_only,
)


def _run_TRANSCRIPTION_qc(
    backend: BaseAsrBackend,
    chunks: list[dict],
    text_results: list[dict],
    log: list[str],
    on_stage: Callable[[str], None] | None = None,
) -> tuple[dict, dict[str, float]]:
    if not ASR_QC_ENABLED:
        return {
            "enabled": False,
            "chunk_count": len(chunks),
            "recoverable_count": 0,
            "warning_count": 0,
            "items": [],
            "recoverable_indices": [],
        }, {"asr_qc_s": 0.0}

    if on_stage:
        on_stage("ASR 质检分析...")

    qc_started = time.perf_counter()
    qc_report = evaluate_asr_text_results_qc(
        chunks,
        text_results,
        is_low_value_text=_is_low_value_text,
        is_context_leak=_is_context_leak,
        backend=backend,
    )
    qc_elapsed = time.perf_counter() - qc_started

    log.append(
        "[qc] context_leak_check={context} repetition_check={repetition} threshold={threshold}".format(
            context=qc_report.get("context_leak_check", "on"),
            repetition=qc_report.get("repetition_check", "on"),
            threshold=qc_report.get("repetition_threshold", 10),
        )
    )
    log.append(
        "ASR QC: recoverable={recoverable}/{total}, warning={warning}".format(
            recoverable=qc_report["recoverable_count"],
            total=qc_report["chunk_count"],
            warning=qc_report["warning_count"],
        )
    )
    log.extend(format_qc_log_items(qc_report))
    if qc_report["recoverable_count"] and not ASR_RECOVERY_ENABLED:
        log.append(
            "ASR Recovery Path: disabled; QC-only prototype keeps original ASR text"
        )

    return qc_report, {"asr_qc_s": qc_elapsed}


def _recover_TRANSCRIPTION_results_if_needed(
    backend: LocalAsrBackend,
    chunks: list[dict],
    text_results: list[dict],
    qc_report: dict,
    log: list[str],
    on_stage: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict[str, float]]:
    recoverable_indices = list(qc_report.get('recoverable_indices', []))
    if not ASR_RECOVERY_ENABLED or not recoverable_indices:
        return text_results, {'asr_recovery_s': 0.0, 'asr_recovered_chunks': 0.0}

    recovery_started = time.perf_counter()
    if on_stage:
        on_stage(f'ASR Recovery Path 调度 {len(recoverable_indices)} 个异常块...')

    timings: dict[str, float] = {
        'asr_recovery_s': 0.0,
        'asr_recovered_chunks': 0.0,
        'asr_recovery_retranscribe_s': 0.0,
    }

    from vad.ffmpeg_backend import FfmpegSilencedetectBackend
    from audio.vad_refine import refine_chunks_via_vad

    recoverable_chunks = [chunks[i] for i in recoverable_indices]
    sub_chunk_paths: list[str] = []
    try:
        refined = refine_chunks_via_vad(
            recoverable_chunks,
            vad_backend=FfmpegSilencedetectBackend(),
            timeout_per_chunk_s=float(os.getenv('ASR_RECOVERY_TIMEOUT_S', '30')),
        )
        sub_chunk_paths = [
            c['path'] for c in refined
            if c['path'] not in {rc['path'] for rc in recoverable_chunks}
        ]
    except Exception as exc:
        log.append(f'ASR Recovery VAD refine failed: {exc}; kept original ASR text')
        timings['asr_recovery_s'] = time.perf_counter() - recovery_started
        return text_results, timings

    log.append(
        f'ASR Recovery VAD: {len(recoverable_chunks)} chunks -> {len(refined)} sub-chunks'
    )

    retranscribe_started = time.perf_counter()
    try:
        flat_results, text_timings = _transcribe_asr_chunks_text_only(
            backend,
            refined,
            'ASR Recovery 重转写',
            on_stage=on_stage,
        )
        timings['asr_recovery_retranscribe_s'] = text_timings.get('text_transcribe_s', 0.0)
    except Exception as exc:
        log.append(f'ASR Recovery retranscribe failed: {exc}; kept original ASR text')
        timings['asr_recovery_s'] = time.perf_counter() - recovery_started
        return text_results, timings

    if not timings['asr_recovery_retranscribe_s']:
        timings['asr_recovery_retranscribe_s'] = time.perf_counter() - retranscribe_started

    groups: dict[int, list[dict]] = {}
    for sub_chunk, sub_result in zip(refined, flat_results):
        parent = int(sub_chunk.get('_vad_parent_index', sub_chunk.get('index', 0)))
        groups.setdefault(parent, []).append(sub_result)

    updated_results = list(text_results)
    recovered_count = 0
    for position in recoverable_indices:
        group = groups.get(position)
        if not group:
            continue
        original_result = text_results[position]
        merged_text = ''.join(
            r.get('text', '').strip() for r in group if r.get('text', '').strip()
        )
        merged_result = dict(group[0])
        merged_result['text'] = merged_text
        result_log = list(merged_result.get('log', []))
        result_log.append('ASR Recovery VAD: text merged from sub-chunk retranscription')
        merged_result['log'] = result_log
        try:
            qc_reasons = qc_report['items'][position].get('reasons', [])
        except (KeyError, IndexError, TypeError):
            qc_reasons = []
        merged_result['recovery'] = {
            'enabled': True,
            'method': 'vad_refine',
            'sub_chunks': len(group),
            'original_text': original_result.get('text', ''),
            'qc_reasons': qc_reasons,
        }
        updated_results[position] = merged_result
        log.append(
            'ASR Recovery VAD replaced chunk {index}: {before} -> {after}'.format(
                index=chunks[position].get('index', position + 1),
                before=(original_result.get('text', '') or '')[:80],
                after=(merged_text or '')[:80],
            )
        )
        recovered_count += 1

    for sub_path in sub_chunk_paths:
        try:
            Path(sub_path).unlink(missing_ok=True)
        except Exception:
            pass

    timings['asr_recovered_chunks'] = float(recovered_count)
    timings['asr_recovery_s'] = time.perf_counter() - recovery_started
    return updated_results, timings
