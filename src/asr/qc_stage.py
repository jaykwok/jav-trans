import time
from typing import Callable

from asr.backends.base import BaseAsrBackend
from asr.qc import (
    asr_qc_enabled,
    evaluate_asr_text_results_qc,
    format_qc_log_items,
)
from asr.transcribe import (
    _is_context_leak,
    _is_low_value_text,
)


def _run_TRANSCRIPTION_qc(
    backend: BaseAsrBackend,
    chunks: list[dict],
    text_results: list[dict],
    log: list[str],
    on_stage: Callable[[str], None] | None = None,
) -> tuple[dict, dict[str, float]]:
    if not asr_qc_enabled():
        return {
            "enabled": False,
            "chunk_count": len(chunks),
            "reject_count": 0,
            "warning_count": 0,
            "generation_error_count": 0,
            "generation_overflow_count": 0,
            "timeout_count": 0,
            "quarantined_count": 0,
            "empty_text_for_speech_count": 0,
            "items": [],
            "rejected_indices": [],
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
        "ASR QC: rejected={reject}/{total}, warning={warning}".format(
            reject=qc_report["reject_count"],
            total=qc_report["chunk_count"],
            warning=qc_report["warning_count"],
        )
    )
    log.extend(format_qc_log_items(qc_report))

    return qc_report, {"asr_qc_s": qc_elapsed}
