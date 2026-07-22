#!/usr/bin/env python3
"""Probe exact Scorer prediction islands with the current 1.7B ASR.

The probe is diagnostic-only. It never merges adjacent argmax islands, adds
context, or changes canonical labels. Human review remains mandatory.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import re
import sys
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


FRAME_HOP_S = 0.02
MODEL_ID = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
PROBE_INPUT_SCHEMA = "speech_scorer_v10_prediction_span_asr_probe_input_v1"
PROBE_RESULT_SCHEMA = "speech_scorer_v10_prediction_span_asr_probe_result_v1"
SUMMARY_SCHEMA = "speech_scorer_v10_prediction_span_asr_probe_summary_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_audio(value: str, *, manifest_path: Path) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [manifest_path.parent / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Scorer ASR probe audio is missing: {value}")


def _frame(span: Mapping[str, Any], side: str) -> int:
    key = f"{side}_frame"
    if key in span:
        return int(span[key])
    return round(float(span[f"{side}_s"]) / FRAME_HOP_S)


def _crop_wav(
    *, source: Path, destination: Path, start_frame: int, end_frame: int
) -> dict[str, Any]:
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("Scorer ASR probe span must be nonempty and ordered")
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        if params.comptype != "NONE":
            raise ValueError("Scorer ASR probe requires uncompressed PCM WAV")
        if params.framerate != 16000 or params.nchannels != 1:
            raise ValueError("Scorer ASR probe requires 16kHz mono WAV")
        start_sample = round(start_frame * FRAME_HOP_S * params.framerate)
        requested_end_sample = round(end_frame * FRAME_HOP_S * params.framerate)
        if start_sample >= params.nframes:
            raise ValueError("Scorer ASR probe span starts beyond source WAV")
        end_sample = min(requested_end_sample, params.nframes)
        reader.setpos(start_sample)
        payload = reader.readframes(end_sample - start_sample)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(payload)
    with wave.open(str(destination), "rb") as check:
        if check.getnframes() != end_sample - start_sample:
            raise ValueError("Scorer ASR probe crop sample count changed")
    return {
        "sample_rate": params.framerate,
        "sample_width": params.sampwidth,
        "channel_count": params.nchannels,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "requested_end_sample": requested_end_sample,
        "clamped_to_source_end": end_sample != requested_end_sample,
        "sample_count": end_sample - start_sample,
        "duration_s": (end_sample - start_sample) / params.framerate,
    }


def prepare_probe_inputs(
    *, selection_path: Path, output_dir: Path
) -> dict[str, Any]:
    rows = _rows(selection_path)
    if not rows:
        raise ValueError("Scorer ASR probe selection is empty")
    prepared = copy.deepcopy(rows)
    source_ids: set[str] = set()
    probe_inputs: list[dict[str, Any]] = []
    crop_dir = output_dir / "crops"
    for row_index, row in enumerate(prepared):
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in source_ids:
            raise ValueError("Scorer ASR probe requires unique nonempty source_id")
        source_ids.add(source_id)
        source_audio = _resolve_audio(str(row.get("audio") or ""), manifest_path=selection_path)
        spans = row.get("prediction_spans")
        if not isinstance(spans, list) or not spans:
            raise ValueError(f"Scorer ASR probe row has no prediction spans: {source_id}")
        for span_index, span in enumerate(spans):
            probe_label = str(span.get("label") or "")
            if probe_label not in {"model_speech", "asr_probe_candidate"}:
                raise ValueError(
                    "Scorer ASR probe accepts only model_speech or asr_probe_candidate spans"
                )
            start_frame = _frame(span, "start")
            end_frame = _frame(span, "end")
            span_id = f"{source_id}::{probe_label}::{start_frame}-{end_frame}"
            destination = crop_dir / f"span-{len(probe_inputs):04d}.wav"
            crop = _crop_wav(
                source=source_audio,
                destination=destination,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            span["start_frame"] = start_frame
            span["end_frame"] = end_frame
            span["start_s"] = start_frame * FRAME_HOP_S
            span["end_s"] = end_frame * FRAME_HOP_S
            span["asr_probe_id"] = span_id
            probe_inputs.append(
                {
                    "schema": PROBE_INPUT_SCHEMA,
                    "probe_id": span_id,
                    "source_id": source_id,
                    "row_index": row_index,
                    "prediction_span_index": span_index,
                    "source_audio": str(source_audio),
                    "source_audio_sha256": _sha256(source_audio),
                    "audio": str(destination.resolve()),
                    "audio_sha256": _sha256(destination),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_s": start_frame * FRAME_HOP_S,
                    "end_s": end_frame * FRAME_HOP_S,
                    **crop,
                }
            )
        row["diagnostic_only"] = True
        row["training_manifest_allowed"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = output_dir / "prepared_selection.jsonl"
    inputs_path = output_dir / "probe_inputs.jsonl"
    _write_rows(prepared_path, prepared)
    _write_rows(inputs_path, probe_inputs)
    return {
        "prepared_rows": prepared,
        "probe_inputs": probe_inputs,
        "prepared_selection": prepared_path,
        "probe_inputs_path": inputs_path,
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"[\s　、。！？!?,.・「」『』（）()\[\]【】~〜～…]+", "", text or "")


def attach_probe_results(
    *, prepared_rows: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for raw in results:
        row = dict(raw)
        probe_id = str(row.get("probe_id") or "")
        if row.get("schema") != PROBE_RESULT_SCHEMA or not probe_id or probe_id in by_id:
            raise ValueError("Scorer ASR probe results are invalid or duplicated")
        by_id[probe_id] = row
    enriched = copy.deepcopy(list(prepared_rows))
    expected_ids: set[str] = set()
    for row in enriched:
        evidence: list[dict[str, Any]] = []
        for span in row["prediction_spans"]:
            probe_id = str(span.get("asr_probe_id") or "")
            expected_ids.add(probe_id)
            if probe_id not in by_id:
                raise ValueError(f"Scorer ASR probe result is missing: {probe_id}")
            result = by_id[probe_id]
            span["asr_probe"] = {
                key: result.get(key)
                for key in (
                    "raw_text",
                    "text",
                    "normalized_text",
                    "nonempty_text",
                    "language",
                    "error_kind",
                    "error_detail",
                    "elapsed_s",
                )
            }
            evidence.append(span["asr_probe"])
        row["asr_probe_summary"] = {
            "span_count": len(evidence),
            "nonempty_text_span_count": sum(bool(item["nonempty_text"]) for item in evidence),
            "error_span_count": sum(bool(item["error_kind"]) for item in evidence),
            "texts_in_workflow_order": [str(item["raw_text"] or "") for item in evidence],
            "diagnostic_only": True,
            "automatic_label_change_allowed": False,
        }
    if expected_ids != set(by_id):
        raise ValueError("Scorer ASR probe results contain foreign span identities")
    return enriched


def _stabilize_context_baseline(torch_module: Any) -> dict[str, Any]:
    from pipeline.memory_safety import reset_shared_vram_baseline, shared_vram_snapshot

    device_index = torch_module.cuda.current_device()
    probe = torch_module.zeros(1, device=f"cuda:{device_index}")
    del probe
    torch_module.cuda.synchronize(device_index)
    torch_module.cuda.empty_cache()
    for _ in range(8):
        shared_vram_snapshot(required=True)
        time.sleep(0.25)
        candidate = reset_shared_vram_baseline(required=True)
        if float(candidate.get("shared_vram_raw_mb") or 0.0) > 0.0:
            return candidate
    raise RuntimeError("Windows shared VRAM counter did not stabilize")


def _memory_snapshot(torch_module: Any, *, stage: str) -> dict[str, Any]:
    from pipeline.memory_safety import runtime_memory_snapshot

    device_index = torch_module.cuda.current_device()
    scale = 1024 * 1024
    total_mb = torch_module.cuda.get_device_properties(device_index).total_memory / scale
    free_bytes, _ = torch_module.cuda.mem_get_info(device_index)
    snapshot = {
        "stage": stage,
        "device_index": int(device_index),
        "device_name": torch_module.cuda.get_device_name(device_index),
        "physical_vram_total_mb": round(total_mb, 3),
        "physical_vram_budget_mb": round(total_mb * 0.95, 3),
        "cuda_allocated_mb": round(torch_module.cuda.memory_allocated(device_index) / scale, 3),
        "cuda_reserved_mb": round(torch_module.cuda.memory_reserved(device_index) / scale, 3),
        "cuda_max_allocated_mb": round(torch_module.cuda.max_memory_allocated(device_index) / scale, 3),
        "cuda_max_reserved_mb": round(torch_module.cuda.max_memory_reserved(device_index) / scale, 3),
        "cuda_free_mb": round(free_bytes / scale, 3),
        **runtime_memory_snapshot(require_shared_vram=True),
    }
    if float(snapshot["cuda_max_allocated_mb"]) > float(snapshot["physical_vram_budget_mb"]):
        raise MemoryError("Scorer ASR probe exceeded 0.95 physical VRAM")
    if float(snapshot.get("shared_vram_mb") or 0.0) > 0.0:
        raise MemoryError(f"Scorer ASR probe shared VRAM spill is a soft OOM: {snapshot}")
    if float(snapshot["physical_ram_used_mb"]) > float(snapshot["physical_ram_budget_mb"]):
        raise MemoryError("Scorer ASR probe exceeded 0.95 physical RAM")
    return snapshot


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda" in message and ("out of memory" in message or "memoryerror" in message)


def run_asr_probe(
    *, probe_inputs: Sequence[Mapping[str, Any]], model_path: Path
) -> dict[str, Any]:
    os.environ.update(
        {
            "ASR_MODEL_ID": MODEL_ID,
            "ASR_MODEL_PATH": str(model_path.resolve()),
            "ASR_DTYPE": "bfloat16",
            "ASR_ATTENTION": "sdpa",
            "ASR_BATCH_SIZE": "1",
            "ASR_MAX_NEW_TOKENS": "64",
            "TRANSCRIPTION_TIMEOUT_S": "180",
            "ASR_LANGUAGE": "Japanese",
            "ASR_FORCE_LANGUAGE": "1",
            "ASR_STAGE_WORKER_RAM_RATIO": "0.95",
        }
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Scorer ASR probe requires CUDA; CPU fallback is forbidden")
    from asr.local_backend import LocalAsrBackend
    from pipeline.memory_safety import reset_shared_vram_baseline, runtime_memory_snapshot

    device_index = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(0.95, device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    context_baseline = _stabilize_context_baseline(torch)
    snapshots = [_memory_snapshot(torch, stage="context_baseline")]
    backend: LocalAsrBackend | None = None
    events: list[str] = []
    results: list[dict[str, Any]] = []
    execution_baseline: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        backend = LocalAsrBackend("cuda")
        backend.load(on_stage=events.append)
        warmup = max(probe_inputs, key=lambda row: int(row["sample_count"]))
        backend.transcribe_texts([str(warmup["audio"])], on_stage=events.append)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        execution_baseline = reset_shared_vram_baseline(required=True)
        execution_baseline["stage"] = "execution_baseline"
        snapshots.append({**execution_baseline, **runtime_memory_snapshot(require_shared_vram=True)})
        torch.cuda.reset_peak_memory_stats(device_index)
        for index, probe in enumerate(probe_inputs, start=1):
            item_started = time.perf_counter()
            raw_text = text = language = error_kind = error_detail = ""
            generation: Mapping[str, Any] = {}
            log: Sequence[str] = ()
            try:
                result = backend.transcribe_texts([str(probe["audio"])], on_stage=events.append)[0]
                raw_text = str(result.get("raw_text") or "")
                text = str(result.get("text") or raw_text)
                language = str(result.get("language") or "")
                generation = dict(result.get("asr_generation") or {})
                log = list(result.get("log") or ())
                error_kind = str(generation.get("error_kind") or "")
                error_detail = str(generation.get("error_detail") or "")
            except Exception as exc:
                if _is_cuda_oom(exc):
                    raise
                error_kind = type(exc).__name__
                error_detail = str(exc)
            normalized = _normalize_text(text or raw_text)
            results.append(
                {
                    "schema": PROBE_RESULT_SCHEMA,
                    "probe_id": probe["probe_id"],
                    "source_id": probe["source_id"],
                    "start_frame": probe["start_frame"],
                    "end_frame": probe["end_frame"],
                    "start_s": probe["start_s"],
                    "end_s": probe["end_s"],
                    "duration_s": probe["duration_s"],
                    "audio": probe["audio"],
                    "audio_sha256": probe["audio_sha256"],
                    "raw_text": raw_text,
                    "text": text,
                    "normalized_text": normalized,
                    "nonempty_text": bool(normalized),
                    "language": language,
                    "error_kind": error_kind,
                    "error_detail": error_detail,
                    "asr_generation": generation,
                    "log": list(log),
                    "elapsed_s": round(time.perf_counter() - item_started, 3),
                }
            )
            snapshots.append(_memory_snapshot(torch, stage=f"span_{index:04d}"))
            print(
                f"probed {index}/{len(probe_inputs)} source_id={probe['source_id']} "
                f"duration={float(probe['duration_s']):.3f}s text={raw_text!r}",
                flush=True,
            )
    finally:
        if backend is not None:
            backend.close()
            del backend
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        snapshots.append(_memory_snapshot(torch, stage="post_release"))
    return {
        "results": results,
        "events": events,
        "context_baseline": context_baseline,
        "execution_baseline": execution_baseline,
        "memory_snapshots": snapshots,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


def run(
    *, selection_path: Path, model_path: Path, output_dir: Path
) -> dict[str, Any]:
    prepared = prepare_probe_inputs(selection_path=selection_path, output_dir=output_dir)
    probe_inputs = prepared["probe_inputs"]
    execution = run_asr_probe(probe_inputs=probe_inputs, model_path=model_path)
    results = execution["results"]
    result_path = output_dir / "probe_results.jsonl"
    _write_rows(result_path, results)
    enriched = attach_probe_results(
        prepared_rows=prepared["prepared_rows"], results=results
    )
    enriched_path = output_dir / "enriched_selection.jsonl"
    _write_rows(enriched_path, enriched)
    durations = [float(row["duration_s"]) for row in probe_inputs]
    selection_roles = {
        str(span.get("selection_role") or "")
        for row in prepared["prepared_rows"]
        for span in row.get("prediction_spans") or ()
    }
    probe_scope = (
        "gemini_outside_complement_pending_asr_only"
        if selection_roles == {"gemini_outside_complement_pending_asr"}
        else "scorer_argmax_speech_islands_only"
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "model_id": MODEL_ID,
        "model_path": str(model_path.resolve()),
        "device": "cuda",
        "dtype": "bfloat16",
        "attention": "sdpa",
        "batch_size": 1,
        "effective_max_new_tokens": 64,
        "selection": _display(selection_path),
        "selection_sha256": _sha256(selection_path),
        "source_count": len(enriched),
        "span_count": len(probe_inputs),
        "span_duration_s": {
            "total": sum(durations),
            "minimum": min(durations),
            "maximum": max(durations),
            "under_100ms_count": sum(value < 0.1 for value in durations),
        },
        "nonempty_text_span_count": sum(bool(row["nonempty_text"]) for row in results),
        "empty_text_span_count": sum(not bool(row["nonempty_text"]) for row in results),
        "error_span_count": sum(bool(row["error_kind"]) for row in results),
        "raw_text_counts": dict(Counter(str(row["raw_text"]) for row in results)),
        "probe_inputs": _display(prepared["probe_inputs_path"]),
        "probe_inputs_sha256": _sha256(prepared["probe_inputs_path"]),
        "probe_results": _display(result_path),
        "probe_results_sha256": _sha256(result_path),
        "enriched_selection": _display(enriched_path),
        "enriched_selection_sha256": _sha256(enriched_path),
        "diagnostic_only": True,
        "training_manifest_allowed": False,
        "automatic_label_change_allowed": False,
        "probe_scope": probe_scope,
        "full_source_semantic_recall_measured": False,
        "manual_gate_status": "pending",
        "events": execution["events"],
        "shared_vram_context_baseline": execution["context_baseline"],
        "shared_vram_execution_baseline": execution["execution_baseline"],
        "memory_snapshots": execution["memory_snapshots"],
        "elapsed_s": execution["elapsed_s"],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            run(
                selection_path=Path(args.selection),
                model_path=Path(args.model_path),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
