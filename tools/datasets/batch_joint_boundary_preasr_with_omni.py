#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.asr.cueqc.label_pre_asr_v10_with_omni import (  # noqa: E402
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    audio_content_part,
    extract_json_object,
    first_env_value,
    is_empty_audio_api_error,
    load_env_file,
    slice_audio_clip,
)
from tools.datasets.label_joint_boundary_preasr_with_omni import (  # noqa: E402
    JOINT_SCHEMA,
    LEGACY_JOINT_PROMPT_VERSION,
    PROMPT_VERSION,
    _build_pre_asr_prompt,
    _collect_completed,
    _collect_response_cache_events,
    _normalize_chunk_decision,
    _read_jsonl,
    _select_chunk_rows,
    _task_units,
    _write_json,
    _write_jsonl,
    is_data_inspection_failed,
)


BATCH_MANIFEST_SCHEMA = "pre_asr_omni_batch_manifest_v1"
BATCH_EXPORT_SCHEMA = "pre_asr_omni_batch_export_v2"
BATCH_IMPORT_SCHEMA = "pre_asr_omni_batch_import_v1"
RAW_SCHEMA = "pre_asr_omni_chunk_raw_v1"
DEFAULT_BATCH_MODEL = "qwen3.5-omni-plus"
SUPPORTED_BATCH_OMNI_MODELS = frozenset({DEFAULT_BATCH_MODEL})
MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_FILE_BYTES = 500 * 1024 * 1024
MAX_BATCH_LINE_BYTES = 6 * 1024 * 1024
DEFAULT_BATCH_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != BATCH_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported batch manifest schema: {payload.get('schema')!r}")
    return payload


def _save_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    _write_json(path, dict(manifest))


def _resolve_batch_model(requested: str) -> str:
    model = requested.strip() or DEFAULT_BATCH_MODEL
    if model not in SUPPORTED_BATCH_OMNI_MODELS:
        supported = ", ".join(sorted(SUPPORTED_BATCH_OMNI_MODELS))
        raise ValueError(
            f"model {model!r} is not supported by Alibaba Cloud Batch omni; "
            f"current supported model: {supported}. Real-time Flash labels must "
            "not be silently submitted as Batch Plus labels."
        )
    return model


def _provider_client(args: argparse.Namespace):
    from openai import OpenAI

    load_env_file(args.env_file)
    _key_name, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY (or another configured Omni API key) is required")
    _url_name, configured_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    base_url = args.base_url.strip() or configured_url or DEFAULT_BATCH_BASE_URL
    return OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout_s)


def _object_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"provider returned unsupported object: {type(value).__name__}")


def _candidate_index(
    *,
    dataset: Path,
    output: Path,
    chunk_limit: int,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(dataset / "source_windows.jsonl"):
        window_id = str(row["window_id"])
        all_chunk_rows = _read_jsonl(Path(row["pre_asr_candidates"]))
        chunk_rows = (
            sorted(all_chunk_rows, key=lambda item: int(item["chunk_index"]))
            if chunk_limit == 0
            else _select_chunk_rows(
                all_chunk_rows,
                limit=chunk_limit,
                seed=f"{window_id}:chunk",
            )
        )
        for position, candidate in enumerate(chunk_rows):
            candidate_id = str(candidate["candidate_id"])
            index[candidate_id] = {
                "window": row,
                "candidate": candidate,
                "prompt_id": f"p{position:03d}",
                "request_audio": (
                    output
                    / "request_audio"
                    / "pre_asr"
                    / f"{candidate_id}.wav"
                ),
            }
    return index


def _ensure_request_audio(item: Mapping[str, Any]) -> Path:
    request_audio = Path(item["request_audio"])
    if not request_audio.exists():
        slice_audio_clip(
            source_audio=Path(str(item["window"]["audio_wav"])),
            row=item["candidate"],
            output_path=request_audio,
            fmt="wav",
            bitrate="",
            sample_rate=16000,
            force=False,
        )
    return request_audio


def _batch_body(
    *,
    model: str,
    prompt: str,
    audio_path: Path,
    audio_content_mode: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "enable_thinking": False,
        "temperature": 0,
        "max_tokens": max(1, int(max_tokens)),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    audio_content_part(
                        audio_path,
                        fmt="wav",
                        mode=audio_content_mode,
                    ),
                ],
            }
        ],
        "modalities": ["text"],
    }


def _request_lines(
    *,
    args: argparse.Namespace,
    index: Mapping[str, Mapping[str, Any]],
    output: Path,
    model: str,
    only_custom_ids: set[str] | None = None,
) -> Iterable[tuple[str, bytes]]:
    for candidate_id, item in sorted(index.items()):
        if only_custom_ids is not None and candidate_id not in only_custom_ids:
            continue
        window_id = str(item["window"]["window_id"])
        raw_path = output / "raw_responses" / "pre_asr" / f"{candidate_id}.json"
        if raw_path.exists() and not args.include_existing_raw:
            continue
        if (
            (output / "joint_labels" / f"{window_id}.json").exists()
            and not args.include_completed_windows
        ):
            continue
        request_audio = _ensure_request_audio(item)
        prompt = _build_pre_asr_prompt(
            item["candidate"],
            item_id=str(item["prompt_id"]),
        )
        line = {
            "custom_id": candidate_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": _batch_body(
                model=model,
                prompt=prompt,
                audio_path=request_audio,
                audio_content_mode=args.audio_content_mode,
                max_tokens=args.max_tokens,
            ),
        }
        encoded = _json_bytes(line)
        if len(candidate_id) > 256:
            raise ValueError(f"custom_id exceeds 256 characters: {candidate_id!r}")
        if len(encoded) > args.max_line_bytes:
            raise ValueError(
                f"request {candidate_id!r} is {len(encoded)} bytes; "
                f"Batch line limit is {args.max_line_bytes} bytes"
            )
        yield candidate_id, encoded


def _read_custom_ids(paths: Iterable[str]) -> set[str] | None:
    custom_ids: set[str] = set()
    used = False
    for raw_path in paths:
        used = True
        for row in _read_jsonl(Path(raw_path)):
            custom_id = str(row.get("custom_id") or "")
            if custom_id:
                custom_ids.add(custom_id)
    return custom_ids if used else None


def prepare_batch(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    dataset = Path(args.dataset_dir)
    output = Path(args.output_dir)
    batch_dir = Path(args.batch_dir)
    manifest_path = Path(args.manifest)
    model = _resolve_batch_model(args.model)
    index = _candidate_index(
        dataset=dataset,
        output=output,
        chunk_limit=args.max_runtime_chunks,
    )
    only_custom_ids = _read_custom_ids(args.only_custom_ids_from)
    batch_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    current_handle = None
    current_path: Path | None = None
    current_count = 0
    current_bytes = 0
    exported_ids: set[str] = set()

    def close_shard() -> None:
        nonlocal current_handle, current_path, current_count, current_bytes
        if current_handle is None or current_path is None:
            return
        current_handle.close()
        shards.append(
            {
                "index": len(shards),
                "input_path": str(current_path.resolve()),
                "request_count": current_count,
                "byte_count": current_bytes,
                "sha256": _sha256(current_path),
                "status": "prepared",
                "input_file_id": "",
                "batch_id": "",
                "output_file_id": "",
                "error_file_id": "",
                "output_path": "",
                "error_path": "",
            }
        )
        current_handle = None
        current_path = None
        current_count = 0
        current_bytes = 0

    try:
        for candidate_id, encoded in _request_lines(
            args=args,
            index=index,
            output=output,
            model=model,
            only_custom_ids=only_custom_ids,
        ):
            if candidate_id in exported_ids:
                raise ValueError(f"duplicate custom_id: {candidate_id!r}")
            needs_new_shard = current_handle is None or (
                current_count > 0
                and (
                    current_count >= args.max_requests_per_file
                    or current_bytes + len(encoded) > args.max_file_bytes
                )
            )
            if needs_new_shard:
                close_shard()
                current_path = batch_dir / f"requests-{len(shards):05d}.jsonl"
                current_handle = current_path.open("wb")
            current_handle.write(encoded)
            current_count += 1
            current_bytes += len(encoded)
            exported_ids.add(candidate_id)
    finally:
        close_shard()

    manifest = {
        "schema": BATCH_MANIFEST_SCHEMA,
        "export_schema": BATCH_EXPORT_SCHEMA,
        "provider": "aliyun_bailian_openai_batch",
        "dataset_dir": str(dataset.resolve()),
        "output_dir": str(output.resolve()),
        "batch_dir": str(batch_dir.resolve()),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "enable_thinking": False,
        "audio_content_mode": args.audio_content_mode,
        "max_tokens": args.max_tokens,
        "candidate_count": len(index),
        "exported_count": len(exported_ids),
        "request_unit": "pre_asr_chunk",
        "limits": {
            "max_requests_per_file": args.max_requests_per_file,
            "max_file_bytes": args.max_file_bytes,
            "max_line_bytes": args.max_line_bytes,
        },
        "shards": shards,
    }
    _save_manifest(manifest_path, manifest)


def submit_batch(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    client = _provider_client(args)
    for shard in manifest["shards"]:
        if shard.get("batch_id"):
            continue
        input_path = Path(shard["input_path"])
        if _sha256(input_path) != shard["sha256"]:
            raise RuntimeError(f"prepared shard changed after validation: {input_path}")
        with input_path.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        input_file_id = str(uploaded.id)
        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=args.completion_window,
            metadata={
                "ds_name": f"{args.task_name}-s{int(shard['index']):05d}",
                "ds_description": "Pre-ASR CueQC one-chunk one-label Batch job",
                "manifest_sha256": _sha256(manifest_path),
                "prompt_version": str(manifest["prompt_version"]),
            },
        )
        payload = _object_dict(batch)
        shard.update(
            {
                "input_file_id": input_file_id,
                "batch_id": str(payload["id"]),
                "status": str(payload.get("status") or "submitted"),
                "provider": payload,
            }
        )
        _save_manifest(manifest_path, manifest)


def refresh_status(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    client = _provider_client(args)
    for shard in manifest["shards"]:
        batch_id = str(shard.get("batch_id") or "")
        if not batch_id:
            continue
        payload = _object_dict(client.batches.retrieve(batch_id))
        shard.update(
            {
                "status": str(payload.get("status") or "unknown"),
                "output_file_id": str(payload.get("output_file_id") or ""),
                "error_file_id": str(payload.get("error_file_id") or ""),
                "request_counts": payload.get("request_counts"),
                "provider": payload,
            }
        )
    _save_manifest(manifest_path, manifest)
    return manifest


def status_batch(args: argparse.Namespace) -> None:
    manifest = refresh_status(args)
    counts = Counter(str(shard.get("status") or "unknown") for shard in manifest["shards"])
    print(json.dumps({"shard_count": len(manifest["shards"]), "statuses": counts}, ensure_ascii=False))


def _download_file(client: Any, file_id: str, path: Path) -> None:
    response = client.files.content(file_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(response, "write_to_file"):
        response.write_to_file(path)
        return
    content = getattr(response, "content", None)
    if content is None and hasattr(response, "read"):
        content = response.read()
    if not isinstance(content, bytes):
        raise TypeError(f"unsupported file content response: {type(response).__name__}")
    path.write_bytes(content)


def download_batch(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = refresh_status(args) if args.refresh else _load_manifest(manifest_path)
    client = _provider_client(args)
    batch_dir = Path(manifest["batch_dir"])
    for shard in manifest["shards"]:
        shard_index = int(shard["index"])
        for kind in ("output", "error"):
            file_id = str(shard.get(f"{kind}_file_id") or "")
            existing_path = str(shard.get(f"{kind}_path") or "")
            if not file_id or (existing_path and Path(existing_path).exists()):
                continue
            path = batch_dir / f"{kind}-{shard_index:05d}.jsonl"
            _download_file(client, file_id, path)
            shard[f"{kind}_path"] = str(path.resolve())
        _save_manifest(manifest_path, manifest)


def resume_batch(args: argparse.Namespace) -> None:
    manifest = _load_manifest(Path(args.manifest))
    if any(not shard.get("batch_id") for shard in manifest["shards"]):
        submit_batch(args)
    while True:
        manifest = refresh_status(args)
        statuses = {str(shard.get("status") or "unknown") for shard in manifest["shards"]}
        if statuses <= {"completed", "failed", "expired", "cancelled"}:
            break
        if args.no_wait:
            return
        time.sleep(args.poll_seconds)
    download_batch(argparse.Namespace(**{**vars(args), "refresh": False}))


def _batch_error_text(line: Mapping[str, Any], response: Mapping[str, Any] | None) -> str:
    parts = []
    for value in (line.get("error"), response.get("error") if response else None):
        if value:
            parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    body = response.get("body") if response else None
    if isinstance(body, Mapping) and body.get("error"):
        parts.append(json.dumps(body.get("error"), ensure_ascii=False, sort_keys=True))
    return " ".join(parts)


def _parse_batch_line(line: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    response = line.get("response")
    if not isinstance(response, Mapping):
        response = {}
    status_code = int(response.get("status_code") or line.get("status_code") or 200)
    body = response.get("body") if isinstance(response.get("body"), Mapping) else line.get("body")
    if not isinstance(body, Mapping):
        body = {}
    error_text = _batch_error_text(line, response)
    if status_code >= 400 or error_text:
        error = RuntimeError(error_text or f"batch status_code={status_code}")
        if is_empty_audio_api_error(error):
            parsed = {
                "label": "drop",
                "confidence": 1.0,
                "semantic_speech_detected": False,
                "flags": ["short_fragment"],
                "reason": "Batch API reported the audio is empty; mark this chunk as drop.",
            }
            raw = {
                "stream": False,
                "content": "",
                "error": error_text,
                "local_fallback": "empty_audio_to_definite_drop",
                "batch_line": dict(line),
            }
            return parsed, raw
        if is_data_inspection_failed(error):
            parsed = {
                "label": "unsure",
                "confidence": 1.0,
                "semantic_speech_detected": False,
                "flags": ["api_rejected"],
                "reason": (
                    "Batch API rejected this chunk with data_inspection_failed; "
                    "exclude it from training."
                ),
            }
            raw = {
                "stream": False,
                "content": "",
                "error": error_text,
                "local_fallback": "data_inspection_failed_to_ambiguous_ignore",
                "batch_line": dict(line),
            }
            return parsed, raw
        raise RuntimeError(error_text or f"batch status_code={status_code}")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("batch response body has no choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else {}
    content = str((message or {}).get("content") or choices[0].get("text") or "")
    parsed = extract_json_object(content)
    raw = {
        "stream": False,
        "content": content,
        "ids": [str(body["id"])] if body.get("id") else [],
        "models": [str(body["model"])] if body.get("model") else [],
        "finish_reasons": [
            str(choice.get("finish_reason"))
            for choice in choices
            if isinstance(choice, Mapping) and choice.get("finish_reason")
        ],
        "usage": body.get("usage"),
        "batch_line": dict(line),
    }
    return parsed, raw


def _label_row(
    *,
    output: Path,
    window: Mapping[str, Any],
    candidate: Mapping[str, Any],
    prompt_id: str,
    parsed: Mapping[str, Any],
    raw: Mapping[str, Any],
    model: str,
    keep_confidence: float,
    drop_confidence: float,
) -> dict[str, Any]:
    decision = _normalize_chunk_decision(
        parsed,
        keep_confidence=keep_confidence,
        drop_confidence=drop_confidence,
    )
    window_id = str(window["window_id"])
    segment_label = str(decision["label"])
    segment_path = (
        output
        / "pre_asr"
        / "audio_wav"
        / segment_label
        / f"{window_id}-chunk{int(candidate['chunk_index']):05d}.wav"
    )
    slice_audio_clip(
        source_audio=Path(str(window["audio_wav"])),
        row=candidate,
        output_path=segment_path,
        fmt="wav",
        bitrate="",
        sample_rate=16000,
        force=False,
    )
    request_audio = output / "request_audio" / "pre_asr" / f"{candidate['candidate_id']}.wav"
    label = {
        "schema": "pre_asr_omni_label_v1",
        "sample_id": str(candidate["sample_id"]),
        "candidate_id": str(candidate["candidate_id"]),
        "audio_id": str(candidate["audio_id"]),
        "video_id": str(candidate["video_id"]),
        "window_id": window_id,
        "chunk_index": int(candidate["chunk_index"]),
        "start": float(candidate["start"]),
        "end": float(candidate["end"]),
        "duration_s": float(candidate["duration_s"]),
        "label": segment_label,
        "display_decision": (
            "keep"
            if segment_label == "definite_keep"
            else "drop"
            if segment_label == "definite_drop"
            else "ambiguous_ignore"
        ),
        "training_label_included": segment_label
        in {"definite_keep", "definite_drop"},
        "label_source": f"omni:{model}",
        "omni_label": decision["omni_label"],
        "omni_confidence": decision["confidence"],
        "omni_semantic_speech_detected": decision["semantic_speech_detected"],
        "omni_flags": decision["flags"],
        "omni_reason": str(parsed.get("reason") or ""),
        "prompt_id": prompt_id,
        "prompt_version": PROMPT_VERSION,
        "audio": str(segment_path.resolve()),
        "audio_format": "wav",
        "omni_request_audio": str(request_audio.resolve()),
        "omni_request_audio_format": "wav",
        "omni_request_audio_scope": "chunk",
        "feature_schema": str(candidate.get("feature_schema") or ""),
        "runtime_adapter": str(candidate.get("runtime_adapter") or ""),
    }
    if raw.get("local_fallback"):
        label["local_fallback"] = str(raw.get("local_fallback"))
    return label


def _write_window_if_complete(
    *,
    output: Path,
    item_by_candidate_id: Mapping[str, Mapping[str, Any]],
    window_id: str,
    model: str,
    keep_confidence: float,
    drop_confidence: float,
) -> bool:
    joint_path = output / "joint_labels" / f"{window_id}.json"
    if joint_path.exists():
        return False
    items = [
        item
        for item in item_by_candidate_id.values()
        if str(item["window"]["window_id"]) == window_id
    ]
    if not items:
        return False
    raw_payloads = []
    for item in items:
        candidate_id = str(item["candidate"]["candidate_id"])
        raw_path = output / "raw_responses" / "pre_asr" / f"{candidate_id}.json"
        if not raw_path.exists():
            return False
        raw_payloads.append(json.loads(raw_path.read_text(encoding="utf-8")))
    labels = []
    response_usage = []
    response_finish_reasons = []
    for item, raw_payload in zip(items, raw_payloads, strict=True):
        raw = raw_payload.get("response") or {}
        labels.append(
            _label_row(
                output=output,
                window=item["window"],
                candidate=item["candidate"],
                prompt_id=str(item["prompt_id"]),
                parsed=raw_payload.get("parsed") or {},
                raw=raw,
                model=model,
                keep_confidence=keep_confidence,
                drop_confidence=drop_confidence,
            )
        )
        response_usage.append(
            {
                "request_kind": "pre_asr_chunk",
                "prompt_id": str(item["prompt_id"]),
                "usage": raw.get("usage"),
                "response_cache": raw_payload.get("response_cache"),
            }
        )
        response_finish_reasons.append(
            {
                "request_kind": "pre_asr_chunk",
                "prompt_id": str(item["prompt_id"]),
                "finish_reasons": raw.get("finish_reasons"),
            }
        )
    window = items[0]["window"]
    _write_json(
        joint_path,
        {
            "schema": JOINT_SCHEMA,
            "window_id": window_id,
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "request_mode": "separate_single_task",
            "label_task": "pre_asr",
            "request_dirs": {
                "split": str((output / "requests" / "split").resolve()),
                "pre_asr": str((output / "requests" / "pre_asr").resolve()),
            },
            "audio_mp3_32k": str(Path(window["omni_mp3_32k"]).resolve()),
            "training_audio_wav": str(Path(window["audio_wav"]).resolve()),
            "split_labels": [],
            "pre_asr_labels": labels,
            "missed_boundaries": [],
            "response_usage": response_usage,
            "response_finish_reasons": response_finish_reasons,
        },
    )
    return True


def _write_summary(*, dataset: Path, output: Path, model: str, args: argparse.Namespace) -> None:
    split_rows, pre_asr_rows, missed_rows = _collect_completed(output)
    _write_jsonl(output / "pre_asr_labels.jsonl", pre_asr_rows)
    pre_asr_counts = Counter(str(row["label"]) for row in pre_asr_rows)
    _write_json(
        output / "summary.json",
        {
            "schema": "joint_boundary_preasr_omni_summary_v1",
            "dataset_dir": str(dataset.resolve()),
            "window_count": len(_read_jsonl(dataset / "source_windows.jsonl")),
            "completed_window_count": len(list((output / "joint_labels").glob("*.json"))),
            "attempted_this_run": 0,
            "errors_this_run": len(list((output / "errors").glob("*.json"))),
            "split_label_count": len(split_rows),
            "split_labels": {},
            "missed_boundary_count": len(missed_rows),
            "missed_boundary_window_count": 0,
            "pre_asr_label_count": len(pre_asr_rows),
            "pre_asr_labels": dict(pre_asr_counts),
            "omni_audio_format": "wav",
            "training_audio_format": "wav",
            "sample_rate": 16000,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "legacy_joint_prompt_version": LEGACY_JOINT_PROMPT_VERSION,
            "request_mode": "separate_single_task",
            "label_task": "pre_asr",
            **_task_units("pre_asr"),
            "max_requests_per_minute": 0.0,
            "write_dataset_labels": False,
            "single_request_per_window": False,
            "request_granularity": {
                "split": "one_window_split_task_per_request",
                "pre_asr": "one_chunk_per_request",
                "missed_boundary": "disabled",
            },
            "split_omni_audio_scope": "window",
            "pre_asr_omni_audio_scope": "chunk",
            "semantic_split_dataset": str((dataset / "semantic_split").resolve()),
            "pre_asr_dataset": str((dataset / "pre_asr").resolve()),
            "response_cache": {
                "enabled": True,
                "root": str((output / "response_cache").resolve()),
                "events": _collect_response_cache_events(output),
            },
            "batch_import": {
                "schema": BATCH_IMPORT_SCHEMA,
                "results_jsonl": [str(path.resolve()) for path in args.result_paths],
            },
        },
    )


def _manifest_result_paths(manifest_path: Path) -> list[Path]:
    manifest = _load_manifest(manifest_path)
    paths: list[Path] = []
    for shard in manifest["shards"]:
        for key in ("output_path", "error_path"):
            value = str(shard.get(key) or "")
            if value:
                path = Path(value)
                if path.exists():
                    paths.append(path)
    return paths


def import_batch(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    dataset = Path(args.dataset_dir)
    output = Path(args.output_dir)
    model = _resolve_batch_model(args.model)
    args.result_paths = (
        _manifest_result_paths(Path(args.manifest))
        if args.manifest
        else [Path(path) for path in args.results_jsonl]
    )
    if not args.result_paths:
        raise RuntimeError("no downloaded Batch output/error JSONL files were found")
    index = _candidate_index(
        dataset=dataset,
        output=output,
        chunk_limit=args.max_runtime_chunks,
    )
    imported = 0
    skipped_existing_raw = 0
    errors = []
    retry_rows = []
    touched_windows: set[str] = set()
    for result_path in args.result_paths:
        for line_number, line in enumerate(_read_jsonl(result_path), start=1):
            candidate_id = str(line.get("custom_id") or "")
            item = index.get(candidate_id)
            if item is None:
                errors.append({"file": str(result_path), "line": line_number, "custom_id": candidate_id, "error": "unknown_custom_id"})
                continue
            raw_path = output / "raw_responses" / "pre_asr" / f"{candidate_id}.json"
            if raw_path.exists() and not args.overwrite_raw:
                skipped_existing_raw += 1
                continue
            try:
                parsed, raw = _parse_batch_line(line)
            except Exception as exc:  # noqa: BLE001
                error = {
                    "file": str(result_path),
                    "line": line_number,
                    "custom_id": candidate_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                errors.append(error)
                retry_rows.append({**error, "batch_line": line})
                continue
            _write_json(
                raw_path,
                {
                    "schema": RAW_SCHEMA,
                    "window_id": str(item["window"]["window_id"]),
                    "candidate_id": candidate_id,
                    "prompt_id": str(item["prompt_id"]),
                    "parsed": parsed,
                    "response": raw,
                    "response_cache": {"event": "batch_import"},
                },
            )
            imported += 1
            touched_windows.add(str(item["window"]["window_id"]))
    completed_windows = 0
    for window_id in sorted(touched_windows):
        completed_windows += int(
            _write_window_if_complete(
                output=output,
                item_by_candidate_id=index,
                window_id=window_id,
                model=model,
                keep_confidence=args.keep_confidence,
                drop_confidence=args.drop_confidence,
            )
        )
    _write_json(
        Path(args.summary_json),
        {
            "schema": BATCH_IMPORT_SCHEMA,
            "dataset_dir": str(dataset.resolve()),
            "output_dir": str(output.resolve()),
            "results_jsonl": [str(path.resolve()) for path in args.result_paths],
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "imported_raw_count": imported,
            "skipped_existing_raw": skipped_existing_raw,
            "completed_window_count": completed_windows,
            "error_count": len(errors),
            "errors": errors[:50],
            "retry_custom_ids_jsonl": str((output / "batch_retry_custom_ids.jsonl").resolve()),
        },
    )
    _write_jsonl(output / "batch_retry_custom_ids.jsonl", retry_rows)
    _write_summary(dataset=dataset, output=output, model=model, args=args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable Pre-ASR Omni labels through Alibaba Cloud OpenAI-compatible Batch."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    dataset_common = argparse.ArgumentParser(add_help=False)
    dataset_common.add_argument("--dataset-dir", required=True)
    dataset_common.add_argument("--output-dir", required=True)
    dataset_common.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    dataset_common.add_argument("--model", default=DEFAULT_BATCH_MODEL)
    dataset_common.add_argument(
        "--max-runtime-chunks",
        type=int,
        default=0,
        help="Maximum chunks per source window; 0 labels every final chunk (default).",
    )
    provider_common = argparse.ArgumentParser(add_help=False)
    provider_common.add_argument("--manifest", required=True)
    provider_common.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    provider_common.add_argument("--base-url", default="")
    provider_common.add_argument("--timeout-s", type=float, default=120.0)

    prepare = subparsers.add_parser("prepare", parents=[dataset_common])
    prepare.add_argument("--batch-dir", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--audio-content-mode", default=os.getenv("OMNI_AUDIO_CONTENT_MODE", "input_audio"))
    prepare.add_argument("--max-tokens", type=int, default=256)
    prepare.add_argument("--max-requests-per-file", type=int, default=MAX_BATCH_REQUESTS)
    prepare.add_argument("--max-file-bytes", type=int, default=MAX_BATCH_FILE_BYTES)
    prepare.add_argument("--max-line-bytes", type=int, default=MAX_BATCH_LINE_BYTES)
    prepare.add_argument("--only-custom-ids-from", action="append", default=[])
    prepare.add_argument("--include-existing-raw", action="store_true")
    prepare.add_argument("--include-completed-windows", action="store_true")
    prepare.set_defaults(func=prepare_batch)

    submit = subparsers.add_parser("submit", parents=[provider_common])
    submit.add_argument("--task-name", required=True)
    submit.add_argument("--completion-window", default="24h")
    submit.set_defaults(func=submit_batch)

    status = subparsers.add_parser("status", parents=[provider_common])
    status.set_defaults(func=status_batch)

    download = subparsers.add_parser("download", parents=[provider_common])
    download.add_argument("--refresh", action=argparse.BooleanOptionalAction, default=True)
    download.set_defaults(func=download_batch)

    resume = subparsers.add_parser("resume", parents=[provider_common])
    resume.add_argument("--task-name", required=True)
    resume.add_argument("--completion-window", default="24h")
    resume.add_argument("--poll-seconds", type=float, default=60.0)
    resume.add_argument("--no-wait", action="store_true")
    resume.set_defaults(func=resume_batch)

    ingest = subparsers.add_parser("import", parents=[dataset_common])
    ingest.add_argument("--manifest", default="")
    ingest.add_argument("--results-jsonl", action="append", default=[])
    ingest.add_argument("--summary-json", required=True)
    ingest.add_argument("--keep-confidence", type=float, default=0.80)
    ingest.add_argument("--drop-confidence", type=float, default=0.90)
    ingest.add_argument("--overwrite-raw", action="store_true")
    ingest.set_defaults(func=import_batch)
    args = parser.parse_args(argv)
    if hasattr(args, "max_runtime_chunks") and args.max_runtime_chunks < 0:
        parser.error("--max-runtime-chunks must be non-negative")
    for name in ("max_requests_per_file", "max_file_bytes", "max_line_bytes"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "poll_seconds") and args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.command == "import" and not args.manifest and not args.results_jsonl:
        parser.error("import requires --manifest or at least one --results-jsonl")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
