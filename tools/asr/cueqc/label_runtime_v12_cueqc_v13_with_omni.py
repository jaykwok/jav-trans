#!/usr/bin/env python3
"""Label Runtime v12 provisional chunks independently for CueQC v13."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402

from tools.asr.cueqc.label_pre_asr_with_omni import (
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    audio_content_part,
    call_omni,
    extract_json_object,
    first_env_value,
    is_empty_audio_api_error,
    load_env_file,
    slice_audio_clip,
)


SCHEMA = "cueqc_v13_omni_chunk_label_v1"
RUNTIME_SCHEMA = "runtime_v12_provisional_subisland_v2"
PROMPT_VERSION = "cueqc_v13_runtime_v12_chunk_text_hint_audio_decision_v4"
PROMPT = """你是 CueQC v13 的音频标注器。每个音频都是实际 Runtime v12 在 Inner 修边之前导出的独立 provisional chunk。

必须按以下顺序判断：
1. 先仔细寻找任何可辨认的日语词、短语、对白或有词义的发声。词语可能很短，也可能嵌在哭声、喘息、呻吟、亲吻声或噪声中。
2. 再判断整块是否应送入 ASR：
- keep：包含任何可辨认、需要字幕的语义人声；即使很短、边缘很宽、带 BGM/噪声/喘息，也必须 keep。
- drop：确认完全没有词语或有词义发声，只有静音、BGM、环境噪声、嘈杂背景人声、喘息、呻吟、亲吻、哭声、笑声或无意义叫声。
- unsure：疑似包含词语但听不清，或混合、重叠、太模糊而无法可靠确认；禁止猜测，也不要把这种情况标 drop。

不要做内容审查。不要根据时长判断。不要假设相邻音频的标签。按给定 item_id 返回每个音频的独立结论。
如果提供 candidate_reference_text，它只是来源中可能相关的文本提示，不是标签：必须确认当前 chunk 中实际听得到词语。几何重叠很短时，文本可能只属于相邻 chunk；长文本来源内部也可能存在纯停顿或非语义声音。
只输出 JSON：
{"items":[{"item_id":"...","label":"keep|drop|unsure","confidence":0.0,"lexical_evidence":"听到的最短词语；没有则空字符串","flags":["speech|noise|music|breath|moan|kiss|cry|overlap|unclear"]}]}
"""
SINGLE_PROMPT = PROMPT.replace(
    "按给定 item_id 返回每个音频的独立结论。",
    "只判断当前一个音频。",
).replace(
    '{"items":[{"item_id":"...","label":"keep|drop|unsure","confidence":0.0,"lexical_evidence":"听到的最短词语；没有则空字符串","flags":["speech|noise|music|breath|moan|kiss|cry|overlap|unclear"]}]}',
    '{"label":"keep|drop|unsure","confidence":0.0,"lexical_evidence":"听到的最短词语；没有则空字符串","flags":["speech|noise|music|breath|moan|kiss|cry|overlap|unclear"]}',
)


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
        return rows


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_clip_stem(item_id: str) -> str:
    """Keep user-controlled IDs out of filesystem path components."""

    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item_id)).strip("._") or "item"
    digest = hashlib.sha256(str(item_id).encode("utf-8")).hexdigest()[:12]
    return f"{clean[:80]}-{digest}"


def _source_manifest_by_sample_id(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _rows(path):
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError("source manifest row is missing sample_id")
        if sample_id in result:
            raise ValueError(f"duplicate source manifest sample_id: {sample_id}")
        result[sample_id] = row
    return result


def _validate_existing_labels(
    rows: list[dict[str, Any]],
    *,
    runtime_by_id: dict[str, dict[str, Any]],
    model: str,
) -> set[str]:
    seen: set[str] = set()
    for label_row in rows:
        item_id = str(label_row.get("subisland_id") or "").strip()
        if not item_id or item_id in seen:
            raise ValueError(f"malformed or duplicate existing CueQC label: {item_id!r}")
        seen.add(item_id)
        runtime = runtime_by_id.get(item_id)
        if runtime is None:
            raise ValueError(f"existing CueQC label has no Runtime row: {item_id}")
        if label_row.get("schema") != SCHEMA:
            raise ValueError("existing CueQC labels use a stale schema")
        if str(label_row.get("prompt_version") or "") != PROMPT_VERSION:
            raise ValueError("existing CueQC labels use a stale prompt version")
        if str(label_row.get("model") or "") != model:
            raise ValueError("existing CueQC labels use a different Omni model")
        if str(label_row.get("source_id") or "") != str(runtime.get("source_id") or ""):
            raise ValueError(f"existing CueQC label source mismatch: {item_id}")
        if str(label_row.get("source_partition") or "") != str(runtime.get("source_partition") or ""):
            raise ValueError(f"existing CueQC label partition mismatch: {item_id}")
        for key in ("audio", "sample_id", "source_audio_sha256", "source_audio_size"):
            if str(label_row.get(key)) != str(runtime.get(key)):
                raise ValueError(f"existing CueQC label {key} mismatch: {item_id}")
        for key in (
            "semantic_split_weights_sha256",
            "inner_edge_refiner_weights_sha256",
            "boundary_serialization_contract_id",
        ):
            if str(label_row.get(key) or "") != str(runtime.get(key) or ""):
                raise ValueError(
                    f"existing CueQC label {key} mismatch: {item_id}"
                )
        try:
            start = float(label_row["start_s"])
            end = float(label_row["end_s"])
            duration = float(label_row["duration_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"existing CueQC label has invalid coordinates: {item_id}") from exc
        if not all(math.isfinite(value) for value in (start, end, duration)):
            raise ValueError(f"existing CueQC label has non-finite coordinates: {item_id}")
        if not math.isclose(start, float(runtime["start_s"]), abs_tol=1e-6) or not math.isclose(
            end, float(runtime["end_s"]), abs_tol=1e-6
        ):
            raise ValueError(f"existing CueQC label coordinate mismatch: {item_id}")
        if not math.isclose(duration, float(runtime["duration_s"]), abs_tol=1e-6):
            raise ValueError(f"existing CueQC label duration mismatch: {item_id}")
        if str(label_row.get("label") or "").lower() not in {"keep", "drop", "unsure"}:
            raise ValueError(f"existing CueQC label has invalid label: {item_id}")
    return seen


def _validate_runtime_rows(
    rows: list[dict[str, Any]],
    *,
    require_audio_files: bool = False,
) -> None:
    seen: set[str] = set()
    source_partitions: dict[str, set[str]] = {}
    core_owners: dict[str, str] = {}
    upstream_shas: dict[str, set[str]] = {
        "semantic_split_weights_sha256": set(),
        "inner_edge_refiner_weights_sha256": set(),
    }
    for row in rows:
        item_id = str(row.get("subisland_id") or "").strip()
        if not item_id or item_id in seen:
            raise ValueError(f"duplicate or missing Runtime v12 subisland_id: {item_id!r}")
        seen.add(item_id)
        if row.get("schema") != RUNTIME_SCHEMA:
            raise ValueError("CueQC v13 teacher requires fresh Runtime v12 chunks")
        if row.get("inner_execution_status") != "deferred_until_cueqc_keep":
            raise ValueError(
                "CueQC v13 teacher requires provisional chunks before Inner inference"
            )
        source_id = str(row.get("source_id") or "").strip()
        sample_id = str(row.get("sample_id") or "").strip()
        partition = str(row.get("source_partition") or "").strip()
        if not source_id or not sample_id or partition not in {"train", "val", "test"}:
            raise ValueError(
                "CueQC v13 teacher requires frozen source_id and source partition"
            )
        if row.get("training_manifest_allowed") is not True:
            raise ValueError("CueQC v13 teacher requires an approved runtime manifest")
        source_partitions.setdefault(source_id, set()).add(partition)
        if str(row.get("boundary_serialization_contract_id") or "") != ACOUSTIC_BINARY_V12_CONTRACT.contract_id:
            raise ValueError("CueQC v13 teacher requires the current Boundary serialization contract")
        for key in ("audio", "source_audio_sha256"):
            if not str(row.get(key) or "").strip():
                raise ValueError(f"CueQC v13 runtime row is missing {key}")
        try:
            start = float(row["start_s"])
            end = float(row["end_s"])
            duration = float(row["duration_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"CueQC v13 runtime row has invalid coordinates: {item_id}") from exc
        if not all(math.isfinite(value) for value in (start, end, duration)) or end <= start:
            raise ValueError(f"CueQC v13 runtime row has invalid coordinates: {item_id}")
        if not math.isclose(duration, end - start, abs_tol=1e-5):
            raise ValueError(f"CueQC v13 runtime row duration mismatch: {item_id}")
        core_ids = row.get("source_core_ids")
        if not isinstance(core_ids, list) or len(core_ids) != len(set(str(value) for value in core_ids)):
            raise ValueError(f"CueQC v13 runtime row has invalid source_core_ids: {item_id}")
        for core_id in core_ids:
            core = str(core_id).strip()
            if not core:
                raise ValueError(f"CueQC v13 runtime row has an empty source_core_id: {item_id}")
            previous = core_owners.get(core)
            if previous is not None and previous != item_id:
                raise ValueError(f"CueQC v13 core is reused by subislands: {core}")
            core_owners[core] = item_id
        if require_audio_files and not Path(str(row["audio"])).is_file():
            raise FileNotFoundError(f"CueQC runtime audio not found for {item_id}: {row['audio']}")
        for key in (
            "semantic_split_weights_sha256",
            "inner_edge_refiner_weights_sha256",
        ):
            value = str(row.get(key) or "").lower()
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"CueQC v13 runtime row is missing exact {key}")
            upstream_shas[key].add(value)
        candidate = row.get("pre_asr_candidate") or {}
        if not ACOUSTIC_BINARY_V12_CONTRACT.matches(
            candidate.get("boundary_contract_id")
        ):
            raise ValueError("CueQC v13 teacher requires the current Boundary contract")
        if candidate.get("schema") != "pre_asr_cueqc_features_v10":
            raise ValueError("CueQC v13 teacher requires the current feature schema")
        candidate_start = candidate.get("start")
        candidate_end = candidate.get("end")
        if candidate_start is not None and candidate_end is not None:
            if not math.isclose(float(candidate_start), start, abs_tol=1e-5) or not math.isclose(
                float(candidate_end), end, abs_tol=1e-5
            ):
                raise ValueError(f"CueQC v13 candidate/runtime coordinate mismatch: {item_id}")
    leaked = [source_id for source_id, values in source_partitions.items() if len(values) != 1]
    if leaked:
        raise ValueError(f"CueQC source identity crosses partitions: {sorted(leaked)[:3]}")
    if any(len(values) != 1 for values in upstream_shas.values()):
        raise ValueError("CueQC v13 runtime rows mix upstream checkpoint identities")


def _normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    return label if label in {"keep", "drop", "unsure"} else "unsure"


def _confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, parsed))


def _batch_call(
    *,
    items: list[tuple[str, Path, str]],
    model: str,
    api_key: str,
    base_url: str,
    audio_content_mode: str,
    audio_format: str,
    timeout_s: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from openai import OpenAI

    content: list[dict[str, Any]] = [{"type": "text", "text": PROMPT}]
    for item_id, audio, reference in items:
        item_text = f"item_id={item_id}"
        if reference:
            item_text += "\n" + reference
        content.append({"type": "text", "text": item_text})
        content.append(
            audio_content_part(audio, fmt=audio_format, mode=audio_content_mode)
        )
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_s}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    stream = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max(192, len(items) * 96),
        messages=[{"role": "user", "content": content}],
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
    )
    text_parts: list[str] = []
    usage = None
    response_model = ""
    for chunk in stream:
        payload = chunk.model_dump(mode="json")
        usage = payload.get("usage") or usage
        response_model = str(payload.get("model") or response_model)
        choices = getattr(chunk, "choices", None) or []
        if choices:
            text_parts.append(str(getattr(choices[0].delta, "content", None) or ""))
    content_text = "".join(text_parts)
    parsed = extract_json_object(content_text)
    rows = parsed.get("items")
    if not isinstance(rows, list):
        raise ValueError("batch Omni response missing items array")
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("item_id"):
            continue
        item_id = str(row["item_id"])
        if item_id in by_id:
            duplicate_ids.add(item_id)
        else:
            by_id[item_id] = dict(row)
    if duplicate_ids:
        raise ValueError(
            "batch Omni response contains duplicate item_id values: "
            f"{sorted(duplicate_ids)}"
        )
    expected = {item_id for item_id, _audio, _reference in items}
    if set(by_id) != expected:
        raise ValueError(
            f"batch Omni response ids mismatch: expected={sorted(expected)} got={sorted(by_id)}"
        )
    return by_id, {
        "content": content_text,
        "usage": usage,
        "model": response_model,
        "item_ids": sorted(expected),
    }


def _retry(callable_, *, attempts: int = 6):
    for attempt in range(attempts):
        try:
            return callable_()
        except Exception as exc:
            message = str(exc).lower()
            retryable = any(token in message for token in ("429", "503", "timeout", "rate"))
            if not retryable or attempt + 1 == attempts:
                raise
            delay = min(30, 5 * (attempt + 1))
            print(f"omni_retry={attempt + 1}/{attempts - 1} delay_s={delay}", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _multi_audio_unsupported(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "multiple audio inputs are not supported",
            "multiple input_audio",
            "only one audio",
            "at most one audio",
            "single audio only",
            "does not support multiple audio",
        )
    )


def _moderation_rejected(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "data_inspection_failed",
            "inappropriate content",
            "content moderation",
        )
    )


def _reference_context(
    row: dict[str, Any], source: dict[str, Any] | None
) -> str:
    if source is None:
        return ""
    start = float(row["start_s"])
    end = float(row["end_s"])
    references: list[dict[str, Any]] = []
    for core in source.get("core_spans") or []:
        overlap = max(
            0.0,
            min(end, float(core["end_s"])) - max(start, float(core["start_s"])),
        )
        if overlap <= 0.0:
            continue
        references.append(
            {
                "text": str(core.get("text") or ""),
                "chunk_overlap_s": round(overlap, 6),
                "source_core_duration_s": round(
                    float(core["end_s"]) - float(core["start_s"]), 6
                ),
            }
        )
    if not references:
        return "candidate_reference_text=[]"
    return "candidate_reference_text=" + json.dumps(references, ensure_ascii=False)


def run(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    _model_name, configured_model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or configured_model or "qwen3.5-omni-plus"
    _key_name, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_name, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    if not api_key:
        raise ValueError("Omni API key is not configured")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.jsonl"
    raw_path = output_dir / "raw_responses.jsonl"
    runtime_path = Path(args.runtime_chunks).resolve()
    runtime_rows = _rows(runtime_path)
    _validate_runtime_rows(runtime_rows, require_audio_files=True)
    runtime_by_id = {str(row["subisland_id"]): row for row in runtime_rows}
    _model_for_labels = model
    existing = _validate_existing_labels(
        _rows(labels_path), runtime_by_id=runtime_by_id, model=_model_for_labels
    )
    rows = [row for row in runtime_rows if str(row["subisland_id"]) not in existing]
    sources: dict[str, dict[str, Any]] = {}
    source_manifest_path: Path | None = None
    if args.source_manifest:
        source_manifest_path = Path(args.source_manifest).resolve()
        sources = _source_manifest_by_sample_id(source_manifest_path)
        for row in runtime_rows:
            sample_id = str(row["sample_id"])
            source = sources.get(sample_id)
            if source is None:
                raise ValueError(f"source manifest has no Runtime sample_id: {sample_id}")
            for key in ("source_id", "source_partition"):
                if str(source.get(key) or "") and str(source.get(key)) != str(row.get(key) or ""):
                    raise ValueError(f"source manifest {key} mismatch for {sample_id}")
    audio_fingerprints: dict[str, tuple[str, int]] = {}
    for row in runtime_rows:
        audio_path = Path(str(row["audio"])).resolve()
        key = str(audio_path).lower()
        fingerprint = audio_fingerprints.get(key)
        if fingerprint is None:
            if not audio_path.is_file():
                raise FileNotFoundError(f"CueQC runtime audio not found: {audio_path}")
            fingerprint = (_sha256(audio_path), int(audio_path.stat().st_size))
            audio_fingerprints[key] = fingerprint
        expected_sha = str(row.get("source_audio_sha256") or "").lower()
        expected_size = int(row.get("source_audio_size") or -1)
        if fingerprint != (expected_sha, expected_size):
            raise ValueError(f"Runtime source audio changed for {row['subisland_id']}")
    if args.max_items > 0:
        rows = rows[: args.max_items]
    clip_dir = output_dir / "audio_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    batch_mode = args.batch_size > 1
    counts: Counter[str] = Counter()
    for offset in range(0, len(rows), max(1, args.batch_size)):
        batch = rows[offset : offset + max(1, args.batch_size)]
        clips: list[tuple[str, Path, str]] = []
        responses: dict[str, dict[str, Any]] = {}
        for row in batch:
            item_id = str(row["subisland_id"])
            clip = clip_dir / f"{_safe_clip_stem(item_id)}.{args.audio_format}"
            slice_audio_clip(
                source_audio=Path(str(row["audio"])),
                row={
                    "start": float(row["start_s"]),
                    "end": float(row["end_s"]),
                    "duration_s": float(row["duration_s"]),
                },
                output_path=clip,
                fmt=args.audio_format,
                bitrate=args.audio_bitrate,
                sample_rate=16000,
                force=False,
            )
            clips.append(
                (
                    item_id,
                    clip,
                    _reference_context(row, sources.get(str(row["sample_id"]))),
                )
            )

        if batch_mode and len(clips) > 1:
            try:
                batch_responses, raw = _retry(
                    lambda: _batch_call(
                        items=clips,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                        audio_content_mode=args.audio_content_mode,
                        audio_format=args.audio_format,
                        timeout_s=args.timeout_s,
                    )
                )
                responses.update(batch_responses)
                _append(raw_path, {"schema": "cueqc_v13_omni_batch_raw_v1", **raw})
            except Exception as exc:
                disable_batch_mode = _multi_audio_unsupported(exc)
                if disable_batch_mode:
                    batch_mode = False
                _append(
                    raw_path,
                    {
                        "schema": "cueqc_v13_omni_batch_fallback_v1",
                        "item_ids": [item_id for item_id, _clip, _reference in clips],
                        "model": model,
                        "prompt_version": PROMPT_VERSION,
                        "error": str(exc),
                        "fallback": "single_audio_requests",
                        "batch_mode_disabled": disable_batch_mode,
                    },
                )
        for item_id, clip, reference in clips:
            if item_id in responses:
                continue
            try:
                parsed, raw = _retry(
                    lambda clip=clip, reference=reference: call_omni(
                        audio_path=clip,
                        fmt=args.audio_format,
                        audio_content_mode=args.audio_content_mode,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                        timeout_s=args.timeout_s,
                        store_stream_chunks=False,
                        prompt=(SINGLE_PROMPT + "\n" + reference if reference else SINGLE_PROMPT),
                        max_tokens=128,
                    )
                )
            except Exception as exc:
                if _moderation_rejected(exc):
                    parsed = {
                        "label": "unsure",
                        "confidence": 1.0,
                        "flags": ["moderation_rejected"],
                    }
                    raw = {
                        "error": str(exc),
                        "local_route": "moderation_rejection_to_unsure",
                    }
                elif not is_empty_audio_api_error(exc):
                    raise
                else:
                    parsed = {
                        "label": "drop",
                        "confidence": 1.0,
                        "flags": ["empty_audio"],
                    }
                    raw = {
                        "error": str(exc),
                        "local_route": "empty_audio_to_drop",
                    }
            responses[item_id] = parsed
            _append(
                raw_path,
                {"schema": "cueqc_v13_omni_single_raw_v1", "item_id": item_id, **raw},
            )

        for row in batch:
            item_id = str(row["subisland_id"])
            response = responses[item_id]
            label = _normalize_label(response.get("label"))
            counts[label] += 1
            flags = response.get("flags")
            _append(
                labels_path,
                {
                    "schema": SCHEMA,
                    "prompt_version": PROMPT_VERSION,
                    "model": model,
                    "sample_id": str(row["sample_id"]),
                    "subisland_id": item_id,
                    "source_id": str(row["source_id"]),
                    "source_partition": str(row["source_partition"]),
                    "audio": str(row["audio"]),
                    "source_audio_sha256": str(row["source_audio_sha256"]),
                    "source_audio_size": int(row["source_audio_size"]),
                    "semantic_split_weights_sha256": str(
                        row["semantic_split_weights_sha256"]
                    ),
                    "inner_edge_refiner_weights_sha256": str(
                        row["inner_edge_refiner_weights_sha256"]
                    ),
                    "boundary_serialization_contract_id": str(
                        row["boundary_serialization_contract_id"]
                    ),
                    "start_s": float(row["start_s"]),
                    "end_s": float(row["end_s"]),
                    "duration_s": float(row["duration_s"]),
                    "label": label,
                    "confidence": _confidence(response.get("confidence")),
                    "lexical_evidence": str(response.get("lexical_evidence") or ""),
                    "flags": list(flags) if isinstance(flags, list) else [],
                    "label_source": "omni_text_hint_audio_decision_independent_runtime_v12_chunk_v4",
                    "parent_label_inherited": False,
                },
            )
        print(
            f"cueqc_v13_omni={min(offset + len(batch), len(rows))}/{len(rows)} "
            f"batch_mode={batch_mode} counts={dict(counts)}",
            flush=True,
        )
    summary = {
        "schema": "cueqc_v13_omni_label_summary_v1",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "requested_count": len(rows),
        "total_label_count": len(_rows(labels_path)),
        "label_counts": dict(Counter(row["label"] for row in _rows(labels_path))),
        "batch_mode_final": batch_mode,
        "labels": str(labels_path),
        "runtime_chunks": str(runtime_path),
        "runtime_chunks_sha256": _sha256(runtime_path),
        "runtime_schema": RUNTIME_SCHEMA,
        "source_manifest": str(source_manifest_path) if source_manifest_path else "",
        "source_manifest_sha256": (
            _sha256(source_manifest_path) if source_manifest_path else ""
        ),
        "semantic_split_weights_sha256": next(
            iter({str(row["semantic_split_weights_sha256"]) for row in runtime_rows}), ""
        ),
        "inner_edge_refiner_weights_sha256": next(
            iter({str(row["inner_edge_refiner_weights_sha256"]) for row in runtime_rows}), ""
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-chunks", required=True)
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", default="")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--audio-bitrate", default="32k")
    parser.add_argument("--audio-format", choices=("mp3", "wav"), default="mp3")
    parser.add_argument("--audio-content-mode", default="input_audio")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
