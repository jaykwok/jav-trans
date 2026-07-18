#!/usr/bin/env python3
"""Label Runtime v12 provisional chunks independently for CueQC v13."""
from __future__ import annotations

import argparse
import json
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
        return [json.loads(line) for line in handle if line.strip()]


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate_runtime_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        item_id = str(row.get("subisland_id") or "").strip()
        if not item_id or item_id in seen:
            raise ValueError(f"duplicate or missing Runtime v12 subisland_id: {item_id!r}")
        seen.add(item_id)
        if row.get("schema") != "runtime_v12_provisional_subisland_v1":
            raise ValueError("CueQC v13 teacher requires fresh Runtime v12 chunks")
        candidate = row.get("pre_asr_candidate") or {}
        if not ACOUSTIC_BINARY_V12_CONTRACT.matches(
            candidate.get("boundary_contract_id")
        ):
            raise ValueError("CueQC v13 teacher requires the current Boundary contract")
        if candidate.get("schema") != "pre_asr_cueqc_features_v10":
            raise ValueError("CueQC v13 teacher requires the current feature schema")


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
    by_id = {
        str(row.get("item_id") or ""): dict(row)
        for row in rows
        if isinstance(row, dict) and row.get("item_id")
    }
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
    labels_path = output_dir / "labels.jsonl"
    raw_path = output_dir / "raw_responses.jsonl"
    existing = {str(row["subisland_id"]) for row in _rows(labels_path)}
    runtime_rows = _rows(Path(args.runtime_chunks))
    _validate_runtime_rows(runtime_rows)
    rows = [row for row in runtime_rows if str(row["subisland_id"]) not in existing]
    sources = (
        {str(row["sample_id"]): row for row in _rows(Path(args.source_manifest))}
        if args.source_manifest
        else {}
    )
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
            if round(float(row["end_s"]), 6) <= round(float(row["start_s"]), 6):
                responses[item_id] = {
                    "label": "drop",
                    "confidence": 1.0,
                    "flags": ["empty_audio"],
                }
                _append(
                    raw_path,
                    {
                        "schema": "cueqc_v13_empty_audio_local_v1",
                        "item_id": item_id,
                        "local_route": "zero_length_audio_to_drop",
                    },
                )
                continue
            clip = clip_dir / f"{item_id}.{args.audio_format}"
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
                    "source_partition": str(row.get("source_partition") or "train"),
                    "audio": str(row["audio"]),
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
