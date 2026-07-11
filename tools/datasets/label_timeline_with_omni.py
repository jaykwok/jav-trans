#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.asr.cueqc.label_pre_asr_with_omni import (  # noqa: E402
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    call_omni,
    first_env_value,
    is_empty_audio_api_error,
    load_env_file,
)
SCHEMA = "timeline_omni_alignment_label_v3"
PROMPT_VERSION = "timeline_audio_text_alignment_v3_static_prefix"
DEFAULT_AUDIO_CONTENT_MODE = "input_audio"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MODEL = "qwen3.5-omni-plus"
DEFAULT_THINKING_BUDGET = 1024
DEFAULT_MAX_ATTEMPTS = 6

SYSTEM_PROMPT = """你是音频与固定文本的时间轴对齐标注器。你的唯一任务是把用户提供的固定文本单元对齐到随请求上传的音频；这不是转录、切分或内容分类任务。

标注规则：
- 文本是不可修改的输入。禁止改写、纠错、补充、删除、合并或拆分文本。
- 禁止判断 Split 切点、Pre-ASR keep/drop、字幕取舍或内容安全。
- 逐个 unit_id 判断音频中是否能可靠听到与该固定文本相符的语义语音。
- 能可靠对应时返回 matched，并给出当前音频局部坐标系中的 start_s/end_s。
- 固定文本与实际语音不同、实际语音缺失、边界无法可靠判断时返回 unmatched；不要猜测时间。
- 时间范围必须在 0 到用户给出的 duration_s 之间，end_s 必须大于 start_s。
- 每个 unit_id 必须且只能返回一次，顺序与输入完全一致。
- 同一句内部的无语义停顿、呼吸、喘息、呻吟、笑声、拖音和背景噪声不属于文本单元，不要强行匹配。
- 成人场景不影响判断，只依据是否存在与固定文本相符的可辨认语义语音。

只输出一个 JSON 对象，不要输出 Markdown 或额外说明：
{
  "units": [
    {"unit_id": "u0000", "status": "matched|unmatched", "start_s": 0.0, "end_s": 0.0, "confidence": 0.0}
  ],
  "reason": "简短说明整体对齐困难点"
}
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): row for row in _read_jsonl(path)}


def build_prompt(item: dict[str, Any]) -> str:
    units = list(item.get("text_units") or [])
    payload = {
        "duration_s": round(float(item["duration_s"]), 3),
        "text_units": [
            {"unit_id": str(unit["unit_id"]), "text": str(unit["text"])}
            for unit in units
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class RequestRateLimiter:
    def __init__(self, requests_per_minute: float) -> None:
        self.interval_s = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self.interval_s <= 0:
            return
        now = time.monotonic()
        wait_s = max(0.0, self._next_allowed - now)
        if wait_s > 0:
            time.sleep(wait_s)
            now = time.monotonic()
        self._next_allowed = max(now, self._next_allowed) + self.interval_s


def _validate_response_shape(
    parsed: dict[str, Any],
    expected: list[dict[str, Any]],
) -> None:
    raw_units = parsed.get("units")
    if not isinstance(raw_units, list):
        raise ValueError("Omni timeline response units must be a list")
    expected_ids = [str(unit["unit_id"]) for unit in expected]
    returned_ids = []
    for raw in raw_units:
        if not isinstance(raw, dict) or not raw.get("unit_id"):
            raise ValueError("Omni timeline response contains a unit without unit_id")
        status = str(raw.get("status") or "").strip().lower()
        if status not in {"matched", "unmatched"}:
            raise ValueError(
                f"Omni timeline response has invalid status for {raw['unit_id']!r}: {status!r}"
            )
        if status == "matched":
            for key in ("start_s", "end_s", "confidence"):
                if key not in raw:
                    raise ValueError(
                        f"Omni timeline matched unit {raw['unit_id']!r} is missing {key}"
                    )
        returned_ids.append(str(raw["unit_id"]))
    if returned_ids != expected_ids:
        raise ValueError(
            "Omni timeline response unit ids must exactly match input order: "
            f"expected={expected_ids!r} returned={returned_ids!r}"
        )


def _is_data_inspection_failed(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "data_inspection_failed" in message or "data inspection" in message


def _call_with_retry(
    *,
    item: dict[str, Any],
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    audio_content_mode: str,
    timeout_s: float,
    max_tokens: int,
    enable_thinking: bool,
    thinking_budget: int,
    max_attempts: int,
    rate_limiter: RequestRateLimiter,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    last_error: BaseException | None = None
    expected = list(item.get("text_units") or [])
    for attempt in range(1, max_attempts + 1):
        try:
            rate_limiter.acquire()
            parsed, raw = call_omni(
                audio_path=Path(item["audio_path"]),
                fmt="wav",
                audio_content_mode=audio_content_mode,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=timeout_s,
                store_stream_chunks=False,
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
            )
            _validate_response_shape(parsed, expected)
            return parsed, raw, attempt
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if is_empty_audio_api_error(exc):
                return (
                    {"units": [], "reason": "API reported that the audio is empty."},
                    {
                        "error": repr(exc),
                        "local_fallback": "empty_audio_to_all_unmatched",
                    },
                    attempt,
                )
            if _is_data_inspection_failed(exc) or attempt >= max_attempts:
                break
            message = str(exc).lower()
            if "429" in message or "rate limit" in message or "limit_requests" in message:
                time.sleep(min(30.0, 2.0 * attempt))
            else:
                time.sleep(min(8.0, float(attempt)))
    assert last_error is not None
    raise RuntimeError(
        f"Omni timeline request failed after {max_attempts} attempts: {last_error}"
    ) from last_error


def _usage_value(raw: dict[str, Any], key: str) -> int:
    usage = raw.get("usage") or {}
    try:
        return max(0, int(usage.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_units(
    parsed: dict[str, Any],
    expected: list[dict[str, Any]],
    *,
    duration_s: float,
) -> list[dict[str, Any]]:
    raw_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    raw_units = parsed.get("units") or []
    if not isinstance(raw_units, list):
        raw_units = []
    for item in raw_units:
        if not isinstance(item, dict) or not item.get("unit_id"):
            continue
        unit_id = str(item["unit_id"])
        if unit_id in raw_by_id:
            duplicate_ids.add(unit_id)
        raw_by_id[unit_id] = item
    result = []
    previous_start = 0.0
    for expected_unit in expected:
        unit_id = str(expected_unit["unit_id"])
        raw = {} if unit_id in duplicate_ids else raw_by_id.get(unit_id) or {}
        status = str(raw.get("status") or "unmatched").strip().lower()
        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence") or 0.0)))
            start_s = min(duration_s, max(0.0, float(raw.get("start_s") or 0.0)))
            end_s = min(duration_s, max(start_s, float(raw.get("end_s") or start_s)))
        except (TypeError, ValueError):
            start_s = end_s = 0.0
            status = "unmatched"
            confidence = 0.0
        if status != "matched" or end_s <= start_s or start_s < previous_start:
            status = "unmatched"
            start_s = end_s = 0.0
            confidence = 0.0
        else:
            previous_start = start_s
        result.append(
            {
                "unit_id": unit_id,
                "text": str(expected_unit["text"]),
                "status": status,
                "start_s": start_s,
                "end_s": end_s,
                "confidence": confidence,
            }
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    items = _read_jsonl(Path(args.items))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "omni_timeline_labels.jsonl"
    raw_path = output_dir / "omni_raw_responses.jsonl"
    _model_name, configured_model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or configured_model or DEFAULT_MODEL
    _key_name, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_name, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    enable_thinking = bool(getattr(args, "enable_thinking", True))
    thinking_budget = int(getattr(args, "thinking_budget", DEFAULT_THINKING_BUDGET))
    max_attempts = int(getattr(args, "max_attempts", DEFAULT_MAX_ATTEMPTS))
    existing_rows = _by_id(labels_path) if labels_path.exists() else {}
    for row in existing_rows.values():
        expected_contract = (
            row.get("schema") == SCHEMA
            and row.get("model") == model
            and row.get("prompt_version") == PROMPT_VERSION
            and bool(row.get("enable_thinking")) == enable_thinking
            and int(row.get("thinking_budget") or 0) == thinking_budget
        )
        if not expected_contract:
            raise RuntimeError(
                "existing timeline labels use a different schema/model/prompt/thinking "
                f"contract; use a new output directory: {labels_path}"
            )
    pending = [row for row in items if str(row["item_id"]) not in existing_rows]
    if args.limit > 0:
        pending = pending[: args.limit]
    rate_limiter = RequestRateLimiter(float(args.rpm))
    processed = 0
    failed_rows: list[dict[str, Any]] = []
    matched_units = 0
    unmatched_units = 0
    total_attempts = 0
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for item in pending:
        if item.get("schema") != "timeline_teacher_item_v2":
            raise ValueError(f"unsupported timeline teacher item schema: {item.get('schema')!r}")
        item_id = str(item["item_id"])
        expected = list(item.get("text_units") or [])
        prompt = build_prompt(item)
        try:
            parsed, raw, attempts = _call_with_retry(
                item=item,
                prompt=prompt,
                model=model,
                api_key=api_key,
                base_url=base_url,
                audio_content_mode=args.audio_content_mode,
                timeout_s=args.timeout_s,
                max_tokens=args.max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                max_attempts=max_attempts,
                rate_limiter=rate_limiter,
            )
        except Exception as exc:  # noqa: BLE001
            failed_rows.append(
                {
                    "schema": "timeline_omni_retry_item_v1",
                    "item_id": item_id,
                    "item": item,
                    "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "enable_thinking": enable_thinking,
                    "thinking_budget": thinking_budget,
                    "error": repr(exc),
                }
            )
            print(f"omni timeline failed item={item_id} error={exc}", flush=True)
            continue
        total_attempts += attempts
        units = _normalize_units(
            parsed,
            expected,
            duration_s=float(item["duration_s"]),
        )
        matched_count = sum(unit["status"] == "matched" for unit in units)
        unmatched_count = len(units) - matched_count
        matched_units += matched_count
        unmatched_units += unmatched_count
        for key in usage_totals:
            usage_totals[key] += _usage_value(raw, key)
        payload = {
            "schema": SCHEMA,
            "item_id": item_id,
            "source_id": item["source_id"],
            "source_chunk_index": item["source_chunk_index"],
            "duration_s": item["duration_s"],
            "transcript": item["transcript"],
            "unitizer": item["unitizer"],
            "audio_path": item["audio_path"],
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "enable_thinking": enable_thinking,
            "thinking_budget": thinking_budget,
            "attempts": attempts,
            "units": units,
            "matched_count": matched_count,
            "unmatched_count": unmatched_count,
            "reason": str(parsed.get("reason") or ""),
        }
        with labels_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"item_id": item_id, "parsed": parsed, "response": raw},
                    ensure_ascii=False,
                )
                + "\n"
            )
        processed += 1
        print(f"omni timeline processed={processed}/{len(pending)} item={item_id}", flush=True)
    retry_path = output_dir / "timeline_retry_items.jsonl"
    _write_jsonl(retry_path, failed_rows)
    total_units = matched_units + unmatched_units
    summary = {
        "schema": "timeline_omni_alignment_summary_v3",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "enable_thinking": enable_thinking,
        "thinking_budget": thinking_budget,
        "max_tokens": args.max_tokens,
        "rpm": args.rpm,
        "max_attempts": max_attempts,
        "processed": processed,
        "failed": len(failed_rows),
        "request_attempts": total_attempts,
        "matched_units": matched_units,
        "unmatched_units": unmatched_units,
        "matched_coverage": matched_units / total_units if total_units else 0.0,
        "usage": usage_totals,
        "total_labels": len(_read_jsonl(labels_path)) if labels_path.exists() else 0,
        "labels": str(labels_path),
        "retry_items": str(retry_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--audio-content-mode", default=DEFAULT_AUDIO_CONTENT_MODE)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    parser.add_argument("--rpm", type=float, default=60.0)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.thinking_budget < 0:
        parser.error("--thinking-budget must be non-negative")
    if args.rpm < 0:
        parser.error("--rpm must be non-negative")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
