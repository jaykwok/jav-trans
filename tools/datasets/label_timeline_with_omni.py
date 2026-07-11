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


SCHEMA = "timeline_omni_alignment_label_v1"
PROMPT_VERSION = "timeline_audio_text_alignment_v1"
DEFAULT_AUDIO_CONTENT_MODE = "input_audio"
DEFAULT_MAX_TOKENS = 4096


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): row for row in _read_jsonl(path)}


def build_prompt(forced_row: dict[str, Any]) -> str:
    units = [
        {"unit_id": unit["unit_id"], "text": unit["text"]}
        for unit in forced_row.get("word_units") or []
    ]
    return f"""你是音频与固定文本的时间轴对齐标注器。本次唯一任务是：把给定文本单元对齐到音频时间，不做转录。

硬约束：
- 不要改写、纠错、补充或删除文本。
- 不判断字幕是否保留，不判断切分点，不做内容审查。
- start_s/end_s 使用当前音频片段内的秒数，范围 0 到 {float(forced_row['duration_s']):.3f}。
- 能明确听到并对应时标 matched；无法可靠对应时标 unmatched。
- 每个 unit_id 必须且只能返回一次，顺序保持不变。
- 同一句中的停顿、喘息或拖音不属于任何文本单元，不要强行匹配。

固定文本单元：
{json.dumps(units, ensure_ascii=False)}

只输出 JSON：
{{
  "units": [
    {{"unit_id": "u0000", "status": "matched|unmatched", "start_s": 0.0, "end_s": 0.0, "confidence": 0.0}}
  ],
  "reason": "简短说明整体对齐困难点"
}}
"""


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
    forced_rows = _read_jsonl(Path(args.forced_labels))
    item_index = _by_id(Path(args.items))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "omni_timeline_labels.jsonl"
    raw_path = output_dir / "omni_raw_responses.jsonl"
    existing = set(_by_id(labels_path)) if labels_path.exists() else set()
    pending = [row for row in forced_rows if str(row["item_id"]) not in existing]
    if args.limit > 0:
        pending = pending[: args.limit]
    _model_name, configured_model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or configured_model or "qwen3.5-omni-flash"
    _key_name, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_name, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    processed = 0
    for forced_row in pending:
        item_id = str(forced_row["item_id"])
        item = item_index[item_id]
        expected = list(forced_row.get("word_units") or [])
        try:
            parsed, raw = call_omni(
                audio_path=Path(item["audio_path"]),
                fmt="wav",
                audio_content_mode=args.audio_content_mode,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=args.timeout_s,
                store_stream_chunks=False,
                prompt=build_prompt(forced_row),
                max_tokens=args.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            parsed = {"units": [], "reason": str(exc)}
            raw = {"error": repr(exc)}
            if not is_empty_audio_api_error(exc):
                raise
        units = _normalize_units(
            parsed,
            expected,
            duration_s=float(forced_row["duration_s"]),
        )
        payload = {
            "schema": SCHEMA,
            "item_id": item_id,
            "source_id": forced_row["source_id"],
            "source_chunk_index": forced_row["source_chunk_index"],
            "duration_s": forced_row["duration_s"],
            "transcript": forced_row["transcript"],
            "audio_path": item["audio_path"],
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "units": units,
            "matched_count": sum(unit["status"] == "matched" for unit in units),
            "unmatched_count": sum(unit["status"] != "matched" for unit in units),
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
        if args.rpm > 0:
            time.sleep(60.0 / args.rpm)
    summary = {
        "schema": "timeline_omni_alignment_summary_v1",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "max_tokens": args.max_tokens,
        "processed": processed,
        "total_labels": len(_read_jsonl(labels_path)) if labels_path.exists() else 0,
        "labels": str(labels_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--forced-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--audio-content-mode", default=DEFAULT_AUDIO_CONTENT_MODE)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--rpm", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
