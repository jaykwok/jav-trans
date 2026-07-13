#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.asr.cueqc.label_pre_asr_with_omni import (  # noqa: E402
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    call_omni,
    first_env_value,
    load_env_file,
)


SCHEMA = "semantic_speech_outer_teacher_v1"
PROMPT_VERSION = "semantic_foreground_speech_spans_no_transcript_v2"
LABELS = ("discardable", "semantic_target", "unsure")

SYSTEM_PROMPT = """你是日语语义前景语音标注器。每次请求只处理一段完整音频，并把整段时间轴连续、无重叠地分成三类：

- semantic_target：清楚可辨、具有语言语义、值得进入字幕的前景人声。短词、句尾 mora 和助词只要可辨也属于 semantic_target。
- discardable：纯背景音乐、环境/机械噪声、纯喘息、呻吟、亲吻声、笑声、无意义叫声或短促非词 vocalization，以及远处/嘈杂/不可辨且无字幕价值的背景人声。
- unsure：疑似包含词语但听不清、与 semantic_target 重叠而无法可靠分离，或无法确定是否有语义内容。

高召回不等于把纯 BGM 或非语言 vocalization 标成语音。只要可能有真实词语但证据不足，必须 unsure，不能硬判 discardable。不要转录、引用或改写任何具体台词；reason 只能描述“清楚前景语言/疑似词语/重叠/非语言 vocalization/BGM”等声学与可辨识性证据。不要判断内部语义切分，不要输出最终 Split timing。

输出严格 JSON：
{"sample_id":"...","segments":[{"start_s":0.0,"end_s":1.2,"label":"discardable|semantic_target|unsure","confidence":0.0,"reason":"简短声学理由"}]}

segments 必须从 0.0 开始、覆盖到用户提供的 duration_s，按时间排序、首尾相接且无重叠。相邻同类应合并。不要输出 Markdown。"""


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate(parsed: dict[str, Any], sample: dict[str, Any]) -> list[dict[str, Any]]:
    if str(parsed.get("sample_id") or "") != str(sample["sample_id"]):
        raise ValueError("sample_id mismatch")
    rows = list(parsed.get("segments") or ())
    if not rows:
        raise ValueError("segments must be non-empty")
    duration = float(sample["duration_s"])
    previous = 0.0
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        start = float(row["start_s"])
        end = float(row["end_s"])
        label = str(row["label"])
        confidence = float(row["confidence"])
        reason = str(row.get("reason") or "")
        if label not in LABELS:
            raise ValueError(f"unknown semantic speech label: {label}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if any(mark in reason for mark in ('"', "“", "”", "「", "」", "『", "』")):
            raise ValueError("reason must not quote or transcribe speech")
        if abs(start - previous) > 0.25:
            raise ValueError(f"segments must be contiguous at index {index}")
        if end <= start:
            raise ValueError("segment end must be after start")
        validated.append(
            {
                "start_s": start,
                "end_s": end,
                "label": label,
                "confidence": confidence,
                "reason": reason,
            }
        )
        previous = end
    if abs(previous - duration) > 0.25:
        raise ValueError("segments must cover duration_s")
    return validated


def run(args: argparse.Namespace) -> None:
    samples = _rows(Path(args.samples))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "semantic_speech_labels.jsonl"
    raw_path = output_dir / "omni_raw_responses.jsonl"
    load_env_file(DEFAULT_ENV_FILE)
    _key_env, configured_api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_env, configured_base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    api_key = args.api_key.strip() or configured_api_key
    base_url = args.base_url.strip() or configured_base_url
    if not api_key:
        raise RuntimeError("Omni API key is required")
    completed: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for sample in samples:
        error: Exception | None = None
        for attempt in range(1, args.max_attempts + 1):
            parsed, raw = call_omni(
                audio_path=Path(sample["audio"]),
                fmt="wav",
                audio_content_mode="input_audio",
                model=args.model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=args.timeout_s,
                store_stream_chunks=False,
                prompt=json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "duration_s": sample["duration_s"],
                    },
                    ensure_ascii=False,
                ),
                system_prompt=SYSTEM_PROMPT,
                max_tokens=2048,
                enable_thinking=True,
                thinking_budget=args.thinking_budget,
            )
            raw_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "attempt": attempt,
                    "response": raw,
                }
            )
            try:
                segments = _validate(parsed, sample)
                completed.append(
                    {
                        "schema": SCHEMA,
                        "prompt_version": PROMPT_VERSION,
                        "model": args.model,
                        "sample_id": sample["sample_id"],
                        "audio": sample["audio"],
                        "duration_s": sample["duration_s"],
                        "source": sample.get("source") or "",
                        "audit_focus": sample.get("audit_focus") or "",
                        "segments": segments,
                    }
                )
                error = None
                break
            except Exception as exc:
                error = exc
        if error is not None:
            raise error
        labels_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in completed),
            encoding="utf-8",
        )
        raw_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_rows),
            encoding="utf-8",
        )
        print(f"semantic_speech_teacher={len(completed)}/{len(samples)}", flush=True)
    summary = {
        "schema": SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "sample_count": len(samples),
        "completed_count": len(completed),
        "labels": list(LABELS),
        "output": str(labels_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label semantic foreground speech spans.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="qwen3.5-omni-plus")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--thinking-budget", type=int, default=2048)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
