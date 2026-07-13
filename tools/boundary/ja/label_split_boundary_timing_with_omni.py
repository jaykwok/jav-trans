#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.asr.cueqc.label_pre_asr_with_omni import (  # noqa: E402
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    call_omni,
    first_env_value,
    load_env_file,
)


SCHEMA = "semantic_split_v3_boundary_timing_label_v1"
PROMPT_VERSION = "semantic_split_v3_omni_plus_boundary_timing_preserve_speech_v1"
DEFAULT_MODEL = "qwen3.5-omni-plus"

SYSTEM_PROMPT = """你是日语 ASR 分句边界 Timing 标注器。本次唯一任务是：对一个已经确认存在的语义分句边界，定位不会丢失任何语音音节的安全共享切割时间。

用户会上传完整自然 speech island，并给出一个 coarse_time_s，只用于指出要处理的那一个语义边界附近。不要寻找或报告其它边界，不要重新判断是否应该语义分句，不要转录或修改文本。

请精确定位：
- left_speech_end_s：左侧语义单元最后一个可听音节/尾音真正结束的时间。
- right_speech_start_s：右侧语义单元第一个可听音节真正开始的时间。
- safe_cut_time_s：左右 chunk 共用的唯一切割时间。存在静音间隙时放在间隙中部；间隙极短时优先完整保留左右两侧全部音节。

硬要求：
- 不能把 safe_cut_time_s 放在左侧句尾、拖音、尾辅音或呼吸性尾音尚未结束的位置。
- 不能把右侧句首的首音节留到左 chunk。
- 不因噪声、音乐、喘息或内容类型进行审查。
- 如果左右语音重叠，确实不存在无损共享切点，status 返回 unsure，不要伪造精确值。
- 所有时间均相对完整 island 起点，单位秒。

只输出 JSON，不要 Markdown：
{
  "boundary_id": "island#b000",
  "status": "ok|unsure",
  "left_speech_end_s": 1.23,
  "right_speech_start_s": 1.45,
  "safe_cut_time_s": 1.34,
  "confidence": 0.95,
  "flags": ["short_gap", "overlap", "tail_vowel", "breath"],
  "reason": "简短中文理由"
}
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_boundaries(refined_labels: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    for label in _read_jsonl(refined_labels):
        for index, cut in enumerate(label.get("cuts") or []):
            rows.append(
                {
                    "boundary_id": f"{label['island_id']}#b{index:03d}",
                    "island_id": label["island_id"],
                    "window_id": label["window_id"],
                    "duration_s": float(label["duration_s"]),
                    "coarse_time_s": float(cut.get("teacher_time_s", cut["time_s"])),
                    "projected_time_s": float(cut["projected_candidate_time_s"]),
                    "refined_time_s": float(cut["time_s"]),
                    "refiner_delta_s": float(cut["cut_refiner_delta_s"]),
                }
            )
    rows.sort(key=lambda row: (row["refiner_delta_s"], row["boundary_id"]))
    return rows[:limit]


def _validate(parsed: dict[str, Any], selected: dict[str, Any]) -> None:
    if str(parsed.get("boundary_id") or "") != selected["boundary_id"]:
        raise ValueError("boundary_id does not match request")
    status = str(parsed.get("status") or "")
    if status not in {"ok", "unsure"}:
        raise ValueError("status must be ok or unsure")
    confidence = float(parsed.get("confidence") or 0.0)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if status == "unsure":
        return
    left = float(parsed["left_speech_end_s"])
    right = float(parsed["right_speech_start_s"])
    cut = float(parsed["safe_cut_time_s"])
    duration = float(selected["duration_s"])
    if not 0.0 <= left <= cut <= right <= duration:
        raise ValueError("timing must satisfy 0 <= left_end <= cut <= right_start <= duration")


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "selected_boundaries.jsonl"
    if selected_path.exists() and args.resume_selection:
        selected = _read_jsonl(selected_path)
    else:
        selected = select_boundaries(Path(args.refined_labels), int(args.max_boundaries))
        selected_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
            encoding="utf-8",
        )
    _model_env, configured_model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or configured_model or DEFAULT_MODEL
    _key_env, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_env, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    labels_path = output / "boundary_timing_labels.jsonl"
    raw_path = output / "omni_raw_responses.jsonl"
    existing = {str(row["boundary_id"]) for row in _read_jsonl(labels_path)}
    next_request = 0.0
    failures = []
    for item in selected:
        if item["boundary_id"] in existing:
            continue
        audio_name = str(item["island_id"]).replace("#", "__") + ".wav"
        audio_path = Path(args.request_audio_dir) / audio_name
        prompt = json.dumps(
            {
                "boundary_id": item["boundary_id"],
                "duration_s": round(float(item["duration_s"]), 3),
                "coarse_time_s": round(float(item["coarse_time_s"]), 3),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        last_error: Exception | None = None
        for attempt in range(1, int(args.max_attempts) + 1):
            wait_s = max(0.0, next_request - time.monotonic())
            if wait_s:
                time.sleep(wait_s)
            next_request = time.monotonic() + (60.0 / args.rpm if args.rpm > 0 else 0.0)
            try:
                parsed, raw = call_omni(
                    audio_path=audio_path,
                    fmt="wav",
                    audio_content_mode="input_audio",
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    timeout_s=args.timeout_s,
                    store_stream_chunks=False,
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    max_tokens=args.max_tokens,
                    enable_thinking=True,
                    thinking_budget=args.thinking_budget,
                )
                _validate(parsed, item)
                row = {
                    "schema": SCHEMA,
                    "prompt_version": PROMPT_VERSION,
                    "model": model,
                    "thinking_budget": args.thinking_budget,
                    "attempts": attempt,
                    **item,
                    "status": parsed["status"],
                    "left_speech_end_s": parsed.get("left_speech_end_s"),
                    "right_speech_start_s": parsed.get("right_speech_start_s"),
                    "safe_cut_time_s": parsed.get("safe_cut_time_s"),
                    "confidence": float(parsed.get("confidence") or 0.0),
                    "flags": list(parsed.get("flags") or []),
                    "reason": str(parsed.get("reason") or ""),
                }
                with labels_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                with raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"boundary_id": item["boundary_id"], "parsed": parsed, "response": raw}, ensure_ascii=False) + "\n")
                print(f"timing smoke boundary={item['boundary_id']} status={parsed['status']} cut={parsed.get('safe_cut_time_s')}", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < int(args.max_attempts):
                    time.sleep(min(8.0, float(attempt)))
        else:
            failures.append({"boundary_id": item["boundary_id"], "error": repr(last_error)})
    (output / "retry_boundaries.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8"
    )
    labels = _read_jsonl(labels_path)
    summary = {
        "schema": "semantic_split_v3_boundary_timing_summary_v1",
        "selected_boundaries": len(selected),
        "label_count": len(labels),
        "ok_count": sum(row["status"] == "ok" for row in labels),
        "unsure_count": sum(row["status"] == "unsure" for row in labels),
        "failed_count": len(failures),
        "prompt_version": PROMPT_VERSION,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-labels", required=True)
    parser.add_argument("--request-audio-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-boundaries", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--rpm", type=float, default=60.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--resume-selection", action="store_true")
    args = parser.parse_args()
    if args.max_boundaries <= 0 or args.max_attempts <= 0 or args.rpm < 0:
        parser.error("invalid boundary count, attempts, or RPM")
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
