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
    slice_audio_clip,
)


SCHEMA = "semantic_split_v3_speech_island_label_v1"
SELECTION_SCHEMA = "semantic_split_v3_speech_island_selection_v1"
PROMPT_VERSION = "semantic_split_v3_omni_plus_speech_island_cut_eager_complete_v3"
DEFAULT_MODEL = "qwen3.5-omni-plus"
TARGET_DURATIONS_S = (2.0, 5.0, 9.0, 14.0, 35.0)

SYSTEM_PROMPT = """你是日语 ASR Semantic Split 数据标注器。本次唯一任务是：在上传的一个完整 speech island 内，找出全部适合拆成短而自然的 ASR/字幕单元的边界。

speech island 的起止已经由上游声学模型确定。你不需要判断 speech/non-speech，也不要转录、改写或总结音频。

目标粒度：
- 不要求左右都是语法上的完整长句。自然短句、分句、独立语块、感叹、称呼、应答和话轮都可以独立成字幕。
- 优先把长 island 拆成较短、单一表达意图的单元，减少整段字幕过早显示。
- 时长不是硬规则；是否切取决于语义与自然停顿，而不是固定秒数。

应输出 cut 的条件：
- 自然停顿前后的紧邻语音都能被听懂，并可分别作为自然 ASR/字幕单元，即使它们只是短句或分句。
- 说完一个表达意图后进入下一个表达、问答之间、独立回应、话轮切换，应切。
- 短暂停顿可以是切点：只要停顿位于可独立显示的语块边界，而不是词法或句法连接内部。

不得输出 cut 的情况：
- 词语内部、名词/主语与助词之间、助词与其支配成分之间、活用或固定搭配连接内部。
- 左侧或右侧只是明显未完成的半个词、助词、助动词，单独送入 ASR 会失去可理解内容。
- 同一自然语块内部仅因呼吸、喘息、呻吟、笑声、拖音、犹豫或重复出现短暂停顿。
- 仅因为静音、噪声、音色变化、情绪变化或 speaker change；仍需确认它同时是自然语块边界。
- island 的 0 秒起点和 duration_s 终点不是内部切点，不要返回。

必须主动搜索并返回 island 内全部自然语块切点，按 time_s 严格升序。不要因为整段属于同一主题或同一句复句就合并成过长单元。没有合适切点时 cuts 才返回空数组。时间单位为相对 island 起点的秒，尽量落在实际停顿或语音边缘中心。

完整性复核（仍然只做语义切分这一项任务）：
- 在形成最终 JSON 前，先按时间从头到尾检查每次表达意图、独立短句、应答和话轮的结束位置。
- 再从尾到头复核一遍相邻语块，确认没有因为后半段较长、切点较密或非语言发声穿插而遗漏自然边界。
- 复核只用于补齐遗漏，不得降低上述句法保护，也不得为了增加数量制造切点。

只输出 JSON，不要 Markdown：
{
  "island_id": "island-id",
  "cuts": [
    {
      "time_s": 1.23,
      "confidence": 0.95,
      "left_complete": true,
      "right_complete": true,
      "boundary_type": "sentence|turn|question_answer|other",
      "reason": "简短中文理由"
    }
  ],
  "complete_search": true,
  "reason": "对整座 island 的一句话说明"
}
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def collect_islands(reexport_dir: Path) -> list[dict[str, Any]]:
    islands = []
    for source in _read_jsonl(reexport_dir / "source_windows.jsonl"):
        audit_rows = _read_jsonl(Path(source["boundary_audit"]))
        chunks = sorted(
            _read_jsonl(Path(source["pre_asr_candidates"])),
            key=lambda row: int(row["chunk_index"]),
        )
        groups: list[list[dict[str, Any]]] = []
        for chunk in chunks:
            if not groups:
                groups.append([chunk])
                continue
            previous = groups[-1][-1]
            same_outer_island = abs(
                float(chunk["raw_start"]) - float(previous["raw_end"])
            ) <= 0.02
            if same_outer_island:
                groups[-1].append(chunk)
            else:
                groups.append([chunk])
        for island_index, group in enumerate(groups):
            start = float(group[0]["acoustic_start"])
            end = float(group[-1]["acoustic_end"])
            duration = end - start
            if duration <= 0.1:
                continue
            candidates = []
            for row in audit_rows:
                time_s = float(row["time_s"])
                if not start < time_s < end:
                    continue
                candidates.append(
                    {
                        "feature_index": int(row.get("frame", -1)),
                        "time_s": time_s,
                        "relative_time_s": time_s - start,
                        "kind": str(row.get("kind") or ""),
                        "accepted": bool(row.get("accepted")),
                        "current_label": str(row.get("label") or ""),
                        "p_cut": float(row.get("p_cut") or 0.0),
                    }
                )
            candidates_by_time = {
                round(float(row["relative_time_s"]), 3): row for row in candidates
            }
            islands.append(
                {
                    "schema": SELECTION_SCHEMA,
                    "island_id": f"{source['window_id']}#outer{island_index:03d}",
                    "window_id": str(source["window_id"]),
                    "video_id": str(source["video_id"]),
                    "span_index": island_index,
                    "span_start_s": start,
                    "span_end_s": end,
                    "duration_s": duration,
                    "source_audio": str(source["audio_wav"]),
                    "candidates": sorted(
                        candidates_by_time.values(),
                        key=lambda row: row["relative_time_s"],
                    ),
                }
            )
    return islands


def select_smoke_islands(
    islands: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    pool = [row for row in islands if row["candidates"]]
    if limit <= 0 or limit >= len(pool):
        return sorted(pool, key=lambda row: (row["duration_s"], row["island_id"]))
    targets = list(TARGET_DURATIONS_S[:limit])
    if limit > len(targets):
        low = min(row["duration_s"] for row in pool)
        high = max(row["duration_s"] for row in pool)
        step = (high - low) / max(1, limit - 1)
        targets = [low + step * index for index in range(limit)]
    selected = []
    used: set[str] = set()
    for target in targets:
        row = min(
            (item for item in pool if item["island_id"] not in used),
            key=lambda item: (
                abs(float(item["duration_s"]) - target),
                -sum(bool(c["accepted"]) for c in item["candidates"]),
                str(item["island_id"]),
            ),
        )
        selected.append(row)
        used.add(str(row["island_id"]))
    return selected


def _validate_response(parsed: dict[str, Any], island: dict[str, Any]) -> None:
    if str(parsed.get("island_id") or "") != island["island_id"]:
        raise ValueError("Omni island_id does not match request")
    if parsed.get("complete_search") is not True:
        raise ValueError("Omni did not confirm complete_search")
    cuts = parsed.get("cuts")
    if not isinstance(cuts, list):
        raise ValueError("cuts must be an array")
    previous = 0.0
    duration = float(island["duration_s"])
    for cut in cuts:
        time_s = float(cut["time_s"])
        confidence = float(cut["confidence"])
        if not previous < time_s < duration:
            raise ValueError("cut times must be strictly ordered inside the island")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("cut confidence must be in [0, 1]")
        previous = time_s


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "selected_islands.jsonl"
    if selected_path.exists() and args.resume_selection:
        selected = _read_jsonl(selected_path)
    else:
        selected = select_smoke_islands(
            collect_islands(Path(args.reexport_dir)), int(args.max_islands)
        )
        selected_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
            encoding="utf-8",
        )
    if args.prepare_only:
        return {"selected_islands": len(selected), "durations_s": [round(row["duration_s"], 3) for row in selected]}

    _model_env, configured_model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or configured_model or DEFAULT_MODEL
    _key_env, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_env, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    labels_path = output / "omni_island_labels.jsonl"
    raw_path = output / "omni_raw_responses.jsonl"
    existing = {str(row["island_id"]) for row in _read_jsonl(labels_path)}
    failures = []
    processed = 0
    next_request = 0.0
    for island in selected:
        island_id = str(island["island_id"])
        if island_id in existing:
            continue
        audio_path = output / "request_audio" / f"{island_id.replace('#', '__')}.wav"
        slice_audio_clip(
            source_audio=Path(island["source_audio"]),
            row={"start": island["span_start_s"], "end": island["span_end_s"]},
            output_path=audio_path,
            fmt="wav",
            bitrate="256k",
            sample_rate=16000,
            force=False,
        )
        prompt = json.dumps(
            {"island_id": island_id, "duration_s": round(island["duration_s"], 3)},
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
                _validate_response(parsed, island)
                row = {
                    "schema": SCHEMA,
                    "prompt_version": PROMPT_VERSION,
                    "model": model,
                    "thinking_budget": args.thinking_budget,
                    "attempts": attempt,
                    **{key: island[key] for key in ("island_id", "window_id", "video_id", "span_index", "span_start_s", "span_end_s", "duration_s")},
                    "cuts": parsed["cuts"],
                    "complete_search": True,
                    "reason": str(parsed.get("reason") or ""),
                }
                _append_jsonl(labels_path, row)
                _append_jsonl(raw_path, {"island_id": island_id, "parsed": parsed, "response": raw})
                processed += 1
                print(f"island smoke processed={processed}/{len(selected)} id={island_id} cuts={len(parsed['cuts'])}", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < int(args.max_attempts):
                    time.sleep(min(8.0, float(attempt)))
        else:
            failures.append({"island_id": island_id, "error": repr(last_error)})
    (output / "retry_islands.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8"
    )
    labels = _read_jsonl(labels_path)
    summary = {
        "schema": "semantic_split_v3_speech_island_summary_v1",
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "selected_islands": len(selected),
        "label_count": len(labels),
        "failed_islands": len(failures),
        "total_cuts": sum(len(row["cuts"]) for row in labels),
        "durations_s": [round(row["duration_s"], 3) for row in selected],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reexport-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-islands", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--rpm", type=float, default=60.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume-selection", action="store_true")
    args = parser.parse_args()
    if args.max_islands <= 0 or args.max_attempts <= 0 or args.rpm < 0:
        parser.error("invalid island count, attempts, or RPM")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
