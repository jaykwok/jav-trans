#!/usr/bin/env python3
"""Classify multi-safe runs without sample-specific repair specifications."""
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
from tools.boundary.ja.evaluate_semantic_anchor_proposer_audit import (  # noqa: E402
    safe_runs,
)


SCHEMA = "multi_safe_event_repair_teacher_v1"
SELECTION_SCHEMA = "multi_safe_event_repair_selection_v1"
SUMMARY_SCHEMA = "multi_safe_event_repair_summary_v1"
PROMPT_VERSION = "multi_safe_event_scope_text_audio_v1"
DEFAULT_MODEL = "qwen3.5-omni-plus"
DECISIONS = ("semantic_split", "acoustic_continue", "outer_only", "unsure")


SYSTEM_PROMPT = """你是日语 Semantic Split 多安全区修复标注器。每次请求只处理一个已经由人工或独立声学 teacher 确认“波形上不会截字”的候选点。你的任务不是重新判断 safe，也不要输出时间戳；只判断这个 safe 点在通用边界链中的职责。

用户会给出完整可信 reference_text、当前按顺序排列的 semantic text_units、当前 event 的左右 unit id，以及一条在候选点叠加 12ms 短 tick 的完整自适应邻域音频。tick 只标位置，不插入时间。

decision 定义：
- semantic_split：tick 两侧对应两个自然、可独立送入 ASR/显示字幕的最小完整语义单元。返回紧邻边界的 left_text/right_text；二者直接拼接必须是 reference_text 中唯一出现的连续原文，禁止改写。
- acoustic_continue：候选声学上安全，但只是同一最小完整语义单元内部的停顿、呼吸或附属断续，语义应继续。left_text/right_text 必须为空字符串。
- outer_only：候选位于首个语义语音之前、最后语义语音之后，或只把脱离目标语义的呼吸/呻吟/亲吻/BGM/环境声与最近语义内容分开；它属于 Scorer/Outer 的边缘监督，不是内部 Split。left_text/right_text 必须为空字符串。
- unsure：无法可靠判断职责或无法把候选对应到可信文本边界。left_text/right_text 必须为空字符串。

不要因为当前 event 的左右 unit 很大就强迫候选属于它；多 safe run 正可能意味着候选实际位于 unit 内另一个边界或位于非语义外缘。也不要因为存在停顿就自动 semantic_split；左右文本必须各自满足最小完整语义要求。

只输出 JSON，不要 Markdown：
{"candidate_id":"...","decision":"semantic_split|acoustic_continue|outer_only|unsure","left_text":"","right_text":"","confidence":0.0,"reason":"简短理由"}
"""


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _index_unique(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in indexed:
            raise ValueError(f"duplicate {key}: {value}")
        indexed[value] = row
    return indexed


def select_multi_safe_run_representatives(
    *, events: Path, verdicts: Path, timeline_labels: Path
) -> list[dict[str, Any]]:
    event_rows = _index_unique(_rows(events), "event_key")
    timeline_rows = _index_unique(_rows(timeline_labels), "sample_id")
    selected: list[dict[str, Any]] = []
    for verdict in _rows(verdicts):
        event_key = str(verdict["event_key"])
        runs = safe_runs(list(verdict.get("candidates") or []))
        if len(runs) <= 1:
            continue
        if event_key not in event_rows:
            raise ValueError(f"candidate verdict references unknown event: {event_key}")
        event = event_rows[event_key]
        sample_id = str(event["sample_id"])
        if sample_id not in timeline_rows:
            raise ValueError(f"event references unknown timeline sample: {sample_id}")
        timeline = timeline_rows[sample_id]
        reference_text = str(timeline["reference_text"])
        if reference_text != str(event["reference_text"]):
            raise ValueError(f"reference text mismatch for event: {event_key}")
        candidates = {
            str(row["candidate_id"]): row for row in event.get("candidates") or []
        }
        text_units = list(timeline.get("text_units") or [])
        units = _index_unique(text_units, "unit_id")
        left_unit_id = str(event["left_unit_id"])
        right_unit_id = str(event["right_unit_id"])
        if left_unit_id not in units or right_unit_id not in units:
            raise ValueError(f"event unit ids are absent from timeline: {event_key}")
        current_left = str(units[left_unit_id]["text"])
        current_right = str(units[right_unit_id]["text"])
        if current_left != str(event.get("left_text") or "") or current_right != str(
            event.get("right_text") or ""
        ):
            raise ValueError(f"event text does not match timeline units: {event_key}")
        for run_index, run in enumerate(runs):
            representative_truth = max(
                run,
                key=lambda row: float(
                    candidates[str(row["candidate_id"])]["proposer_probability"]
                ),
            )
            candidate_id = str(representative_truth["candidate_id"])
            candidate = candidates[candidate_id]
            selected.append(
                {
                    "schema": SELECTION_SCHEMA,
                    "selection_id": f"{event_key}__run{run_index:02d}",
                    "sample_id": sample_id,
                    "event_key": event_key,
                    "candidate_id": candidate_id,
                    "safe_run_index": run_index,
                    "safe_run_candidate_ids": [
                        str(item["candidate_id"]) for item in run
                    ],
                    "candidate_time_s": float(candidate["time_s"]),
                    "candidate_audio": str(candidate["tick_audio"]),
                    "reference_text": reference_text,
                    "current_left_unit_id": left_unit_id,
                    "current_right_unit_id": right_unit_id,
                    "current_left_text": current_left,
                    "current_right_text": current_right,
                    "current_text_units": text_units,
                    "proposer_probability": float(candidate["proposer_probability"]),
                    "selection_contract": "highest_learned_proposer_probability_per_separated_safe_run_v1",
                }
            )
    return selected


def build_prompt(selection: dict[str, Any], *, validation_feedback: str = "") -> str:
    payload: dict[str, Any] = {
        "candidate_id": str(selection["candidate_id"]),
        "reference_text": str(selection["reference_text"]),
        "current_text_units": selection["current_text_units"],
        "current_event": {
            "left_unit_id": str(selection["current_left_unit_id"]),
            "right_unit_id": str(selection["current_right_unit_id"]),
            "left_text": str(selection["current_left_text"]),
            "right_text": str(selection["current_right_text"]),
        },
        "candidate_contract": "acoustically_safe_tick_marker_without_time_insertion",
        "task": "classify_scope_and_return_exact_text_boundary_only_if_semantic_split",
    }
    if validation_feedback:
        payload["previous_response_validation_error"] = validation_feedback
    return json.dumps(payload, ensure_ascii=False)


def _text_boundary_offset(reference_text: str, left_text: str, right_text: str) -> int:
    joined = left_text + right_text
    starts: list[int] = []
    cursor = 0
    while True:
        index = reference_text.find(joined, cursor)
        if index < 0:
            break
        starts.append(index)
        cursor = index + 1
    if len(starts) != 1:
        raise ValueError("semantic repair text pair must occur exactly once in reference_text")
    return starts[0] + len(left_text)


def validate_response(
    parsed: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any]:
    candidate_id = str(selection["candidate_id"])
    if str(parsed.get("candidate_id") or "") != candidate_id:
        raise ValueError("candidate_id mismatch")
    decision = str(parsed.get("decision") or "")
    if decision not in DECISIONS:
        raise ValueError("invalid repair decision")
    left_text = str(parsed.get("left_text") or "")
    right_text = str(parsed.get("right_text") or "")
    boundary_offset: int | None = None
    if decision == "semantic_split":
        if not left_text or not right_text:
            raise ValueError("semantic_split requires left_text and right_text")
        boundary_offset = _text_boundary_offset(
            str(selection["reference_text"]), left_text, right_text
        )
    elif left_text or right_text:
        raise ValueError("non-semantic repair decisions require empty text sides")
    confidence = float(parsed.get("confidence"))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return {
        "decision": decision,
        "left_text": left_text,
        "right_text": right_text,
        "boundary_text_offset": boundary_offset,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or ""),
    }


def evaluate_against_manual_truth(
    labels: list[dict[str, Any]], manual_truth: Path
) -> dict[str, Any]:
    truth = {
        (str(row["source_event_key"]), str(row["candidate_id"])): row
        for row in _rows(manual_truth)
    }
    comparisons: list[dict[str, Any]] = []
    for label in labels:
        key = (str(label["event_key"]), str(label["candidate_id"]))
        manual = truth[key]
        decision_match = str(label["decision"]) == str(manual["decision"])
        boundary_match = True
        if manual["decision"] == "semantic_split":
            manual_offset = _text_boundary_offset(
                str(label["reference_text"]),
                str(manual["left_text"]),
                str(manual["right_text"]),
            )
            boundary_match = int(label["boundary_text_offset"]) == manual_offset
        comparisons.append(
            {
                "selection_id": str(label["selection_id"]),
                "event_key": key[0],
                "candidate_id": key[1],
                "teacher_decision": str(label["decision"]),
                "manual_decision": str(manual["decision"]),
                "decision_match": decision_match,
                "boundary_match": boundary_match,
                "strict_match": decision_match and boundary_match,
            }
        )
    return {
        "comparison_count": len(comparisons),
        "strict_match_count": sum(row["strict_match"] for row in comparisons),
        "strict_accuracy": sum(row["strict_match"] for row in comparisons)
        / max(1, len(comparisons)),
        "comparisons": comparisons,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_multi_safe_run_representatives(
        events=Path(args.events),
        verdicts=Path(args.candidate_verdicts),
        timeline_labels=Path(args.timeline_labels),
    )
    if args.expected_count and len(selected) != int(args.expected_count):
        raise ValueError(
            f"expected {args.expected_count} multi-safe run representatives, got {len(selected)}"
        )
    _write(output_dir / "selected_repairs.jsonl", selected)
    _model_env, configured_model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or configured_model or DEFAULT_MODEL
    _key_env, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_env, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    if not api_key:
        raise RuntimeError("Omni API key is required")
    labels_path = output_dir / "repair_labels.jsonl"
    raw_path = output_dir / "omni_raw_responses.jsonl"
    existing = {str(row["selection_id"]): row for row in _rows(labels_path)}
    for selection in selected:
        selection_id = str(selection["selection_id"])
        if selection_id in existing:
            continue
        last_error: Exception | None = None
        feedback = ""
        for attempt in range(1, int(args.max_attempts) + 1):
            parsed: dict[str, Any] | None = None
            raw: dict[str, Any] | None = None
            try:
                parsed, raw = call_omni(
                    audio_path=Path(selection["candidate_audio"]),
                    fmt="wav",
                    audio_content_mode="input_audio",
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    timeout_s=float(args.timeout_s),
                    store_stream_chunks=False,
                    prompt=build_prompt(selection, validation_feedback=feedback),
                    system_prompt=SYSTEM_PROMPT,
                    max_tokens=int(args.max_tokens),
                    enable_thinking=True,
                    thinking_budget=int(args.thinking_budget),
                )
                validated = validate_response(parsed, selection)
                last_error = None
            except Exception as error:  # noqa: BLE001
                last_error = error
                feedback = str(error) if isinstance(error, ValueError) else ""
                _append(
                    raw_path,
                    {
                        "selection_id": selection_id,
                        "attempt": attempt,
                        "error": repr(error),
                        "parsed": parsed,
                        "response": raw,
                    },
                )
                if attempt < int(args.max_attempts):
                    time.sleep(min(8.0, float(attempt)))
                continue
            _append(
                raw_path,
                {
                    "selection_id": selection_id,
                    "attempt": attempt,
                    "parsed": parsed,
                    "response": raw,
                },
            )
            label = {
                "schema": SCHEMA,
                "prompt_version": PROMPT_VERSION,
                "model": model,
                **selection,
                **validated,
                "attempts": attempt,
            }
            _append(labels_path, label)
            existing[selection_id] = label
            print(
                f"multi_safe_repair={len(existing)}/{len(selected)} "
                f"selection_id={selection_id} decision={validated['decision']}",
                flush=True,
            )
            break
        if last_error is not None:
            raise RuntimeError(
                f"multi-safe repair teacher failed for {selection_id}: {last_error}"
            ) from last_error
        if float(args.request_interval_s) > 0:
            time.sleep(float(args.request_interval_s))

    labels = _rows(labels_path)
    comparison = (
        evaluate_against_manual_truth(labels, Path(args.manual_truth))
        if args.manual_truth
        else {}
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "selection_count": len(selected),
        "label_count": len(labels),
        "selection_contract": "all_separated_safe_runs_without_sample_specific_specs",
        **comparison,
        "training_ready": bool(comparison)
        and float(comparison["strict_accuracy"]) == 1.0,
        "labels": str(labels_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a generic Omni scope teacher on every separated safe run."
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--candidate-verdicts", required=True)
    parser.add_argument("--timeline-labels", required=True)
    parser.add_argument("--manual-truth", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--model", default="")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking-budget", type=int, default=768)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-interval-s", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
