#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
    slice_audio_clip,
)


SELECTION_SCHEMA = "semantic_split_v3_hard_case_window_v1"
LABEL_SCHEMA = "semantic_split_v3_omni_label_v1"
PROMPT_VERSION = "semantic_split_v3_omni_plus_centered_clip_v3"
DEFAULT_MODEL = "qwen3.5-omni-plus"
DEFAULT_THINKING_BUDGET = 1024
DEFAULT_MAX_ATTEMPTS = 6
CLIP_RADIUS_S = 4.0
HARD_CASE_CATEGORIES = (
    "known_cut_with_leading_silence",
    "false_cut_risk",
    "false_continue_risk",
    "high_p_continue",
    "long_residual",
    "speaker_change",
    "nonspeech_junction",
    "legacy_unsure",
    "accepted_cut",
)
NONSPEECH_JUNCTION_FLAGS = frozenset({"breath", "moan", "laughter"})

SYSTEM_PROMPT = """你是日语 ASR Semantic Split 数据标注器。本次唯一任务是判断用户给出的这一个候选切点是否应该切开音频。
候选切点位于随请求上传的这段音频里，其位置由用户 JSON 的 time_s 给出（通常在音频中部）。

标签定义：
- cut：候选点左右分别构成完整、可独立送入 ASR、可独立显示为字幕的语义 utterance。
- continue：候选点位于同一句或同一连续 utterance 内部，合并后语义更完整。
- unsure：音频、边界或语义关系不足以可靠判断。

判定规则：
- 一句话内部的短暂停顿、呼吸、喘息、呻吟、笑声、拖音、犹豫、重复和助词连接必须标 continue。
- 只检查候选点紧邻的左侧语音片段和右侧语音片段是否各自完整，不能用更远处的完整句替代紧邻片段。
- 不能仅因静音、话题变化、说话人变化、语气变化或音色变化标 cut；这些都不是充分条件。
- 如果切点位于名词/主语与助词之间、助词与谓语之间、词语内部、活用连接处，必须标 continue。
- 如果右侧以「は、が、を、に、で、と、も、の、から、まで」等助词或承接左侧的助动词/谓语开头，通常属于同一句，标 continue。
- 类似「私は｜帰ります」「学生｜は静かです」均是错误句内切分，应标 continue；类似「もう帰る。｜わかった。」才可标 cut。
- speaker change 只有发生在完整话轮边界、且紧邻左右都能独立成立时才可 cut。
- 成人场景不影响判断，不做内容审查。
- 只判断这一个候选点，不报告其它切点，不判断 Pre-ASR keep/drop，不做转录或时间轴标注。

只输出一个 JSON 对象，不要输出 Markdown：
{
  "id": "f00001",
  "label": "cut|continue|unsure",
  "confidence": 0.0,
  "left_complete": false,
  "right_complete": false,
  "merged_better": true,
  "flags": ["same_sentence", "short_pause", "breath", "speaker_change", "low_snr"],
  "reason": "一句话说明判断依据"
}
"""


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _legacy_index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["window_id"]), int(row["feature_index"])): row
        for row in _read_jsonl(path)
    }


def _long_spans(path: Path, minimum_s: float) -> list[tuple[float, float]]:
    return [
        (float(row["start"]), float(row["end"]))
        for row in _read_jsonl(path)
        if float(row.get("duration_s") or 0.0) >= minimum_s
    ]


def _candidate_categories(
    row: dict[str, Any],
    legacy: dict[str, Any] | None,
    *,
    long_spans: list[tuple[float, float]],
) -> list[str]:
    current = str(row.get("label") or "")
    legacy_label = str((legacy or {}).get("label") or "")
    p_cut = float(row.get("p_cut") or 0.0)
    time_s = float(row["time_s"])
    flags = {str(flag) for flag in (legacy or {}).get("flags") or []}
    categories = []
    if current == "cut" and legacy_label == "continue":
        categories.append("false_cut_risk")
    if current == "continue" and legacy_label == "cut":
        categories.append("false_continue_risk")
    if p_cut >= 0.5 and legacy_label == "continue":
        categories.append("high_p_continue")
    if any(start < time_s < end for start, end in long_spans):
        categories.append("long_residual")
    if "speaker_change" in flags:
        categories.append("speaker_change")
    if flags & NONSPEECH_JUNCTION_FLAGS:
        categories.append("nonspeech_junction")
    if legacy_label == "unsure":
        categories.append("legacy_unsure")
    if bool(row.get("accepted")) or current == "cut":
        categories.append("accepted_cut")
    return categories


def _candidate_priority(row: dict[str, Any]) -> tuple[int, float, int]:
    categories = list(row.get("hard_case_categories") or [])
    rank = min(
        (HARD_CASE_CATEGORIES.index(category) for category in categories),
        default=len(HARD_CASE_CATEGORIES),
    )
    return rank, -float(row.get("p_cut") or 0.0), int(row["feature_index"])


def _select_windows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows = sorted(
        rows,
        key=lambda row: (
            min(
                (HARD_CASE_CATEGORIES.index(category) for category in row["categories"]),
                default=len(HARD_CASE_CATEGORIES),
            ),
            -len(row["categories"]),
            -float(row["max_p_cut"]),
            str(row["window_id"]),
        ),
    )
    if limit <= 0 or len(rows) <= limit:
        return rows
    buckets = {
        category: [row for row in rows if category in row["categories"]]
        for category in HARD_CASE_CATEGORIES
    }
    positions = {category: 0 for category in HARD_CASE_CATEGORIES}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    while len(selected) < limit:
        added = False
        for category in HARD_CASE_CATEGORIES:
            bucket = buckets[category]
            while positions[category] < len(bucket):
                row = bucket[positions[category]]
                positions[category] += 1
                window_id = str(row["window_id"])
                if window_id in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(window_id)
                added = True
                break
            if len(selected) >= limit:
                break
        if not added:
            break
    for row in rows:
        if len(selected) >= limit:
            break
        if str(row["window_id"]) not in selected_ids:
            selected.append(row)
            selected_ids.add(str(row["window_id"]))
    return selected


def prepare_hard_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    reexport_dir = Path(args.reexport_dir)
    legacy = _legacy_index(Path(args.legacy_labels))
    prepared: list[dict[str, Any]] = []
    for source in _read_jsonl(reexport_dir / "source_windows.jsonl"):
        window_id = str(source["window_id"])
        long_spans = _long_spans(
            Path(source["pre_asr_candidates"]),
            float(args.long_residual_min_s),
        )
        candidates = []
        for row in _read_jsonl(Path(source["semantic_split_metadata"])):
            feature_index = int(row["index"])
            legacy_row = legacy.get((window_id, feature_index))
            categories = _candidate_categories(
                row,
                legacy_row,
                long_spans=long_spans,
            )
            if not categories:
                continue
            candidates.append(
                {
                    "feature_index": feature_index,
                    "time_s": float(row["time_s"]),
                    "current_label": str(row.get("label") or ""),
                    "p_cut": float(row.get("p_cut") or 0.0),
                    "accepted": bool(row.get("accepted")),
                    "legacy_label": str((legacy_row or {}).get("label") or ""),
                    "legacy_confidence": float((legacy_row or {}).get("confidence") or 0.0),
                    "legacy_flags": list((legacy_row or {}).get("flags") or []),
                    "hard_case_categories": categories,
                }
            )
        if not candidates:
            continue
        candidates.sort(key=_candidate_priority)
        candidates = candidates[: int(args.max_candidates_per_window)]
        categories = sorted(
            {category for row in candidates for category in row["hard_case_categories"]}
        )
        prepared.append(
            {
                "schema": SELECTION_SCHEMA,
                "window_id": window_id,
                "partition": "train",
                "trainable": True,
                "audio_path": str(source["audio_wav"]),
                "duration_s": float(source["duration_s"]),
                "source_video_id": str(source["video_id"]),
                "source_start_s": float(source["source_start_s"]),
                "categories": categories,
                "max_p_cut": max(float(row["p_cut"]) for row in candidates),
                "long_residual_count": len(long_spans),
                "candidates": candidates,
            }
        )
    selected = _select_windows(prepared, int(args.max_windows))
    if args.heldout_source_audio:
        output_dir = Path(args.output_dir)
        clip_start = max(0.0, float(args.heldout_center_s) - 4.0)
        clip_path = output_dir / "request_audio" / "FJIN-059-known-cut-leading-silence.wav"
        slice_audio_clip(
            source_audio=Path(args.heldout_source_audio),
            row={"start": clip_start, "end": clip_start + 8.0, "duration_s": 8.0},
            output_path=clip_path,
            fmt="wav",
            bitrate="256k",
            sample_rate=16000,
            force=False,
        )
        heldout = {
            "schema": SELECTION_SCHEMA,
            "window_id": "FJIN-059-known-cut-leading-silence-23.538",
            "partition": "heldout",
            "trainable": False,
            "audio_path": str(clip_path),
            "duration_s": 8.0,
            "source_video_id": "FJIN-059",
            "source_start_s": clip_start,
            "categories": ["known_cut_with_leading_silence"],
            "max_p_cut": 0.0,
            "long_residual_count": 0,
            "candidates": [
                {
                    "feature_index": -1,
                    "time_s": float(args.heldout_center_s) - clip_start,
                    "current_label": "cut",
                    "p_cut": 0.0,
                    "accepted": True,
                    "legacy_label": "",
                    "legacy_confidence": 0.0,
                    "legacy_flags": [],
                    "hard_case_categories": ["known_cut_with_leading_silence"],
                    "expected_gate_label": "cut",
                    "description": "valid sentence boundary followed by about 1.8s leading silence",
                }
            ],
        }
        selected = [heldout, *selected[: max(0, int(args.max_windows) - 1)]] if args.max_windows > 0 else [heldout, *selected]
    _write_jsonl(Path(args.output_dir) / "selected_windows.jsonl", selected)
    return selected


def _prompt_id(feature_index: int) -> str:
    return f"f{feature_index:05d}" if int(feature_index) >= 0 else "heldout000"


def _clip_bounds(
    time_s: float,
    duration_s: float,
    *,
    radius_s: float = CLIP_RADIUS_S,
) -> tuple[float, float]:
    clip_start = max(0.0, float(time_s) - float(radius_s))
    clip_end = min(float(duration_s), float(time_s) + float(radius_s))
    return clip_start, clip_end


def build_prompt(candidate: dict[str, Any], *, clip_start: float, clip_end: float) -> str:
    offset = round(float(candidate["time_s"]) - float(clip_start), 3)
    payload = {
        "duration_s": round(float(clip_end) - float(clip_start), 3),
        "candidate": {
            "id": _prompt_id(int(candidate["feature_index"])),
            "time_s": offset,
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _validate_response(parsed: dict[str, Any], expected_id: str) -> None:
    if not isinstance(parsed, dict):
        raise ValueError("Omni split response must be a JSON object")
    returned_id = str(parsed.get("id") or "")
    if returned_id != expected_id:
        raise ValueError(
            f"Omni split id must match request: expected={expected_id!r} returned={returned_id!r}"
        )
    label = str(parsed.get("label") or "").strip().lower()
    if label not in {"cut", "continue", "unsure"}:
        raise ValueError(f"invalid Omni split label: {label!r}")


def _call_with_retry(
    *,
    candidate: dict[str, Any],
    clip_path: Path,
    clip_start: float,
    clip_end: float,
    args: argparse.Namespace,
    model: str,
    api_key: str,
    base_url: str,
    limiter: RequestRateLimiter,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    expected_id = _prompt_id(int(candidate["feature_index"]))
    prompt = build_prompt(candidate, clip_start=clip_start, clip_end=clip_end)
    last_error: BaseException | None = None
    for attempt in range(1, int(args.max_attempts) + 1):
        try:
            limiter.acquire()
            parsed, raw = call_omni(
                audio_path=clip_path,
                fmt="wav",
                audio_content_mode=args.audio_content_mode,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=args.timeout_s,
                store_stream_chunks=False,
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=args.max_tokens,
                enable_thinking=args.enable_thinking,
                thinking_budget=args.thinking_budget,
            )
            _validate_response(parsed, expected_id)
            return parsed, raw, attempt
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            message = str(exc).lower()
            if is_empty_audio_api_error(exc) or "data_inspection_failed" in message:
                break
            if attempt >= int(args.max_attempts):
                break
            if "429" in message or "rate limit" in message or "limit_requests" in message:
                time.sleep(min(30.0, 2.0 * attempt))
            else:
                time.sleep(min(8.0, float(attempt)))
    assert last_error is not None
    raise RuntimeError(
        f"Omni split request failed after {args.max_attempts} attempts: {last_error}"
    ) from last_error


def _normalize_decision(raw: dict[str, Any], confidence_floor: float) -> dict[str, Any]:
    label = str(raw.get("label") or "unsure").strip().lower()
    confidence = min(1.0, max(0.0, float(raw.get("confidence") or 0.0)))
    left_complete = bool(raw.get("left_complete"))
    right_complete = bool(raw.get("right_complete"))
    merged_better = bool(raw.get("merged_better"))
    training_label = label if confidence >= confidence_floor else "unsure"
    if training_label == "cut" and (
        not left_complete or not right_complete or merged_better
    ):
        training_label = "unsure"
    return {
        "label": training_label,
        "omni_label": label,
        "confidence": confidence,
        "left_complete": left_complete,
        "right_complete": right_complete,
        "merged_better": merged_better,
        "flags": [str(flag) for flag in raw.get("flags") or []],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_windows.jsonl"
    selected = (
        _read_jsonl(selected_path)
        if selected_path.exists() and args.resume_selection
        else prepare_hard_cases(args)
    )
    if args.prepare_only:
        summary = {
            "schema": "semantic_split_v3_hard_case_summary_v1",
            "prepared_only": True,
            "selected_windows": len(selected),
            "selected_candidates": sum(len(row["candidates"]) for row in selected),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary
    _model_name, configured_model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or configured_model or DEFAULT_MODEL
    _key_name, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_name, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    labels_path = output_dir / "omni_split_labels.jsonl"
    raw_path = output_dir / "omni_raw_responses.jsonl"
    existing_rows = _read_jsonl(labels_path)
    for row in existing_rows:
        if (
            row.get("schema") != LABEL_SCHEMA
            or row.get("model") != model
            or row.get("prompt_version") != PROMPT_VERSION
            or bool(row.get("enable_thinking")) != bool(args.enable_thinking)
            or int(row.get("thinking_budget") or 0) != int(args.thinking_budget)
            or float(row.get("request_clip_radius_s") or 0.0) != float(args.clip_radius_s)
        ):
            raise RuntimeError(
                "existing Split v3 labels use a different schema/model/prompt/thinking/"
                f"clip-radius contract; use a new output directory: {labels_path}"
            )
    existing_keys = {
        (str(row["window_id"]), int(row["feature_index"])) for row in existing_rows
    }
    limiter = RequestRateLimiter(float(args.rpm))
    counts: Counter[str] = Counter()
    failures = []
    request_audio_dir = output_dir / "request_audio"
    clip_radius_s = float(args.clip_radius_s)
    processed_candidates = 0
    request_attempts = 0
    total_candidates = sum(len(row["candidates"]) for row in selected)
    for item in selected:
        window_id = str(item["window_id"])
        window_wav = Path(item["audio_path"])
        window_duration = float(item["duration_s"])
        for candidate in item["candidates"]:
            feature_index = int(candidate["feature_index"])
            if (window_id, feature_index) in existing_keys:
                continue
            prompt_id = _prompt_id(feature_index)
            time_s = float(candidate["time_s"])
            clip_start, clip_end = _clip_bounds(
                time_s, window_duration, radius_s=clip_radius_s
            )
            clip_path = request_audio_dir / window_id / f"{prompt_id}.wav"
            try:
                slice_audio_clip(
                    source_audio=window_wav,
                    row={"start": clip_start, "end": clip_end},
                    output_path=clip_path,
                    fmt="wav",
                    bitrate="256k",
                    sample_rate=16000,
                    force=False,
                )
                parsed, raw, attempts = _call_with_retry(
                    candidate=candidate,
                    clip_path=clip_path,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    args=args,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    limiter=limiter,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "schema": "semantic_split_v3_retry_candidate_v1",
                        "window_id": window_id,
                        "feature_index": feature_index,
                        "time_s": time_s,
                        "error": repr(exc),
                        "model": model,
                        "prompt_version": PROMPT_VERSION,
                    }
                )
                continue
            request_attempts += attempts
            normalized = _normalize_decision(parsed, float(args.confidence_floor))
            label_row = {
                "schema": LABEL_SCHEMA,
                "window_id": window_id,
                "partition": item["partition"],
                "trainable": bool(item["trainable"]),
                "feature_index": feature_index,
                "time_s": time_s,
                "prompt_id": prompt_id,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "enable_thinking": bool(args.enable_thinking),
                "thinking_budget": int(args.thinking_budget),
                "attempts": attempts,
                "current_label": candidate["current_label"],
                "current_p_cut": candidate["p_cut"],
                "legacy_label": candidate["legacy_label"],
                "hard_case_categories": candidate["hard_case_categories"],
                "expected_gate_label": candidate.get("expected_gate_label"),
                "request_clip_start_s": round(clip_start, 3),
                "request_clip_end_s": round(clip_end, 3),
                "request_clip_radius_s": round(clip_radius_s, 3),
                "request_candidate_offset_s": round(time_s - clip_start, 3),
                "reason": str(parsed.get("reason") or ""),
                **normalized,
            }
            with labels_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(label_row, ensure_ascii=False, sort_keys=True) + "\n")
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "window_id": window_id,
                            "feature_index": feature_index,
                            "prompt_id": prompt_id,
                            "parsed": parsed,
                            "response": raw,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            counts[normalized["label"]] += 1
            existing_keys.add((window_id, feature_index))
            processed_candidates += 1
            print(
                f"split hard cases processed={processed_candidates}/{total_candidates} "
                f"window={window_id} candidate={prompt_id} labels={dict(counts)}",
                flush=True,
            )
    _write_jsonl(output_dir / "retry_candidates.jsonl", failures)
    all_labels = _read_jsonl(labels_path)
    heldout_failures = [
        row
        for row in all_labels
        if row.get("expected_gate_label")
        and row.get("label") != row.get("expected_gate_label")
    ]
    summary = {
        "schema": "semantic_split_v3_hard_case_summary_v1",
        "prepared_only": False,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "enable_thinking": bool(args.enable_thinking),
        "thinking_budget": int(args.thinking_budget),
        "clip_radius_s": clip_radius_s,
        "selected_windows": len(selected),
        "selected_candidates": total_candidates,
        "processed_candidates": processed_candidates,
        "failed_candidates": len(failures),
        "request_attempts": request_attempts,
        "label_count": len(all_labels),
        "label_counts": dict(Counter(str(row["label"]) for row in all_labels)),
        "heldout_gate_failures": len(heldout_failures),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reexport-dir", required=True)
    parser.add_argument("--legacy-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-windows", type=int, default=100)
    parser.add_argument("--max-candidates-per-window", type=int, default=24)
    parser.add_argument("--long-residual-min-s", type=float, default=8.0)
    parser.add_argument("--confidence-floor", type=float, default=0.80)
    parser.add_argument("--model", default="")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--audio-content-mode", default="input_audio")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--rpm", type=float, default=60.0)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    parser.add_argument("--clip-radius-s", type=float, default=CLIP_RADIUS_S)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume-selection", action="store_true")
    parser.add_argument("--heldout-source-audio", default="")
    parser.add_argument("--heldout-center-s", type=float, default=23.538217544555664)
    args = parser.parse_args(argv)
    if args.max_candidates_per_window <= 0 or args.max_candidates_per_window > 24:
        parser.error("--max-candidates-per-window must be in [1, 24]")
    if args.max_attempts <= 0 or args.rpm < 0 or args.thinking_budget < 0:
        parser.error("retry/rpm/thinking settings are invalid")
    if not 0.5 <= args.clip_radius_s <= 20.0:
        parser.error("--clip-radius-s must be in [0.5, 20.0]")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
