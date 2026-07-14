#!/usr/bin/env python3
"""Label reusable semantic timelines from trusted text plus full source audio.

The teacher deliberately stops before model-specific compilation.  A single
response identifies the smallest natural subtitle/ASR units in the trusted
text and aligns those units to the source audio.  Local deterministic code
then exposes separate views for Scorer, Outer Refiner, Semantic Split, and
Inner Refiner without pretending that one label means the same thing for all
four models.
"""
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
    load_env_file,
)


SCHEMA = "semantic_timeline_teacher_v1"
SUMMARY_SCHEMA = "semantic_timeline_teacher_summary_v1"
PROMPT_VERSION = "semantic_timeline_text_then_audio_alignment_v1"
DEFAULT_MODEL = "qwen3.5-omni-plus"
TEXT_KINDS = ("semantic", "nonsemantic", "unsure")
ALIGNMENT_STATUSES = ("matched", "not_audible", "unsure")


SYSTEM_PROMPT = """你是日语语义时间轴离线监督标注器。用户会在一次请求中提供一条完整音频、完整可信参考文本和音频时长。你的输出将被分别编译成 Semantic Speech Scorer、Outer Edge Refiner、Semantic Split 和 Inner Edge Refiner 的训练视图，因此必须区分“文本语义单元”与“波形安全切割区”；不要为了让整段成为一个单元而合并，也不要为了增加切点而过切。

在同一个响应里依次完成两步：

第一步：按原始顺序拆分 reference_text。
- text_unit.kind 只能是 semantic / nonsemantic / unsure。
- semantic：清楚的词句、应答、称呼或独立语块，具有语言语义和字幕价值。
- nonsemantic：喘息、呻吟、亲吻声、笑声、无意义叫声、短促非词拟声或只有拉长音的片段。
- unsure：文本本身无法可靠判断是否为词语。
- 对 semantic 内容，单位应是“最小但完整、自然、可独立送入 ASR 或显示为字幕的语义单元”。句号、问号、感叹号通常形成边界；逗号不自动形成边界，但当左右是自然独立语块、表达意图转换、独立补充或自然分句时应拆开。
- 允许自然分句以连接形式结束，只要结合语气它本身可作为短字幕，例如「そんなこんなで割とあっさり繋がりは作れたんだけど、」与「まだ全部じゃない」。
- 「それと、もう一つ。」与「彼は、たぶん甘えたくなかったんだと思う」应是两个 semantic units。
- 「昨年の外部生の受け入れに続き、」与「本年の男女共学化へとこぎ着けられたことに協力感謝する」应是两个 semantic units。
- 不得在词中、活用中、助词连接中、句尾 mora 前或单纯呼吸停顿处拆分；不要把最小完整语义句单元继续切碎。
- 所有 unit.text 依次直接拼接后必须逐字符等于 reference_text；禁止改写、纠错、补充、删除或重排。相邻 unit 可以同为 semantic，因为它们代表不同的可切语义单元。

第二步：结合完整音频，对每个 kind=semantic 的 unit 分别做时间对齐。
- matched：能可靠听到固定文本，返回该 unit 实际语音的 start_s/end_s。
- not_audible：参考文本在音频中听不到，坐标为 null。
- unsure：疑似存在但重叠、低信噪比或边界无法可靠判断，坐标为 null；禁止猜时间。
- matched span 按文本顺序排列。若相邻说话真实重叠，可以时间重叠，不要伪造静音；本地编译会让 Inner Refiner abstain。
- 这些时间是 semantic timeline，不是最终 Split cut，也不是 Inner safe-zone 标签。不要输出单一 shared cut time、keep_span、最终 chunk 或固定时长规则。
- 背景音乐、环境噪声、喘息、呻吟、亲吻声、无字幕价值背景人声不属于 semantic span；语义时间轴的可靠差集会由本地代码作为 Scorer 的 nonsemantic frame 监督，但不会直接当作 CueQC drop 或 Inner safe。
- unsure_audio_spans 只记录确实无法可靠归类或对齐的局部音频；没有则为空数组。

只输出一个 JSON 对象，不要 Markdown：
{
  "sample_id": "...",
  "text_units": [
    {"unit_id":"u00","text":"...","kind":"semantic|nonsemantic|unsure","confidence":0.0,"reason":"简短理由"}
  ],
  "semantic_alignments": [
    {"unit_id":"u00","status":"matched|not_audible|unsure","start_s":0.0,"end_s":0.0,"confidence":0.0,"reason":"简短理由"}
  ],
  "unsure_audio_spans": [
    {"start_s":0.0,"end_s":0.0,"reason":"简短理由"}
  ],
  "reason":"整体说明"
}
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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(sample: dict[str, Any], *, validation_feedback: str = "") -> str:
    payload: dict[str, Any] = {
        "sample_id": str(sample["sample_id"]),
        "duration_s": round(float(sample["duration_s"]), 6),
        "reference_text": str(sample["reference_text"]),
        "task_order": [
            "segment_reference_text_into_minimal_complete_semantic_units",
            "align_each_semantic_unit_to_full_audio",
        ],
        "timeline_contract": "semantic_units_not_shared_cut_times",
        "downstream_contract": {
            "scorer": "semantic spans plus confident audio complement",
            "outer_refiner": "first and last semantic speech edges",
            "semantic_split": "ordered semantic events projected to proposer candidates",
            "inner_refiner": "requires separate candidate safe-zone labels",
            "cueqc": "not labeled by this request",
        },
    }
    if validation_feedback:
        payload["previous_response_validation_error"] = validation_feedback
        payload["retry_instruction"] = (
            "Correct the schema or semantic-unit error without changing the audio or reference text."
        )
    return json.dumps(payload, ensure_ascii=False)


def _confidence(value: Any, field: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} confidence must be in [0, 1]")
    return result


def _nullable_span(
    raw: dict[str, Any], *, duration_s: float, field: str
) -> tuple[float | None, float | None]:
    start = raw.get("start_s")
    end = raw.get("end_s")
    if start is None or end is None:
        if start is not None or end is not None:
            raise ValueError(f"{field} must use two coordinates or two nulls")
        return None, None
    start_s = float(start)
    end_s = float(end)
    if not 0.0 <= start_s < end_s <= duration_s:
        raise ValueError(f"{field} must be inside 0..duration_s")
    return start_s, end_s


def _merged_spans(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[list[float]] = []
    for start_s, end_s in sorted(spans):
        if result and start_s <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end_s)
        else:
            result.append([start_s, end_s])
    return [(start_s, end_s) for start_s, end_s in result]


def _complement_spans(
    spans: Iterable[tuple[float, float]], *, duration_s: float
) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    cursor = 0.0
    for start_s, end_s in _merged_spans(spans):
        if start_s > cursor:
            result.append({"start_s": cursor, "end_s": start_s})
        cursor = max(cursor, end_s)
    if cursor < duration_s:
        result.append({"start_s": cursor, "end_s": duration_s})
    return result


def _derive_model_views(
    *,
    units: list[dict[str, Any]],
    alignments: list[dict[str, Any]],
    unsure_spans: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    by_id = {row["unit_id"]: row for row in alignments}
    semantic_units = [row for row in units if row["kind"] == "semantic"]
    matched = [row for row in alignments if row["status"] == "matched"]
    has_unsure = bool(unsure_spans) or any(
        row["status"] == "unsure" for row in alignments
    )

    events: list[dict[str, Any]] = []
    for event_index, (left_unit, right_unit) in enumerate(
        zip(semantic_units, semantic_units[1:], strict=False)
    ):
        left = by_id[left_unit["unit_id"]]
        right = by_id[right_unit["unit_id"]]
        if left["status"] != "matched" or right["status"] != "matched":
            events.append(
                {
                    "event_id": f"e{event_index:02d}",
                    "left_unit_id": left_unit["unit_id"],
                    "right_unit_id": right_unit["unit_id"],
                    "status": "unsure",
                    "interval_start_s": None,
                    "interval_end_s": None,
                    "overlap": None,
                }
            )
            continue
        left_end = float(left["end_s"])
        right_start = float(right["start_s"])
        events.append(
            {
                "event_id": f"e{event_index:02d}",
                "left_unit_id": left_unit["unit_id"],
                "right_unit_id": right_unit["unit_id"],
                "status": "matched",
                "interval_start_s": left_end,
                "interval_end_s": right_start,
                "overlap": right_start < left_end,
            }
        )

    if has_unsure:
        membership = {"status": "unsure", "start_s": None, "end_s": None}
        complement: list[dict[str, float]] = []
    elif matched:
        membership = {
            "status": "matched",
            "start_s": min(float(row["start_s"]) for row in matched),
            "end_s": max(float(row["end_s"]) for row in matched),
        }
        complement = _complement_spans(
            ((float(row["start_s"]), float(row["end_s"])) for row in matched),
            duration_s=duration_s,
        )
    else:
        membership = {"status": "none", "start_s": None, "end_s": None}
        complement = [{"start_s": 0.0, "end_s": duration_s}]

    semantic_spans = [
        {
            "unit_id": row["unit_id"],
            "start_s": row["start_s"],
            "end_s": row["end_s"],
        }
        for row in matched
    ]
    outer_target = (
        {
            "status": "matched",
            "left_speech_start_s": membership["start_s"],
            "right_speech_end_s": membership["end_s"],
        }
        if membership["status"] == "matched"
        else {
            "status": membership["status"],
            "left_speech_start_s": None,
            "right_speech_end_s": None,
        }
    )
    return {
        "semantic_timeline": semantic_spans,
        "semantic_events": events,
        "scorer_view": {
            "semantic_spans": semantic_spans,
            "nonsemantic_complement_spans": complement,
            "source_membership": membership,
            "complement_usage": "semantic_speech_scorer_frame_negative_only",
        },
        "outer_refiner_view": outer_target,
        "split_view": {
            "events": events,
            "timing_usage": "ordered_event_projection_to_runtime_proposer_candidates",
        },
        "inner_refiner_view": {
            "status": "requires_candidate_safe_zone_teacher",
            "semantic_event_intervals": events,
        },
        "cueqc_view": {"status": "not_labeled_until_new_chunks_are_exported"},
    }


def validate_response(
    parsed: dict[str, Any], sample: dict[str, Any]
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    duration_s = float(sample["duration_s"])
    reference_text = str(sample["reference_text"])
    if str(parsed.get("sample_id") or "") != sample_id:
        raise ValueError("sample_id mismatch")

    raw_units = parsed.get("text_units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ValueError("text_units must be a non-empty list")
    units: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_units):
        if not isinstance(raw, dict):
            raise ValueError("every text unit must be an object")
        unit_id = str(raw.get("unit_id") or "")
        if unit_id != f"u{index:02d}":
            raise ValueError("text unit ids must be contiguous u00, u01, ...")
        text = str(raw.get("text") or "")
        if not text:
            raise ValueError("text unit text must be non-empty")
        kind = str(raw.get("kind") or "")
        if kind not in TEXT_KINDS:
            raise ValueError("invalid text unit kind")
        units.append(
            {
                "unit_id": unit_id,
                "text": text,
                "kind": kind,
                "confidence": _confidence(raw.get("confidence"), unit_id),
                "reason": str(raw.get("reason") or ""),
            }
        )
    if "".join(row["text"] for row in units) != reference_text:
        raise ValueError("text unit concatenation must exactly equal reference_text")

    semantic_ids = [row["unit_id"] for row in units if row["kind"] == "semantic"]
    raw_alignments = parsed.get("semantic_alignments")
    if not isinstance(raw_alignments, list):
        raise ValueError("semantic_alignments must be a list")
    returned_ids = [str(row.get("unit_id") or "") for row in raw_alignments]
    if returned_ids != semantic_ids:
        raise ValueError("semantic alignment ids must exactly match semantic text units")
    alignments: list[dict[str, Any]] = []
    previous_start = -1.0
    for raw in raw_alignments:
        if not isinstance(raw, dict):
            raise ValueError("every semantic alignment must be an object")
        unit_id = str(raw.get("unit_id") or "")
        status = str(raw.get("status") or "")
        if status not in ALIGNMENT_STATUSES:
            raise ValueError("invalid semantic alignment status")
        start_s, end_s = _nullable_span(
            raw, duration_s=duration_s, field=f"semantic alignment {unit_id}"
        )
        if status == "matched":
            if start_s is None or end_s is None:
                raise ValueError("matched semantic alignment requires coordinates")
            if start_s < previous_start:
                raise ValueError("matched semantic alignments must follow text order")
            previous_start = start_s
        elif start_s is not None or end_s is not None:
            raise ValueError("non-matched semantic alignment must use null coordinates")
        alignments.append(
            {
                "unit_id": unit_id,
                "status": status,
                "start_s": start_s,
                "end_s": end_s,
                "confidence": _confidence(raw.get("confidence"), unit_id),
                "reason": str(raw.get("reason") or ""),
            }
        )

    raw_unsure = parsed.get("unsure_audio_spans")
    if not isinstance(raw_unsure, list):
        raise ValueError("unsure_audio_spans must be a list")
    unsure_spans: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_unsure):
        if not isinstance(raw, dict):
            raise ValueError("every unsure audio span must be an object")
        start_s, end_s = _nullable_span(
            raw, duration_s=duration_s, field=f"unsure audio span {index}"
        )
        if start_s is None or end_s is None:
            raise ValueError("unsure audio span requires coordinates")
        unsure_spans.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "reason": str(raw.get("reason") or ""),
            }
        )

    return {
        "text_units": units,
        "semantic_alignments": alignments,
        "unsure_audio_spans": unsure_spans,
        "reason": str(parsed.get("reason") or ""),
        **_derive_model_views(
            units=units,
            alignments=alignments,
            unsure_spans=unsure_spans,
            duration_s=duration_s,
        ),
    }


def _select_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in args.samples:
        for row in _rows(Path(source)):
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                continue
            seen.add(sample_id)
            samples.append(row)
    requested = list(args.sample_id or [])
    if requested:
        by_id = {str(row["sample_id"]): row for row in samples}
        missing = [sample_id for sample_id in requested if sample_id not in by_id]
        if missing:
            raise ValueError(f"unknown sample ids: {missing}")
        samples = [by_id[sample_id] for sample_id in requested]
    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]
    if not samples:
        raise ValueError("no semantic timeline samples selected")
    return samples


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    samples = _select_samples(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_samples.jsonl"
    labels_path = output_dir / "semantic_timeline_labels.jsonl"
    raw_path = output_dir / "omni_raw_responses.jsonl"
    _write_jsonl(selected_path, samples)

    _model_env, configured_model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or configured_model or DEFAULT_MODEL
    _key_env, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_env, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    if not api_key:
        raise RuntimeError("Omni API key is required")

    existing = {str(row["sample_id"]): row for row in _rows(labels_path)}
    for row in existing.values():
        if (
            row.get("schema") != SCHEMA
            or row.get("prompt_version") != PROMPT_VERSION
            or row.get("model") != model
        ):
            raise RuntimeError(
                "existing labels use another schema/model/prompt contract; use a new output directory"
            )
    pending = [row for row in samples if str(row["sample_id"]) not in existing]
    for position, sample in enumerate(pending, start=1):
        sample_id = str(sample["sample_id"])
        last_error: Exception | None = None
        validation_feedback = ""
        for attempt in range(1, int(args.max_attempts) + 1):
            parsed: dict[str, Any] | None = None
            raw: dict[str, Any] | None = None
            try:
                parsed, raw = call_omni(
                    audio_path=Path(sample["audio"]),
                    fmt=Path(sample["audio"]).suffix.lstrip(".") or "wav",
                    audio_content_mode="input_audio",
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    timeout_s=float(args.timeout_s),
                    store_stream_chunks=False,
                    prompt=build_prompt(
                        sample, validation_feedback=validation_feedback
                    ),
                    system_prompt=SYSTEM_PROMPT,
                    max_tokens=int(args.max_tokens),
                    enable_thinking=True,
                    thinking_budget=int(args.thinking_budget),
                )
                validated = validate_response(parsed, sample)
                last_error = None
            except Exception as error:  # noqa: BLE001
                last_error = error
                validation_feedback = str(error) if isinstance(error, ValueError) else ""
                _append(
                    raw_path,
                    {
                        "sample_id": sample_id,
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
                    "sample_id": sample_id,
                    "attempt": attempt,
                    "parsed": parsed,
                    "response": raw,
                },
            )
            label = {
                "schema": SCHEMA,
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "request_contract": "single_full_audio_plus_trusted_reference_text",
                "sample_id": sample_id,
                "audio": str(sample["audio"]),
                "duration_s": float(sample["duration_s"]),
                "source": str(sample.get("source") or ""),
                "audit_focus": str(sample.get("audit_focus") or ""),
                "reference_text": str(sample["reference_text"]),
                "attempts": attempt,
                **validated,
            }
            _append(labels_path, label)
            existing[sample_id] = label
            print(
                f"semantic_timeline={len(existing)}/{len(samples)} "
                f"sample_id={sample_id} units={len(validated['text_units'])} "
                f"semantic={len(validated['semantic_timeline'])} "
                f"events={len(validated['semantic_events'])}",
                flush=True,
            )
            break
        if last_error is not None:
            raise RuntimeError(
                f"semantic timeline labeling failed for {sample_id}: {last_error}"
            ) from last_error
        if position < len(pending) and float(args.request_interval_s) > 0:
            time.sleep(float(args.request_interval_s))

    labels = _rows(labels_path)
    kind_counts = Counter(
        unit["kind"] for row in labels for unit in row.get("text_units") or []
    )
    status_counts = Counter(
        alignment["status"]
        for row in labels
        for alignment in row.get("semantic_alignments") or []
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "requested_samples": len(samples),
        "labeled_samples": len(labels),
        "unique_source_count": len({str(row.get("source") or "") for row in labels}),
        "text_kind_counts": {kind: kind_counts[kind] for kind in TEXT_KINDS},
        "alignment_status_counts": {
            status: status_counts[status] for status in ALIGNMENT_STATUSES
        },
        "semantic_event_count": sum(
            len(row.get("semantic_events") or []) for row in labels
        ),
        "labels": str(labels_path),
        "selected_samples": str(selected_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label reusable semantic timelines for Scorer, Outer, Split, and Inner dataset compilation."
    )
    parser.add_argument("--samples", action="append", required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--thinking-budget", type=int, default=1536)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-interval-s", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.thinking_budget < 0:
        parser.error("--thinking-budget must be non-negative")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
