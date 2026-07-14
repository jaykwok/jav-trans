#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.asr.cueqc.label_pre_asr_with_omni import (
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    call_omni,
    first_env_value,
    load_env_file,
)


SCHEMA = "semantic_source_text_alignment_teacher_v1"
SUMMARY_SCHEMA = "semantic_source_text_alignment_summary_v1"
PROMPT_VERSION = "semantic_source_text_then_audio_alignment_v1"
DEFAULT_MODEL = "qwen3.5-omni-plus"
TEXT_KINDS = ("semantic", "nonsemantic", "unsure")
ALIGNMENT_STATUSES = ("matched", "not_audible", "unsure")
KEEP_STATUSES = ("matched", "none", "unsure")
CONTEXT_CLASSES = (
    "attached_nonsemantic",
    "semantic_edge",
    "external_only",
    "unsure",
    "not_applicable",
)


SYSTEM_PROMPT = """你是日语短 source utterance 的离线监督标注器。用户会在一次请求中提供一条完整短音频、它的完整可信参考文本和音频时长。你必须在这一次响应里严格按两个步骤完成任务；不要要求第二次请求。

第一步：只分析参考文本的语义组成。
- 按原始顺序把 reference_text 拆成连续 text_units，kind 只能是 semantic / nonsemantic / unsure。
- semantic：清楚的词、助词、应答词或句子，具有语言语义和字幕价值。
- nonsemantic：文本中表示喘息、呻吟、亲吻声、笑声、无意义叫声、短促非词拟声或拉长音的部分。
- unsure：仅当该文本片段本身无法可靠判断是否是词语时使用。
- 不要把连续的正常词句按单词切碎；只在 semantic / nonsemantic / unsure 类别发生变化时拆开。标点和省略号必须保留在相邻原文片段中。
- 所有 unit.text 依次直接拼接后必须逐字符等于原始 reference_text；禁止改写、纠错、补充、删除或重排。

第二步：结合完整音频做对齐与上下文归属。
- 只为第一步 kind=semantic 的 unit 返回 semantic_alignments，unit_id 和顺序必须完全一致。
- matched：音频中能可靠听到该固定文本，返回局部坐标 start_s/end_s。
- not_audible：参考文本对应内容在音频中听不到，start_s/end_s 返回 null。
- unsure：疑似存在但重叠、低信噪比或边界无法可靠判断，start_s/end_s 返回 null；禁止猜时间。
- matched span 必须按文本顺序排列且互不重叠，坐标位于 0..duration_s。
- 再返回一个 keep_span。它是 source-membership teacher，不是最终 Outer edge，也不是 Split 切点：必须包含所有 matched semantic span，并自适应包含属于同一个前景 utterance 的前导/尾随喘息、呻吟、亲吻声等附属非语义声音；同时排除仅有 BGM、环境/机械噪声和远处背景人声的外部区域。不得使用固定秒数 margin。
- 第一个/最后一个 semantic span 不能直接冒充 keep_span；必须独立听音判断两侧上下文属于 attached_nonsemantic、semantic_edge、external_only 还是 unsure。
- 若没有任何 matched semantic unit，keep_span.status=none；若语义或归属无法可靠判断，keep_span.status=unsure。不要为了给出区间而强行猜测。
- unsure_audio_spans 只记录确实需要 abstain 的局部区域；没有则返回空数组。
- 这是离线 teacher alignment。不要输出最终 Split cut，不要判断 chunk 时长，不要把参考文本当作 runtime 输入，不要输出 Markdown。

只输出一个 JSON 对象：
{
  "sample_id": "...",
  "text_units": [
    {"unit_id":"u00","text":"...","kind":"semantic|nonsemantic|unsure","confidence":0.0,"reason":"简短理由"}
  ],
  "semantic_alignments": [
    {"unit_id":"u00","status":"matched|not_audible|unsure","start_s":0.0,"end_s":0.0,"confidence":0.0,"reason":"简短理由"}
  ],
  "keep_span": {
    "status":"matched|none|unsure",
    "start_s":0.0,
    "end_s":0.0,
    "leading_context":"attached_nonsemantic|semantic_edge|external_only|unsure|not_applicable",
    "trailing_context":"attached_nonsemantic|semantic_edge|external_only|unsure|not_applicable",
    "confidence":0.0,
    "reason":"简短理由"
  },
  "unsure_audio_spans": [
    {"start_s":0.0,"end_s":0.0,"reason":"简短理由"}
  ],
  "reason":"整体说明"
}

当 alignment 不是 matched，或 keep_span.status 不是 matched 时，对应 start_s/end_s 使用 null，不要使用 0.0 占位。
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


def build_prompt(sample: dict[str, Any]) -> str:
    return json.dumps(
        {
            "sample_id": str(sample["sample_id"]),
            "duration_s": round(float(sample["duration_s"]), 6),
            "reference_text": str(sample["reference_text"]),
            "task_order": [
                "split_reference_text_by_semantic_kind",
                "align_semantic_units_and_assign_keep_span",
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _confidence(value: Any, field: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} confidence must be in [0, 1]")
    return result


def _matched_span(
    row: dict[str, Any], *, duration_s: float, field: str
) -> tuple[float, float]:
    start_s = float(row.get("start_s"))
    end_s = float(row.get("end_s"))
    if not 0.0 <= start_s < end_s <= duration_s:
        raise ValueError(f"{field} span must satisfy 0 <= start < end <= duration")
    return start_s, end_s


def _require_null_span(row: dict[str, Any], field: str) -> None:
    if row.get("start_s") is not None or row.get("end_s") is not None:
        raise ValueError(f"{field} non-matched span must use null coordinates")


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
    if "".join(unit["text"] for unit in units) != reference_text:
        raise ValueError("text unit concatenation must exactly equal reference_text")

    semantic_ids = [unit["unit_id"] for unit in units if unit["kind"] == "semantic"]
    raw_alignments = parsed.get("semantic_alignments")
    if not isinstance(raw_alignments, list):
        raise ValueError("semantic_alignments must be a list")
    returned_ids = [str(row.get("unit_id") or "") for row in raw_alignments]
    if returned_ids != semantic_ids:
        raise ValueError("semantic alignment ids must exactly match semantic text units")
    alignments: list[dict[str, Any]] = []
    previous_end = 0.0
    for raw in raw_alignments:
        if not isinstance(raw, dict):
            raise ValueError("every semantic alignment must be an object")
        unit_id = str(raw["unit_id"])
        status = str(raw.get("status") or "")
        if status not in ALIGNMENT_STATUSES:
            raise ValueError("invalid semantic alignment status")
        start_s: float | None = None
        end_s: float | None = None
        if status == "matched":
            start_s, end_s = _matched_span(
                raw, duration_s=duration_s, field=f"alignment {unit_id}"
            )
            if start_s < previous_end:
                raise ValueError("matched semantic alignments must not overlap")
            previous_end = end_s
        else:
            _require_null_span(raw, f"alignment {unit_id}")
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

    raw_keep = parsed.get("keep_span")
    if not isinstance(raw_keep, dict):
        raise ValueError("keep_span must be an object")
    keep_status = str(raw_keep.get("status") or "")
    if keep_status not in KEEP_STATUSES:
        raise ValueError("invalid keep span status")
    matched_alignments = [row for row in alignments if row["status"] == "matched"]
    keep_start: float | None = None
    keep_end: float | None = None
    if keep_status == "matched":
        keep_start, keep_end = _matched_span(
            raw_keep, duration_s=duration_s, field="keep"
        )
        if not matched_alignments:
            raise ValueError("matched keep span requires at least one matched semantic unit")
        if keep_start > float(matched_alignments[0]["start_s"]) or keep_end < float(
            matched_alignments[-1]["end_s"]
        ):
            raise ValueError("keep span must contain every matched semantic alignment")
    else:
        _require_null_span(raw_keep, "keep")
        if keep_status == "none" and matched_alignments:
            raise ValueError("keep span cannot be none when semantic units are matched")
    leading_context = str(raw_keep.get("leading_context") or "")
    trailing_context = str(raw_keep.get("trailing_context") or "")
    if leading_context not in CONTEXT_CLASSES or trailing_context not in CONTEXT_CLASSES:
        raise ValueError("invalid keep span context class")
    keep_span = {
        "status": keep_status,
        "start_s": keep_start,
        "end_s": keep_end,
        "leading_context": leading_context,
        "trailing_context": trailing_context,
        "confidence": _confidence(raw_keep.get("confidence"), "keep"),
        "reason": str(raw_keep.get("reason") or ""),
    }

    raw_unsure = parsed.get("unsure_audio_spans")
    if not isinstance(raw_unsure, list):
        raise ValueError("unsure_audio_spans must be a list")
    unsure_spans: list[dict[str, Any]] = []
    previous_unsure_end = 0.0
    for index, raw in enumerate(raw_unsure):
        if not isinstance(raw, dict):
            raise ValueError("every unsure audio span must be an object")
        start_s, end_s = _matched_span(
            raw, duration_s=duration_s, field=f"unsure span {index}"
        )
        if start_s < previous_unsure_end:
            raise ValueError("unsure audio spans must be ordered and non-overlapping")
        previous_unsure_end = end_s
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
        "keep_span": keep_span,
        "unsure_audio_spans": unsure_spans,
        "reason": str(parsed.get("reason") or ""),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    samples = _rows(Path(args.samples))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "text_alignment_labels.jsonl"
    raw_path = output_dir / "omni_raw_responses.jsonl"
    _model_env, configured_model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or configured_model or DEFAULT_MODEL
    _key_env, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_env, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    if not api_key:
        raise RuntimeError("Omni API key is required")

    existing_rows = {str(row["sample_id"]): row for row in _rows(labels_path)}
    for row in existing_rows.values():
        if (
            row.get("schema") != SCHEMA
            or row.get("prompt_version") != PROMPT_VERSION
            or row.get("model") != model
        ):
            raise RuntimeError(
                "existing labels use another schema/model/prompt contract; use a new output directory"
            )
    pending = [row for row in samples if str(row["sample_id"]) not in existing_rows]
    if int(args.limit) > 0:
        pending = pending[: int(args.limit)]

    for position, sample in enumerate(pending, start=1):
        sample_id = str(sample["sample_id"])
        validation_error: Exception | None = None
        for attempt in range(1, int(args.max_attempts) + 1):
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
                    prompt=build_prompt(sample),
                    system_prompt=SYSTEM_PROMPT,
                    max_tokens=int(args.max_tokens),
                    enable_thinking=True,
                    thinking_budget=int(args.thinking_budget),
                )
                validated = validate_response(parsed, sample)
                validation_error = None
            except Exception as error:  # noqa: BLE001
                validation_error = error
                _append(
                    raw_path,
                    {
                        "sample_id": sample_id,
                        "attempt": attempt,
                        "error": repr(error),
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
                "request_contract": "single_full_audio_plus_reference_text",
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
            existing_rows[sample_id] = label
            print(
                f"text_alignment={len(existing_rows)}/{len(samples)} "
                f"sample_id={sample_id} text_units={len(validated['text_units'])} "
                f"matched={sum(row['status'] == 'matched' for row in validated['semantic_alignments'])} "
                f"keep={validated['keep_span']['status']}",
                flush=True,
            )
            break
        if validation_error is not None:
            raise RuntimeError(
                f"semantic source text alignment failed for {sample_id}: {validation_error}"
            ) from validation_error
        if position < len(pending) and float(args.request_interval_s) > 0:
            time.sleep(float(args.request_interval_s))

    labels = _rows(labels_path)
    kind_counts = Counter(
        unit["kind"] for row in labels for unit in row.get("text_units") or []
    )
    alignment_counts = Counter(
        alignment["status"]
        for row in labels
        for alignment in row.get("semantic_alignments") or []
    )
    keep_counts = Counter(str(row["keep_span"]["status"]) for row in labels)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "request_contract": "one_request_per_full_source",
        "requested_samples": len(samples),
        "labeled_samples": len(labels),
        "text_kind_counts": {kind: kind_counts[kind] for kind in TEXT_KINDS},
        "alignment_status_counts": {
            status: alignment_counts[status] for status in ALIGNMENT_STATUSES
        },
        "keep_status_counts": {status: keep_counts[status] for status in KEEP_STATUSES},
        "labels": str(labels_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label semantic text units, audio alignment, and membership keep span in one Omni request."
    )
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking-budget", type=int, default=1024)
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
