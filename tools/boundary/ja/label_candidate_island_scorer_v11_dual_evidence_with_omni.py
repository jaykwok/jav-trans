#!/usr/bin/env python3
"""Preaudit Scorer v11 with independent Protect and Remove evidence passes.

The two Omni requests never observe each other's output.  Their evidence is
merged into partial three-state canonical supervision:

* Protect only -> inside_candidate
* Remove only -> outside_candidate
* overlap or neither -> unsure (ignore=-100 downstream)

This tool only creates reviewable preaudit data.  It never promotes labels to
training truth by itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from tools.asr.cueqc.label_pre_asr_with_omni import (  # noqa: E402
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    GEMINI_THINKING_LEVELS,
    audio_content_mode_for_profile,
    build_omni_request_body,
    call_omni,
    first_env_value,
    load_env_file,
    normalize_openai_compat_base_url,
    redact_omni_request_preview,
)
from tools.boundary.ja.label_candidate_island_scorer_v11_with_omni import (  # noqa: E402
    FRAME_HOP_S,
    _resolve_audio,
    _rows,
    _sha256,
    _write_progress,
)
from tools.omni.timestamp_contract import (  # noqa: E402
    TIMESTAMP_CONTRACT_ID,
    TIMESTAMP_PROMPT_CONTRACT_ZH,
    parse_mmss_span,
    timestamp_request_contract,
)


SCHEMA = "candidate_island_scorer_v11_dual_evidence_preaudit_v1"
SUMMARY_SCHEMA = "candidate_island_scorer_v11_dual_evidence_summary_v1"
PROMPT_PROFILE = "dual-evidence-protect-remove-v1"
PROTECT_PROMPT_VERSION = (
    "candidate_island_scorer_v11_protect_evidence_v3_high_recall_mmss_mmm"
)
REMOVE_PROMPT_VERSION = (
    "candidate_island_scorer_v11_remove_evidence_v2_mmss_mmm"
)
PROMPT_VERSION = f"{PROTECT_PROMPT_VERSION}__{REMOVE_PROMPT_VERSION}"
TEACHER_EXECUTION_CONTRACT_ID = (
    "gemini_openrouter_reasoning_require_parameters_v1"
)
OUTSIDE_CATEGORIES = frozenset(
    {
        "breathing",
        "moan",
        "cry",
        "kiss",
        "action",
        "impact",
        "silence",
        "music",
        "ambience",
        "mechanical",
        "other",
    }
)


PROTECT_SYSTEM_PROMPT = """你是 1.7B Scorer v11 的“候选对话保护通道”预审 teacher。这是高召回保护通道：漏掉可能的语言波形，比暂时多保留非语言声音代价更高。

Scorer 位于 Proposal、Split、CueQC 和 Inner 之前。你的唯一任务是找出必须完整送给下游的连续候选对话岛。你不切句、不区分说话人，也不判断最终字幕是否值得保留。

请先完整听取 source，再输出 protected_spans。

应标记 protected_span：
1. 能听出明确或具有现实可能性属于日语词语、音拍组合、对白、应答、耳语、口吃、残缺发音、词首、词尾或语言尾音。
2. 「あ、ん、はぁ」等短音、哭腔、气声或呻吟式发声，只要可能承担应答、感叹、呼唤、强调或词语残片功能，就必须保护；不要求先辨认出具体词义。
3. 同一轮连续对话中的相邻语言锚点，应保持在同一个连续波形包络中。
4. 夹在同一轮对话内部、无法独立删除的停顿、呼吸、呻吟、动作声或撞击声，应随对话一起保护。
5. 被音乐、动作声、喘息或噪声遮盖，但仍可能存在语言的部分，应保护。

不得仅因为声音来自人、情绪连续、场景持续或存在呻吟喘息，就扩展 protected_span。

如果中间存在边界清楚、能够独立删除的非语言事件，不要用它连接两轮对话。句子切分属于后续 Split，不属于你。

只有在当前局部能够确认是独立纯非语言声音、且与任何可能对白包络无关时，才可以不标记。无法可靠排除语言或交流功能时，宁可保护。

未标记区域只表示本通道没有给出保护证据，不代表 outside；最终是否可删除由另一次完全独立的 Remove 通道决定。

边界应覆盖完整词头、词尾、吸气起始、衰减尾音以及同一发声单元的完整声学包络。宁可略宽，也不能削掉边缘；但不要添加固定时间缓冲。

输出当前完整 source 的 0-based 局部坐标。区间按时间排序、不得重叠。允许空数组。

只输出 JSON：
{
  "source_id": "...",
  "protected_spans": [
    {
      "start_ts": "00:00.000",
      "end_ts": "00:01.000",
      "reason": "需要保护的简短声学证据"
    }
  ],
  "overall_reason": "简短整体判断"
}
""" + "\n" + TIMESTAMP_PROMPT_CONTRACT_ZH


REMOVE_SYSTEM_PROMPT = """你是 1.7B Scorer v11 的“安全删除通道”预审 teacher。

你的唯一任务是找出具有明确正面声学证据、能够从当前完整 source 中独立删除的非语言连续区间。不要标注 inside，不切句，不使用 ASR 文本。

可以标记 safe_outside_span 的内容包括：
- 纯静音、环境底噪、纯器乐、机械声；
- 独立的动作声、撞击声、摩擦声、亲吻声；
- 明确非词化、没有交流功能的呼吸、喘息、呻吟、哭声或笑声。

声音类别本身不能决定标签。只有同时满足以下条件才能输出：
1. 当前可听证据明确支持它主要是非语言事件；
2. 听不出词语、语言音拍组合、应答或残缺语言；
3. 它在声学上能够独立移除；
4. 删除不会截断相邻词头、词尾、耳语或连续对话波形。

不要求逻辑上证明“绝对不可能存在语言”，但只要存在实际可听的语言可能性，就不要输出。

如果候选两端接近疑似语言，应向内部收缩；收缩后仍不能形成独立区间，就省略。不要使用固定缓冲或固定时长。

「あ、ん、はぁ」等短音必须根据实际交流功能判断：明确属于非词化生理发声时可以输出；可能是应答、音节或词语残片时省略。

未标记区域只是 unresolved，不代表 inside。允许空数组，但不得为了避免空数组而猜测。

输出当前完整 source 的 0-based 局部坐标。区间按时间排序、不得重叠。

只输出 JSON：
{
  "source_id": "...",
  "safe_outside_spans": [
    {
      "start_ts": "00:00.000",
      "end_ts": "00:01.000",
      "category": "breathing|moan|cry|kiss|action|impact|silence|music|ambience|mechanical|other",
      "reason": "可以独立安全删除的简短声学证据"
    }
  ],
  "overall_reason": "简短整体判断"
}
""" + "\n" + TIMESTAMP_PROMPT_CONTRACT_ZH


def _request_prompt(
    row: Mapping[str, Any],
    *,
    pass_name: str,
    feedback: str = "",
) -> str:
    payload: dict[str, Any] = {
        "source_id": str(row["source_id"]),
        **timestamp_request_contract(float(row["duration_s"])),
        "pass": pass_name,
    }
    if feedback:
        payload["previous_validation_error"] = feedback
    return json.dumps(payload, ensure_ascii=False)


def _normalize_evidence_spans(
    parsed: Mapping[str, Any],
    *,
    field: str,
    label: str,
    duration_s: float,
    frame_count: int,
) -> list[dict[str, Any]]:
    if field not in parsed:
        raise ValueError(f"teacher response must contain {field}")
    raw_spans = parsed[field]
    if not isinstance(raw_spans, list):
        raise ValueError(f"teacher response {field} must be an array")
    result: list[dict[str, Any]] = []
    previous_end_frame = 0
    for index, raw in enumerate(raw_spans):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{field} item {index} is not an object")
        start, end = parse_mmss_span(
            raw,
            field=f"{field} item {index}",
            duration_s=duration_s,
        )
        assert start is not None and end is not None
        start_frame = 0 if start <= 0.0 else round(start / FRAME_HOP_S)
        end_frame = (
            frame_count
            if duration_s - end < FRAME_HOP_S
            else round(end / FRAME_HOP_S)
        )
        if (
            start_frame < previous_end_frame
            or start_frame < 0
            or end_frame > frame_count
            or end_frame <= start_frame
        ):
            raise ValueError(
                f"{field} item {index} has invalid/overlapping frame coordinates: "
                f"start_frame={start_frame}, end_frame={end_frame}, "
                f"previous_end_frame={previous_end_frame}, frame_count={frame_count}"
            )
        previous_end_frame = end_frame
        span = {
            "label": label,
            "start_s": round(start_frame * FRAME_HOP_S, 6),
            "end_s": round(end_frame * FRAME_HOP_S, 6),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "reason": str(raw.get("reason") or ""),
        }
        if field == "safe_outside_spans":
            category = str(raw.get("category") or "other")
            if category not in OUTSIDE_CATEGORIES:
                raise ValueError(
                    f"safe_outside_spans item {index} has unsupported category: {category}"
                )
            span["category"] = category
        result.append(span)
    return result


def _runs(
    states: list[str],
    *,
    state: str,
    label: str,
    reason: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    start: int | None = None
    for index, value in enumerate([*states, "__end__"]):
        if value == state and start is None:
            start = index
        elif value != state and start is not None:
            result.append(
                {
                    "label": label,
                    "start_s": round(start * FRAME_HOP_S, 6),
                    "end_s": round(index * FRAME_HOP_S, 6),
                    "start_frame": start,
                    "end_frame": index,
                    "reason": reason,
                }
            )
            start = None
    return result


def merge_dual_evidence(
    *,
    protected_spans: list[Mapping[str, Any]],
    safe_outside_spans: list[Mapping[str, Any]],
    frame_count: int,
) -> dict[str, list[dict[str, Any]]]:
    if frame_count <= 0:
        raise ValueError("dual-evidence merge requires a positive frame_count")
    protected = [False] * frame_count
    removable = [False] * frame_count
    for span in protected_spans:
        start, end = int(span["start_frame"]), int(span["end_frame"])
        protected[start:end] = [True] * (end - start)
    for span in safe_outside_spans:
        start, end = int(span["start_frame"]), int(span["end_frame"])
        removable[start:end] = [True] * (end - start)

    states: list[str] = []
    for has_protect, has_remove in zip(protected, removable):
        if has_protect and not has_remove:
            states.append("inside")
        elif has_remove and not has_protect:
            states.append("outside")
        elif has_protect and has_remove:
            states.append("conflict")
        else:
            states.append("unresolved")

    islands = _runs(
        states,
        state="inside",
        label="inside_candidate",
        reason="Protect-only evidence from the independent dialogue pass",
    )
    final_outside = _runs(
        states,
        state="outside",
        label="outside_candidate",
        reason="Remove-only evidence from the independent safe-deletion pass",
    )
    conflicts = _runs(
        states,
        state="conflict",
        label="unsure",
        reason="Protect and Remove evidence overlap; ignore instead of choosing a side",
    )
    unresolved = _runs(
        states,
        state="unresolved",
        label="unsure",
        reason="Neither independent pass supplied positive evidence",
    )
    unsure = sorted(
        [*conflicts, *unresolved],
        key=lambda span: (int(span["start_frame"]), int(span["end_frame"])),
    )
    return {
        "islands": islands,
        "safe_outside_spans": final_outside,
        "unsure_spans": unsure,
        "conflict_spans": conflicts,
        "unresolved_spans": unresolved,
    }


def _selected_rows(
    rows: list[dict[str, Any]],
    *,
    source_ids: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if source_ids:
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("--source-id values must be unique")
        index = {str(row["source_id"]): row for row in rows}
        missing = [source_id for source_id in source_ids if source_id not in index]
        if missing:
            raise ValueError(f"manifest is missing requested source_id values: {missing}")
        selected = [index[source_id] for source_id in source_ids]
    else:
        selected = rows
    return selected[:limit] if limit > 0 else selected


def _resolve_verified_audio(
    row: Mapping[str, Any],
    *,
    manifest: Path,
    audio_root: Path,
) -> Path:
    try:
        audio = _resolve_audio(str(row["audio"]), manifest=manifest)
    except FileNotFoundError:
        audio = (audio_root / f"{row['source_id']}.wav").resolve()
        if not audio.is_file():
            raise FileNotFoundError(
                f"missing manifest audio and exact source-id fallback: "
                f"{row['audio']!r}, {audio}"
            )
    expected_sha = str(row.get("audio_sha256") or "")
    actual_sha = _sha256(audio)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f"audio identity mismatch for {row['source_id']}: "
            f"expected={expected_sha}, actual={actual_sha}, path={audio}"
        )
    return audio


def _resume_index(
    path: Path,
    *,
    model: str,
    provider_profile: str,
    reasoning_effort: str,
    enable_thinking: bool,
    thinking_budget: int,
    max_tokens: int,
    exclude_reasoning: bool,
    require_provider_parameters: bool,
    response_format: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(row["source_id"]): row
        for row in _rows(path)
        if row.get("schema") == SCHEMA
        and row.get("model") == model
        and row.get("prompt_version") == PROMPT_VERSION
        and row.get("teacher_timestamp_contract_id") == TIMESTAMP_CONTRACT_ID
        and row.get("provider_profile") == provider_profile
        and row.get("reasoning_effort") == reasoning_effort
        and bool(row.get("enable_thinking")) == enable_thinking
        and int(row.get("thinking_budget") or 0) == thinking_budget
        and int(row.get("max_tokens") or 0) == max_tokens
        and bool(row.get("exclude_reasoning")) == exclude_reasoning
        and bool(row.get("require_provider_parameters"))
        == require_provider_parameters
        and row.get("teacher_execution_contract_id")
        == TEACHER_EXECUTION_CONTRACT_ID
        and dict(row.get("response_format") or {}) == dict(response_format)
    }


def _response_reasoning_tokens(raw: Mapping[str, Any]) -> int:
    usage = raw.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return int(details.get("reasoning_tokens") or 0)


def _response_reasoning_evidence(raw: Mapping[str, Any]) -> dict[str, Any]:
    tokens = _response_reasoning_tokens(raw)
    signature_count = int(raw.get("reasoning_signature_count") or 0)
    text_chunk_count = int(raw.get("reasoning_text_chunk_count") or 0)
    character_count = int(raw.get("reasoning_character_count") or 0)
    signature_formats = sorted(
        {str(value) for value in raw.get("reasoning_signature_formats") or ()}
    )
    valid_gemini_signature = (
        signature_count > 0 and "google-gemini-v1" in signature_formats
    )
    visible_reasoning = text_chunk_count > 0 or character_count > 0
    return {
        "reasoning_tokens": tokens,
        "reasoning_signature_count": signature_count,
        "reasoning_signature_formats": signature_formats,
        "reasoning_text_chunk_count": text_chunk_count,
        "reasoning_character_count": character_count,
        # A signature proves that the Gemini thinking transport was active, but
        # it does not prove that this request actually consumed high-effort
        # reasoning. Training-grade evidence therefore requires positive usage
        # tokens or visible reasoning text; signature-only remains diagnostic.
        "reasoning_transport_evidence_present": (
            tokens > 0 or valid_gemini_signature or visible_reasoning
        ),
        "reasoning_evidence_present": tokens > 0 or visible_reasoning,
    }


def _call_pass(
    *,
    row: Mapping[str, Any],
    audio: Path,
    pass_name: str,
    system_prompt: str,
    expected_field: str,
    evidence_label: str,
    duration_s: float,
    frame_count: int,
    model: str,
    api_key: str,
    base_url: str,
    audio_content_mode: str,
    args: argparse.Namespace,
    raw_path: Path,
    request_number: int,
    request_total: int,
    provider_profile: str,
    reasoning_effort: str,
    require_reasoning_evidence: bool,
    require_provider_parameters: bool,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    feedback = ""
    last_error: Exception | None = None
    for attempt in range(1, args.max_attempts + 1):
        parsed: dict[str, Any] | None = None
        raw: dict[str, Any] | None = None
        try:
            print(
                f"dual_evidence_request={request_number}/{request_total} "
                f"pass={pass_name} source_id={row['source_id']} "
                f"attempt={attempt}/{args.max_attempts}",
                flush=True,
            )
            parsed, raw = call_omni(
                audio_path=audio,
                fmt=audio.suffix.lstrip(".") or "wav",
                audio_content_mode=audio_content_mode,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=args.timeout_s,
                store_stream_chunks=False,
                prompt=_request_prompt(row, pass_name=pass_name, feedback=feedback),
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                enable_thinking=args.enable_thinking,
                thinking_budget=args.thinking_budget,
                provider_profile=provider_profile,
                reasoning_effort=reasoning_effort,
                exclude_reasoning=bool(args.exclude_reasoning),
                require_provider_parameters=require_provider_parameters,
                response_format=(
                    {"type": "json_object"}
                    if provider_profile == "gemini"
                    else None
                ),
                print_request=(
                    args.print_request and request_number <= 2 and attempt == 1
                ),
            )
            if str(parsed.get("source_id") or "") != str(row["source_id"]):
                raise ValueError(
                    f"{pass_name} response source_id mismatch: "
                    f"{parsed.get('source_id')!r} != {row['source_id']!r}"
                )
            if expected_field not in parsed:
                raise ValueError(f"{pass_name} response must contain {expected_field}")
            normalized = _normalize_evidence_spans(
                parsed,
                field=expected_field,
                label=evidence_label,
                duration_s=duration_s,
                frame_count=frame_count,
            )
            reasoning_evidence = _response_reasoning_evidence(raw)
            if (
                require_reasoning_evidence
                and not reasoning_evidence["reasoning_evidence_present"]
            ):
                raise ValueError(
                    "training-grade OpenRouter reasoning requires positive "
                    "reasoning tokens or visible reasoning_details; a "
                    "google-gemini-v1 thought signature alone is insufficient"
                )
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "source_id": str(row["source_id"]),
                            "pass": pass_name,
                            "attempt": attempt,
                            "parsed": parsed,
                            "response": raw,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            return parsed, raw, normalized
        except Exception as error:  # noqa: BLE001
            last_error = error
            feedback = str(error)
            print(
                f"dual_evidence_error pass={pass_name} source_id={row['source_id']} "
                f"attempt={attempt}/{args.max_attempts} "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "source_id": str(row["source_id"]),
                            "pass": pass_name,
                            "attempt": attempt,
                            "error": repr(error),
                            "parsed": parsed,
                            "response": raw,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            if attempt < args.max_attempts:
                time.sleep(min(8.0, float(attempt)))
    assert last_error is not None
    raise last_error


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.manifest).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    audio_root = Path(args.audio_root)
    if not audio_root.is_absolute():
        audio_root = PROJECT_ROOT / audio_root
    audio_root = audio_root.resolve()
    env_file = Path.home() / ".config" / "omni" / str(args.env_file)
    load_env_file(env_file)
    _model_env, model_from_env = first_env_value(tuple(args.model_env.split(",")))
    model = args.model or model_from_env
    _key_env, api_key = first_env_value(tuple(args.api_key_env.split(",")))
    _url_env, raw_base_url = first_env_value(tuple(args.base_url_env.split(",")))
    base_url = normalize_openai_compat_base_url(raw_base_url)
    if not model or not api_key:
        raise RuntimeError("Omni model and API key are required")
    profile_name = env_file.name.lower()
    audio_content_mode = audio_content_mode_for_profile(profile_name)
    reasoning_effort = (
        str(args.reasoning_effort).lower() if args.enable_thinking else "none"
    ) if profile_name == "gemini" else ""
    effective_thinking_budget = (
        int(args.thinking_budget) if profile_name == "qwen" else 0
    )
    effective_max_tokens = int(args.max_tokens) if args.max_tokens > 0 else (
        8192 if profile_name == "gemini" else 2048
    )
    exclude_reasoning = bool(args.exclude_reasoning)
    require_provider_parameters = bool(
        profile_name == "gemini" and "openrouter.ai" in base_url.lower()
    )
    reasoning_metadata: dict[str, Any] = {
        "provider_profile": profile_name,
        "enable_thinking": bool(args.enable_thinking),
        "reasoning_transport": (
            "openrouter_reasoning_effort_to_google_thinking_level"
            if profile_name == "gemini"
            else "qwen_enable_thinking_budget"
        ),
        "max_tokens": effective_max_tokens,
        # Default false: OpenRouter returns usage.reasoning_tokens more reliably
        # when reasoning text is not excluded; exclude=true is optional.
        "exclude_reasoning": exclude_reasoning,
        "require_provider_parameters": require_provider_parameters,
        "teacher_execution_contract_id": TEACHER_EXECUTION_CONTRACT_ID,
        "response_format": (
            {"type": "json_object"} if profile_name == "gemini" else {}
        ),
        "omitted_sampling_parameters": ["temperature", "top_p", "top_k"],
    }
    if profile_name == "gemini":
        reasoning_metadata.update(
            {
                "reasoning_effort": reasoning_effort,
                "gemini_thinking_level": reasoning_effort,
            }
        )
    else:
        reasoning_metadata["thinking_budget"] = effective_thinking_budget
    require_reasoning_evidence = bool(
        args.require_reasoning_evidence
        and profile_name == "gemini"
        and args.enable_thinking
    )

    rows = _selected_rows(
        _rows(manifest),
        source_ids=list(args.source_id),
        limit=args.limit,
    )
    if args.preview_requests:
        if not rows:
            raise ValueError("request preview requires at least one source")
        row = rows[0]
        duration_s = float(row["duration_s"])
        audio = _resolve_verified_audio(
            row,
            manifest=manifest,
            audio_root=audio_root,
        )
        previews: list[dict[str, Any]] = []
        for pass_name, system_prompt in (
            ("protect", PROTECT_SYSTEM_PROMPT),
            ("remove", REMOVE_SYSTEM_PROMPT),
        ):
            request_body, extra_body = build_omni_request_body(
                audio_path=audio,
                fmt=audio.suffix.lstrip(".") or "wav",
                audio_content_mode=audio_content_mode,
                model=model,
                prompt=_request_prompt(row, pass_name=pass_name),
                system_prompt=system_prompt,
                max_tokens=effective_max_tokens,
                enable_thinking=args.enable_thinking,
                thinking_budget=effective_thinking_budget,
                provider_profile=profile_name,
                reasoning_effort=reasoning_effort,
                exclude_reasoning=exclude_reasoning,
                require_provider_parameters=require_provider_parameters,
                response_format=(
                    {"type": "json_object"}
                    if profile_name == "gemini"
                    else None
                ),
            )
            previews.append(
                {
                    "pass": pass_name,
                    **redact_omni_request_preview(
                        request_body=request_body,
                        extra_body=extra_body,
                        provider_profile=profile_name,
                        base_url=base_url,
                    ),
                }
            )
        preview_payload = {
            "schema": "candidate_island_scorer_v11_request_preview_v1",
            "source_id": str(row["source_id"]),
            "duration_s": duration_s,
            "audio_sha256": _sha256(audio),
            "audio_content_mode": audio_content_mode,
            **reasoning_metadata,
            "requests": previews,
            "network_request_sent": False,
        }
        preview_path = output / "request_preview.json"
        preview_path.write_text(
            json.dumps(
                preview_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {**preview_payload, "request_preview": str(preview_path)}
    labels_path = output / "preaudit.jsonl"
    raw_path = output / "raw_responses.jsonl"
    if args.retry_failed_closed and labels_path.is_file():
        retained = [
            row
            for row in _rows(labels_path)
            if not bool(row.get("teacher_failed_closed"))
        ]
        labels_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in retained
            ),
            encoding="utf-8",
        )
    existing = _resume_index(
        labels_path,
        model=model,
        provider_profile=profile_name,
        reasoning_effort=reasoning_effort,
        enable_thinking=bool(args.enable_thinking),
        thinking_budget=effective_thinking_budget,
        max_tokens=effective_max_tokens,
        exclude_reasoning=exclude_reasoning,
        require_provider_parameters=require_provider_parameters,
        response_format=(
            {"type": "json_object"} if profile_name == "gemini" else {}
        ),
    )
    pending = [row for row in rows if str(row["source_id"]) not in existing]
    progress_path = output / "progress.json"
    request_total = len(pending) * 2
    request_number = 0
    started = time.perf_counter()
    _write_progress(
        progress_path,
        {
            "schema": "candidate_island_scorer_v11_dual_evidence_progress_v1",
            "status": "running",
            "model": model,
            **reasoning_metadata,
            "require_reasoning_evidence": require_reasoning_evidence,
            "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
            "completed": len(existing),
            "total": len(rows),
            "pending": len(pending),
            "request_completed": 0,
            "request_total": request_total,
        },
    )

    for row_index, row in enumerate(pending, start=1):
        source_id = str(row["source_id"])
        duration_s = float(row["duration_s"])
        frame_count = int(row.get("frame_count") or round(duration_s / FRAME_HOP_S))
        audio = _resolve_verified_audio(
            row,
            manifest=manifest,
            audio_root=audio_root,
        )
        protect_parsed: dict[str, Any] | None = None
        remove_parsed: dict[str, Any] | None = None
        failure: Exception | None = None
        try:
            request_number += 1
            protect_parsed, protect_raw, protected_evidence = _call_pass(
                row=row,
                audio=audio,
                pass_name="protect",
                system_prompt=PROTECT_SYSTEM_PROMPT,
                expected_field="protected_spans",
                evidence_label="inside_candidate",
                duration_s=duration_s,
                frame_count=frame_count,
                model=model,
                api_key=api_key,
                base_url=base_url,
                audio_content_mode=audio_content_mode,
                args=args,
                raw_path=raw_path,
                request_number=request_number,
                request_total=request_total,
                provider_profile=profile_name,
                reasoning_effort=reasoning_effort,
                require_reasoning_evidence=require_reasoning_evidence,
                require_provider_parameters=require_provider_parameters,
                max_tokens=effective_max_tokens,
            )
            if args.request_interval_s > 0:
                time.sleep(args.request_interval_s)
            request_number += 1
            remove_parsed, remove_raw, remove_evidence = _call_pass(
                row=row,
                audio=audio,
                pass_name="remove",
                system_prompt=REMOVE_SYSTEM_PROMPT,
                expected_field="safe_outside_spans",
                evidence_label="outside_candidate",
                duration_s=duration_s,
                frame_count=frame_count,
                model=model,
                api_key=api_key,
                base_url=base_url,
                audio_content_mode=audio_content_mode,
                args=args,
                raw_path=raw_path,
                request_number=request_number,
                request_total=request_total,
                provider_profile=profile_name,
                reasoning_effort=reasoning_effort,
                require_reasoning_evidence=require_reasoning_evidence,
                require_provider_parameters=require_provider_parameters,
                max_tokens=effective_max_tokens,
            )
            merged = merge_dual_evidence(
                protected_spans=protected_evidence,
                safe_outside_spans=remove_evidence,
                frame_count=frame_count,
            )
            label = {
                "schema": SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "prompt_profile": PROMPT_PROFILE,
                "prompt_version": PROMPT_VERSION,
                "protect_prompt_version": PROTECT_PROMPT_VERSION,
                "remove_prompt_version": REMOVE_PROMPT_VERSION,
                "source_id": source_id,
                "partition": str(row.get("partition") or ""),
                "frame_count": frame_count,
                "frame_hop_s": FRAME_HOP_S,
                "duration_s": duration_s,
                "audio": str(audio),
                "manifest_audio": str(row["audio"]),
                "audio_sha256": _sha256(audio),
                "model": model,
                "env_file_name": profile_name,
                **reasoning_metadata,
                "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
                "protect_reasoning": _response_reasoning_evidence(protect_raw),
                "remove_reasoning": _response_reasoning_evidence(remove_raw),
                "base_url_host": (
                    base_url.split("/", 3)[2] if "://" in base_url else base_url
                ),
                "protect_overall_reason": str(
                    protect_parsed.get("overall_reason") or ""
                ),
                "remove_overall_reason": str(
                    remove_parsed.get("overall_reason") or ""
                ),
                "protected_evidence_spans": protected_evidence,
                "remove_evidence_spans": remove_evidence,
                **merged,
                "merge_contract": (
                    "protect_only=inside; remove_only=outside; "
                    "overlap_or_neither=unsure"
                ),
                "unmarked_semantics": "unsure_ignore_minus_100",
                "reviewed_full_source": False,
                "preaudit_provenance": f"omni:{model}:independent_dual_evidence",
                "human_review_required": True,
                "training_manifest_allowed": False,
            }
        except Exception as error:  # noqa: BLE001
            failure = error
            label = {
                "schema": SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "prompt_profile": PROMPT_PROFILE,
                "prompt_version": PROMPT_VERSION,
                "protect_prompt_version": PROTECT_PROMPT_VERSION,
                "remove_prompt_version": REMOVE_PROMPT_VERSION,
                "source_id": source_id,
                "partition": str(row.get("partition") or ""),
                "frame_count": frame_count,
                "frame_hop_s": FRAME_HOP_S,
                "duration_s": duration_s,
                "audio": str(audio),
                "manifest_audio": str(row["audio"]),
                "audio_sha256": _sha256(audio),
                "model": model,
                "env_file_name": profile_name,
                **reasoning_metadata,
                "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
                "protect_reasoning": {
                    "reasoning_tokens": 0,
                    "reasoning_signature_count": 0,
                    "reasoning_signature_formats": [],
                    "reasoning_evidence_present": False,
                    "reasoning_transport_evidence_present": False,
                },
                "remove_reasoning": {
                    "reasoning_tokens": 0,
                    "reasoning_signature_count": 0,
                    "reasoning_signature_formats": [],
                    "reasoning_evidence_present": False,
                    "reasoning_transport_evidence_present": False,
                },
                "protected_evidence_spans": [],
                "remove_evidence_spans": [],
                "islands": [],
                "safe_outside_spans": [],
                "conflict_spans": [],
                "unresolved_spans": [
                    {
                        "label": "unsure",
                        "start_s": 0.0,
                        "end_s": round(frame_count * FRAME_HOP_S, 6),
                        "start_frame": 0,
                        "end_frame": frame_count,
                        "reason": (
                            "dual-evidence request failed; exclude entire source "
                            "from supervision"
                        ),
                    }
                ],
                "unsure_spans": [
                    {
                        "label": "unsure",
                        "start_s": 0.0,
                        "end_s": round(frame_count * FRAME_HOP_S, 6),
                        "start_frame": 0,
                        "end_frame": frame_count,
                        "reason": (
                            f"teacher failed closed: {type(error).__name__}: {error}"
                        ),
                    }
                ],
                "teacher_failed_closed": True,
                "failure": f"{type(error).__name__}: {error}",
                "reviewed_full_source": False,
                "preaudit_provenance": f"omni:{model}:dual_evidence_failure",
                "human_review_required": True,
                "training_manifest_allowed": False,
            }

        with labels_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
        existing[source_id] = label
        inside_frames = sum(
            int(span["end_frame"]) - int(span["start_frame"])
            for span in label["islands"]
        )
        outside_frames = sum(
            int(span["end_frame"]) - int(span["start_frame"])
            for span in label["safe_outside_spans"]
        )
        unsure_frames = frame_count - inside_frames - outside_frames
        elapsed = time.perf_counter() - started
        completed = len(existing)
        run_completed = row_index
        eta = max(
            0.0,
            (len(pending) - run_completed)
            * elapsed
            / max(run_completed, 1),
        )
        print(
            f"dual_evidence_source={run_completed}/{len(pending)} "
            f"source_id={source_id} inside={inside_frames/frame_count:.3f} "
            f"outside={outside_frames/frame_count:.3f} "
            f"unsure={unsure_frames/frame_count:.3f} "
            f"conflicts={len(label.get('conflict_spans') or ())} "
            f"failed_closed={failure is not None} eta_s={eta:.0f}",
            flush=True,
        )
        _write_progress(
            progress_path,
            {
                "schema": "candidate_island_scorer_v11_dual_evidence_progress_v1",
                "status": "running",
                "provider_profile": profile_name,
                "model": model,
                "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
                "completed": completed,
                "total": len(rows),
                "pending": len(rows) - completed,
                "request_completed": request_number,
                "request_total": request_total,
                "last_source_id": source_id,
                "last_inside_ratio": inside_frames / frame_count,
                "last_outside_ratio": outside_frames / frame_count,
                "last_unsure_ratio": unsure_frames / frame_count,
                "elapsed_s": round(elapsed, 3),
                "eta_s": round(eta, 3),
            },
        )
        if row_index < len(pending) and args.request_interval_s > 0:
            time.sleep(args.request_interval_s)

    result_rows = [existing[str(row["source_id"])] for row in rows]
    totals = {
        "frame_count": sum(int(row["frame_count"]) for row in result_rows),
        "inside_frames": sum(
            int(span["end_frame"]) - int(span["start_frame"])
            for row in result_rows
            for span in row.get("islands") or ()
        ),
        "outside_frames": sum(
            int(span["end_frame"]) - int(span["start_frame"])
            for row in result_rows
            for span in row.get("safe_outside_spans") or ()
        ),
        "conflict_frames": sum(
            int(span["end_frame"]) - int(span["start_frame"])
            for row in result_rows
            for span in row.get("conflict_spans") or ()
        ),
        "failed_closed_count": sum(
            bool(row.get("teacher_failed_closed")) for row in result_rows
        ),
        "protect_reasoning_tokens": sum(
            int((row.get("protect_reasoning") or {}).get("reasoning_tokens") or 0)
            for row in result_rows
        ),
        "remove_reasoning_tokens": sum(
            int((row.get("remove_reasoning") or {}).get("reasoning_tokens") or 0)
            for row in result_rows
        ),
        "protect_reasoning_signature_count": sum(
            int(
                (row.get("protect_reasoning") or {}).get(
                    "reasoning_signature_count"
                )
                or 0
            )
            for row in result_rows
        ),
        "remove_reasoning_signature_count": sum(
            int(
                (row.get("remove_reasoning") or {}).get(
                    "reasoning_signature_count"
                )
                or 0
            )
            for row in result_rows
        ),
        "protect_reasoning_evidence_count": sum(
            bool(
                (row.get("protect_reasoning") or {}).get(
                    "reasoning_evidence_present"
                )
            )
            for row in result_rows
        ),
        "remove_reasoning_evidence_count": sum(
            bool(
                (row.get("remove_reasoning") or {}).get(
                    "reasoning_evidence_present"
                )
            )
            for row in result_rows
        ),
    }
    totals["unsure_frames"] = (
        totals["frame_count"] - totals["inside_frames"] - totals["outside_frames"]
    )
    denominator = max(totals["frame_count"], 1)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "prompt_profile": PROMPT_PROFILE,
        "prompt_version": PROMPT_VERSION,
        "protect_prompt_version": PROTECT_PROMPT_VERSION,
        "remove_prompt_version": REMOVE_PROMPT_VERSION,
        "model": model,
        "env_file_name": profile_name,
        **reasoning_metadata,
        "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
        "require_reasoning_evidence": require_reasoning_evidence,
        "reasoning_contract_satisfied": (
            not require_reasoning_evidence
            or (
                totals["failed_closed_count"] == 0
                and totals["protect_reasoning_evidence_count"] == len(result_rows)
                and totals["remove_reasoning_evidence_count"] == len(result_rows)
            )
        ),
        "audio_content_mode": audio_content_mode,
        "base_url_host": (
            base_url.split("/", 3)[2] if "://" in base_url else base_url
        ),
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "source_ids": [str(row["source_id"]) for row in rows],
        "source_count": len(result_rows),
        **totals,
        "inside_ratio": totals["inside_frames"] / denominator,
        "outside_ratio": totals["outside_frames"] / denominator,
        "unsure_ratio": totals["unsure_frames"] / denominator,
        "conflict_ratio": totals["conflict_frames"] / denominator,
        "manual_review_required": True,
        "training_manifest_allowed": False,
        "labels": str(labels_path),
        "raw_responses": str(raw_path),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_progress(
        progress_path,
        {
            "schema": "candidate_island_scorer_v11_dual_evidence_progress_v1",
            "status": "completed",
            "model": model,
            **reasoning_metadata,
            "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
            "require_reasoning_evidence": require_reasoning_evidence,
            "completed": len(result_rows),
            "total": len(rows),
            "pending": 0,
            "request_completed": request_number,
            "request_total": request_total,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "summary": str(summary_path),
        },
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--audio-root",
        default="datasets/train/omni-joint-boundary-preasr-v2/audio_wav",
        help=(
            "Exact source-id WAV fallback used only when manifest-local audio is "
            "absent; SHA-256 must still match the manifest."
        ),
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Repeat to run a fixed source subset in the supplied order.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--env-file",
        default="gemini",
        choices=("qwen", "gemini"),
    )
    parser.add_argument("--api-key-env", default=",".join(DEFAULT_API_KEY_ENV_CANDIDATES))
    parser.add_argument("--model-env", default="OMNI_MODEL,QWEN_OMNI_MODEL")
    parser.add_argument("--base-url-env", default=",".join(DEFAULT_BASE_URL_ENV_CANDIDATES))
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help=(
            "0 selects the provider default used by this tool: Gemini=8192, "
            "Qwen=2048."
        ),
    )
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument(
        "--reasoning-effort",
        choices=GEMINI_THINKING_LEVELS,
        default="medium",
        help="Gemini/OpenRouter only; Qwen uses --thinking-budget instead.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--exclude-reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "OpenRouter only: if true, ask the provider to hide reasoning text "
            "(reasoning.exclude=true). Default false so usage.reasoning_tokens "
            "and reasoning_details stay visible for the Gemini thinking gate."
        ),
    )
    parser.add_argument(
        "--require-reasoning-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For Gemini with thinking enabled, require positive reasoning tokens, "
            "visible reasoning_details, or a valid google-gemini-v1 thought signature."
        ),
    )
    parser.add_argument(
        "--preview-requests",
        action="store_true",
        help="Write redacted Protect/Remove request bodies without network calls.",
    )
    parser.add_argument(
        "--print-request",
        action="store_true",
        help="Print the first Protect/Remove request bodies with audio redacted.",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-interval-s", type=float, default=0.5)
    parser.add_argument(
        "--retry-failed-closed",
        action="store_true",
        help="Drop only prior failed-closed rows from the resumable label file and retry them.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
