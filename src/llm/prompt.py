import hashlib
import json
import os
import re
import sys


PROMPT_VERSION = "v2.5"
_LEADING_SPEAKER_RE = re.compile(
    r"^\s*(?:男|女|男性|女性|男优|女优|スタッフ|撮影者|カメラマン|"
    r"[A-Za-z][A-Za-z ._-]{0,20})\s*[：:]\s*"
)
_JSON_OUTPUT_LABEL = "LLM JSON output"
COMPACT_SYSTEM_PROMPT = os.getenv("COMPACT_SYSTEM_PROMPT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _translator_global(name: str, default):
    module = sys.modules.get("llm.translator")
    if module is None:
        return default
    return getattr(module, name, default)


def _compute_prompt_signature(
    extra_glossary: str = "",
    *,
    glossary: str = "",
    target_lang: str = "简体中文",
    character_reference: str = "",
) -> str:
    prompt_version = _translator_global("PROMPT_VERSION", PROMPT_VERSION)
    model_name_default = _translator_global("LLM_MODEL_NAME", "")
    model_name = os.getenv("LLM_MODEL_NAME", model_name_default).strip()
    compact = "1" if _translator_global("COMPACT_SYSTEM_PROMPT", COMPACT_SYSTEM_PROMPT) else "0"
    payload = (
        f"{prompt_version}\n{target_lang.strip()}\n{glossary.strip()}\n"
        f"{extra_glossary.strip()}\n{(character_reference or '').strip()}\n"
        f"{model_name}\ncompact={compact}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _translation_cache_key(
    batch_index: int,
    batch_segments: list[dict],
    *,
    extra_glossary: str = "",
    glossary: str = "",
    target_lang: str = "简体中文",
    character_reference: str = "",
) -> str:
    source_text = "|".join(
        str(seg.get("ja_text") or seg.get("text") or seg.get("ja") or "")
        for seg in batch_segments
    )
    source_sig = hashlib.sha1(source_text.encode("utf-8")).hexdigest()[:8]
    prompt_sig = _compute_prompt_signature(
        extra_glossary,
        glossary=glossary,
        target_lang=target_lang,
        character_reference=character_reference,
    )
    return f"{prompt_sig}::{batch_index}::{source_sig}"


def _normalize_source_text(text: str) -> str:
    cleaned = re.sub(r"[ \t]+", " ", (text or "").strip())
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())
    cleaned = re.sub(r"(.)\1{4,}", r"\1\1\1", cleaned)
    return cleaned.strip()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _serialize_segments(
    segments: list[dict],
    *,
    start_index: int = 0,
    explicit_ids: list[int] | None = None,
    compact: bool = False,
) -> str:
    payload = []
    for idx, seg in enumerate(segments):
        start = _safe_float(seg.get("start"))
        end = _safe_float(seg.get("end"))
        ja_text = _normalize_source_text(seg.get("text", ""))
        gender = seg.get("gender")
        if gender == "M":
            ja_text = f"[M]{ja_text}"
        elif gender == "F":
            ja_text = f"[F]{ja_text}"
        item_id = explicit_ids[idx] if explicit_ids is not None else start_index + idx
        payload.append(
            {
                "id": item_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration_sec": round(max(0.0, end - start), 3),
                "ja": ja_text,
            }
        )
    if compact:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_full_segments_summary(segments: list[dict], *, limit_chars: int = 1800) -> str:
    lines = []
    for idx, seg in enumerate(segments):
        start = _safe_float(seg.get("start"))
        text = _normalize_source_text(seg.get("text", ""))
        if not text:
            continue
        lines.append(f"{idx}: {start:.2f}s {text}")
    summary = "\n".join(lines)
    if len(summary) > limit_chars:
        return summary[:limit_chars].rstrip() + "\n..."
    return summary


_SYSTEM_PROMPT_FULL = (
    "你是专业的日语成人视频字幕翻译，目标语言是{target_lang}。\n"
    "本片译文要体现主动撩拨的语气，露骨直接；遇到性器官性行为时优先用粗口口语，不用阴茎性交插入等书面语。\n"
    "性器官相关词汇优先统一使用：男性器官用“肉棒”，女性器官用“小穴”；避免其他漂移译法。\n"
    "你会收到全片字幕 JSON，以及本次需要翻译的 requested_ids；必须基于全局上下文完成指定 id 翻译。\n"
    "本视频出场人物全名参考：【{character_reference}】。\n"
    "要求：\n"
    "1. 翻译要自然、口语化、适合字幕阅读，避免书面腔。\n"
    "2. 成人、暧昧、调情、呻吟、下流语气要保留原本强度，不要净化、弱化或说教。\n"
    "3. 人名不要翻译成中文；如果原文出现人物姓名，直接输出罗马音，格式用 Title Case，并用空格分隔名和姓，例如 Aya Onami。原文是汉字姓名时也必须按日语读音罗马音化，不要输出中文汉字或中文读法。\n"
    "4. {name_boundary}\n"
    "5. {name_homophone}\n"
    "6. 允许根据前后文修正明显 ASR 误听，但不要编造未出现的信息。\n"
    "7. 输入中部分日文前可能带 [M]（男声）或 [F]（女声）声学标签，请利用此信息理解对话切换、调整语气和人称（例如男声用更直白的命令式，女声用更贴合的女性口吻）。**译文中绝对不要保留或输出任何说话人前缀**——[M]/[F] 仅作为输入提示，输出时只给纯净的中文翻译。\n"
    "8. 每条输入必须单独翻译，不能合并、拆分、漏译、调换顺序。\n"
    "9. 输出尽量短，贴近屏幕阅读节奏；短促呻吟和语气词也要简短自然。\n"
    "10. 如果一行里大部分是呻吟、喘息、重复语气词，只保留清晰语义核心，重复部分可以压缩；映射参考：あんっ/はあん 译 啊嗯啊，気持ちいい 译 好舒服要爽死了，イッちゃう/イク 译 要去了要射了，避免感觉很舒服即将达到高潮等翻译腔。\n"
    "11. 结构化 JSON 输出要求 prompt 明确包含 json 字样；最终只输出合法 JSON 对象。\n"
    '12. 你必须只输出 JSON：{{"translations":[{{"id":0,"text":"..."}}]}}，条数必须严格匹配本次任务要求。\n'
    "13. 最终 content 不能为空；即使开启思考模式，也必须把完整 JSON 对象写进最终 content。\n"
    "14. 不要输出 Markdown，不要解释，不要额外字段；思考过程不要写进最终 content。\n\n"
    "EXAMPLE JSON OUTPUT:\n"
    '{{"translations":[{{"id":0,"text":"第一句中文翻译"}},{{"id":1,"text":"第二句中文翻译"}}]}}'
)

_SYSTEM_PROMPT_COMPACT = (
    "你是日语成人视频字幕译者，目标语言是{target_lang}。保持口语、露骨语气和人名罗马音；"
    "汉字人名也要按日语读音罗马音化，不输出中文汉字名；"
    "性器官相关词汇优先统一使用肉棒/小穴；可修正明显ASR误听；每条独立翻译，不合并、不漏译、不调序。"
    '只输出合法 JSON：{{"translations":[{{"id":0,"text":"..."}}]}}。'
)


def _build_system_prompt(
    expected_count: int,
    character_reference: str,
    *,
    target_lang: str,
    glossary: str,
    compact: bool = False,
    extra_glossary: str = "",
    full_template: str | None = None,
    compact_template: str | None = None,
) -> str:
    del expected_count
    name_guidance = _build_character_name_guidance(character_reference)
    template = (
        (compact_template or _SYSTEM_PROMPT_COMPACT)
        if compact
        else (full_template or _SYSTEM_PROMPT_FULL)
    )
    effective_target_lang = (target_lang or "简体中文").strip() or "简体中文"
    prompt = template.format(
        target_lang=effective_target_lang,
        character_reference=character_reference,
        name_boundary=name_guidance["boundary"],
        name_homophone=name_guidance["homophone"],
    )
    effective_glossary = (glossary or "").strip()
    if effective_glossary:
        prompt += f"\n\n以下词汇表必须严格遵守，不得自行创造译名：\n{effective_glossary}"
    extra_glossary = (extra_glossary or "").strip()
    if extra_glossary:
        prompt += (
            "\n\n<glossary>\n"
            "本片已确定译法（必须沿用）：\n"
            f"{extra_glossary}\n"
            "</glossary>"
        )
    return prompt


def _build_translation_messages(
    source_payload: str,
    expected_count: int,
    compact_system_prompt: bool = False,
    extra_glossary: str = "",
    target_lang: str = "简体中文",
    glossary: str = "",
    character_reference: str | None = None,
    system_prompt: str | None = None,
) -> list[dict]:
    if system_prompt is None:
        system_prompt = _build_system_prompt(
            expected_count,
            (character_reference or "").strip(),
            target_lang=target_lang,
            glossary=glossary,
            compact=compact_system_prompt,
            extra_glossary=extra_glossary,
        )

    user_parts = [
        "【任务】把下面 JSON 数组里的日文字幕逐条翻译成中文字幕。",
        "每个元素的 `id` 必须原样保留，`text` 只能是翻译结果本身。",
        f"必须恰好返回 {expected_count} 条翻译，不要多也不要少。",
        f"【待翻译字幕 JSON】\n{source_payload}",
    ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _format_requested_ids(requested_ids: list[int]) -> str:
    return json.dumps([int(idx) for idx in requested_ids], ensure_ascii=False)


def _build_requested_ids_task(
    requested_ids: list[int],
    *,
    extra_glossary: str = "",
    warmup: bool = False,
) -> str:
    expected_count = len(requested_ids)
    task = [
        "【本次任务】",
        f"requested_ids = {_format_requested_ids(requested_ids)}",
    ]
    if warmup:
        task.extend(
            [
                "这是缓存预热请求，不要翻译任何字幕。",
                '只输出合法 JSON：{"translations":[]}',
            ]
        )
    else:
        task.extend(
            [
                "只翻译 requested_ids 中列出的字幕。",
                f"必须只返回这些 id，恰好 {expected_count} 条，不要返回其他 id。",
                '每个 `text` 只能是翻译结果本身，输出 JSON：{"translations":[{"id":0,"text":"..."}]}',
            ]
        )
    if extra_glossary.strip():
        task.append("注意：必须严格使用 System Prompt 中 <glossary> 标签内的术语表翻译。")
    return "\n".join(task)


def _build_batch_messages(
    batch_segments: list[dict],
    full_segments_summary: str | list[dict],
    batch_offset: int,
    character_reference: str,
    expected_count: int,
    batch_index: int = 0,
    extra_glossary: str = "",
    target_lang: str = "简体中文",
    glossary: str = "",
    source_payload_override: str | None = None,
    full_source_payload: str | None = None,
    requested_ids: list[int] | None = None,
    warmup: bool = False,
    compact_system_prompt_enabled: bool = False,
) -> list[dict]:
    if requested_ids is None:
        requested_ids = list(range(batch_offset, batch_offset + expected_count))

    source_payload = source_payload_override or _serialize_segments(
        batch_segments,
        start_index=batch_offset,
    )
    messages = _build_translation_messages(
        source_payload=source_payload,
        expected_count=expected_count,
        compact_system_prompt=(
            compact_system_prompt_enabled and batch_index > 0 and full_source_payload is None
        ),
        extra_glossary=extra_glossary,
        target_lang=target_lang,
        glossary=glossary,
        character_reference=character_reference,
    )
    if full_source_payload is not None:
        messages[1]["content"] = "\n\n".join(
            [
                "【全片字幕 JSON】",
                full_source_payload,
                _build_requested_ids_task(
                    requested_ids,
                    extra_glossary=extra_glossary,
                    warmup=warmup,
                ),
            ]
        )
        return messages

    if isinstance(full_segments_summary, str):
        summary = full_segments_summary
    else:
        summary = _build_full_segments_summary(full_segments_summary)

    messages[0]["content"] = (
        messages[0]["content"]
        + "\n\n全片字幕概览（仅作上下文连贯参考，不要翻译，原 id 不在本批的不要返回）：\n"
        + summary
    )
    messages[1]["content"] = (
        "【任务】把下面当前批次 JSON 数组里的日文字幕逐条翻译成中文字幕。\n"
        "每个元素的 `id` 是全片全局 id，必须原样保留；只返回本批 id，不要返回概览里的其他 id。\n"
        "每个 `text` 只能是翻译结果本身。\n"
        f"【当前批次字幕 JSON】\n{source_payload}\n\n"
        + _build_requested_ids_task(
            requested_ids,
            extra_glossary=extra_glossary,
            warmup=warmup,
        )
    )
    return messages


def _build_character_name_guidance(character_reference: str) -> dict[str, str]:
    normalized = (character_reference or "").strip()
    return {
        "boundary": (
            f"智能拆解边界：请自动根据日本姓名习惯识别人物全名的姓氏和名字边界。"
            f"本片参考名“{normalized}”只作为人名参考；剧中如果只称呼姓氏或名字，只翻译实际出现的部分，"
            "不要强行补全全名。人名用罗马音输出。"
        ),
        "homophone": (
            "ASR 同音纠错：源日文文本由语音识别生成，可能把人物姓名识别成同音字、谐音字或近音词。"
            f"只有当原文称呼的读音明显接近参考名“{normalized}”时，才纠正为该参考名对应的罗马音；"
            "不要为了统一人物而把不同汉字姓氏或不同读音的称呼强行合并。"
            "除非同一局部上下文有明确证据表明 ASR 把同一个名字写错。"
            "纠错只限明显人名称呼，不要把普通名词误改成人名。"
        ),
    }
