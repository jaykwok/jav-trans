import json
import re

from llm.glossary import normalize_glossary_text
from llm.context import SourceContext


# v4.0: semantic fidelity precedes display compression; each request carries
# nearby source evidence and confirmed continuation units. Repeated dialogue
# is no longer a mandatory glossary. Invalidate entries made under v3.4.
PROMPT_VERSION = "v4.0"
_LEADING_ROLE_LABEL_RE = re.compile(
    r"^\s*(?:男|女|男性|女性|男优|女优|スタッフ|撮影者|カメラマン|"
    r"[A-Za-z][A-Za-z ._-]{0,20})\s*[：:]\s*"
)
_JSON_OUTPUT_LABEL = "LLM JSON output"


def _normalize_source_text(text: str) -> str:
    cleaned = re.sub(r"[ \t]+", " ", (text or "").strip())
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())
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
        item_id = explicit_ids[idx] if explicit_ids is not None else start_index + idx
        item = {
            "id": item_id,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration_sec": round(max(0.0, end - start), 3),
            "ja": ja_text,
        }
        # Emitted only when true. Most cues are whole utterances, and a pair of
        # `false` on every line would cost prompt tokens on every batch of every
        # film to say nothing.
        if seg.get("continues_from_previous"):
            item["cont_prev"] = True
        if seg.get("continues_into_next"):
            item["cont_next"] = True
        payload.append(item)
    if compact:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(payload, ensure_ascii=False, indent=2)


_SYSTEM_PROMPT_FULL = (
    "你是专业日语字幕译者，目标语言是{target_lang}。\n"
    "优先级：原文含义完整准确 > 用户词汇表 > 原句语气与口吻 > 简洁和显示格式。\n"
    "保留否定、条件、程度、时间、数量以及谁对谁做什么；不为了缩短字幕省略关键信息。"
    "上下文只用于消歧与衔接，不补写原文没有的人物、动作、关系或确定性，不替 ASR 猜改原文。\n"
    "语域和情绪强度跟随原句：露骨、粗俗或亲昵的原文如实表达，不净化不说教；"
    "也不因为视频题材或场景自行加强语气、添加修饰或把普通对白色情化。\n"
    "你会收到全片参考、聚焦上下文和本次 requested_ids。只输出 requested_ids 对应译文。\n"
    "本视频出场人物全名参考：【{character_reference}】。\n"
    "要求：\n"
    "1. 使用自然口语。简洁以不丢信息为前提；16 字是排版的每行目标，不是每条译文的语义上限。"
    "标点、全半角与两行布局由后续排版统一，不要为满足格式删词。\n"
    "2. 保留疑问、请求、命令和陈述的区别；不要凭空补出日语省略的主语或宾语。\n"
    "3. 真实重复可以用自然中文表达其强调、催促或犹豫，不一律只保留一次；"
    "只在有依据判断为机器循环时概括，不增加原文没有的重复。\n"
    "4. 人名在读音有依据时按日语读音罗马音化，用 Title Case，只保留原文实际称呼的部分；"
    "没有读音依据时保留原名，不虚构姓名读音，不把普通名词认成人名。\n"
    "5. {name_boundary}\n"
    "6. {name_homophone}\n"
    "7. 相同日文短句在不同语境下可以有不同译法；已有译文是参考，不是必须照抄的词汇表。\n"
    "8. 字幕单元已按日语原文确定，逐条翻译；semantic_units 只用于理解相邻分句的完整语义。"
    "每个 requested id 返回一条，只表达本单元的原文，不把别条内容搬进来，不改时间轴。\n"
    "9. 带 cont_prev / cont_next 的条目是长句在原文阶段按分句和停顿切分后的相邻单元："
    "cont_prev 表示它接着上一条没说完的话，cont_next 表示这句话在下一条继续。"
    "这类片段按上下文译成能前后衔接的半句，各自都不要补成完整句子，也不要在片段结尾加收句的标点；"
    "缺少标记只表示没有已确认的连接信息，不保证它在语法上独立完整。\n"
    "风格示例（只示范语义和口吻，不套用到其他句子）：\n"
    "誰も来ないわけではない。 → 并非没人来。\n"
    "もう一度だけ確認して。 → 再确认一次就好。\n"
    "10. 只输出合法 JSON，不要 Markdown、不要解释、不要额外字段；思考过程不要写进最终 content。"
    '最终 content 必须是完整 JSON 对象，形如 {{"translations":[{{"id":0,"text":"..."}}]}}，条数严格匹配本次任务，且不能为空。'
)

_SYSTEM_PROMPT_COMPACT = (
    "你是日语字幕译者，目标语言是{target_lang}。含义完整准确优先于压缩和排版；"
    "保留否定、条件、施受关系和原句语气，不加戏，不猜改原文，不一律折叠真实重复。"
    "有读音依据的人名按日语读音罗马音化，缺少依据保留原名。"
    "词汇表必须遵守；同一短句允许随语境变化。"
    "字幕已按日语原文断句，按 requested_ids 逐条翻译，不重分、不漏译、不重复、不改时间轴。"
    "semantic_units 仅帮助理解相邻分句，不要各自补成完整句。"
    "cont_prev/cont_next 表示已确认的续句；无标记不保证语法完整。"
    '只输出合法 JSON：{{"translations":[{{"id":0,"text":"..."}}]}}。'
)


def _build_system_prompt(
    character_reference: str,
    *,
    target_lang: str,
    glossary: str,
    compact: bool = False,
    extra_glossary: str = "",
    full_template: str | None = None,
    compact_template: str | None = None,
) -> str:
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
    effective_glossary = normalize_glossary_text(glossary)
    if any(marker in effective_target_lang.lower() for marker in ("中文", "chinese", "zh")):
        prompt += (
            "\n原文明确提及性器官、且用户词汇表没有另行指定时，中文默认用词为「肉棒」「小穴」；"
            "只用于对应的原文含义，不据此补词或加强语气。"
        )
    if effective_glossary:
        prompt += f"\n\n以下词汇表必须严格遵守，不得自行创造译名：\n{effective_glossary}"
    # Optional rendering examples belong after the stable prefix, and must
    # never acquire the authority of the user's terminology.
    del extra_glossary
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
            (character_reference or "").strip(),
            target_lang=target_lang,
            glossary=glossary,
            compact=compact_system_prompt,
            extra_glossary=extra_glossary,
        )
    effective_target_lang = (target_lang or "简体中文").strip() or "简体中文"

    user_parts = [
        f"【任务】把下面 JSON 数组里的日文字幕逐条翻译成{effective_target_lang}字幕。",
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


def _build_extra_glossary_block(extra_glossary: str) -> str:
    """Optional, nonbinding rendering examples after the stable source prefix."""
    effective = normalize_glossary_text(extra_glossary)
    if not effective:
        return ""
    return (
        "<rendering_examples>\n"
        "其他语境的译法示例，仅供参考；不得覆盖本句含义或用户词汇表：\n"
        f"{effective}\n"
        "</rendering_examples>"
    )


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
    block = _build_extra_glossary_block(extra_glossary)
    if block:
        task.append(block)
    return "\n".join(task)


def _build_batch_messages(
    batch_segments: list[dict],
    full_segments_summary: str,
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
    source_context: SourceContext | None = None,
) -> list[dict]:
    if requested_ids is None:
        requested_ids = list(range(expected_count))

    focus = (
        "【聚焦上下文：semantic_units 先整体理解，context_only 不输出】\n"
        + json.dumps(source_context.focus(requested_ids), ensure_ascii=False, separators=(",", ":"))
        if source_context is not None and requested_ids and not warmup
        else ""
    )

    source_payload = source_payload_override or _serialize_segments(batch_segments)
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
                focus,
            ]
        )
        return messages

    # Callers provide the already bounded full-source summary as a string.
    summary = full_segments_summary
    effective_target_lang = (target_lang or "简体中文").strip() or "简体中文"

    messages[0]["content"] = (
        messages[0]["content"]
        + "\n\n全片字幕概览（仅作上下文连贯参考，不要翻译，原 id 不在本批的不要返回）：\n"
        + summary
    )
    messages[1]["content"] = (
        f"【任务】把下面当前批次 JSON 数组里的日文字幕逐条翻译成{effective_target_lang}字幕。\n"
        "每个元素的 `id` 是全片全局 id，必须原样保留；只返回本批 id，不要返回概览里的其他 id。\n"
        "每个 `text` 只能是翻译结果本身。\n"
        f"【当前批次字幕 JSON】\n{source_payload}\n\n{focus}\n\n"
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
            f"本片参考名“{normalized}”只作为人名参考；剧中如果只称呼姓氏或名字，只翻译实际出现的部分，"
            "不要强行补全全名。仅在读音和姓名边界有依据时罗马音化；缺少依据保留原名。"
        ),
        "homophone": (
            "人物参考只用于识别字幕文本中已经明确出现的人名；不要根据参考名推测、补全或替换其他词。"
            "不要为了统一人物而把不同汉字姓氏或不同读音的称呼强行合并。"
            "不要把普通名词误改成人名。"
        ),
    }
