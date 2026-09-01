import json
import re

from llm.glossary import normalize_glossary_text


# v3.2: cues carry `cont_prev`/`cont_next` when the display-duration split cut a
# sentence in half, and the system prompt says what to do with them. Bumped
# because the cache key is `id@version` - translations produced without the
# markers are the exact ones this change exists to replace.
# v3.4: the per-film extracted terms moved from the system prompt to the tail of
# the user task, so term extraction, prefix warmup and every batch share one
# cacheable prefix. Bumped because the cache key is `id@version` and the prompt
# the old entries were produced under no longer exists.
PROMPT_VERSION = "v3.4"
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
    "你是专业的日语成人视频字幕翻译，目标语言是{target_lang}。\n"
    "翻译风格随场景自适应，不要给全片套同一种腔调：\n"
    "- 情欲场景（性行为、调情挑逗、下流对话、呻吟喘息）：露骨直接，用口语粗话，保留原文的挑逗和情色强度，不净化不弱化不说教；"
    "性器官相关词汇统一——男性器官用“肉棒”，女性器官用“小穴”，不要漂移成其他译法或书面语（如阴茎、性交、插入）。\n"
    "- 非情欲对白（剧情、采访、日常、过场）：自然生活化口语，就事论事，不要强行加撩拨腔或色情味。\n"
    "你会收到全片字幕 JSON，以及本次需要翻译的 requested_ids；必须基于全局上下文完成指定 id 翻译。\n"
    "本视频出场人物全名参考：【{character_reference}】。\n"
    "要求：\n"
    "1. 译文自然、口语化、贴合字幕阅读节奏，避免书面腔；输出尽量短，短促呻吟和语气词也要简短自然。\n"
    "2. 标点遵循 Netflix 简中字幕规范：译文里不用逗号和句号，句中停顿用一个空格；省略号只用单个“…”；"
    "问句叹句用全角“？”“！”且不连用（不出现？！、！！）；数字用半角；每条译文尽量不超过 16 个汉字；"
    "同一条原文把同一短语连说多遍时只译一遍，不要复读。\n"
    "3. 情欲与非情欲的语气按上面场景规则处理；呻吟、喘息、下流语气保留原本情绪强度，译成适合字幕的自然短句，"
    "不要按固定词表替换、删除或净化，只有明显机器循环才可在译文中适度概括。\n"
    "4. 人名不要翻译成中文；如果原文出现人物姓名，直接输出罗马音，格式用 Title Case，并用空格分隔名和姓，例如 Aya Onami。原文是汉字姓名时也必须按日语读音罗马音化，不要输出中文汉字或中文读法。\n"
    "5. {name_boundary}\n"
    "6. {name_homophone}\n"
    "7. 全片上下文只用于翻译连贯、指代判断、口吻一致和术语一致；不要修改、补全或纠正日文原文。\n"
    "8. 每条输入必须单独翻译，不能合并、拆分、漏译、调换顺序。\n"
    "9. 带 cont_prev / cont_next 的条目是同一句话被字幕时长切开后的片段："
    "cont_prev 表示它接着上一条没说完的话，cont_next 表示这句话在下一条继续。"
    "这类片段按上下文译成能前后衔接的半句，各自都不要补成完整句子，也不要在片段结尾加收句的标点；"
    "没有这两个标记的条目就是完整的一句，照常翻译。\n"
    "10. 只输出合法 JSON，不要 Markdown、不要解释、不要额外字段；思考过程不要写进最终 content。"
    '最终 content 必须是完整 JSON 对象，形如 {{"translations":[{{"id":0,"text":"..."}}]}}，条数严格匹配本次任务，且不能为空。\n\n'
    "风格示例（仅示范语气与用词，不是本次待译内容）：\n"
    "気持ちいい… → 好舒服…\n"
    "もっと奥まで欲しいの → 想要你插到更里面\n"
    "太いおちんちんで突いて → 用你那根粗肉棒狠狠捅我\n"
    "そこはダメぇ… → 那里不行啦…\n"
    "今日は撮影ありがとうございました → 今天拍摄辛苦了 谢谢\n"
    "えっと、次はどうすればいい？ → 呃 接下来我该怎么做？\n\n"
    "EXAMPLE JSON OUTPUT:\n"
    '{{"translations":[{{"id":0,"text":"第一句中文翻译"}},{{"id":1,"text":"第二句中文翻译"}}]}}'
)

_SYSTEM_PROMPT_COMPACT = (
    "你是日语成人视频字幕译者，目标语言是{target_lang}。语气随场景自适应："
    "情欲场景露骨口语、保留强度、性器官统一肉棒/小穴；非情欲对白自然生活化口语、不强加色情腔。"
    "保持人名罗马音；汉字人名也要按日语读音罗马音化，不输出中文汉字名；"
    "不用逗号句号，停顿用空格；省略号单个…；全角？！且不连用；"
    "每条独立翻译，不合并、不漏译、不调序。"
    "带 cont_prev/cont_next 的是同一句被切开的片段，译成能衔接的半句，不要各自补成完整句。"
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
    if effective_glossary:
        prompt += f"\n\n以下词汇表必须严格遵守，不得自行创造译名：\n{effective_glossary}"
    # The per-film extracted terms deliberately do NOT go here. Everything the
    # provider caches is a token prefix, so a block that only exists once the
    # extraction has returned splits the film's own requests into two prefixes:
    # the extraction pays the full-source miss, then the first batch pays it
    # again because the system message no longer matches. Keeping the system
    # prompt free of them lets one request warm the prefix for all of them -
    # see _build_requested_ids_task, which carries the block instead.
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
    """The per-film extracted terms, as a task-tail block rather than a system one.

    Sits after the full-film payload so that everything before it is shared by
    the extraction request and every batch, which is what lets a single request
    warm the provider's prefix cache for the whole film.
    """
    effective = normalize_glossary_text(extra_glossary)
    if not effective:
        return ""
    return (
        "<glossary>\n"
        "本片已确定译法（必须沿用，与上面的词汇表冲突时以词汇表为准）：\n"
        f"{effective}\n"
        "</glossary>"
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
        task.append("注意：必须严格使用上面 <glossary> 标签内的术语表翻译。")
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
) -> list[dict]:
    if requested_ids is None:
        requested_ids = list(range(expected_count))

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
            "人物参考只用于识别字幕文本中已经明确出现的人名；不要根据参考名推测、补全或替换其他词。"
            "不要为了统一人物而把不同汉字姓氏或不同读音的称呼强行合并。"
            "不要把普通名词误改成人名。"
        ),
    }
