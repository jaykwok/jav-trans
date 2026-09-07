import json

from llm import translator
def test_prompt_version_is_current():
    """The cache key is `id@version`, so this constant is what decides whether a
    prompt change reaches anyone who already has a cache."""
    assert translator.PROMPT_VERSION == "v4.0"


def test_system_prompt_no_male_prefix_example():
    assert "男：" not in translator._SYSTEM_PROMPT_FULL


def test_system_prompt_tone_is_scene_conditional():
    # Tone follows the source, never a genre-based instruction to intensify it.
    full = translator._SYSTEM_PROMPT_FULL
    assert "本片译文要体现主动撩拨的语气" not in full
    assert "语域和情绪强度跟随原句" in full
    assert "不因为视频题材或场景自行加强语气" in full


def test_system_prompt_includes_style_examples():
    # Few-shot JA->ZH style pairs guide register better than abstract rules.
    full = translator._SYSTEM_PROMPT_FULL
    assert "风格示例" in full
    assert " → " in full


def test_user_prompt_uses_target_language_label():
    messages = translator._build_translation_messages(
        source_payload='[{"id":0,"ja":"好き"}]',
        expected_count=1,
        target_lang="繁體中文",
        glossary="",
        character_reference="",
    )

    assert "翻译成繁體中文字幕" in messages[1]["content"]
    assert "翻译成中文字幕" not in messages[1]["content"]


def test_system_prompt_does_not_authorize_asr_rewrites():
    prompt = translator._build_system_prompt(
        character_reference="小那海あや",
        target_lang="简体中文",
        glossary="",
    )

    assert "允许根据前后文修正明显 ASR 误听" not in prompt
    assert "可修正明显ASR误听" not in prompt
    assert "ASR 同音纠错" not in prompt
    assert "同音误听" not in prompt
    assert "上下文漂移" not in prompt
    assert "不要根据参考名推测、补全或替换其他词" in prompt


def test_system_prompt_fixes_requested_genital_terms_without_kikuka():
    prompt = translator._build_system_prompt(
        character_reference="",
        target_lang="简体中文",
        glossary="",
    )

    assert "肉棒" in prompt
    assert "小穴" in prompt
    assert "菊花" not in prompt
    assert "淫穴" not in prompt
    assert "骚逼" not in prompt
    assert "用户词汇表没有另行指定时" in prompt


def test_fidelity_and_evidenced_names_precede_display_constraints():
    prompt = translator._build_system_prompt("山田", target_lang="简体中文", glossary="")
    assert "原文含义完整准确 > 用户词汇表" in prompt
    assert "不是每条译文的语义上限" in prompt
    assert "不一律只保留一次" in prompt
    assert "缺少依据保留原名" in prompt
    assert "智能拆解" not in prompt


def test_leading_role_label_regex_strips_human_male_prefix():
    assert translator._LEADING_ROLE_LABEL_RE.sub("", "男：xxx", count=1) == "xxx"


def test_leading_role_label_regex_strips_human_female_prefix():
    assert translator._LEADING_ROLE_LABEL_RE.sub("", "女：xxx", count=1) == "xxx"


def test_leading_role_label_regex_strips_ascii_name_prefix():
    assert translator._LEADING_ROLE_LABEL_RE.sub("", "Aya：xxx", count=1) == "xxx"


def test_leading_role_label_regex_does_not_strip_normal_chinese_prefix():
    text = "今天天气：晴朗"
    assert translator._LEADING_ROLE_LABEL_RE.sub("", text, count=1) == text


def test_leading_role_label_regex_only_strips_at_start():
    text = "hello 男：yyy"
    assert translator._LEADING_ROLE_LABEL_RE.sub("", text, count=1) == text


def _serialized_ja(segment: dict) -> str:
    payload = json.loads(translator._serialize_segments([segment]))
    return payload[0]["ja"]


def test_serialize_segments_does_not_add_acoustic_prefixes():
    assert _serialized_ja({"start": 0.0, "end": 1.0, "text": "来て"}) == "来て"
    assert _serialized_ja({"start": 0.0, "end": 1.0, "text": "いい"}) == "いい"


def test_serialize_segments_preserves_repeated_vocalizations():
    ja = _serialized_ja({"start": 0.0, "end": 1.0, "text": "ああああああ"})
    assert ja == "ああああああ"


def test_normalize_translation_text_strips_human_role_labels():
    assert translator._normalize_translation_text("男：过来") == "过来"
    assert translator._normalize_translation_text("Aya：过来") == "过来"
