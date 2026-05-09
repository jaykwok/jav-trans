import json

from llm import translator
def test_prompt_version_is_v25():
    assert translator.PROMPT_VERSION == "v2.5"


def test_system_prompt_no_male_prefix_example():
    assert "男：" not in translator._SYSTEM_PROMPT_FULL


def test_system_prompt_mentions_acoustic_tags():
    assert "[M]" in translator._SYSTEM_PROMPT_FULL


def test_system_prompt_fixes_requested_genital_terms_without_kikuka():
    prompt = translator._build_system_prompt(
        expected_count=2,
        character_reference="",
        target_lang="简体中文",
        glossary="",
    )

    assert "肉棒" in prompt
    assert "小穴" in prompt
    assert "菊花" not in prompt
    assert "淫穴" not in prompt
    assert "骚逼" not in prompt


def test_leading_speaker_regex_strips_male_prefix():
    assert translator._LEADING_SPEAKER_RE.sub("", "男：xxx", count=1) == "xxx"


def test_leading_speaker_regex_strips_female_prefix():
    assert translator._LEADING_SPEAKER_RE.sub("", "女：xxx", count=1) == "xxx"


def test_leading_speaker_regex_strips_ascii_name_prefix():
    assert translator._LEADING_SPEAKER_RE.sub("", "Aya：xxx", count=1) == "xxx"


def test_leading_speaker_regex_does_not_strip_normal_chinese_prefix():
    text = "今天天气：晴朗"
    assert translator._LEADING_SPEAKER_RE.sub("", text, count=1) == text


def test_leading_speaker_regex_only_strips_at_start():
    text = "hello 男：yyy"
    assert translator._LEADING_SPEAKER_RE.sub("", text, count=1) == text


def _serialized_ja(segment: dict) -> str:
    payload = json.loads(translator._serialize_segments([segment]))
    return payload[0]["ja"]


def test_serialize_segments_prefixes_male_gender():
    assert _serialized_ja({"start": 0.0, "end": 1.0, "text": "来て", "gender": "M"}).startswith("[M]")


def test_serialize_segments_prefixes_female_gender():
    assert _serialized_ja({"start": 0.0, "end": 1.0, "text": "いい", "gender": "F"}).startswith("[F]")


def test_serialize_segments_omits_prefix_for_none_gender():
    ja = _serialized_ja({"start": 0.0, "end": 1.0, "text": "いい", "gender": None})
    assert not ja.startswith("[M]")
    assert not ja.startswith("[F]")

