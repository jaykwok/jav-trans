import json

import pytest

from llm import translator
def test_extra_glossary_rides_the_task_tail_not_the_system_prompt():
    """The system prompt has to be byte-identical before and after extraction,
    because the extraction request is what warms the provider's prefix cache for
    every batch behind it. A block that only appears once the terms are known
    would split the film into two prefixes and buy the full source twice."""
    prompt = translator._build_system_prompt(
        character_reference="",
        target_lang="简体中文",
        glossary="",
        extra_glossary="あなた-你\n肉棒-肉棒",
    )
    assert "<glossary>" not in prompt
    assert "あなた-你" not in prompt
    assert prompt == translator._build_system_prompt(
        character_reference="", target_lang="简体中文", glossary=""
    )

    messages = translator._build_batch_messages(
        [{"start": 0.0, "end": 1.0, "text": "あなた"}],
        "0: あなた",
        "",
        1,
        target_lang="简体中文",
        glossary="",
        extra_glossary="あなた-你",
    )
    assert "<glossary>" not in messages[0]["content"]
    assert "<glossary>" in messages[1]["content"]
    assert "あなた-你" in messages[1]["content"]
    assert "注意：必须严格使用上面 <glossary> 标签内的术语表翻译。" in messages[1]["content"]


def test_the_extraction_request_shares_the_batch_prefix():
    """One request has to serve as both extraction and warmup, which only works
    if everything up to the end of the full-film payload is identical."""
    payload = '[{"id":0,"text":"あなた"}]'
    extraction = translator._build_glossary_extraction_messages(
        full_source_payload=payload,
        target_lang="简体中文",
        glossary="ちんぽ-肉棒",
        character_reference="",
    )
    batch = translator._build_batch_messages(
        [{"start": 0.0, "end": 1.0, "text": "あなた"}],
        "",
        "",
        1,
        target_lang="简体中文",
        glossary="ちんぽ-肉棒",
        extra_glossary="あなた-你",
        full_source_payload=payload,
        requested_ids=[0],
    )

    assert extraction[0]["content"] == batch[0]["content"]
    shared = f"【全片字幕 JSON】\n\n{payload}\n\n"
    assert extraction[1]["content"].startswith(shared)
    assert batch[1]["content"].startswith(shared)
    assert '{"terms"' in extraction[1]["content"]


def test_glossary_cache_file_skips_chat(monkeypatch, tmp_path):
    cache_path = tmp_path / "translation_global_glossary.json"
    cache_path.write_text(
        json.dumps({"terms": [{"ja": "あなた", "zh": "你"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    def fail_chat(*_args, **_kwargs):
        raise AssertionError("_chat should not be called when glossary cache exists")

    monkeypatch.setattr(translator, "_chat", fail_chat)

    assert translator.extract_global_glossary(["あなた"], str(cache_path)) == [
        {"ja": "あなた", "zh": "你"}
    ]


def test_a_target_that_is_not_chinese_is_dropped():
    """These pairs are fed back into the batch prompt as settled translations, so
    a Latin or kana target is an instruction to leave Japanese on screen. The
    first run that shared the translation system prompt with the extractor
    inherited its romanise-unknown-names rule, returned `ジェイ-Jay` and
    `シルス-Sirusu`, and the base pass then echoed 239 of 1,595 cues verbatim."""
    terms = [
        {"ja": "ジェイ", "zh": "Jay"},
        {"ja": "シルス", "zh": "Sirusu"},
        {"ja": "おなみ", "zh": "オナミ"},
        {"ja": "ちんぽ", "zh": "肉棒"},
        {"ja": "イク", "zh": "高潮"},
    ]

    filtered = translator._filter_global_glossary_terms(terms)

    assert filtered == [{"ja": "ちんぽ", "zh": "肉棒"}, {"ja": "イク", "zh": "高潮"}]


def test_the_extraction_task_overrides_the_inherited_name_rule():
    """The task tail is the only part of the request that differs from a batch,
    so it is the only place the inherited system rules can be overridden."""
    messages = translator._build_glossary_extraction_messages(
        full_source_payload="[]",
        target_lang="简体中文",
        glossary="",
        character_reference="",
    )
    task = messages[1]["content"]

    assert "罗马音化" in messages[0]["content"]
    assert "本任务不适用上面关于人名罗马音化的规则" in task
    assert "不得出现日文假名、罗马字或英文" in task


def test_glossary_denoise_filters_invalid_and_caps_to_15():
    raw_terms = [
        {"ja": "123456789", "zh": "长"},
        {"ja": "あなた,ね", "zh": "你"},
        {"ja": "好き", "zh": "喜欢。"},
    ]
    raw_terms.extend({"ja": f"語{i}", "zh": f"词{i}"} for i in range(20))

    filtered = translator._filter_global_glossary_terms(raw_terms)

    assert len(filtered) == 15
    assert filtered[0] == {"ja": "語0", "zh": "词0"}
    assert filtered[-1] == {"ja": "語14", "zh": "词14"}


def test_extracted_terms_never_contradict_the_user_glossary():
    """Measured on sample-v 2026-08-24. With a glossary of `ちんぽ-肉棒` the
    extractor mined the film's own wording and proposed 鸡巴 for the same word
    under five spellings; exact-key matching caught two and injected the rest,
    so the prompt asserted both mappings at once and 6 of 37 cues came back 鸡巴.
    Variants have to lose to the user's glossary, mosaic spellings included."""
    terms = [
        {"ja": "ちんぽ", "zh": "鸡巴"},
        {"ja": "ちんちん", "zh": "鸡巴"},
        {"ja": "おちんぽ", "zh": "鸡巴"},
        {"ja": "ち○ぽ", "zh": "鸡巴"},
        {"ja": "おち○ちん", "zh": "鸡巴"},
        {"ja": "おち○ぽ", "zh": "鸡巴"},
        {"ja": "まんこ", "zh": "小穴"},
        {"ja": "中出し", "zh": "内射"},
    ]

    formatted = translator._format_global_glossary_terms(
        terms, glossary="ちんぽ-肉棒, おちんちん-肉棒"
    )

    assert "鸡巴" not in formatted
    assert formatted.splitlines() == ["まんこ-小穴", "中出し-内射"]


def test_an_extracted_variant_that_agrees_with_the_glossary_is_kept():
    """Suppression is about contradiction, not about the word. A variant mapping
    to the same target reinforces the glossary, so dropping it only loses a hint."""
    terms = [
        {"ja": "おちんぽ", "zh": "肉棒"},
        {"ja": "ち○ぽ", "zh": "阴茎"},
    ]

    formatted = translator._format_global_glossary_terms(terms, glossary="ちんぽ-肉棒")

    assert formatted.splitlines() == ["おちんぽ-肉棒"]


def test_extracted_terms_are_untouched_without_a_glossary():
    terms = [{"ja": "ちんぽ", "zh": "鸡巴"}, {"ja": "まんこ", "zh": "小穴"}]

    formatted = translator._format_global_glossary_terms(terms, glossary="")

    assert formatted.splitlines() == ["ちんぽ-鸡巴", "まんこ-小穴"]


def test_glossary_preextract_failure_returns_empty(monkeypatch, tmp_path):
    def fail_chat(*_args, **_kwargs):
        raise RuntimeError("api failed")

    monkeypatch.setattr(translator, "_chat", fail_chat)

    assert translator.extract_global_glossary(
        ["あなた"],
        str(tmp_path / "translation_global_glossary.json"),
    ) == []


def test_prompt_signature_changes_with_extra_glossary():
    first = translator._compute_prompt_signature(extra_glossary="あなた-你")
    second = translator._compute_prompt_signature(extra_glossary="あなた-您")

    assert first != second


def test_global_glossary_cache_path_uses_source_text_hash(tmp_path):
    cache_path = str(tmp_path / "translation_cache.jsonl")
    first = translator._global_glossary_cache_path_for_texts(cache_path, ["あなた"])
    second = translator._global_glossary_cache_path_for_texts(cache_path, ["わたし"])

    assert first != second
    assert first.endswith(".json")
    assert "translation_global_glossary." in first

