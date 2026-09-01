from llm import global_glossary, translator


def test_a_repeated_line_with_a_dominant_rendering_is_settled():
    ja = ["気持ちいい…", "気持ちいい…", "気持ちいい…", "うん"]
    zh = ["好舒服…", "好舒服…", "爽…", "嗯"]

    terms = global_glossary.derive_settled_glossary(ja, zh)

    assert terms == [{"ja": "気持ちいい…", "zh": "好舒服…"}]


def test_a_line_seen_only_once_is_not_settled():
    terms = global_glossary.derive_settled_glossary(["こんにちは"], ["你好"])

    assert terms == []


def test_a_line_the_base_pass_rendered_inconsistently_is_not_settled():
    """No majority, no settled answer to hand back - picking whichever
    rendering happened to be more common by one would be a guess wearing the
    same clothes as the LLM-guessed pairs this module replaced."""
    ja = ["だめ", "だめ"]
    zh = ["不行", "不要"]

    terms = global_glossary.derive_settled_glossary(ja, zh)

    assert terms == []


def test_a_target_that_is_not_chinese_is_dropped():
    """Fed back into the repair prompt as a settled translation, so a Latin or
    kana target is an instruction to leave Japanese on screen."""
    ja = ["ジェイ", "ジェイ"]
    zh = ["Jay", "Jay"]

    terms = global_glossary.derive_settled_glossary(ja, zh)

    assert terms == []


def test_a_line_or_rendering_carrying_the_pair_delimiter_is_dropped():
    """`parse_glossary_pairs` splits each formatted item on its first "-", so a
    line containing one would be split at the wrong point downstream."""
    ja = ["これ-それ", "これ-それ"]
    zh = ["这个-那个", "这个-那个"]

    terms = global_glossary.derive_settled_glossary(ja, zh)

    assert terms == []


def test_settled_terms_are_capped_and_sorted_by_recurrence():
    ja_texts = []
    zh_texts = []
    for i in range(25):
        line = f"台词{i}"
        count = 25 - i
        ja_texts.extend([line] * count)
        zh_texts.extend([f"翻译{i}"] * count)

    terms = global_glossary.derive_settled_glossary(ja_texts, zh_texts)

    assert len(terms) == 20
    assert terms[0] == {"ja": "台词0", "zh": "翻译0"}
    assert terms[-1] == {"ja": "台词19", "zh": "翻译19"}


def test_extracted_terms_never_contradict_the_user_glossary():
    """Measured on sample-v 2026-08-24 against the old LLM extractor: it mined
    five spellings of the same word and proposed 鸡巴 for all of them while the
    user's glossary said 肉棒, and the prompt asserted both at once. The
    suppression logic that caught it is unchanged by the switch to a
    deterministic source - it still has to hold for whatever produces the
    candidate pairs."""
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

    formatted = global_glossary._format_global_glossary_terms(
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

    formatted = global_glossary._format_global_glossary_terms(terms, glossary="ちんぽ-肉棒")

    assert formatted.splitlines() == ["おちんぽ-肉棒"]


def test_extracted_terms_are_untouched_without_a_glossary():
    terms = [{"ja": "ちんぽ", "zh": "鸡巴"}, {"ja": "まんこ", "zh": "小穴"}]

    formatted = global_glossary._format_global_glossary_terms(terms, glossary="")

    assert formatted.splitlines() == ["ちんぽ-鸡巴", "まんこ-小穴"]


def test_resolve_settled_glossary_writes_the_same_artifact_shape_quality_reads(tmp_path):
    """`pipeline/quality.py` globs `translation_global_glossary.*.json` and reads
    `{"terms":[{"ja":...,"zh":...}]}` - that contract predates this rewrite and
    must not move, only who writes it and when."""
    cache_path = str(tmp_path / "translation_cache.jsonl")
    segments = [{"text": "気持ちいい…"}, {"text": "気持ちいい…"}]
    zh_texts = ["好舒服…", "好舒服…"]

    block = translator.resolve_settled_glossary(segments, zh_texts, cache_path, "")

    assert block == "気持ちいい…-好舒服…"
    written = list(tmp_path.glob("translation_global_glossary.*.json"))
    assert len(written) == 1
    import json

    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload == {"terms": [{"ja": "気持ちいい…", "zh": "好舒服…"}]}


def test_resolve_settled_glossary_issues_no_request(monkeypatch, tmp_path):
    def fail_chat(*_args, **_kwargs):
        raise AssertionError("resolve_settled_glossary must not call the model")

    monkeypatch.setattr(translator, "_chat", fail_chat)

    block = translator.resolve_settled_glossary(
        [{"text": "こんにちは"}], ["你好"], str(tmp_path / "translation_cache.jsonl"), ""
    )

    assert block == ""


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
