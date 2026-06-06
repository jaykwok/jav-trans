from llm import translator
def _segments() -> list[dict]:
    return [
        {"start": 0.0, "end": 1.0, "text": "いい"},
        {"start": 1.0, "end": 2.0, "text": "もっと"},
    ]


def test_translation_cache_key_same_input_same_glossary():
    first = translator._translation_cache_key(0, _segments(), glossary="健太（男主）")
    second = translator._translation_cache_key(0, _segments(), glossary="健太（男主）")

    assert first == second


def test_translation_cache_key_normalizes_glossary_spacing():
    first = translator._translation_cache_key(0, _segments(), glossary=" 健太 - 男主 ")
    second = translator._translation_cache_key(0, _segments(), glossary="健太-男主")

    assert first == second


def test_translation_cache_key_changes_with_glossary():
    first = translator._translation_cache_key(0, _segments(), glossary="健太-男主")
    second = translator._translation_cache_key(0, _segments(), glossary="小那海-女主")

    assert first != second


def test_translation_cache_key_changes_with_target_lang():
    first = translator._translation_cache_key(
        0,
        _segments(),
        glossary="健太（男主）",
        target_lang="简体中文",
    )
    second = translator._translation_cache_key(
        0,
        _segments(),
        glossary="健太（男主）",
        target_lang="繁體中文",
    )

    assert first != second


def test_translation_cache_key_changes_with_prompt_version(monkeypatch):
    first = translator._translation_cache_key(0, _segments(), glossary="健太（男主）")

    monkeypatch.setattr(translator, "PROMPT_VERSION", "v-test-next")
    second = translator._translation_cache_key(0, _segments(), glossary="健太（男主）")

    assert first != second


def test_translation_cache_key_changes_with_timing():
    first = translator._translation_cache_key(0, _segments(), glossary="")
    changed = [
        {"start": 0.0, "end": 1.5, "text": "いい"},
        {"start": 1.5, "end": 2.0, "text": "もっと"},
    ]
    second = translator._translation_cache_key(0, changed, glossary="")

    assert first != second


def test_translation_cache_key_ignores_non_timing_metadata():
    first = translator._translation_cache_key(0, _segments(), glossary="")
    changed = [
        {"start": 0.0, "end": 1.0, "text": "いい", "source_note": "ignored"},
        {"start": 1.0, "end": 2.0, "text": "もっと"},
    ]
    second = translator._translation_cache_key(0, changed, glossary="")

    assert first == second


def test_translation_memory_key_survives_timing_changes():
    first = translator._translation_memory_key(
        " いい\n",
        glossary="健太（男主）",
        target_lang="简体中文",
    )
    second = translator._translation_memory_key(
        "いい",
        glossary="健太（男主）",
        target_lang="简体中文",
    )

    assert first == second


def test_translation_memory_key_changes_with_glossary():
    first = translator._translation_memory_key(
        "いい",
        glossary="健太-Kenta",
        target_lang="简体中文",
    )
    second = translator._translation_memory_key(
        "いい",
        glossary="小那海-Aya",
        target_lang="简体中文",
    )

    assert first != second


def test_translation_memory_key_changes_with_target_lang():
    first = translator._translation_memory_key(
        "いい",
        glossary="",
        target_lang="简体中文",
    )
    second = translator._translation_memory_key(
        "いい",
        glossary="",
        target_lang="繁體中文",
    )

    assert first != second


def test_translation_memory_key_changes_with_character_reference():
    first = translator._translation_memory_key(
        "いい",
        glossary="",
        target_lang="简体中文",
        character_reference="Aya",
    )
    second = translator._translation_memory_key(
        "いい",
        glossary="",
        target_lang="简体中文",
        character_reference="Mio",
    )

    assert first != second


def test_translation_memory_key_changes_with_prompt_version(monkeypatch):
    first = translator._translation_memory_key("いい", glossary="", target_lang="简体中文")

    monkeypatch.setattr(translator, "PROMPT_VERSION", "v-test-next")
    second = translator._translation_memory_key("いい", glossary="", target_lang="简体中文")

    assert first != second


def test_translation_memory_rejects_low_information_sources():
    assert not translator._translation_memory_source_is_cacheable("")
    assert not translator._translation_memory_source_is_cacheable("あ")
    assert not translator._translation_memory_source_is_cacheable("ああああ")
    assert translator._translation_memory_source_is_cacheable("もっと来て")

