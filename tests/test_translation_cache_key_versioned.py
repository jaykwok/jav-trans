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


def test_translation_cache_key_changes_with_glossary():
    first = translator._translation_cache_key(0, _segments(), glossary="健太（男主）")
    second = translator._translation_cache_key(0, _segments(), glossary="小那海（女主）")

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

    monkeypatch.setattr(translator, "PROMPT_VERSION", "v2.6")
    second = translator._translation_cache_key(0, _segments(), glossary="健太（男主）")

    assert first != second

