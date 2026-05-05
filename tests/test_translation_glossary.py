from llm import translator
def test_translation_glossary_empty_omits_glossary_section():

    prompt = translator._build_system_prompt(
        expected_count=2,
        character_reference="",
        target_lang="简体中文",
        glossary="",
    )

    assert "词汇表" not in prompt


def test_translation_glossary_appends_user_terms():
    glossary = "健太（男主）, 小那海（女主）"

    prompt = translator._build_system_prompt(
        expected_count=2,
        character_reference="",
        target_lang="简体中文",
        glossary=glossary,
    )

    assert "以下词汇表必须严格遵守，不得自行创造译名：" in prompt
    assert glossary in prompt

