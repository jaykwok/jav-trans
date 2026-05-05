from llm import translator
def test_default_full_prompt_same_length_for_batch0_and_batch1(monkeypatch):
    monkeypatch.setattr(translator, "COMPACT_SYSTEM_PROMPT", False)

    batch0_prompt = translator._build_system_prompt(
        expected_count=5,
        character_reference="",
        target_lang="简体中文",
        glossary="",
        compact=False,
    )
    batch1_prompt = translator._build_system_prompt(
        expected_count=5,
        character_reference="",
        target_lang="简体中文",
        glossary="",
        compact=False,
    )

    assert len(batch0_prompt) == len(batch1_prompt)
    assert batch0_prompt == batch1_prompt


def test_compact_prompt_shorter_for_batch1_when_enabled(monkeypatch):
    monkeypatch.setattr(translator, "COMPACT_SYSTEM_PROMPT", True)

    batch0_prompt = translator._build_system_prompt(
        expected_count=5,
        character_reference="",
        target_lang="简体中文",
        glossary="",
        compact=False,
    )
    batch1_prompt = translator._build_system_prompt(
        expected_count=5,
        character_reference="",
        target_lang="简体中文",
        glossary="",
        compact=True,
    )

    assert len(batch1_prompt) < len(batch0_prompt)
    assert "EXAMPLE JSON OUTPUT" in batch0_prompt
    assert "EXAMPLE JSON OUTPUT" not in batch1_prompt

