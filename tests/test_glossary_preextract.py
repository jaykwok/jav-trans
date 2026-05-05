import json

import pytest

from llm import translator
def test_extra_glossary_injected_into_system_prompt_and_batch_user_message():
    prompt = translator._build_system_prompt(
        expected_count=2,
        character_reference="",
        target_lang="简体中文",
        glossary="",
        extra_glossary="あなた \u2192 你\n肉棒 \u2192 肉棒",
    )
    assert "<glossary>" in prompt
    assert "あなた \u2192 你" in prompt

    messages = translator._build_batch_messages(
        [{"start": 0.0, "end": 1.0, "text": "あなた"}],
        "0: あなた",
        0,
        "",
        1,
        target_lang="简体中文",
        glossary="",
        extra_glossary="あなた \u2192 你",
    )
    assert "<glossary>" in messages[0]["content"]
    assert "注意：必须严格使用 System Prompt 中 <glossary> 标签内的术语表翻译。" in messages[1]["content"]


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


def test_glossary_preextract_failure_returns_empty(monkeypatch, tmp_path):
    def fail_chat(*_args, **_kwargs):
        raise RuntimeError("api failed")

    monkeypatch.setattr(translator, "_chat", fail_chat)

    assert translator.extract_global_glossary(
        ["あなた"],
        str(tmp_path / "translation_global_glossary.json"),
    ) == []


def test_prompt_signature_changes_with_extra_glossary():
    first = translator._compute_prompt_signature(extra_glossary="あなた \u2192 你")
    second = translator._compute_prompt_signature(extra_glossary="あなた \u2192 您")

    assert first != second

