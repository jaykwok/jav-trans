"""The 简繁 post-filter: it must fire on exactly the two Chinese targets, get the
context-dependent mappings right, and never touch anything else.

The pass exists because the target language is a prompt request, not a
guarantee - a line returned in the wrong script is fluent, correct, and passes
every other check, so nothing downstream would catch it.
"""

import pytest

from llm import zh_variant


class TestWhichTargetsConvert:
    def test_the_three_labels_the_web_page_actually_sends(self):
        """These strings are the `<option value>`s in index.html. If they drift,
        the whole pass silently becomes a no-op."""
        assert zh_variant.target_variant("简体中文") == "simplified"
        assert zh_variant.target_variant("繁體中文") == "traditional"
        assert zh_variant.target_variant("English") is None

    @pytest.mark.parametrize(
        "label",
        ["zh-CN", "zh_Hans", "Simplified Chinese", "简体", "zh-SG"],
    )
    def test_hand_written_simplified_labels(self, label):
        assert zh_variant.target_variant(label) == "simplified"

    @pytest.mark.parametrize(
        "label",
        ["zh-TW", "zh_Hant", "Traditional Chinese", "正體中文", "zh-HK"],
    )
    def test_hand_written_traditional_labels(self, label):
        assert zh_variant.target_variant(label) == "traditional"

    @pytest.mark.parametrize("label", ["", "   ", None, "日本語", "Korean", "zh"])
    def test_unreadable_targets_leave_text_alone(self, label):
        """Guessing a variant would rewrite output nobody asked to rewrite. Bare
        `zh` is deliberately in here: it names no variant."""
        assert zh_variant.target_variant(label) is None
        assert zh_variant.converter_for(label) is None


class TestTheMappingsThatNeedContext:
    """A per-character table gets these backwards, which is why the reference
    OpenCC dictionaries are a dependency instead of a dict literal."""

    @pytest.mark.parametrize(
        "simplified, traditional",
        [
            ("她的头发很长", "她的頭髮很長"),
            ("我发现了", "我發現了"),
            ("在房间里面", "在房間裡面"),
            ("干净", "乾淨"),
            ("你在干什么", "你在幹什麼"),
        ],
    )
    def test_both_directions(self, simplified, traditional):
        assert zh_variant.convert(traditional, "简体中文") == simplified
        assert zh_variant.convert(simplified, "繁體中文") == traditional

    def test_traditional_uses_the_tw_hk_variant_forms(self):
        """Generic `s2t` writes 裏面 / 這裏; anyone who picks 繁體中文 expects
        裡面 / 這裡. Pinned because the config name is the only thing that
        decides it and it is a one-character diff away from wrong."""
        assert zh_variant.convert("这里的房间里面", "繁體中文") == "這裡的房間裡面"

    def test_traditional_does_not_swap_vocabulary(self):
        """`s2twp` would make this 軟體. Changing word choice is not this pass's
        job - it converts script, and only script."""
        assert zh_variant.convert("软件", "繁體中文") == "軟件"

    def test_simplified_folds_both_traditional_variant_forms(self):
        assert zh_variant.convert("在房間裡面", "简体中文") == "在房间里面"
        assert zh_variant.convert("在房間裏面", "简体中文") == "在房间里面"

    def test_same_character_maps_two_ways_by_context(self):
        """发 -> 髮 in 头发 but 發 in 发现; 干 -> 乾 in 干净 but 幹 in 干什么."""
        out = zh_variant.convert("头发/发现/干净/干什么", "繁體中文")
        assert out == "頭髮/發現/乾淨/幹什麼"

    def test_already_in_the_requested_variant_is_a_no_op(self):
        assert zh_variant.convert("她的头发很长", "简体中文") == "她的头发很长"
        assert zh_variant.convert("她的頭髮很長", "繁體中文") == "她的頭髮很長"

    @pytest.mark.parametrize("target", ["简体中文", "繁體中文"])
    def test_idempotent(self, target):
        for source in ("她的头发很长", "她的頭髮很長", "干净的房间里面"):
            once = zh_variant.convert(source, target)
            assert zh_variant.convert(once, target) == once


class TestWhatMustSurviveUntouched:
    @pytest.mark.parametrize(
        "text",
        ["Hello, world!", "ああっ…だめ", "田中さん", "OK！ 100%", "", "…", "🙂"],
    )
    @pytest.mark.parametrize("target", ["简体中文", "繁體中文", "English"])
    def test_non_han_text(self, text, target):
        assert zh_variant.convert(text, target) == text

    def test_english_target_never_converts(self):
        assert zh_variant.convert("她的頭髮很長", "English") == "她的頭髮很長"
        assert zh_variant.convert("她的头发很长", "English") == "她的头发很长"


def test_a_missing_dependency_degrades_instead_of_failing_a_run(monkeypatch):
    """A venv built before this dependency existed must still translate. The
    conversion is cosmetic; a run is not."""
    monkeypatch.setattr(zh_variant, "_converters", {})
    monkeypatch.setattr(zh_variant, "_unavailable", False)
    import builtins

    real_import = builtins.__import__

    def _no_opencc(name, *args, **kwargs):
        if name == "opencc":
            raise ImportError("No module named 'opencc'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_opencc)
    assert zh_variant.converter_for("简体中文") is None
    assert zh_variant.convert("她的頭髮很長", "简体中文") == "她的頭髮很長"


class TestItReachesTranslatedOutput:
    """Unit-testing the converter proves nothing if the engine never calls it.
    These go through `translate_segments`, i.e. the same path a job takes."""

    @staticmethod
    def _reply_with(text: str):
        import json as _json

        def fake_chat(messages, expected_count=0, on_progress=None, **_kwargs):
            return _json.dumps(
                {"translations": [{"id": i, "text": text} for i in range(expected_count)]},
                ensure_ascii=False,
            )

        return fake_chat

    @staticmethod
    def _segments(count: int) -> list[dict]:
        return [
            {"start": float(i), "end": float(i) + 1.0, "text": f"ja-{i}"}
            for i in range(count)
        ]

    def test_traditional_reply_is_simplified_when_the_user_asked_for_简体(
        self, monkeypatch
    ):
        """Hy-MT2 answers in 繁體 for a share of lines no matter what the prompt
        says, and a wrong-script line passes every other guard."""
        from llm import translator

        monkeypatch.setattr(translator, "_chat", self._reply_with("她的頭髮很長"))
        zh_texts, _timings, _retries = translator.translate_segments(
            self._segments(3), max_workers=1, cache_path="", target_lang="简体中文", glossary=""
        )
        assert zh_texts == ["她的头发很长"] * 3

    def test_simplified_reply_is_traditionalized_when_the_user_asked_for_繁體(
        self, monkeypatch
    ):
        from llm import translator

        monkeypatch.setattr(translator, "_chat", self._reply_with("她的头发很长"))
        zh_texts, _timings, _retries = translator.translate_segments(
            self._segments(3), max_workers=1, cache_path="", target_lang="繁體中文", glossary=""
        )
        assert zh_texts == ["她的頭髮很長"] * 3

    def test_english_target_passes_the_reply_through(self, monkeypatch):
        from llm import translator

        monkeypatch.setattr(translator, "_chat", self._reply_with("Her hair is long"))
        zh_texts, _timings, _retries = translator.translate_segments(
            self._segments(2), max_workers=1, cache_path="", target_lang="English", glossary=""
        )
        assert zh_texts == ["Her hair is long"] * 2

    def test_a_degenerate_reply_still_counts_as_missing(self, monkeypatch):
        """Regression guard on where the conversion was inserted: routing fresh
        output through the normalizer here would turn a truthy-but-empty reply
        into "", and the engine's missing-check is `is None`, so "" would count
        as answered and an empty line would reach the screen."""
        from llm import translator

        monkeypatch.setattr(translator, "_chat", self._reply_with('"'))
        with pytest.raises(Exception):
            translator.translate_segments(
                self._segments(2),
                max_workers=1,
                cache_path="",
                target_lang="简体中文",
                glossary="",
                api_retries=0,
            )
