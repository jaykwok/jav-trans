from __future__ import annotations

from whisper.prealign import clean_text_for_aligner, prepare_text_for_alignment


def test_prepare_text_for_alignment_keeps_display_and_cleans_aligner_text():
    result = prepare_text_for_alignment("  あっっっ♡www 気持ちいい〜  ")

    assert result.display_text == "あっっっ♡www 気持ちいい〜"
    assert result.align_text == "あっっ 気持ちいい"
    assert result.removed_for_alignment is True
    assert result.empty_after_cleaning is False
    assert result.flags == [
        "removed_decoration",
        "removed_laugh_marker",
        "removed_punctuation",
        "compacted_kana_repeat",
    ]


def test_prepare_text_for_alignment_marks_empty_after_cleaning():
    result = prepare_text_for_alignment("~~~♡ｗｗｗ!!!")

    assert result.display_text == "~~~♡www!!!"
    assert result.align_text == ""
    assert result.empty_after_cleaning is True
    assert "empty_after_alignment_cleaning" in result.flags
    assert clean_text_for_aligner("~~~♡ｗｗｗ!!!") == ""


def test_prepare_text_for_alignment_compacts_repeated_phrase_for_display():
    result = prepare_text_for_alignment("いやいやいやいやいや")

    assert result.display_text == "いや、いや"
    assert result.align_text == "いやいや"
