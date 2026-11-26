"""Japanese TTSG presentation rules.

The rules that carry weight here are I.17 (、 and 。 are replaced by spaces of
two different widths) and I.14 (two lines, never three). The wrapper this
replaced broke after 、 and 。 and had no line cap at all, so both are guarded
directly rather than through the writer.
"""

from subtitles.ja_style import (
    FULLWIDTH_SPACE,
    count_banned_ja_punctuation,
    ja_display_units,
    normalize_ja_subtitle_text,
    wrap_ja_subtitle_text,
)


def test_enumeration_comma_becomes_a_halfwidth_space():
    assert normalize_ja_subtitle_text("待って、行こう") == "待って 行こう"


def test_period_becomes_a_fullwidth_space():
    assert normalize_ja_subtitle_text("そうか。行こう") == f"そうか{FULLWIDTH_SPACE}行こう"


def test_sentence_final_period_leaves_no_trailing_space():
    assert normalize_ja_subtitle_text("もういい。") == "もういい"


def test_terminal_marks_become_fullwidth():
    assert normalize_ja_subtitle_text("本当に!?") == "本当に！"
    assert normalize_ja_subtitle_text("やめて!!") == "やめて！"


def test_fullwidth_space_follows_a_terminal_mark_only_mid_line():
    # I.17 asks for the space "when a new sentence starts on the same line", so
    # a mark that ends the subtitle does not get one.
    assert normalize_ja_subtitle_text("本当？そうか") == f"本当？{FULLWIDTH_SPACE}そうか"
    assert normalize_ja_subtitle_text("本当？") == "本当？"


def test_adjacent_replacements_collapse_to_the_wider_space():
    # `。、` would otherwise stack into a gap two units wide; the sentence end
    # is the stronger break, so its full-width space wins.
    assert normalize_ja_subtitle_text("そうか。、行こう") == (
        f"そうか{FULLWIDTH_SPACE}行こう"
    )


def test_normalize_is_idempotent():
    once = normalize_ja_subtitle_text("<i>待って、行こう。本当に!?</i>")
    assert normalize_ja_subtitle_text(once) == once
    assert count_banned_ja_punctuation(once) == 0


def test_display_units_count_by_width():
    # I.5: full-width 1, half-width 0.5, and spaces count as characters.
    assert ja_display_units("あ") == 1.0
    assert ja_display_units("a") == 0.5
    assert ja_display_units(FULLWIDTH_SPACE) == 1.0
    assert ja_display_units(" ") == 0.5


def test_thirteen_units_stays_on_one_line():
    text = "あいうえおかきくけこさしす"
    assert ja_display_units(text) == 13.0
    assert wrap_ja_subtitle_text(text) == text


def test_wrap_never_produces_three_lines():
    """I.14. The wrapper this replaced looped and produced as many as it liked."""
    rendered = wrap_ja_subtitle_text(
        normalize_ja_subtitle_text(
            "お兄さまの、すごく気持ちいいの、もうだめかもしれない、やめてください"
        )
    )
    assert rendered.count("\n") == 1


def test_wrap_prefers_the_space_that_replaced_punctuation():
    rendered = wrap_ja_subtitle_text(
        normalize_ja_subtitle_text("そうですね。わかりました。では行きましょう。")
    )
    assert rendered == f"そうですね{FULLWIDTH_SPACE}わかりました\nでは行きましょう"


def test_wrap_does_not_open_a_line_with_a_small_kana():
    rendered = wrap_ja_subtitle_text("ああああっあああああ", line_max_units=5)
    top, bottom = rendered.split("\n")
    assert not bottom.startswith("っ")


def test_wrap_keeps_a_katakana_word_whole_when_it_can():
    rendered = wrap_ja_subtitle_text(
        normalize_ja_subtitle_text("彼はコンピューターを使って調べた")
    )
    top, bottom = rendered.split("\n")
    assert "コンピューター" in top or "コンピューター" in bottom


def test_a_split_that_fits_beats_a_free_break_that_overflows():
    """I.5 is a must-stay-zero QC gate, so width outranks break quality.

    A free break after ？ leaves a 14-unit bottom line; the only breaks that fit
    are inside the katakana run, which costs 6.0. At the old overflow weight the
    overflow was worth 3.0 and won. Found on a real film, where a 13.5-unit line
    shipped because a free break and a fitting one tied at 3.00.
    """
    text = "あい？アイウエオカキクケコサシスセ"
    assert ja_display_units(text) == 17.0
    lines = wrap_ja_subtitle_text(text).split("\n")
    assert len(lines) == 2
    assert max(ja_display_units(line) for line in lines) <= 13.0


def test_count_banned_ja_punctuation():
    # I.17 bans the glyphs outright, so position is irrelevant - unlike CHS,
    # where 、 is legal mid-sentence.
    assert count_banned_ja_punctuation("待って、行こう。") == 2
    assert count_banned_ja_punctuation("待って 行こう") == 0
    assert count_banned_ja_punctuation("本当に?") == 1
