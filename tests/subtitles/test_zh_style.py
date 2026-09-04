from subtitles.zh_style import (
    count_banned_punctuation,
    normalize_zh_subtitle_text,
    wrap_zh_subtitle_text,
    zh_display_units,
)


def test_normalize_strips_markup_tags():
    assert normalize_zh_subtitle_text("<i>轻一点</i>") == "轻一点"
    assert normalize_zh_subtitle_text("{\\i1}轻一点{\\i0}") == "轻一点"
    assert normalize_zh_subtitle_text('<font color="red">你好</font>') == "你好"


def test_normalize_converts_fullwidth_digits():
    assert normalize_zh_subtitle_text("现在是１２点３０分") == "现在是12点30分"


def test_normalize_halfwidth_terminal_marks_become_fullwidth():
    assert normalize_zh_subtitle_text("真的吗?") == "真的吗？"
    assert normalize_zh_subtitle_text("快住手!") == "快住手！"


def test_normalize_collapses_terminal_mark_combinations():
    assert normalize_zh_subtitle_text("什么？！") == "什么？"
    assert normalize_zh_subtitle_text("什么！？") == "什么！"
    assert normalize_zh_subtitle_text("不要？？？") == "不要？"
    assert normalize_zh_subtitle_text("住手!!") == "住手！"


def test_normalize_unifies_ellipsis_variants_to_single_u2026():
    assert normalize_zh_subtitle_text("等等……") == "等等…"
    assert normalize_zh_subtitle_text("等等。。。") == "等等…"
    assert normalize_zh_subtitle_text("等等...") == "等等…"
    assert normalize_zh_subtitle_text("等等⋯") == "等等…"
    assert normalize_zh_subtitle_text("等等……。。。") == "等等…"


def test_normalize_replaces_comma_and_period_with_space():
    assert normalize_zh_subtitle_text("好了，我们走吧") == "好了 我们走吧"
    assert normalize_zh_subtitle_text("我明白了。原来如此") == "我明白了 原来如此"
    assert normalize_zh_subtitle_text("我明白了。") == "我明白了"


def test_normalize_keeps_numeric_thousands_comma():
    assert normalize_zh_subtitle_text("一共1,000元") == "一共1,000元"


def test_normalize_enumeration_comma_kept_inside_stripped_at_end():
    assert normalize_zh_subtitle_text("苹果、香蕉和橘子") == "苹果、香蕉和橘子"
    assert normalize_zh_subtitle_text("苹果、香蕉、") == "苹果、香蕉"


def test_normalize_converts_paired_halfwidth_quotes():
    assert normalize_zh_subtitle_text('他说"过来"了') == "他说“过来”了"


def test_normalize_collapses_whitespace():
    assert normalize_zh_subtitle_text("你  好　世界 ") == "你 好 世界"


def test_normalize_is_idempotent():
    messy = '<i>他说"你好?!"，然后。。。走了，，１２点!!</i>'
    once = normalize_zh_subtitle_text(messy)
    assert normalize_zh_subtitle_text(once) == once
    assert count_banned_punctuation(once) == 0


def test_normalize_empty_and_none_like():
    assert normalize_zh_subtitle_text("") == ""
    assert normalize_zh_subtitle_text("。") == ""


def test_wrap_keeps_short_line_single():
    assert wrap_zh_subtitle_text("这是一句短字幕") == "这是一句短字幕"


def test_wrap_prefers_break_after_punctuation():
    assert wrap_zh_subtitle_text("他说了很多话…但是我一句都没有听进去") == (
        "他说了很多话…\n但是我一句都没有听进去"
    )


def test_wrap_bottom_heavy_pyramid_without_punctuation():
    assert wrap_zh_subtitle_text("这是一句完全没有标点的超长中文字幕文本内容") == (
        "这是一句完全没有标点\n的超长中文字幕文本内容"
    )


def test_wrap_never_splits_ascii_word():
    wrapped = wrap_zh_subtitle_text("我的朋友叫Alexander我们现在一起出发吧")
    top, bottom = wrapped.split("\n")
    assert "Alexander" in top or "Alexander" in bottom


def test_wrap_overflow_still_two_lines():
    text = "这是一段远远超过三十二个全角字符宽度限制的超长中文字幕文本它依然只能折成两行显示"
    wrapped = wrap_zh_subtitle_text(text)
    assert wrapped.count("\n") == 1


def test_wrap_two_lines_within_limit_when_feasible():
    text = "今天的天气真的非常好我们一起出去外面玩吧"
    wrapped = wrap_zh_subtitle_text(text)
    top, bottom = wrapped.split("\n")
    assert zh_display_units(top) <= 16.0
    assert zh_display_units(bottom) <= 16.0
    assert zh_display_units(top) <= zh_display_units(bottom)


def test_display_units_weights_ascii_lighter():
    assert zh_display_units("你好") == 2.0
    assert zh_display_units("ab") < 2.0


def test_wrap_does_not_end_a_line_with_enumeration_comma():
    # The guide allows 、 mid-sentence but not at the end of a line, and a break
    # is the one place the wrap can create such a line end for itself.
    wrapped = wrap_zh_subtitle_text("苹果、香蕉、橘子都要买一些回来给他")
    top, bottom = wrapped.split("\n")
    assert not top.endswith("、")
    assert count_banned_punctuation(wrapped) == 0


def test_wrap_strips_an_enumeration_comma_it_cannot_avoid():
    # Breaking after 、 costs the same as breaking before one and more than any
    # ordinary position, so this needs ASCII runs on both sides to win at all.
    # When it does win, the 、 goes rather than the rule.
    assert wrap_zh_subtitle_text("Alexanderabc、Christopherwalken") == (
        "Alexanderabc\nChristopherwalken"
    )


def test_wrap_breaks_at_the_space_that_replaced_a_comma():
    assert wrap_zh_subtitle_text(
        normalize_zh_subtitle_text("你去买苹果、香蕉，我留在这里等他们回来")
    ) == "你去买苹果、香蕉\n我留在这里等他们回来"


def test_count_banned_punctuation():
    assert count_banned_punctuation("你好，世界。") == 2
    assert count_banned_punctuation("等等……") == 1
    assert count_banned_punctuation("真的?") == 1
    assert count_banned_punctuation("一共1,000元") == 0
    assert count_banned_punctuation("好 我们走") == 0


def test_count_banned_punctuation_flags_line_final_enumeration_comma():
    assert count_banned_punctuation("苹果、香蕉、\n橘子都要买") == 1
    assert count_banned_punctuation("苹果、") == 1
    assert count_banned_punctuation("苹果、\n香蕉、") == 2
    # Mid-sentence and mid-line is where the guide permits it.
    assert count_banned_punctuation("苹果、香蕉和橘子") == 0
