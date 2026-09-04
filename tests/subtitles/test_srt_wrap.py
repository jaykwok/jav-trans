from subtitles import writer as subtitle
from subtitles.options import SubtitleOptions
from subtitles.zh_style import wrap_zh_subtitle_text, zh_display_units


def test_wrap_zh_keeps_short_text_single_line():
    text = "这是一句短字幕"

    assert wrap_zh_subtitle_text(text) == text


def test_wrap_zh_sixteen_units_stays_single_line():
    text = "一二三四五六七八九十一二三四五六"

    assert zh_display_units(text) == 16.0
    assert wrap_zh_subtitle_text(text) == text


def test_wrap_zh_seventeen_units_breaks_into_two_lines():
    text = "一二三四五六七八九十一二三四五六七"
    wrapped = wrap_zh_subtitle_text(text)
    top, bottom = wrapped.split("\n")

    assert zh_display_units(top) <= 16.0
    assert zh_display_units(bottom) <= 16.0
    assert zh_display_units(top) <= zh_display_units(bottom)


def test_wrap_zh_prefers_punctuation_break():
    assert wrap_zh_subtitle_text("他说了很多话…但是我一句都没有听进去") == (
        "他说了很多话…\n但是我一句都没有听进去"
    )


def test_wrap_zh_never_exceeds_two_lines():
    text = "这是一段远远超过三十二个全角字符宽度限制的超长中文字幕文本它依然只能折成两行显示"
    wrapped = wrap_zh_subtitle_text(text)

    assert wrapped.count("\n") == 1


def test_wrap_zh_flattens_manual_breaks_before_wrapping():
    wrapped = wrap_zh_subtitle_text("短句\n另一段")

    assert wrapped == "短句 另一段"


# The ja line renders under the Japanese guide, which replaces 、 and 。 with
# spaces rather than breaking after them.
def test_ja_render_replaces_punctuation_and_breaks_at_the_space():
    rendered = subtitle._render_ja_subtitle_text(
        "これはとても長い字幕で、句読点で折り返す必要がある",
        options=SubtitleOptions(),
    )

    assert "、" not in rendered
    assert rendered == "これはとても長い字幕で\n句読点で折り返す必要がある"


def test_ja_render_never_exceeds_two_lines():
    """I.14. The wrapper this replaced looped instead, and produced three."""
    rendered = subtitle._render_ja_subtitle_text(
        "お兄さまの、すごく気持ちいいの、もうだめかもしれない、やめてください",
        options=SubtitleOptions(),
    )

    assert rendered.count("\n") == 1
