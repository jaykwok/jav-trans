from subtitles import writer as subtitle
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


# The ja line of bilingual output keeps the legacy wrapper and its
# hiragana→kanji boundary heuristic.
def test_wrap_subtitle_line_uses_punctuation_then_hard_split():
    punctuated = "これはとても長い字幕で、句読点で折り返す必要がある"
    hard = "この字幕には句読点がないので強制的に折り返す"

    assert subtitle._wrap_subtitle_line(punctuated, max_chars=12).startswith(
        "これはとても長い字幕で、\n"
    )
    assert "\n" in subtitle._wrap_subtitle_line(hard, max_chars=10)
