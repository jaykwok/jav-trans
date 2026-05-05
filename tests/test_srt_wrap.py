from subtitles import writer as subtitle
def test_wrap_subtitle_line_keeps_short_text():
    text = "这是一句短字幕"

    assert subtitle._wrap_subtitle_line(text, max_chars=25) == text


def test_wrap_subtitle_line_uses_punctuation_then_hard_split():
    punctuated = "这是很长的一句字幕，需要优先在逗号处折行"
    hard = "这是一句没有任何标点但必须折行的超长字幕"

    assert subtitle._wrap_subtitle_line(punctuated, max_chars=12) == (
        "这是很长的一句字幕，\n需要优先在逗号处折行"
    )
    assert subtitle._wrap_subtitle_line(hard, max_chars=10) == (
        "这是一句没有任何标点\n但必须折行的超长字幕"
    )

