from whisper.model_backend import _post_clean_whisper_text


def test_post_clean_removes_short_parentheses_and_fullwidth_brackets():
    assert _post_clean_whisper_text("いい(笑)気持ち【BGM】") == "いい気持ち"


def test_post_clean_collapses_long_single_character_runs_to_six():
    assert _post_clean_whisper_text("あああああああああああ") == "あ" * 6


def test_post_clean_keeps_normal_text():
    assert _post_clean_whisper_text("気持ちいい") == "気持ちいい"

