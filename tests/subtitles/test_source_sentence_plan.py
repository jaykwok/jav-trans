from copy import deepcopy

import pytest

from subtitles.options import SubtitleOptions
from subtitles.writer import prepare_srt_blocks, write_bilingual_srt


def _source(text, *, gap_after="", gap=0.0, char_s=0.15):
    words = []
    cursor = 0.0
    boundary = text.index(gap_after) + len(gap_after) if gap_after else -1
    for index, char in enumerate(text):
        if index == boundary:
            cursor += gap
        end = cursor if char in "。、！？「」" else cursor + char_s
        words.append({"word": char, "start": cursor, "end": end,
                      "timestamp_kind": "ctc_forced_alignment"})
        cursor = end
    return {"text": text, "ja_text": text, "start": 0.0, "end": cursor, "words": words}


def _plan(block, **options):
    return prepare_srt_blocks([block], options=SubtitleOptions(
        drop_vocalisation_only_cues=False, timing_polish_enabled=False, **options,
    ))


@pytest.mark.parametrize("gap", [0.12, 0.2, 0.8, 5.0])
def test_negation_stays_whole_across_a_pause_even_over_display_targets(gap):
    block = _source("行きたくないわけじゃない。", gap_after="行きたくない", gap=gap)
    original = deepcopy(block)
    cues = _plan(block, max_source_chars=4, max_display_duration_s=1.0)
    assert [cue["ja_text"] for cue in cues] == [block["text"]]
    assert cues[0]["acoustic_start"] == 0.0
    assert cues[0]["acoustic_end"] == block["end"]
    assert cues[0]["duration_soft_cap_violation"]
    assert block == original


def test_each_written_sentence_is_planned_before_translation():
    block = _source("明日は行かない。今日は行く。")
    cues = _plan(block)
    assert [cue["ja_text"] for cue in cues] == ["明日は行かない。", "今日は行く。"]
    assert not cues[0]["continues_into_next"]
    assert not cues[1]["continues_from_previous"]
    assert "".join(word["word"] for cue in cues for word in cue["words"]) == block["text"]


@pytest.mark.parametrize("gap,limit,count", [(0.2, 12, 1), (0.8, 40, 1), (0.8, 12, 2)])
def test_only_overlong_sentences_split_at_completed_source_clauses_with_a_pause(gap, limit, count):
    first, second = "明日の仕事が早く終わったら、", "そちらに寄るよ。"
    block = _source(first + second, gap_after=first, gap=gap)
    cues = _plan(block, max_source_chars=limit)
    assert len(cues) == count
    assert "".join(cue["ja_text"] for cue in cues) == block["text"]
    if count == 2:
        assert [cue["ja_text"] for cue in cues] == [first, second]
        assert cues[0]["text_break_type"] == "clause_with_pause"
        assert cues[1]["start"] == block["words"][len(first)]["start"]
        assert cues[0]["continues_into_next"] and cues[1]["continues_from_previous"]


@pytest.mark.parametrize("text,left", [
    ("行きたくない、わけじゃない。", "行きたくない、"),
    ("私は、明日も駅で待っています。", "私は、"),
    ("パンと牛乳、卵を買ってきてください。", "パンと牛乳、"),
    ("いつ、帰ってくるのか教えてください。", "いつ、"),
    ("昨日話していたあの店に行こう。", "昨日話していた"),
    ("昨日話していた、あの店に行こう。", "昨日話していた、"),
    ("明日の朝までに提出しなければ、いけません。", "明日の朝までに提出しなければ、"),
])
def test_topic_list_relative_clause_and_dependent_tail_are_not_standalone_clauses(text, left):
    cues = _plan(_source(text, gap_after=left, gap=0.8), max_source_chars=5)
    assert [cue["ja_text"] for cue in cues] == [text]


@pytest.mark.parametrize("text,expected", [
    ("「帰る。」と言っていた。待っていよう。", ["「帰る。」と言っていた。", "待っていよう。"]),
    ("「帰ります。」明日また来ます。", ["「帰ります。」", "明日また来ます。"]),
    ("そう…まだ分からない。", ["そう…まだ分からない。"]),
])
def test_quotes_reporting_clauses_and_ellipsis_do_not_create_fragments(text, expected):
    assert [cue["ja_text"] for cue in _plan(_source(text))] == expected


def test_measured_tokens_must_support_the_source_boundary():
    text = "明日は行かない。今日は行く。"
    block = {"text": text, "start": 0.0, "end": 4.0, "words": [
        {"word": text, "start": 0.0, "end": 4.0, "timestamp_kind": "grok_stt_word"}
    ]}
    assert len(_plan(block)) == 1


def test_an_attached_translation_can_never_be_divided_by_source_offsets():
    block = _source("明日は行かない。今日は行く。")
    block["zh_text"] = "明天不去 今天去"
    cues = _plan(block, max_source_chars=4, max_display_duration_s=1.0)
    assert len(cues) == 1 and cues[0]["zh_text"] == block["zh_text"]
    assert cues[0]["subtitle_layout_split_skipped"] == "translation_already_attached"


def test_writing_long_translations_keeps_source_cue_count_and_times(tmp_path):
    cues = _plan(_source("明日は行かない。今日は行く。"))
    for cue in cues:
        cue["zh_text"] = "这是一条已经根据日语原文确定了时间的完整译文即使很长也不能被切成多条字幕"
    path = tmp_path / "sample-a.srt"
    written = write_bilingual_srt(cues, str(path))
    assert len(written) == len(cues) == 2
    assert [(cue["start"], cue["end"]) for cue in written] == [(cue["start"], cue["end"]) for cue in cues]
    assert path.read_text(encoding="utf-8-sig").count("-->") == 2


@pytest.mark.parametrize("middle", ["あっ、", "はい、うん、", "まだ帰りたくない、"])
def test_non_speech_isolation_never_cuts_isolated_reactions_or_protected_speech(middle):
    text = "そうなんですか、" + middle + "それならここで待っています。"
    cues = prepare_srt_blocks([_source(text)], options=SubtitleOptions())
    assert [cue["ja_text"] for cue in cues] == [text]
