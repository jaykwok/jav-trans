from pathlib import Path

from asr import pipeline as asr

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_word_group_does_not_cross_source_chunk_boundary():
    words = [
        {"start": 0.0, "end": 0.4, "word": "よろしくお願いいたします。", "source_chunk_index": 1},
        {"start": 0.5, "end": 0.9, "word": "お願いします。", "source_chunk_index": 2},
    ]

    segments = asr._group_words_to_segments(words)
    postprocessed = asr._postprocess_segments(segments)

    assert [segment["text"] for segment in postprocessed] == [
        "よろしくお願いいたします。",
        "お願いします。",
    ]


def test_word_group_keeps_same_source_chunk_as_one_segment():
    words = [
        {"start": 0.0, "end": 0.4, "word": "そうですね。", "source_chunk_index": 1},
        {"start": 0.5, "end": 1.0, "word": "ありがとうございます。", "source_chunk_index": 1},
    ]

    segments = asr._group_words_to_segments(words)

    assert [segment["text"] for segment in segments] == ["そうですね。ありがとうございます。"]
    assert segments[0]["source_chunk_index"] == 1


def test_postprocess_keeps_same_chunk_fragments_separate():
    segments = [
        {"start": 0.0, "end": 0.5, "text": "本日は", "source_chunk_index": 1},
        {"start": 0.55, "end": 1.2, "text": "ありがとうございます。", "source_chunk_index": 1},
    ]

    postprocessed = asr._postprocess_segments(segments)

    assert [segment["text"] for segment in postprocessed] == ["本日は", "ありがとうございます。"]


def test_postprocess_keeps_short_domain_vocalizations():
    segments = [
        {"start": 0.0, "end": 0.4, "text": "はぁ", "source_chunk_index": 1},
        {"start": 0.5, "end": 0.9, "text": "うん", "source_chunk_index": 1},
        {"start": 1.0, "end": 1.4, "text": "気持ち", "source_chunk_index": 1},
        {"start": 1.5, "end": 1.9, "text": "好き", "source_chunk_index": 1},
        {"start": 2.0, "end": 2.4, "text": "たたたた", "source_chunk_index": 1},
        {"start": 2.0, "end": 2.4, "text": "。！？", "source_chunk_index": 1},
    ]

    postprocessed = asr._postprocess_segments(segments)

    assert [segment["text"] for segment in postprocessed] == [
        "はぁ",
        "うん",
        "気持ち",
        "好き",
        "たたたた",
    ]


def test_postprocess_does_not_hard_split_long_word_backed_segment():
    text = "これは長い同一ASRチャンクです。後段では時間や文字数で分割しません。"
    segments = [
        {
            "start": 10.0,
            "end": 44.0,
            "text": text,
            "source_chunk_index": 7,
            "words": [
                {
                    "start": 10.0,
                    "end": 44.0,
                    "word": text,
                    "source_chunk_index": 7,
                }
            ],
        }
    ]

    postprocessed = asr._postprocess_segments(segments)

    assert len(postprocessed) == 1
    assert postprocessed[0]["start"] == 10.0
    assert postprocessed[0]["end"] == 44.0
    assert postprocessed[0]["text"] == text


def test_postprocess_preserves_word_backed_context_actor_intro(monkeypatch):
    monkeypatch.setenv("ASR_CONTEXT", "小那海あや")
    words = [
        {"start": 10.46, "end": 10.86, "word": "...小那海", "source_chunk_index": 0},
        {"start": 10.86, "end": 10.86, "word": "あや", "source_chunk_index": 0},
        {"start": 10.86, "end": 11.18, "word": "です。", "source_chunk_index": 0},
        {"start": 11.18, "end": 11.50, "word": "よろしく", "source_chunk_index": 0},
        {"start": 11.50, "end": 11.50, "word": "お", "source_chunk_index": 0},
        {"start": 11.50, "end": 11.90, "word": "願い", "source_chunk_index": 0},
        {"start": 11.90, "end": 12.06, "word": "し", "source_chunk_index": 0},
        {"start": 12.06, "end": 12.30, "word": "ます", "source_chunk_index": 0},
    ]

    postprocessed = asr._postprocess_segments(asr._group_words_to_segments(words))

    assert [segment["text"] for segment in postprocessed] == [
        "...小那海あやです。よろしくお願いします"
    ]


def test_postprocess_does_not_drop_context_like_text_without_word_timing(monkeypatch):
    monkeypatch.setenv("ASR_CONTEXT", "小那海あや")
    segments = [
        {
            "start": 10.46,
            "end": 11.18,
            "text": "...小那海あやです。",
            "source_chunk_index": 0,
            "words": [],
        }
    ]

    postprocessed = asr._postprocess_segments(segments)

    assert [segment["text"] for segment in postprocessed] == ["...小那海あやです。"]


def test_postprocess_locks_next_start_when_repairing_overlap():
    segments = [
        {"start": 0.0, "end": 1.2, "text": "前の字幕", "source_chunk_index": 1},
        {"start": 1.0, "end": 1.5, "text": "次の字幕", "source_chunk_index": 2},
    ]

    postprocessed = asr._postprocess_segments(segments)

    assert [segment["text"] for segment in postprocessed] == ["前の字幕", "次の字幕"]
    assert postprocessed[0]["end"] <= postprocessed[1]["start"]
    assert postprocessed[1]["start"] == 1.0


def test_removed_asr_post_split_runtime_knobs_are_absent():
    removed = (
        "ASR_SEGMENT_HARD_MAX_DURATION",
        "ASR_POSTPROCESS_MAX_CHARS",
        "ASR_POSTPROCESS_MAX_DURATION",
        "ASR_POSTPROCESS_SPLIT_MIN_CHARS",
        "_pick_postprocess_split_index",
        "_split_long_postprocessed_segment",
        "_split_long_postprocessed_segments",
    )
    checked_paths = [
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / "src" / "asr" / "pipeline.py",
        PROJECT_ROOT / "src" / "asr" / "transcribe.py",
        PROJECT_ROOT / "src" / "core" / "config.py",
    ]

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        for key in removed:
            assert key not in text, path
