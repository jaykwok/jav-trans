from whisper import pipeline as asr


def test_word_merge_does_not_cross_source_chunk_boundary():
    words = [
        {"start": 0.0, "end": 0.4, "word": "よろしくお願いいたします。", "source_chunk_index": 1},
        {"start": 0.5, "end": 0.9, "word": "お願いします。", "source_chunk_index": 2},
    ]

    segments = asr._merge_words_to_segments(words)
    postprocessed = asr._postprocess_segments(segments)

    assert [segment["text"] for segment in postprocessed] == [
        "よろしくお願いいたします。",
        "お願いします。",
    ]


def test_word_merge_splits_compact_sentence_turns_inside_chunk():
    words = [
        {"start": 0.0, "end": 0.4, "word": "そうですね。", "source_chunk_index": 1},
        {"start": 0.5, "end": 1.0, "word": "ありがとうございます。", "source_chunk_index": 1},
    ]

    segments = asr._merge_words_to_segments(words)

    assert [segment["text"] for segment in segments] == [
        "そうですね。",
        "ありがとうございます。",
    ]


def test_postprocess_keeps_same_chunk_fragments_mergeable():
    segments = [
        {"start": 0.0, "end": 0.5, "text": "本日は", "source_chunk_index": 1},
        {"start": 0.55, "end": 1.2, "text": "ありがとうございます。", "source_chunk_index": 1},
    ]

    postprocessed = asr._postprocess_segments(segments)

    assert [segment["text"] for segment in postprocessed] == ["本日はありがとうございます。"]
