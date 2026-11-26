"""The DP's extent lookup, and the scan it replaced.

Every candidate edge asks for the spoken edges of a word range, and the original
answer re-sliced the word list and scanned it - O(n) per edge, so with dense
candidates the lookups alone cost more than the O(k^2) scoring they feed.

Precomputing the two directions makes each lookup two array reads. Nothing else
changes, which is exactly what needs holding down: the scan is kept as the
reference definition and the index is required to agree with it everywhere,
including the edges that decide whether a piece exists at all - punctuation-only
ranges, empty ranges, and ranges that start or end on zero-width tokens.
"""

from __future__ import annotations

import random

from subtitles.writer import _LexicalExtents, _exact_lexical_extent


def _word(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end}


def _random_words(rng: random.Random, count: int) -> list[dict]:
    words: list[dict] = []
    cursor = 0.0
    for index in range(count):
        cursor += rng.uniform(0.0, 0.3)
        if rng.random() < 0.35:
            # Zero-width punctuation: present in the text, silent in the audio.
            words.append(_word("、", cursor, cursor))
            continue
        end = cursor + rng.uniform(0.05, 0.4)
        words.append(_word(f"w{index}", cursor, end))
        cursor = end
    return words


def test_the_index_answers_exactly_what_the_scan_answers():
    rng = random.Random(20260906)
    for _ in range(40):
        words = _random_words(rng, rng.randint(0, 24))
        extents = _LexicalExtents(words)
        for start in range(len(words) + 1):
            for end in range(start, len(words) + 1):
                assert extents(start, end) == _exact_lexical_extent(words, start, end)


def test_a_punctuation_only_range_has_no_spoken_edge():
    words = [_word("あ", 0.0, 0.4), _word("、", 0.4, 0.4), _word("。", 0.4, 0.4)]
    extents = _LexicalExtents(words)

    assert extents(1, 3) is None
    assert extents(2, 2) is None
    # A range that merely *ends* in punctuation still has the edges of the word
    # inside it - the silent tokens neither extend nor shorten the cue.
    assert extents(0, 3) == (0.0, 0.4)


def test_the_edges_come_from_the_spoken_words_not_the_range():
    words = [
        _word("「", 0.0, 0.0),
        _word("あ", 0.1, 0.5),
        _word("、", 0.5, 0.5),
        _word("い", 0.9, 1.4),
        _word("」", 1.4, 1.4),
    ]
    extents = _LexicalExtents(words)

    assert extents(0, 5) == (0.1, 1.4)
    assert extents(0, 3) == (0.1, 0.5)
    assert extents(2, 5) == (0.9, 1.4)


def test_a_word_list_with_nothing_spoken_never_yields_an_extent():
    words = [_word("、", 0.2, 0.2), _word("。", 0.4, 0.4)]
    extents = _LexicalExtents(words)

    assert all(
        extents(start, end) is None
        for start in range(3)
        for end in range(start, 3)
    )


def test_an_empty_word_list_is_answerable():
    extents = _LexicalExtents([])

    assert extents(0, 0) is None
