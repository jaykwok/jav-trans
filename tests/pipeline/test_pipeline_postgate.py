"""The post-gate as it sits in the transcription path.

The whole point of moving the gate after the decoder is that its verdicts are
reversible, so the property under test is mostly a negative one: no matter what
it decides, the same chunks come out the other side. A flag is an annotation on
a cue that still exists, and these tests fail if that ever stops being true.

The second property is that `alignment_score` actually reaches the gate. It is
the only hallucination signal in the design - a runaway loop is caught by
`unique_ratio`, but fluent invented text is not - and it travels a long way
(alignment head -> `timing_meta` -> finalize payload -> here), so a silent break
anywhere along it would leave the gate quietly blind rather than failing.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr import pipeline as asr  # noqa: E402
from asr.postgate import POSTGATE_SCHEMA  # noqa: E402


def _chunks(count: int, *, seconds: float = 6.0) -> list[dict]:
    return [
        {"index": index, "start": index * seconds, "end": (index + 1) * seconds}
        for index in range(count)
    ]


def _prepared(texts: list[str], scores: list[float | None] | None = None):
    scores = scores or [None] * len(texts)
    prepared = []
    for text, score in zip(texts, scores):
        result = {"text": text, "raw_text": text, "duration": 6.0}
        if score is not None:
            result["timing_meta"] = {"alignment_score": score}
        prepared.append((result, []))
    return prepared


def _transcript(chunks: list[dict], texts: list[str]) -> list[dict]:
    return [
        {"index": chunk["index"], "text": text, "raw_text": text}
        for chunk, text in zip(chunks, texts)
    ]


def test_clean_text_is_reviewed_and_unflagged():
    texts = ["今日は天気がいいですね", "その本を貸してくれますか"]
    chunks = _chunks(2)
    transcript = _transcript(chunks, texts)
    log: list[str] = []

    report = asr._apply_postgate(chunks, _prepared(texts), transcript, log=log)

    assert report["schema"] == POSTGATE_SCHEMA
    assert report["reviewed"] == 2
    assert report["flagged"] == 0
    assert report["flags"] == {}
    assert all(entry["postgate_flags"] == [] for entry in transcript)
    assert log == []


def test_runaway_repetition_is_flagged_but_the_chunk_survives():
    texts = ["ちゃんとした文章です", "んっ" * 40]
    chunks = _chunks(2)
    transcript = _transcript(chunks, texts)
    log: list[str] = []

    report = asr._apply_postgate(chunks, _prepared(texts), transcript, log=log)

    assert report["flagged"] == 1
    assert "runaway_repetition" in report["flags"]
    # Reversible: the flagged cue is still here, with its text intact.
    assert len(transcript) == 2
    assert transcript[1]["text"] == texts[1]
    assert transcript[1]["postgate_flags"]
    assert transcript[0]["postgate_flags"] == []
    assert any("后置闸标记" in line for line in log)


def test_alignment_score_reaches_the_verdict():
    texts = ["音は合っています", "こちらも合っています"]
    chunks = _chunks(2)
    transcript = _transcript(chunks, texts)

    report = asr._apply_postgate(
        chunks, _prepared(texts, [-1.44, None]), transcript, log=[]
    )

    assert transcript[0]["alignment_score"] == -1.44
    assert transcript[1]["alignment_score"] is None
    # No threshold is calibrated yet, so carrying the score must not by itself
    # turn into a check - see `asr.postgate.PostGateConfig.min_alignment_score`.
    assert report["alignment_score_checked"] == 0
    assert report["flagged"] == 0


def test_mismatched_lengths_review_nothing_rather_than_guess():
    chunks = _chunks(3)
    texts = ["一つ目です", "二つ目です"]
    transcript = _transcript(chunks[:2], texts)

    report = asr._apply_postgate(chunks, _prepared(texts), transcript, log=[])

    assert report == asr._empty_postgate_report()
    assert all("postgate_flags" not in entry for entry in transcript)


def test_flags_follow_chunk_index_not_list_position():
    # Transcript entries are not required to be in chunk order, and quarantined
    # chunks can leave gaps; pairing by position would mislabel the survivors.
    chunks = _chunks(3)
    texts = ["正常な行です", "んっ" * 40, "これも正常です"]
    transcript = list(reversed(_transcript(chunks, texts)))

    asr._apply_postgate(chunks, _prepared(texts), transcript, log=[])

    by_index = {entry["index"]: entry for entry in transcript}
    assert by_index[1]["postgate_flags"]
    assert by_index[0]["postgate_flags"] == []
    assert by_index[2]["postgate_flags"] == []


def test_flags_reach_the_segments_the_subtitle_layer_filters():
    transcript = [
        {"index": 0, "postgate_flags": []},
        {"index": 1, "postgate_flags": ["runaway_repetition"]},
    ]
    segments = [
        {"words": [{"source_chunk_index": 0}]},
        {"words": [{"source_chunk_index": 1}]},
        # Straddles the clean and the flagged chunk: the union is what a
        # downstream filter needs to see.
        {"words": [{"source_chunk_index": 0}, {"source_chunk_index": 1}]},
    ]

    annotated = asr._annotate_segments_with_postgate(segments, transcript)

    assert "postgate_flags" not in annotated[0]
    assert annotated[1]["postgate_flags"] == ["runaway_repetition"]
    assert annotated[2]["postgate_flags"] == ["runaway_repetition"]


def test_segments_without_flags_are_left_untouched():
    segments = [{"words": [{"source_chunk_index": 0}]}]
    annotated = asr._annotate_segments_with_postgate(
        segments, [{"index": 0, "postgate_flags": []}]
    )
    assert annotated[0] == {"words": [{"source_chunk_index": 0}]}


def test_empty_report_is_shaped_like_a_real_one():
    empty = asr._empty_postgate_report()
    report = asr._apply_postgate(
        _chunks(1), _prepared(["何か話しています"]), _transcript(_chunks(1), ["何か話しています"]), log=[]
    )
    assert set(empty) == set(report)
