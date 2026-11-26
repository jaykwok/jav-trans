"""The calibration set is the only thing standing between a prompt and 456 spans.

Its whole value is that the number it produces was not available while the
prompt was being written. Three things have to hold for that, and each fails
silently:

  * the two halves must not share a video. Same speaker, same room, same mixing;
    the project already lost a model to a split that let provenance leak.
  * the halves must be balanced on WORD-POSITIVE clips, not on total size. Recall
    over those clips is the number the decision turns on, and a half holding 21
    of them cannot clear an 0.85 lower bound even at 21/21.
  * the teacher must hear the same audio file the human heard, so the clips are
    reused from the audit pages rather than recut.

The scoring side has one job beyond arithmetic: not letting a teacher that
answers `words` to everything pass on a perfect recall score.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.audits.build_word_definition_calibration import (  # noqa: E402
    split_by_video,
)
from tools.audits.evaluate_word_teacher_calibration import (  # noqa: E402
    RECALL_FLOOR,
    SPECIFICITY_FLOOR,
    join,
    report,
    wilson,
)
from tools.datasets.label_drop_spans_words import (  # noqa: E402
    LABELS,
    PROMPT,
    RESPONSE_SCHEMA,
    normalize_label,
    parse_response,
)


def _item(index: int, video: str, human: str) -> dict:
    return {
        "item_id": f"a:clip-{index:03d}",
        "video_id": video,
        "human": human,
        "stratum": "drop_duration_weighted",
        "type_label": "non_semantic_vocal",
        "clip_duration_s": 4.0,
    }


# --------------------------------------------------------------------------
# the split


def test_no_video_reaches_both_halves() -> None:
    items = [_item(i, f"v{i // 4}", "words" if i % 3 else "no_words") for i in range(40)]
    development, holdout = split_by_video(items, seed=1)
    assert {i["video_id"] for i in development} & {i["video_id"] for i in holdout} == set()
    assert len(development) + len(holdout) == len(items)


def test_the_halves_are_balanced_on_word_positives_not_on_size() -> None:
    """One video carries many positives; balancing on size would hand them all
    to one half and leave the other unable to measure recall."""
    items = [_item(i, "big", "words") for i in range(20)]
    items += [_item(100 + i, f"v{i}", "words") for i in range(20)]
    items += [_item(200 + i, f"w{i}", "no_words") for i in range(40)]
    development, holdout = split_by_video(items, seed=1)
    counts = [
        sum(1 for i in half if i["human"] == "words")
        for half in (development, holdout)
    ]
    assert abs(counts[0] - counts[1]) <= 2, counts


def test_a_half_can_actually_clear_the_gate_it_is_judged_by() -> None:
    """The earlier size-balanced split left 21 positives on one side, where a
    flawless score still bottoms out at 0.845 - below the gate of the day."""
    positives = 28
    low, _ = wilson(positives, positives)
    assert low >= RECALL_FLOOR
    assert wilson(21, 21)[0] < 0.85, "the sizing failure this test guards against"


def test_the_split_is_deterministic() -> None:
    items = [_item(i, f"v{i // 3}", "words" if i % 2 else "no_words") for i in range(30)]
    first = split_by_video(items, seed=7)
    second = split_by_video(items, seed=7)
    assert [i["item_id"] for i in first[0]] == [i["item_id"] for i in second[0]]
    assert [i["item_id"] for i in first[1]] == [i["item_id"] for i in second[1]]


# --------------------------------------------------------------------------
# the prompt


def test_the_prompt_states_the_mixture_rule() -> None:
    """The defect being fixed: v4's drop EXAMPLES (breathing, moaning) beat its
    drop DEFINITION the moment a clip holds both a moan and a word."""
    assert "混合就算 words" in PROMPT
    assert "同时出现" in PROMPT
    assert "只有整段自始至终没有任何词" in PROMPT


def test_the_prompt_drops_the_intelligibility_requirement() -> None:
    """v4 asked for 可辨认的 semantic speech; the humans were told the opposite."""
    assert "听不懂意思也算 words" in PROMPT
    assert "可辨认" not in PROMPT


def test_the_prompt_counts_fragments_and_backchannels() -> None:
    assert "半个词" in PROMPT and "词的残片" in PROMPT
    for backchannel in ("うん", "はい"):
        assert backchannel in PROMPT


def test_the_prompt_forbids_answering_the_old_question() -> None:
    """v4 conflated 'has words' with 'is worth sending to ASR'."""
    assert "不判断该不该识别" in PROMPT


def test_the_schema_constrains_structure_but_not_the_answer() -> None:
    """An enum would hide an off-vocabulary answer, and a schema is needed at
    all only because its absence wraps the JSON in markdown fences."""
    assert RESPONSE_SCHEMA["properties"]["label"] == {"type": "string"}
    assert "enum" not in json.dumps(RESPONSE_SCHEMA)
    assert RESPONSE_SCHEMA["required"] == ["label"]


def test_an_off_vocabulary_answer_is_surfaced_not_coerced() -> None:
    label, raw = normalize_label("probably speech")
    assert label == "" and raw == "probably speech"


def test_known_inflections_normalise() -> None:
    for given, expected in (("A", "words"), ("no_speech", "no_words"),
                            ("Uncertain", "unsure")):
        assert normalize_label(given)[0] == expected
    assert set(LABELS) == {"words", "no_words", "unsure"}


def test_parse_response_rejects_a_non_object() -> None:
    with pytest.raises(ValueError):
        parse_response(["words"])


# --------------------------------------------------------------------------
# scoring


def _score(pairs: list[tuple[str, str]]) -> dict:
    items, answers = [], []
    for index, (human, teacher) in enumerate(pairs):
        items.append(_item(index, f"v{index}", human))
        answers.append({"item_id": f"a:clip-{index:03d}", "label": teacher})
    return report(join(items, answers), name="t")


def test_a_teacher_that_always_says_words_cannot_pass_on_recall_alone() -> None:
    """Recall 1.00, specificity 0.00 - the trivially gamed case."""
    pairs = [("words", "words")] * 28 + [("no_words", "words")] * 44
    result = _score(pairs)
    assert result["words_recall"]["rate"] == 1.0
    assert result["no_words_specificity"]["rate"] == 0.0
    assert result["verdict"] == "unusable"
    assert "特异度" in result["basis"]


def test_a_teacher_clearing_both_floors_is_trusted() -> None:
    pairs = [("words", "words")] * 28 + [("no_words", "no_words")] * 40
    pairs += [("no_words", "words")] * 4
    result = _score(pairs)
    assert result["verdict"] == "trusted"
    assert result["words_recall"]["ci95"][0] >= RECALL_FLOOR
    assert result["no_words_specificity"]["ci95"][0] >= SPECIFICITY_FLOOR


def test_a_recall_interval_straddling_the_floor_stays_undecided() -> None:
    pairs = [("words", "words")] * 22 + [("words", "no_words")] * 6
    pairs += [("no_words", "no_words")] * 40
    assert _score(pairs)["verdict"] == "undecided"


def test_a_clearly_poor_recall_is_rejected() -> None:
    pairs = [("words", "words")] * 12 + [("words", "no_words")] * 16
    pairs += [("no_words", "no_words")] * 44
    result = _score(pairs)
    assert result["verdict"] == "unusable"
    assert "人工分批听" in result["basis"]


def test_teacher_unsure_is_reported_and_never_redistributed() -> None:
    pairs = [("words", "unsure")] * 4 + [("words", "words")] * 24
    pairs += [("no_words", "no_words")] * 44
    result = _score(pairs)
    assert result["teacher_unsure"] == 4
    assert result["words_recall"]["n"] == 24, "unsure must not count as a miss"


def test_the_projection_says_what_accepting_the_teacher_buys() -> None:
    pairs = [("words", "words")] * 21 + [("words", "no_words")] * 7
    pairs += [("no_words", "no_words")] * 44
    projected = _score(pairs)["projected"]
    assert projected["false_drop_rate_after"] == pytest.approx(0.32 * 0.25, abs=1e-3)
    assert projected["minutes_recovered"] > 0


def test_disagreements_are_listed_with_what_the_teacher_heard() -> None:
    items = [_item(0, "v0", "words")]
    answers = [{"item_id": "a:clip-000", "label": "no_words", "heard": "只有喘息"}]
    result = report(join(items, answers), name="t")
    assert result["disagreements"][0]["heard"] == "只有喘息"


def test_answers_from_the_other_half_are_ignored_not_an_error() -> None:
    """Both runs append to their own file, but a shared file must not crash."""
    items = [_item(0, "v0", "words")]
    answers = [
        {"item_id": "a:clip-000", "label": "words"},
        {"item_id": "b:clip-999", "label": "words"},
    ]
    assert report(join(items, answers), name="t")["clips"] == 1


def test_a_duplicated_answer_is_an_error() -> None:
    items = [_item(0, "v0", "words")]
    answers = [{"item_id": "a:clip-000", "label": "words"}] * 2
    with pytest.raises(ValueError, match="duplicate"):
        join(items, answers)
