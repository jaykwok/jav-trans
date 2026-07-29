"""The type track is being re-annotated by listening; pin what that depends on.

Two stem-matching errors in a row (`non_speech`, then bare `noise`) came from
reading the teacher's decision words as descriptions of sound. The replacement
avoids that by construction, and these tests hold the construction in place:

  * the closed set must contain no generic catch-all, since that is the exact
    shape of both previous failures;
  * the model returns a category and nothing else, with the type derived here,
    so a category that disagrees with its type is unrepresentable;
  * anything outside the set becomes `unsure` rather than being coerced, and is
    reported so the vocabulary can be judged rather than silently trusted;
  * the run is resumable, because the free tier covers roughly 440 requests a
    day against ~770 windows, so it will be interrupted.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.datasets.label_typed_span_types import (  # noqa: E402
    CATEGORIES,
    CATEGORY_TYPES,
    build_prompt,
    load_done,
    parse_response,
    window_spans,
)


def test_no_generic_catch_all_in_the_closed_set() -> None:
    """`noise` and `non_speech` are the two words that already went wrong.

    Both name the absence of speech rather than a source, so the annotator must
    not be offered anything of that shape. A sound with no concrete name is
    `uncertain`, which costs one span instead of mislabelling it.
    """
    for banned in ("noise", "non_speech", "no_speech", "non_vocal", "other", "sound"):
        assert banned not in CATEGORY_TYPES


def test_every_category_maps_to_exactly_one_type() -> None:
    assert set(CATEGORY_TYPES.values()) == {
        "speech",
        "non_semantic_vocal",
        "non_vocal",
        "unsure",
    }
    assert CATEGORY_TYPES["uncertain"] == "unsure"


def test_human_produced_categories_are_never_typed_non_vocal() -> None:
    """The failure being designed out: calling a human sound `non_vocal`."""
    for category in ("breath", "moan", "cry", "laugh", "scream", "kiss", "cough"):
        assert CATEGORY_TYPES[category] == "non_semantic_vocal"


def test_prompt_lists_the_closed_set_and_the_spans() -> None:
    spans = [
        {"id": "s000", "start_s": 1.25, "end_s": 1.75},
        {"id": "s001", "start_s": 4.0, "end_s": 4.5},
    ]
    prompt = build_prompt(75.0, spans)
    for category in CATEGORIES:
        assert category in prompt
    assert "s000" in prompt and "s001" in prompt
    assert "75.000" in prompt
    # the instruction that the two stem failures violated
    assert "不要用" in prompt


def test_parse_keeps_only_ids_that_were_asked_about() -> None:
    spans = [{"id": "s000"}, {"id": "s001"}]
    parsed = {
        "span_categories": [
            {"id": "s000", "category": "moan"},
            {"id": "s999", "category": "music"},
        ]
    }
    categories, off = parse_response(parsed, spans)
    assert categories == {"s000": "moan"}
    assert off == []


def test_parse_reports_off_vocabulary_instead_of_coercing() -> None:
    spans = [{"id": "s000"}, {"id": "s001"}]
    parsed = {
        "span_categories": [
            {"id": "s000", "category": "noise"},
            {"id": "s001", "category": "MUSIC"},
        ]
    }
    categories, off = parse_response(parsed, spans)
    assert off == ["noise"]
    assert categories == {"s001": "music"}, "casing is normalised, meaning is not"


def test_a_span_the_model_skipped_becomes_unsure_not_a_guess() -> None:
    spans = [{"id": "s000"}, {"id": "s001"}]
    categories, _ = parse_response(
        {"span_categories": [{"id": "s000", "category": "water"}]}, spans
    )
    assert "s001" not in categories
    assert CATEGORY_TYPES.get(categories.get("s001", ""), "unsure") == "unsure"


@pytest.mark.parametrize(
    "parsed", [None, [], {"nope": []}, {"span_categories": "moan"}]
)
def test_malformed_responses_raise_rather_than_return_junk(parsed) -> None:
    with pytest.raises(ValueError):
        parse_response(parsed, [{"id": "s000"}])


def test_only_the_requested_source_labels_are_typed() -> None:
    """Keep spans are already typed by the segmentation and must not be asked about."""
    row = {
        "spans": [
            {"source_label": "definite_keep", "type": "speech"},
            {"source_label": "definite_drop", "type": "non_vocal"},
            {"source_label": "ambiguous_ignore", "type": "unsure"},
            {"source_label": "definite_drop", "type": "non_semantic_vocal"},
        ]
    }
    spans = window_spans(row, ("definite_drop",))
    assert [s["id"] for s in spans] == ["s000", "s001"]
    assert all(s["source_label"] == "definite_drop" for s in spans)


def test_span_ids_are_positional_and_unique() -> None:
    row = {"spans": [{"source_label": "definite_drop"} for _ in range(12)]}
    identifiers = [s["id"] for s in window_spans(row, ("definite_drop",))]
    assert identifiers[0] == "s000" and identifiers[-1] == "s011"
    assert len(set(identifiers)) == 12


def test_resume_skips_completed_windows(tmp_path: Path) -> None:
    """The free tier cannot finish this in one run, so resume is load-bearing."""
    path = tmp_path / "out.jsonl"
    path.write_text(
        json.dumps({"window_id": "a-w00"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"window_id": "b-w01"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    assert load_done(path) == {"a-w00", "b-w01"}


def test_resume_survives_a_half_written_final_line(tmp_path: Path) -> None:
    """A run killed mid-write must not poison the resume set."""
    path = tmp_path / "out.jsonl"
    path.write_text(
        json.dumps({"window_id": "a-w00"}, ensure_ascii=False) + "\n{\"window_id\": ",
        encoding="utf-8",
    )
    assert load_done(path) == {"a-w00"}


def test_resume_on_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_done(tmp_path / "absent.jsonl") == set()


def test_speech_on_a_drop_span_yields_to_the_frozen_segmentation() -> None:
    """A drop span was judged to carry no semantic speech.

    Writing `speech` into the type track there would assert two incompatible
    things at once, so the type yields and the conflict is counted instead.
    """
    from tools.datasets.label_typed_span_types import resolve_type

    for category in ("dialogue", "whisper", "singing"):
        assert resolve_type(category, "definite_drop") == ("unsure", True)
        assert resolve_type(category, "ambiguous_ignore") == ("unsure", True)
        assert resolve_type(category, "definite_keep") == ("speech", False)


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("moan", "non_semantic_vocal"),
        ("music", "non_vocal"),
        ("silence", "non_vocal"),
        ("uncertain", "unsure"),
        ("noise", "unsure"),
        (None, "unsure"),
    ],
)
def test_non_speech_categories_are_unaffected_by_the_source_label(
    category, expected
) -> None:
    from tools.datasets.label_typed_span_types import resolve_type

    resolved, conflict = resolve_type(category, "definite_drop")
    assert resolved == expected
    assert conflict is False


def test_response_schema_constrains_structure_but_not_the_vocabulary() -> None:
    """Schemas were shown to change decisions, not just formatting.

    So this one may only guarantee that the JSON parses. If `category` were an
    enum, an off-vocabulary answer would be unrepresentable and the run could
    no longer report whether the closed set actually fits.
    """
    from tools.datasets.label_typed_span_types import RESPONSE_SCHEMA

    item = RESPONSE_SCHEMA["properties"]["span_categories"]["items"]
    assert item["properties"]["category"] == {"type": "string"}
    assert "enum" not in json.dumps(RESPONSE_SCHEMA)
    assert item["required"] == ["id", "category"]


def test_aliases_only_ever_point_at_an_existing_category() -> None:
    """An alias that added a MEANING would smuggle the catch-all back in.

    The pilot returned `screaming`, `sigh` and `gasp` as off-vocabulary. Two
    were genuinely missing distinctions and were added to the set; the third
    was an inflection of a category already offered. Only the latter kind may
    be aliased, or normalisation becomes a second, unaudited vocabulary.
    """
    from tools.datasets.label_typed_span_types import CATEGORY_ALIASES

    for alias, target in CATEGORY_ALIASES.items():
        assert target in CATEGORY_TYPES, alias
        assert alias not in CATEGORY_TYPES, f"{alias} is a category, not an alias"


def test_inflected_answers_are_folded_onto_their_category() -> None:
    from tools.datasets.label_typed_span_types import parse_response

    spans = [{"id": "s000"}, {"id": "s001"}]
    categories, off = parse_response(
        {
            "span_categories": [
                {"id": "s000", "category": "screaming"},
                {"id": "s001", "category": "Moaning"},
            ]
        },
        spans,
    )
    assert categories == {"s000": "scream", "s001": "moan"}
    assert off == []


def test_the_pilots_missing_distinctions_are_now_first_class() -> None:
    assert CATEGORY_TYPES["sigh"] == "non_semantic_vocal"
    assert CATEGORY_TYPES["gasp"] == "non_semantic_vocal"


def test_a_generic_word_is_still_not_rescuable_by_alias() -> None:
    """`noise` must remain off-vocabulary no matter how it is spelled."""
    from tools.datasets.label_typed_span_types import parse_response

    _, off = parse_response(
        {"span_categories": [{"id": "s000", "category": "background noise"}]},
        [{"id": "s000"}],
    )
    assert off == ["background noise"]
