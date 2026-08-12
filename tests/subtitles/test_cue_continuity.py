"""A cue split by display duration is half a sentence, and the translator has to
be told so.

The layout DP cuts a long cue at a measured word gap, which keeps it off the
middle of a word - but it still lands in the middle of a sentence. The
translator then sees two independent lines and closes each one off as a complete
utterance, which is where "…想要你插到" / "更里面" turns into two sentences that
do not join.

Two things have to hold for the fix to work at all: inherited continuation flags
must survive a later exact split, and they have to reach the prompt. Both are
pinned here.
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

from llm import prompt as prompt_module  # noqa: E402
from subtitles import writer  # noqa: E402
from subtitles.options import SubtitleOptions  # noqa: E402


def _long_cue(*, start: float = 0.0, end: float = 30.0, text: str = "") -> dict:
    text = text or "こんにちは" * 20
    words = []
    cursor = start
    for index, char in enumerate(text):
        if index and index % 10 == 0:
            cursor += 0.20
        words.append(
            {
                "word": char,
                "start": cursor,
                "end": cursor + 0.20,
                "timestamp_kind": "ctc_forced_alignment",
            }
        )
        cursor += 0.20
    return {
        "start": start,
        "end": end,
        "text": text,
        "ja_text": text,
        "zh_text": text,
        "words": words,
    }


def _split(block: dict) -> list[dict]:
    return writer._split_long_display_block(block, options=SubtitleOptions())


class TestFlagsOnASplitCue:
    def test_an_unsplit_cue_is_not_marked(self) -> None:
        pieces = _split(_long_cue(start=0.0, end=3.0, text="はい"))
        assert len(pieces) == 1
        assert not pieces[0].get("continues_from_previous")
        assert not pieces[0].get("continues_into_next")

    def test_the_first_piece_only_continues_forward(self) -> None:
        pieces = _split(_long_cue())
        assert len(pieces) > 1
        assert pieces[0]["continues_from_previous"] is False
        assert pieces[0]["continues_into_next"] is True

    def test_the_last_piece_only_continues_backward(self) -> None:
        pieces = _split(_long_cue())
        assert pieces[-1]["continues_from_previous"] is True
        assert pieces[-1]["continues_into_next"] is False

    def test_middle_pieces_continue_both_ways(self) -> None:
        pieces = _split(_long_cue(end=60.0, text="こんにちは" * 40))
        assert len(pieces) >= 3
        for piece in pieces[1:-1]:
            assert piece["continues_from_previous"] is True
            assert piece["continues_into_next"] is True


class TestInheritedContinuation:
    """A later exact split must not forget the parent's outer-edge state."""

    def test_a_second_split_keeps_the_inherited_left_edge(self) -> None:
        already_continued = _long_cue()
        already_continued["continues_from_previous"] = True
        pieces = _split(already_continued)
        assert pieces[0]["continues_from_previous"] is True
        assert pieces[-1]["continues_into_next"] is False

    def test_a_second_split_keeps_the_inherited_right_edge(self) -> None:
        already_continued = _long_cue()
        already_continued["continues_into_next"] = True
        pieces = _split(already_continued)
        assert pieces[0]["continues_from_previous"] is False
        assert pieces[-1]["continues_into_next"] is True

    def test_an_interior_piece_is_marked_regardless_of_the_parent(self) -> None:
        pieces = _split(_long_cue())
        assert pieces[0]["continues_into_next"] is True
        assert pieces[-1]["continues_from_previous"] is True


class TestItReachesThePrompt:
    def test_a_marked_cue_carries_the_flags(self) -> None:
        payload = json.loads(
            prompt_module._serialize_segments(
                [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "もっと奥まで",
                        "continues_into_next": True,
                    },
                    {
                        "start": 2.0,
                        "end": 4.0,
                        "text": "欲しいの",
                        "continues_from_previous": True,
                    },
                ]
            )
        )
        assert payload[0]["cont_next"] is True
        assert "cont_prev" not in payload[0]
        assert payload[1]["cont_prev"] is True
        assert "cont_next" not in payload[1]

    def test_an_ordinary_cue_costs_no_extra_tokens(self) -> None:
        """Most cues are whole utterances. A pair of `false` on every line would
        be paid for on every batch of every film to say nothing."""
        payload = json.loads(
            prompt_module._serialize_segments([{"start": 0.0, "end": 1.0, "text": "はい"}])
        )
        assert set(payload[0]) == {"id", "start", "end", "duration_sec", "ja"}

    @pytest.mark.parametrize("marker", ["cont_prev", "cont_next"])
    def test_the_system_prompt_explains_the_marker(self, marker: str) -> None:
        """A field the model has never been told about is noise in the payload."""
        assert marker in prompt_module._SYSTEM_PROMPT_FULL
        assert marker in prompt_module._SYSTEM_PROMPT_COMPACT

    def test_the_instruction_says_not_to_complete_the_fragment(self) -> None:
        assert "不要各自补成完整句" in prompt_module._SYSTEM_PROMPT_COMPACT
        assert "都不要补成完整句子" in prompt_module._SYSTEM_PROMPT_FULL

    def test_the_numbered_rules_are_not_duplicated(self) -> None:
        """Inserting a rule renumbered the one after it; two rules sharing a
        number is how an instruction gets skipped."""
        full = prompt_module._SYSTEM_PROMPT_FULL
        numbers = [line.split(".", 1)[0] for line in full.split("\n") if line[:2].strip().isdigit()]
        assert len(numbers) == len(set(numbers)), numbers
