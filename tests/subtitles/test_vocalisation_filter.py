from __future__ import annotations

import pytest

from subtitles.options import SubtitleOptions
from subtitles.vocalisation import (
    drop_vocalisation_runs,
    is_non_semantic_vocalisation,
)


class TestClassification:
    @pytest.mark.parametrize(
        "text",
        ["あっ", "あっ、あっ", "んっ", "はぁ…", "んんっ…んんっ…んんっ…",
         "ちゅっ、ちゅっ", "んぐっ", "ちゅぱぁ…ちゅぱぁ…", "あぁー、あぁー",
         "れろっ、れろっ"],
    )
    def test_pure_vocalisation_is_flagged(self, text):
        assert is_non_semantic_vocalisation(text)

    @pytest.mark.parametrize("text", ["ひぐぅううんっ！", "ぷはぁ", "くちゅ"])
    def test_vocalisation_the_lexicon_misses_is_kept_on_purpose(self, text):
        """Accepted cost, recorded so nobody 'fixes' it by widening the class.

        These need `ぐ`, `く`, `ぷ` - consonants that also spell words, which is
        how the first version came to delete `ちんぽ` and `イッちゃう`. Each could
        be added as a morpheme, but that is an endless list that differs per
        film. The run requirement is what absorbs these gaps instead: a missed
        cue simply breaks a run, and a run of them is still caught by whichever
        members the lexicon does know.
        """
        assert not is_non_semantic_vocalisation(text)

    @pytest.mark.parametrize(
        "text",
        ["うん", "うん…", "はい", "はい、あっ", "ええ", "えっ", "ううん"],
    )
    def test_short_replies_survive(self, text):
        """The whole reason this is not a character-class test.

        `うん` is the ordinary casual "yes" and is spelled from exactly the kana
        a moan uses. Deleting it removes an answer from a conversation.
        """
        assert not is_non_semantic_vocalisation(text)

    @pytest.mark.parametrize("text", ["ふふっ", "ふふふ…", "あはは", "えへへ"])
    def test_laughter_survives(self, text):
        # Non-semantic but communicative; a viewer notices when a laugh vanishes.
        assert not is_non_semantic_vocalisation(text)

    @pytest.mark.parametrize(
        "text",
        [
            # Both of these were deleted by the first character-class version.
            "イッちゃう、イッちゃう…はぁ、はぁ、はぁ",
            "はぁはぁ…れろちんぽ、れろちんぽ…",
            # A word sharing kana with the noise list must never decompose.
            "ちんぽ",
            "らめです",
            "いいよ…",
            "そうよね",
            "すごいね",
        ],
    )
    def test_words_built_from_the_same_kana_are_not_vocalisation(self, text):
        assert not is_non_semantic_vocalisation(text)

    @pytest.mark.parametrize(
        "text",
        [
            # The form the filter actually sees: it runs before the render stage
            # normalises `...` to `…`. Matching only the full-width forms dropped
            # 52 cues on a real film where the full set drops 224.
            "ぅ...ちゅぷっ、",
            "ちゅっ...んんっ、",
            "あはぁー...",
            "ん...はぁ、はぁ...ちゅっ。",
        ],
    )
    def test_ascii_punctuation_is_decoration_too(self, text):
        assert is_non_semantic_vocalisation(text)

    def test_empty_text_is_not_vocalisation(self):
        # "Nothing" is not "noise"; what an empty cue means is the caller's call.
        assert not is_non_semantic_vocalisation("")
        assert not is_non_semantic_vocalisation("…、。")

    def test_kanji_settles_it_without_decomposition(self):
        assert not is_non_semantic_vocalisation("あぁ…気持ちいい")


class TestRunFiltering:
    def _blocks(self, *texts: str) -> list[dict]:
        return [{"text": text} for text in texts]

    def test_an_isolated_vocalisation_cue_is_kept(self):
        """Context stands in for a lexicon that cannot be made complete.

        One `あっ` between two lines of dialogue is more likely a reaction than a
        moaning passage, and no word list can tell them apart.
        """
        blocks = self._blocks("そうなんだ", "あっ", "本当に？")
        kept, diagnostics = drop_vocalisation_runs(blocks, min_run=2)

        assert [b["text"] for b in kept] == ["そうなんだ", "あっ", "本当に？"]
        assert diagnostics["vocalisation_cues_flagged"] == 1
        assert diagnostics["vocalisation_cues_dropped"] == 0
        assert diagnostics["vocalisation_cues_kept_as_isolated"] == 1

    def test_a_run_is_dropped_whole(self):
        blocks = self._blocks("そうなんだ", "あっ", "んっ", "はぁ", "本当に？")
        kept, diagnostics = drop_vocalisation_runs(blocks, min_run=2)

        assert [b["text"] for b in kept] == ["そうなんだ", "本当に？"]
        assert diagnostics["vocalisation_cues_dropped"] == 3
        assert diagnostics["vocalisation_runs_dropped"] == 1

    def test_a_reply_inside_a_run_breaks_it(self):
        """A gap in the word list degrades safely: the run simply ends."""
        blocks = self._blocks("あっ", "うん", "んっ")
        kept, _ = drop_vocalisation_runs(blocks, min_run=2)

        assert [b["text"] for b in kept] == ["あっ", "うん", "んっ"]

    def test_a_run_at_the_end_is_still_a_run(self):
        blocks = self._blocks("本当に？", "あっ", "んっ")
        kept, _ = drop_vocalisation_runs(blocks, min_run=2)

        assert [b["text"] for b in kept] == ["本当に？"]

    def test_min_run_one_drops_every_flagged_cue(self):
        blocks = self._blocks("そうなんだ", "あっ", "本当に？")
        kept, diagnostics = drop_vocalisation_runs(blocks, min_run=1)

        assert [b["text"] for b in kept] == ["そうなんだ", "本当に？"]
        assert diagnostics["vocalisation_cues_kept_as_isolated"] == 0

    def test_diagnostics_report_what_was_removed_not_just_a_shorter_list(self):
        # The cues are gone from the return value, so the counts are the only
        # evidence the caller has that anything happened.
        blocks = self._blocks("あっ", "んっ", "そうなんだ")
        _, diagnostics = drop_vocalisation_runs(blocks, min_run=2)

        assert diagnostics["vocalisation_cues_flagged"] == 2
        assert diagnostics["vocalisation_cues_dropped"] == 2
        assert diagnostics["vocalisation_min_run"] == 2


class TestFilterRunsOnFinishedCues:
    def test_it_reads_the_key_the_layout_stage_actually_uses(self):
        """Cues carry `ja_text` here, not `text`.

        Reading only `text` made every cue look empty, and an empty cue is not
        vocalisation - so the filter reported zero flagged on a film with
        hundreds and appeared to be working correctly.
        """
        blocks = [
            {"ja_text": "そうなんだ"},
            {"ja_text": "あっ"},
            {"ja_text": "んっ"},
            {"ja_text": "本当に？"},
        ]
        kept, diagnostics = drop_vocalisation_runs(blocks, min_run=2)

        assert [b["ja_text"] for b in kept] == ["そうなんだ", "本当に？"]
        assert diagnostics["vocalisation_cues_flagged"] == 2

    def test_it_sees_cues_the_layout_produced_not_the_segments_it_started_from(self):
        """The filter has to run after the DP, not before it.

        A first attempt filtered at the top of the prepare stage. Nothing there
        is a cue yet - the blocks are whole ASR segments, and a segment mixing
        dialogue with moaning is not pure vocalisation. On a real film that
        placement dropped zero cues while 349 of the 1983 finished ones
        qualified, and the pipeline looked like it was working.

        Here one long segment splits into several cues, of which a run is pure
        vocalisation. The segment itself never is.
        """
        from subtitles import writer

        segment_text = "本当にそうなんですか、あっ、あっ、んっ、それで大丈夫ですね"
        blocks = [{"ja_text": segment_text, "start": 0.0, "end": 24.0}]

        options = SubtitleOptions()
        assert not is_non_semantic_vocalisation(segment_text)

        diagnostics: dict = {}
        cues = writer.prepare_srt_blocks(
            blocks, options=options, diagnostics=diagnostics
        )

        assert len(cues) > 1, "the DP must actually split this for the test to bite"
        assert "vocalisation_cues_flagged" in diagnostics
        assert all(
            not is_non_semantic_vocalisation(cue.get("text") or "")
            or diagnostics["vocalisation_cues_kept_as_isolated"] > 0
            for cue in cues
        )


class TestOptions:
    def test_the_filter_is_on_by_default_with_a_run_requirement(self):
        options = SubtitleOptions()

        assert options.drop_vocalisation_only_cues is True
        assert options.vocalisation_min_run == 2

    def test_the_settings_reach_the_cache_signature(self):
        """Otherwise a rerun after switching it off would serve filtered cues."""
        signature = SubtitleOptions().signature()

        assert "drop_vocalisation_only_cues" in signature
        assert "vocalisation_min_run" in signature

    def test_it_can_be_switched_off(self, monkeypatch):
        monkeypatch.setenv("SUBTITLE_DROP_VOCALISATION_ONLY_CUES", "0")
        assert SubtitleOptions.from_env().drop_vocalisation_only_cues is False
