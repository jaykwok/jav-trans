from __future__ import annotations

import pytest

from subtitles.options import SubtitleOptions
from subtitles.vocalisation import (
    CueAcoustics,
    classify_cue,
    drop_vocalisation_runs,
    is_non_semantic_vocalisation,
    mixed_cue_split_point,
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
        ["いい", "いい…", "いいっ", "いいー", "あっ、いい", "いい、あんっ",
         "おい", "おーい", "おう", "ふーん", "へぇ", "へえ"],
    )
    def test_high_frequency_words_spelled_from_core_kana_survive(self, text):
        """`いい` is the expensive one, and it cost twice over.

        This module is shared with the CTC training-target builder, so the same
        verdict deleted the cue in production *and* taught the alignment head
        that a breathy `いい` is blank. The archived NSFW strip manifest removed
        it 169 times and kept it 0 times; `いいよ`/`いいの` survived only because
        they carry a kana the core set happens not to have.
        """
        assert not is_non_semantic_vocalisation(text)

    @pytest.mark.parametrize(
        "text", ["はいっ", "いえっ", "はーいっ", "ええっ", "おいっ", "いいっ"]
    )
    def test_an_emphatic_trailing_geminate_does_not_unprotect_a_word(self, text):
        """The hand-written list leaked through its own gaps.

        `はい` was protected and `はいっ` was not, which stripped it 48 times in
        the archived targets. Writing out every emphatic form would double the
        list and the next one would be missed the same way.
        """
        assert not is_non_semantic_vocalisation(text)

    @pytest.mark.parametrize("text", ["いっ", "うっ", "おっ", "いっ、いっ"])
    def test_bare_grunts_are_still_vocalisation(self, text):
        """The deliberate limit of the geminate rule, and of the allow-list.

        `いっ` is as often a truncated `イッ(く)` as a grunt, and nothing in the
        text can say which. It reduces to `い`, which is not protected, so the
        rule cannot promote it - that judgement belongs to the acoustic verdict,
        not to this list.
        """
        assert is_non_semantic_vocalisation(text)

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

    def test_two_consecutive_ii_cues_do_not_form_a_run(self):
        """The failure the run rule cannot absorb once the lexicon is wrong.

        Two neighbouring `いいっ` used to be a run of two and both went. Context
        only rescues a word when its neighbours are words, and in this domain a
        `いい` is usually surrounded by moaning.
        """
        blocks = self._blocks("あっ", "いいっ", "いいっ", "んっ")
        kept, diagnostics = drop_vocalisation_runs(blocks, min_run=2)

        assert [b["text"] for b in kept] == ["あっ", "いいっ", "いいっ", "んっ"]
        assert diagnostics["vocalisation_cues_dropped"] == 0

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
        words = []
        cursor = 0.0
        for char in segment_text:
            if char == "、":
                words.append(
                    {
                        "word": char,
                        "start": cursor,
                        "end": cursor,
                        "timestamp_kind": "ctc_forced_alignment",
                    }
                )
                continue
            words.append(
                {
                    "word": char,
                    "start": cursor,
                    "end": cursor + 0.6,
                    "timestamp_kind": "ctc_forced_alignment",
                }
            )
            cursor += 0.6
        blocks = [{
            "ja_text": segment_text,
            "start": 0.0,
            "end": cursor,
            "words": words,
        }]

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


class TestTheJointVerdict:
    """Text and acoustics together, and which way the combination runs.

    The acoustics only ever ADD a reason to drop. Requiring them to confirm the
    text rule before honouring it was measured on eight films and put 457 cues
    back on screen - sampled, they were `あっ、あっ、あっ`, `あんっ!あんっ!`,
    plain moaning the acoustics simply failed to confirm. Text evidence for a run
    of pure-vocalisation cues is already strong; the frame head is here to reach
    what text cannot see.
    """

    MOAN = CueAcoustics(
        silence=0.05, vocalisation=0.92, speech=0.03, speech_max_run_s=0.0
    )
    TALKING = CueAcoustics(
        silence=0.10, vocalisation=0.15, speech=0.75, speech_max_run_s=1.4
    )

    def test_an_isolated_moan_is_now_reachable(self):
        """Symptom (e). One `あっ` between two lines used to survive because the
        lexicon could not tell a moan from a gasp answering something said, and
        the run rule stood in for that. The acoustics can say it directly."""
        verdict = classify_cue("あっ", self.MOAN)

        assert verdict.drop
        assert verdict.reason == "vocal_text_vocal_audio"

    def test_onomatopoeia_the_lexicon_misses_is_reachable_too(self):
        """Symptom (f). `くちゅ` needs consonants that also spell words, so the
        allow-list can never contain it without deleting `ちんぽ`."""
        verdict = classify_cue("くちゅ", self.MOAN)

        assert verdict.drop
        assert verdict.reason == "kana_text_vocal_audio"

    def test_breathy_speech_written_in_core_kana_is_kept_and_marked(self):
        verdict = classify_cue("あぁー", self.TALKING)

        assert not verdict.drop
        assert verdict.reason == "vocal_text_speech_audio"

    def test_a_protected_word_survives_any_acoustics(self):
        """The rule an early evaluation of this omitted, which promptly deleted
        `ふふふ` and a bare `いい`. Nothing in the sound separates them from a
        moan - that is the whole reason the allow-list exists."""
        for text in ("ふふふ", "いい", "はいっ", "うん"):
            verdict = classify_cue(text, self.MOAN)
            assert not verdict.drop, text
            assert verdict.reason == "protected", text

    def test_kanji_is_never_dropped_by_acoustics(self):
        """The ASR hallucinating a word over moaning is real - one was found by
        ear. But the frame head cannot separate it from ordinary dialogue at a
        usable rate, so the verdict marks and never deletes."""
        verdict = classify_cue("気持ちいい", self.MOAN)

        assert not verdict.drop
        assert verdict.reason == "lexical_text_vocal_audio"

    def test_without_acoustics_it_falls_back_to_the_text_rule(self):
        """A v1 head produces no frame classes, and a promoted head outlives the
        code that trained it - so rolling back must not disable the filter."""
        verdict = classify_cue("あっ", None)

        assert not verdict.drop
        assert verdict.reason == "no_acoustics"

    def test_the_run_rule_keeps_every_drop_it_already_made(self):
        """The additive direction, at the level the filter actually runs.

        These three moans sit in a run of two, which the text rule condemns. The
        acoustics say `speech` for the middle one - and it still goes, because
        re-examining what text already settled is what lost 457 cues.
        """
        blocks = [
            {"text": "そうなんだ"},
            {"text": "あっ", "acoustic_classes": {
                "silence": 0.05, "vocalisation": 0.92, "speech": 0.03,
                "speech_max_run_s": 0.0}},
            {"text": "んっ", "acoustic_classes": {
                "silence": 0.10, "vocalisation": 0.15, "speech": 0.75,
                "speech_max_run_s": 1.4}},
            {"text": "本当に？"},
        ]
        kept, diagnostics = drop_vocalisation_runs(blocks, min_run=2)

        assert [b["text"] for b in kept] == ["そうなんだ", "本当に？"]
        assert diagnostics["vocalisation_cues_dropped"] == 2
        assert diagnostics["vocalisation_cues_dropped_by_acoustics"] == 0

    def test_acoustics_reach_the_isolated_cue_the_run_rule_cannot(self):
        blocks = [
            {"text": "そうなんだ"},
            {"text": "あっ", "acoustic_classes": {
                "silence": 0.05, "vocalisation": 0.92, "speech": 0.03,
                "speech_max_run_s": 0.0}},
            {"text": "本当に？"},
        ]
        kept, diagnostics = drop_vocalisation_runs(blocks, min_run=2)

        assert [b["text"] for b in kept] == ["そうなんだ", "本当に？"]
        assert diagnostics["vocalisation_cues_dropped_by_acoustics"] == 1
        assert diagnostics["vocalisation_cues_kept_as_isolated"] == 0

    def test_a_marked_cue_carries_its_verdict_forward(self):
        """A verdict nobody downstream can read is a detector running for
        nothing - the same failure the post-gate had before its flags reached
        the segments."""
        blocks = [
            {"text": "気持ちいい", "acoustic_classes": {
                "silence": 0.05, "vocalisation": 0.92, "speech": 0.03,
                "speech_max_run_s": 0.0}},
        ]
        kept, diagnostics = drop_vocalisation_runs(blocks, min_run=2)

        assert kept[0]["vocalisation_verdict"] == "lexical_text_vocal_audio"
        assert diagnostics["vocalisation_cues_marked"] == 1

    def test_switching_the_acoustics_off_restores_the_text_only_behaviour(self):
        blocks = [
            {"text": "そうなんだ"},
            {"text": "あっ", "acoustic_classes": {
                "silence": 0.05, "vocalisation": 0.92, "speech": 0.03,
                "speech_max_run_s": 0.0}},
            {"text": "本当に？"},
        ]
        kept, diagnostics = drop_vocalisation_runs(
            blocks, min_run=2, use_acoustics=False
        )

        assert [b["text"] for b in kept] == ["そうなんだ", "あっ", "本当に？"]
        assert diagnostics["vocalisation_cues_kept_as_isolated"] == 1

    def test_a_malformed_acoustic_record_degrades_instead_of_raising(self):
        blocks = [
            {"text": "あっ", "acoustic_classes": {"silence": "?"}},
            {"text": "んっ", "acoustic_classes": []},
        ]
        kept, diagnostics = drop_vocalisation_runs(blocks, min_run=99)

        assert len(kept) == 2
        assert diagnostics["vocalisation_acoustics_available"] == 0


class TestMixedCueDetection:
    """Symptom (g)'s gate.

    `お兄さまの...!あっ、あんっ!` is a second of speech and five of moaning. The
    shares cannot see that - a span average reads the same as six seconds of
    moaning - so the longest speech run is what separates them. This only selects
    the cues worth examining; `TestMixedCueSplitting` covers what may be removed.
    """

    def test_a_mixed_cue_is_recognised(self):
        acoustics = CueAcoustics(
            silence=0.05, vocalisation=0.78, speech=0.17, speech_max_run_s=1.0
        )
        assert mixed_cue_split_point(acoustics, 0.0, 6.0)

    def test_pure_moaning_is_not_mixed(self):
        acoustics = CueAcoustics(
            silence=0.05, vocalisation=0.92, speech=0.03, speech_max_run_s=0.08
        )
        assert not mixed_cue_split_point(acoustics, 0.0, 6.0)

    def test_a_short_moan_inside_speech_is_not_mixed(self):
        """The vocalisation has to be long enough to be worth cutting for; a
        gasp mid-sentence is not a passage."""
        acoustics = CueAcoustics(
            silence=0.05, vocalisation=0.10, speech=0.85, speech_max_run_s=2.0
        )
        assert not mixed_cue_split_point(acoustics, 0.0, 4.0)

    def test_detection_without_a_reader_leaves_the_cue_whole(self):
        """No reader means no fragment measurement, and the split is only ever
        justified by one - so the cue is counted and kept intact."""
        blocks = [
            {"text": "お兄さまの…!あっ、あんっ!", "start": 0.0, "end": 6.0,
             "acoustic_start": 0.0, "acoustic_end": 6.0,
             "acoustic_classes": {"silence": 0.05, "vocalisation": 0.78,
                                  "speech": 0.17, "speech_max_run_s": 1.0}},
        ]
        kept, diagnostics = drop_vocalisation_runs(blocks, min_run=2)

        assert len(kept) == 1
        assert kept[0]["text"] == "お兄さまの…!あっ、あんっ!"
        assert diagnostics["vocalisation_mixed_cues_detected"] == 1
        assert diagnostics["vocalisation_mixed_cues_split"] == 0
        assert diagnostics["vocalisation_cues_dropped"] == 0


def _char_words(text: str, start: float, per_char: float) -> list[dict]:
    """One measured span per character, which is what the aligner emits."""
    return [
        {
            "word": ch,
            "start": start + index * per_char,
            "end": start + (index + 1) * per_char,
        }
        for index, ch in enumerate(text)
    ]


def _reader(speech_before: float):
    """Frame classes that read as speech before `speech_before` and moaning after."""

    def read(start: float, end: float) -> dict:
        if end <= speech_before:
            return {"silence": 0.05, "vocalisation": 0.05, "speech": 0.90,
                    "speech_max_run_s": end - start}
        if start >= speech_before:
            return {"silence": 0.02, "vocalisation": 0.95, "speech": 0.03,
                    "speech_max_run_s": 0.0}
        return {"silence": 0.05, "vocalisation": 0.78, "speech": 0.17,
                "speech_max_run_s": speech_before - start}

    return read


class TestMixedCueSplitting:
    """Symptom (g), acted on.

    The removal criterion is deliberately not new: a fragment goes only when the
    joint verdict applied to that fragment's own re-measured frames returns
    drop. So the split cannot delete anything the filter would have kept had the
    same text arrived as a cue of its own.
    """

    def _mixed_block(self, text="お兄さまの…!あっ、あんっ!あっ、あんっ!"):
        return {
            "text": text,
            "start": 0.0,
            "end": 6.0,
            "acoustic_start": 0.0,
            "acoustic_end": 6.0,
            "display_start": 0.0,
            "display_end": 6.5,
            "words": _char_words(text, 0.0, 6.0 / len(text)),
            "acoustic_classes": {"silence": 0.05, "vocalisation": 0.78,
                                 "speech": 0.17, "speech_max_run_s": 1.0},
        }

    def test_the_moaning_tail_is_removed_and_the_speech_kept(self):
        block = self._mixed_block()
        kept, diagnostics = drop_vocalisation_runs(
            [block], min_run=2, acoustic_reader=_reader(1.5)
        )

        assert len(kept) == 1
        assert kept[0]["text"].startswith("お兄さまの")
        assert "あんっ" not in kept[0]["text"]
        assert diagnostics["vocalisation_mixed_cues_split"] == 1
        assert diagnostics["vocalisation_split_removed_seconds"] > 1.0
        assert kept[0]["vocalisation_split"]["removed_suffix"]
        # The seconds the split deleted, which the shortened cue no longer
        # covers - without them a listening audit of the removal is impossible.
        spans = kept[0]["vocalisation_split"]["removed_spans"]
        assert len(spans) == 1 and spans[0][1] > spans[0][0]

    def test_the_cue_ends_when_the_speech_does(self):
        block = self._mixed_block()
        kept, _ = drop_vocalisation_runs(
            [block], min_run=2, acoustic_reader=_reader(1.5)
        )

        # The window can only shrink, so a split never creates an overlap that
        # the layout had not already allowed.
        assert kept[0]["acoustic_end"] < 6.0
        assert kept[0]["display_end"] <= 6.5
        assert kept[0]["display_start"] >= 0.0

    def test_a_removed_tail_withdraws_the_continuation_claim(self):
        """Same rule as a dropped run: the audio the claim spanned is gone."""
        block = {**self._mixed_block(), "continues_into_next": True}
        kept, _ = drop_vocalisation_runs(
            [block], min_run=2, acoustic_reader=_reader(1.5)
        )

        assert kept[0]["continues_into_next"] is False

    def test_it_never_cuts_into_a_part_that_says_something(self):
        """The walk stops at the first part carrying a word, so a candidate
        fragment can never contain one."""
        text = "あっ、あんっ!やめて…あっ、あんっ!"
        block = {**self._mixed_block(text), "words": _char_words(text, 0.0, 6.0 / len(text))}
        kept, diagnostics = drop_vocalisation_runs(
            [block], min_run=2, acoustic_reader=_reader(0.0)
        )

        assert "やめて" in kept[0]["text"]

    def test_a_fragment_the_acoustics_do_not_condemn_stays(self):
        """The reader hears speech throughout, so nothing may be removed even
        though the text of the tail decomposes."""
        block = self._mixed_block()
        kept, diagnostics = drop_vocalisation_runs(
            [block], min_run=2, acoustic_reader=_reader(99.0)
        )

        assert kept[0]["text"] == block["text"]
        assert diagnostics["vocalisation_mixed_cues_split"] == 0

    def test_a_cue_is_never_emptied_by_a_split(self):
        """All-vocal text is the whole-cue rule's business; the split refuses it
        rather than removing everything and leaving a blank cue."""
        text = "あっ、あんっ!あっ、あんっ!"
        block = {**self._mixed_block(text), "words": _char_words(text, 0.0, 6.0 / len(text))}
        kept, _ = drop_vocalisation_runs(
            [block], min_run=99, acoustic_reader=_reader(0.0)
        )

        assert kept[0]["text"] == text

    def test_a_cue_whose_words_do_not_spell_its_text_is_left_alone(self):
        """Without a character-to-time map there is no defensible cut point."""
        block = self._mixed_block()
        block["words"] = [{"word": "違う", "start": 0.0, "end": 6.0}]
        kept, diagnostics = drop_vocalisation_runs(
            [block], min_run=2, acoustic_reader=_reader(1.5)
        )

        assert kept[0]["text"] == block["text"]
        assert diagnostics["vocalisation_mixed_cues_split"] == 0

    def test_a_protected_fragment_is_not_removable(self):
        """`classify_cue` consults the allow-list first, and the split inherits
        that - nothing in the sound separates a laugh from a moan."""
        text = "だめだって…ふふふ"
        block = {**self._mixed_block(text), "words": _char_words(text, 0.0, 6.0 / len(text))}
        kept, _ = drop_vocalisation_runs(
            [block], min_run=2, acoustic_reader=_reader(0.0)
        )

        assert kept[0]["text"] == text

    def test_the_switch_turns_it_off_without_touching_the_count(self):
        block = self._mixed_block()
        kept, diagnostics = drop_vocalisation_runs(
            [block],
            min_run=2,
            acoustic_reader=_reader(1.5),
            split_mixed_cues=False,
        )

        assert kept[0]["text"] == block["text"]
        assert diagnostics["vocalisation_mixed_cues_detected"] == 1
        assert diagnostics["vocalisation_mixed_cues_split"] == 0

    def test_it_does_not_mutate_the_caller_s_block(self):
        block = self._mixed_block()
        before = dict(block)
        drop_vocalisation_runs([block], min_run=2, acoustic_reader=_reader(1.5))

        assert block == before


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
        # The split rewrites cue text, so serving a cached run made without it
        # would hand back lines this configuration says should not exist.
        assert "vocalisation_split_mixed_cues" in signature

    def test_the_split_is_on_by_default_and_can_be_switched_off(self, monkeypatch):
        assert SubtitleOptions().vocalisation_split_mixed_cues is True
        monkeypatch.setenv("SUBTITLE_VOCALISATION_SPLIT_MIXED_CUES", "0")
        assert SubtitleOptions.from_env().vocalisation_split_mixed_cues is False

    def test_it_can_be_switched_off(self, monkeypatch):
        monkeypatch.setenv("SUBTITLE_DROP_VOCALISATION_ONLY_CUES", "0")
        assert SubtitleOptions.from_env().drop_vocalisation_only_cues is False
