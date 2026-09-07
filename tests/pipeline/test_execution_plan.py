"""A run states what it will do before it starts doing it.

The wait is the reason this exists. A lease is a queue, and a Japanese-only run
used to enter that queue on the way in - behind a local server that was retiring
- and could fail on the timeout of a backend it was never going to send a single
request to. The branch that would have said so was inside the implementation,
after the wait.

So the plan is built from the job's own settings, declares the resources each
stage takes, and is the one place that says how the output is spelled.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.execution_plan import (
    GPU,
    TRANSLATION_BACKEND,
    TRANSLATION_CACHE,
    ExecutionPlan,
)


def _ctx(*, skip_translation: bool = False, subtitle_mode: str = "zh") -> SimpleNamespace:
    return SimpleNamespace(
        skip_translation=skip_translation, subtitle_mode=subtitle_mode
    )


class TestWhatTheRunNeeds:
    def test_a_translating_run_declares_the_backend(self):
        plan = ExecutionPlan.for_run(_ctx(), has_segments=True)

        assert plan.translates is True
        assert plan.needs(TRANSLATION_BACKEND)
        assert plan.needs(TRANSLATION_CACHE)
        assert "translation" in plan.stage_names()
        assert "repair" in plan.stage_names()

    def test_a_japanese_only_run_declares_no_backend(self):
        plan = ExecutionPlan.for_run(_ctx(skip_translation=True), has_segments=True)

        assert plan.translates is False
        assert not plan.needs(TRANSLATION_BACKEND)
        assert plan.stage_names() == ("asr", "layout", "publish")

    def test_a_run_with_nothing_recognised_declares_no_backend(self):
        plan = ExecutionPlan.for_run(_ctx(), has_segments=False)

        assert plan.translates is False
        assert not plan.needs(TRANSLATION_BACKEND)

    def test_every_run_still_needs_the_gpu_and_the_disk(self):
        for plan in (
            ExecutionPlan.for_run(_ctx(), has_segments=True),
            ExecutionPlan.for_run(_ctx(skip_translation=True), has_segments=True),
        ):
            assert plan.needs(GPU)
            assert plan.stage_names()[0] == "asr"
            assert plan.stage_names()[-1] == "publish"


class TestHowTheOutputIsSpelled:
    def test_a_japanese_only_run_writes_the_ja_file_in_the_ja_style(self):
        plan = ExecutionPlan.for_run(_ctx(skip_translation=True), has_segments=True)

        assert plan.subtitle_language == "ja"
        assert plan.srt_filename("sample-a") == "sample-a.ja.srt"

    def test_a_translated_run_writes_the_plain_file(self):
        plan = ExecutionPlan.for_run(_ctx(), has_segments=True)

        assert plan.subtitle_language == "zh"
        assert plan.srt_filename("sample-a") == "sample-a.srt"

    def test_bilingual_is_a_translated_output_option(self):
        """Two lines of the same Japanese is not a bilingual subtitle."""
        translated = ExecutionPlan.for_run(_ctx(subtitle_mode="bilingual"), has_segments=True)
        japanese_only = ExecutionPlan.for_run(
            _ctx(skip_translation=True, subtitle_mode="bilingual"), has_segments=True
        )

        assert translated.bilingual is True
        assert japanese_only.bilingual is False

    def test_an_empty_run_keeps_the_name_it_was_going_to_write(self):
        # Nothing was recognised, but the file that gets written is still the
        # one this job was for - the language did not change, only the content.
        plan = ExecutionPlan.for_run(_ctx(), has_segments=False)

        assert plan.srt_filename("sample-a") == "sample-a.srt"


class TestSettlingItAfterAsr:
    def test_an_empty_transcript_drops_the_translation_stages(self):
        plan = ExecutionPlan.for_asr(_ctx())
        assert plan.needs(TRANSLATION_BACKEND)

        settled = plan.after_asr(segment_count=0)

        assert settled.translates is False
        assert not settled.needs(TRANSLATION_BACKEND)
        assert settled.subtitle_language == plan.subtitle_language

    def test_a_transcript_with_cues_keeps_them(self):
        plan = ExecutionPlan.for_asr(_ctx())

        assert plan.after_asr(segment_count=12) is plan

    def test_a_japanese_only_run_is_unchanged_by_what_asr_found(self):
        plan = ExecutionPlan.for_asr(_ctx(skip_translation=True))

        assert plan.after_asr(segment_count=0) is plan
        assert plan.after_asr(segment_count=12) is plan


def test_a_plan_is_a_value_not_a_handle():
    plan = ExecutionPlan.for_run(_ctx(), has_segments=True)

    with pytest.raises(Exception):
        plan.translates = False  # frozen: a plan cannot change mid-run
