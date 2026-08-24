"""Which thinking tiers exist, and does the request builder agree?

History, because every layer of it is still load-bearing:

The bottom tier was once spelled "minimal", which is a real value on OpenAI,
Gemini and DeepSeek - but on all three it is the smallest *nonzero* thinking
budget, not off. Measured on deepseek-v4-flash, same four cues, bare request:
"minimal" returned 1,453 reasoning tokens in 12.74s while "none" returned 0 in
1.79s. So the UI option labelled 「不思考直出」 was quietly buying a full
reasoning pass. "minimal" is still refused for that reason.

It was then spelled "none" and genuinely switched thinking off, which was worth
roughly 8x on the translation stage. That tier was retired on 2026-08-14
because sample-b's 1,700 cues came back with 171 source echoes (10.1%).

It came back on 2026-08-24, because the failure it was retired for is one a
local detector can find: the repair pass now escalates exactly those ids. The
same change deleted "medium", which was never a value DeepSeek accepted - it
was silently ignored and the request ran at the API default of `high`, so the
shipped default billed at the second-most-expensive tier for ten days. "max"
went with it as the tier nothing here was shown to need.

The tiers are wire values now. That is the invariant this file exists to hold:
`REASONING_EFFORTS` is passed to the provider verbatim, so a name that is not a
real API value cannot be added to it without the request silently changing
meaning.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from core.job_context import _llm_reasoning_effort  # noqa: E402
from llm import settings as llm_settings  # noqa: E402
from llm.settings import (  # noqa: E402
    _escalated_reasoning_effort,
    _normalize_reasoning_effort,
)
from web.models import normalize_llm_reasoning_effort  # noqa: E402


class TestTheThreeTiers:
    @pytest.mark.parametrize("effort", ["none", "low", "high"])
    def test_each_tier_survives_every_normalizer(self, effort: str) -> None:
        """Three modules used to keep their own copy of this list and they
        drifted: `job_context`'s was missing the bottom tier entirely, so a job
        submitted with it ran at the fallback however the request was built."""
        assert _normalize_reasoning_effort(effort) == effort
        assert normalize_llm_reasoning_effort(effort) == effort
        assert _llm_reasoning_effort(effort) == effort

    @pytest.mark.parametrize("junk", ["", None, "minimal", "off", "disabled"])
    def test_unknown_values_fall_back_to_the_default(self, junk: str | None) -> None:
        """"minimal" is deliberately not accepted: it reads like the off switch
        and is not one, which is exactly the confusion that caused the bug."""
        assert _normalize_reasoning_effort(junk) == "low"
        assert normalize_llm_reasoning_effort(junk) == "low"
        assert _llm_reasoning_effort(junk) == "low"

    @pytest.mark.parametrize("retired", ["medium", "max", "xhigh"])
    def test_retired_values_resolve_to_what_they_actually_ran_as(
        self, retired: str
    ) -> None:
        """Not the default: a stored "medium" ran at `high`, because DeepSeek
        ignored it. Clamping it as unknown would move those saved jobs to a
        cheaper tier on re-run and quietly change the result they recorded."""
        assert _normalize_reasoning_effort(retired) == "high"
        assert normalize_llm_reasoning_effort(retired) == "high"
        assert _llm_reasoning_effort(retired) == "high"

    def test_none_is_a_real_tier_again(self) -> None:
        """It must not fall through to the retired-value table: `none` is the
        cheap first pass the whole cost cascade is built on, and resolving it to
        anything else would silently buy thinking for the entire film."""
        assert _normalize_reasoning_effort("none") == "none"
        assert normalize_llm_reasoning_effort("none") == "none"
        assert _llm_reasoning_effort("none") == "none"

    def test_there_is_only_one_copy_of_the_list(self) -> None:
        from core.config import REASONING_EFFORTS

        assert REASONING_EFFORTS == ("none", "low", "high")
        assert _normalize_reasoning_effort is normalize_llm_reasoning_effort
        assert _normalize_reasoning_effort is _llm_reasoning_effort

    def test_case_and_whitespace_are_tolerated(self) -> None:
        assert _normalize_reasoning_effort("  LOW ") == "low"
        assert normalize_llm_reasoning_effort(" Low") == "low"
        assert _normalize_reasoning_effort("  NONE ") == "none"


class TestEscalation:
    """The repair pass reissues one tier up, and the top tier has to terminate."""

    @pytest.mark.parametrize(
        ("base", "escalated"),
        [("none", "low"), ("low", "high"), ("high", "high")],
    )
    def test_one_step_up_and_then_stays(self, base: str, escalated: str) -> None:
        assert _escalated_reasoning_effort(base) == escalated

    def test_a_retired_value_escalates_from_where_it_lands(self) -> None:
        """`medium` normalizes to `high`, which is already the top."""
        assert _escalated_reasoning_effort("medium") == "high"

    def test_escalation_never_leaves_the_supported_set(self) -> None:
        from core.config import REASONING_EFFORTS

        for tier in REASONING_EFFORTS:
            assert _escalated_reasoning_effort(tier) in REASONING_EFFORTS


class TestRepairTier:
    """Escalation is kept only where it is proven necessary, i.e. from `none`.

    Measured on sample-v with a `low` base: repairing at `high` spent 22,585
    reasoning tokens in one request (25% of the film's bill, against 11 base
    requests costing twice that between them); repairing at `low` spent 7,673,
    fixed 146 of 146 flagged cues, and left source echo, residual kana and
    glossary compliance identical at 0 / 0 / 100%.
    """

    @staticmethod
    def _pin(monkeypatch, value: str) -> None:
        monkeypatch.setattr(
            llm_settings, "TRANSLATION_REPAIR_REASONING_EFFORT", value
        )

    def test_the_default_floors_at_low_instead_of_escalating(self, monkeypatch) -> None:
        self._pin(monkeypatch, "")
        assert llm_settings._repair_reasoning_effort("low") == "low"
        assert llm_settings._repair_reasoning_effort("high") == "high"

    def test_none_still_escalates(self, monkeypatch) -> None:
        """Thinking-off left 171 of 1,700 cues echoing the Japanese source, and
        this pass ends in a gate that fails the job over exactly that."""
        self._pin(monkeypatch, "")
        assert llm_settings._repair_reasoning_effort("none") == "low"

    def test_a_pin_buys_back_escalation(self, monkeypatch) -> None:
        self._pin(monkeypatch, "high")
        assert llm_settings._repair_reasoning_effort("low") == "high"
        assert llm_settings._repair_reasoning_effort("none") == "high"

    def test_pinning_none_is_refused(self, monkeypatch) -> None:
        self._pin(monkeypatch, "none")
        assert llm_settings._repair_reasoning_effort("none") == "low"

    def test_an_unusable_pin_falls_back_to_the_rule(self, monkeypatch) -> None:
        """An unrecognised tier must not silently resolve to one that exists -
        `normalize_reasoning_effort` clamps garbage to `low`, which would be
        indistinguishable from someone pinning `low` on purpose."""
        self._pin(monkeypatch, "banana")
        assert llm_settings._repair_reasoning_effort("none") == "low"
        assert llm_settings._repair_reasoning_effort("high") == "high"

    def test_a_retired_pin_lands_where_it_normalizes(self, monkeypatch) -> None:
        self._pin(monkeypatch, "medium")
        assert llm_settings._repair_reasoning_effort("low") == "high"

    def test_the_repair_pass_reads_the_tier(self, monkeypatch) -> None:
        """Wired, not merely defined - the tier has to reach the request."""
        self._pin(monkeypatch, "")
        efforts: list[str] = []

        def fake_chat(_messages, **kwargs):
            efforts.append(kwargs["reasoning_effort"])
            return '{"translations":[{"id":0,"text":"你好"}]}'

        from llm import profiles as profiles_module
        from llm import repair as repair_module

        repair_module.apply_repair_pass(
            [{"text": "これは翻訳されるべきです。"}],
            ["これは翻訳されるべきです。"],
            chat=fake_chat,
            profile=profiles_module.select_profile(),
            batch_size=10,
            reasoning_effort="low",
            target_lang="简体中文",
            glossary="",
            character_reference="",
        )

        assert efforts == ["low"]


class TestItReachesTheWire:
    """Normalizing correctly is useless if the request builder disagrees."""

    @staticmethod
    def _source() -> str:
        return (
            PROJECT_ROOT / "src" / "llm" / "backends" / "openai_compat.py"
        ).read_text(encoding="utf-8")

    def test_the_request_sends_the_tier_verbatim(self) -> None:
        """One field carries the whole axis, `none` included, and it carries the
        normalized tier itself - so `TestTheThreeTiers` above, which pins what
        the normalizer returns, is also pinning what goes on the wire."""
        assert '"reasoning": {"effort": effective_reasoning_effort}' in self._source()

    def test_every_tier_is_a_value_the_api_accepts(self) -> None:
        """The bug this file was rewritten for: `medium` was not in the accepted
        set, was silently ignored rather than refused, and the request ran at the
        API default of `high` for ten days. A tier is only addable here if it is
        already a real `reasoning.effort` value."""
        from core.config import REASONING_EFFORTS

        assert set(REASONING_EFFORTS) <= {"none", "low", "high"}

    def test_the_per_provider_thinking_toggle_is_gone(self) -> None:
        """Chat Completions had no `none` effort, so switching thinking off was
        a second field spelled differently per provider - `thinking.type` on
        DeepSeek, `reasoning.enabled` on OpenRouter - and an unknown spelling was
        dropped rather than refused, which is how the cheapest tier billed as the
        most expensive one. That surface was retired on 2026-08-24; nothing may
        reintroduce a second way to say the same thing."""
        import llm.backends.openai_compat as openai_compat

        for retired in (
            "_chat_reasoning_fields",
            "_chat_response_format",
            "_create_chat_completion",
            "_chat_completions",
        ):
            assert not hasattr(openai_compat, retired), retired
        assert '"reasoning_effort"' not in self._source()


def test_the_ui_offers_exactly_the_three_tiers() -> None:
    html = (PROJECT_ROOT / "src" / "web" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    section = html.split('id="api-reasoning-effort"', 1)[1].split("</select>", 1)[0]
    assert '<option value="none">' in section
    assert '<option value="low"' in section
    assert '<option value="high">' in section
    assert 'value="medium"' not in section
    assert 'value="max"' not in section
    assert 'value="minimal"' not in section
