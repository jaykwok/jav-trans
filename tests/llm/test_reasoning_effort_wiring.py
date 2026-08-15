"""Which thinking tiers exist, and does the request builder agree?

History, because both halves of it are still load-bearing:

The bottom tier was once spelled "minimal", which is a real value on OpenAI,
Gemini and DeepSeek - but on all three it is the smallest *nonzero* thinking
budget, not off. Measured on deepseek-v4-flash, same four cues, bare request:
"minimal" returned 1,453 reasoning tokens in 12.74s while "none" returned 0 in
1.79s. So the UI option labelled 「不思考直出」 was quietly buying a full
reasoning pass. "minimal" is still refused for that reason.

It was then spelled "none" and genuinely switched thinking off, which was worth
roughly 8x on the translation stage. That tier was retired on 2026-08-14, also
on measurement: over sample-b's 1,700 cues, thinking-off left 171 of them
(10.1%) with the Japanese source copied through untranslated, while medium and
max left none. Speed is not worth a tenth of the film in the wrong language, so
the bottom tier is now "low" - the fastest tier that still thinks. A stored
"none" maps to it rather than falling back to "medium", because promoting old
configs to the slowest, most expensive tier is not a safe reading of them.
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
from llm.settings import _normalize_reasoning_effort  # noqa: E402
from web.models import normalize_llm_reasoning_effort  # noqa: E402


class TestTheThreeTiers:
    @pytest.mark.parametrize("effort", ["low", "medium", "max"])
    def test_each_tier_survives_every_normalizer(self, effort: str) -> None:
        """Three modules used to keep their own copy of this list and they
        drifted: `job_context`'s was missing the bottom tier entirely, so a job
        submitted with it ran at 'medium' however the request was built."""
        assert _normalize_reasoning_effort(effort) == effort
        assert normalize_llm_reasoning_effort(effort) == effort
        assert _llm_reasoning_effort(effort) == effort

    @pytest.mark.parametrize(
        "junk", ["", None, "minimal", "high", "xhigh", "off", "disabled"]
    )
    def test_unknown_values_fall_back_to_medium(self, junk: str | None) -> None:
        """"minimal" is deliberately not accepted: it reads like the off switch
        and is not one, which is exactly the confusion that caused the bug."""
        assert _normalize_reasoning_effort(junk) == "medium"
        assert normalize_llm_reasoning_effort(junk) == "medium"
        assert _llm_reasoning_effort(junk) == "medium"

    def test_a_stored_none_reads_as_the_tier_that_replaced_it(self) -> None:
        """Not "medium": every config written before the rename says "none", and
        clamping it as unknown would silently move those runs to the slowest and
        most expensive tier."""
        assert _normalize_reasoning_effort("none") == "low"
        assert normalize_llm_reasoning_effort("none") == "low"
        assert _llm_reasoning_effort("none") == "low"

    def test_there_is_only_one_copy_of_the_list(self) -> None:
        from core.config import REASONING_EFFORTS

        assert set(REASONING_EFFORTS) == {"low", "medium", "max"}
        assert _normalize_reasoning_effort is normalize_llm_reasoning_effort
        assert _normalize_reasoning_effort is _llm_reasoning_effort

    def test_case_and_whitespace_are_tolerated(self) -> None:
        assert _normalize_reasoning_effort("  LOW ") == "low"
        assert normalize_llm_reasoning_effort(" Low") == "low"
        assert _normalize_reasoning_effort("  NONE ") == "low"


class TestItReachesTheWire:
    """Normalizing correctly is useless if the request builder disagrees."""

    @staticmethod
    def _source() -> str:
        return (
            PROJECT_ROOT / "src" / "llm" / "backends" / "openai_compat.py"
        ).read_text(encoding="utf-8")

    def test_the_responses_path_sends_the_tier_verbatim(self) -> None:
        assert '"reasoning": {"effort": effective_reasoning_effort}' in self._source()

    def test_the_chat_path_switches_thinking_on_for_every_tier(self) -> None:
        """Providers on the extra_body convention (GLM/DashScope-style) ignore
        `reasoning_effort`, so this flag is what actually decides. Measured
        against DeepSeek 2026-08-14: with `thinking.type = disabled` the reply
        carried zero reasoning at *every* effort value, "max" included - so a
        tier left mapped to disabled would have had a decorative effort."""
        source = self._source()
        assert '"extra_body": {"thinking": {"type": "enabled"}}' in source
        assert 'effective_effort == "none"' not in source
        assert 'effective_effort == "minimal"' not in source


def test_the_ui_offers_exactly_the_three_tiers() -> None:
    html = (PROJECT_ROOT / "src" / "web" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    section = html.split('id="api-reasoning-effort"', 1)[1].split("</select>", 1)[0]
    assert '<option value="low">' in section
    assert '<option value="medium"' in section
    assert '<option value="max">' in section
    assert 'value="none"' not in section
    assert 'value="minimal"' not in section
