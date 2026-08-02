"""The no-thinking tier has to actually switch thinking off.

It was spelled "minimal", which is a real value on OpenAI, Gemini and DeepSeek -
but on all three it is the smallest *nonzero* thinking budget, not off. The
documented off switch is "none" everywhere. So the UI option labelled
「不思考直出」 was quietly buying a full reasoning pass.

Measured on deepseek-v4-flash, same four cues, bare request: "minimal" returned
1,453 reasoning tokens in 12.74s while "none" returned 0 in 1.79s - so the old
spelling was not off. Through the production streaming path, same 16-cue batch,
both complete 16/16:

    effort "none"       5.53s     262 output tokens
    effort "medium"    45.07s   5,266 output tokens

Reasoning was 95-98% of output tokens on every sampled request, and wall time
is 0.0105 s per output token (R^2 0.993 over 14 requests), so this one word was
worth roughly 8x on the translation stage.
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
    @pytest.mark.parametrize("effort", ["none", "medium", "max"])
    def test_each_tier_survives_every_normalizer(self, effort: str) -> None:
        """Three modules used to keep their own copy of this list and they
        drifted: `job_context`'s was missing the no-thinking tier entirely, so a
        job submitted with it ran at 'medium' however the request was built."""
        assert _normalize_reasoning_effort(effort) == effort
        assert normalize_llm_reasoning_effort(effort) == effort
        assert _llm_reasoning_effort(effort) == effort

    @pytest.mark.parametrize(
        "junk", ["", None, "minimal", "low", "high", "xhigh", "off", "disabled"]
    )
    def test_unknown_values_fall_back_to_medium(self, junk: str | None) -> None:
        """"minimal" is deliberately not accepted: it reads like the off switch
        and is not one, which is exactly the confusion that caused the bug."""
        assert _normalize_reasoning_effort(junk) == "medium"
        assert normalize_llm_reasoning_effort(junk) == "medium"
        assert _llm_reasoning_effort(junk) == "medium"

    def test_there_is_only_one_copy_of_the_list(self) -> None:
        from core.config import REASONING_EFFORTS

        assert set(REASONING_EFFORTS) == {"none", "medium", "max"}
        assert _normalize_reasoning_effort is normalize_llm_reasoning_effort
        assert _normalize_reasoning_effort is _llm_reasoning_effort

    def test_case_and_whitespace_are_tolerated(self) -> None:
        assert _normalize_reasoning_effort("  NONE ") == "none"
        assert normalize_llm_reasoning_effort(" None") == "none"


class TestItReachesTheWire:
    """Normalizing correctly is useless if the request builder disagrees."""

    @staticmethod
    def _source() -> str:
        return (
            PROJECT_ROOT / "src" / "llm" / "backends" / "openai_compat.py"
        ).read_text(encoding="utf-8")

    def test_the_responses_path_sends_the_tier_verbatim(self) -> None:
        assert '"reasoning": {"effort": effective_reasoning_effort}' in self._source()

    def test_the_chat_path_disables_thinking_on_none(self) -> None:
        """Providers on the extra_body convention (GLM/DashScope-style) ignore
        `reasoning_effort` entirely, so "none" has to be spelled twice."""
        source = self._source()
        assert '"type": "disabled" if effective_effort == "none" else "enabled"' in source
        assert 'effective_effort == "minimal"' not in source


def test_the_ui_offers_exactly_the_three_tiers() -> None:
    html = (PROJECT_ROOT / "src" / "web" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    section = html.split('id="api-reasoning-effort"', 1)[1].split("</select>", 1)[0]
    assert '<option value="none">' in section
    assert '<option value="medium"' in section
    assert '<option value="max">' in section
    assert 'value="minimal"' not in section
