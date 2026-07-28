"""The Teacher's prompt identity must have exactly one definition.

A stale duplicate is not a cosmetic problem: on 2026-07-27 the canonical compiler
still pinned `vocal-envelope-single-pass-tristate-v3` while the labeler had moved
to v4/v7, so a fully paid-for 144-source label set would have been rejected at
compile time - after the human audit, and after the API spend.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.boundary.ja import vocal_envelope_scorer_v12_calibration as calibration
from tools.boundary.ja import compile_vocal_envelope_scorer_v12_canonical as compiler
from tools.boundary.ja import label_vocal_envelope_scorer_v12_with_omni as labeler
from tools.boundary.ja.vocal_envelope_scorer_v12_teacher_contract import (
    PROMPT_PROFILE,
    PROMPT_VERSION,
)

CONSUMERS = (
    "tools/boundary/ja/label_vocal_envelope_scorer_v12_with_omni.py",
    "tools/boundary/ja/compile_vocal_envelope_scorer_v12_canonical.py",
    "tools/boundary/ja/vocal_envelope_scorer_v12_calibration.py",
)


def test_prompt_identity_has_one_source_of_truth() -> None:
    assert labeler.PROMPT_PROFILE == PROMPT_PROFILE
    assert labeler.PROMPT_VERSION == PROMPT_VERSION
    assert compiler.EXPECTED_PROMPT_PROFILE == PROMPT_PROFILE
    assert compiler.EXPECTED_PROMPT_VERSION == PROMPT_VERSION
    assert calibration.CALIBRATION_TEACHER_CONTRACT["prompt_profile"] == PROMPT_PROFILE
    assert calibration.CALIBRATION_TEACHER_CONTRACT["prompt_version"] == PROMPT_VERSION


def test_labeler_output_is_accepted_by_the_compiler_contract() -> None:
    """What the labeler stamps must be what the compiler demands back."""
    assert labeler.PROMPT_PROFILE == compiler.EXPECTED_PROMPT_PROFILE
    assert labeler.PROMPT_VERSION == compiler.EXPECTED_PROMPT_VERSION
    assert labeler.EXPECTED_REASONING == compiler.EXPECTED_REASONING
    assert labeler.EXPECTED_MAX_TOKENS == compiler.EXPECTED_MAX_TOKENS
    for profile, contract in compiler.PROVIDER_CONTRACTS.items():
        assert labeler.PROVIDER_CONTRACTS[profile]["model"] == contract["model"]
        assert (
            labeler.PROVIDER_CONTRACTS[profile]["execution_contract"]
            == contract["execution_contract"]
        )


def test_no_module_relitigates_the_prompt_version_as_a_literal() -> None:
    """Only the contract module may spell the version strings out."""
    for relative in CONSUMERS:
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert PROMPT_VERSION not in text, (
            f"{relative} hardcodes the prompt version; import it from "
            "vocal_envelope_scorer_v12_teacher_contract instead"
        )
        assert f'"{PROMPT_PROFILE}"' not in text, (
            f"{relative} hardcodes the prompt profile; import it from "
            "vocal_envelope_scorer_v12_teacher_contract instead"
        )


def test_prompt_version_matches_the_frozen_prompt_content() -> None:
    """The version string and the prompt fingerprint must move together."""
    from tools.boundary.ja.vocal_envelope_scorer_v12_teacher_contract import (
        SYSTEM_PROMPT_SHA256,
        text_sha256,
    )

    assert text_sha256(labeler.TRISTATE_SYSTEM_PROMPT) == SYSTEM_PROMPT_SHA256
    assert "v7" in PROMPT_VERSION, (
        "prompt content changed without bumping the version string"
    )
