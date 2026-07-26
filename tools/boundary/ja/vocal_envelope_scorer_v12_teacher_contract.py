"""Frozen content identity for the Scorer v12 vocal-envelope Teacher."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import hashlib
import json
import math
from typing import Any, Mapping


TEACHER_TASK_CONTRACT_ID = "human_vocal_event_envelope_single_pass_tristate_v3"
SCORER_V12_TIME_GRID_CONTRACT_ID = "scorer_v12_10ms_wire_20ms_frame_v1"
SCORER_V12_LOCAL_TIMESTAMP_STEP_S = 0.01
SCORER_V12_FRAME_HOP_S = 0.02

# These hashes are deliberately frozen separately from the prompt version string.
# Changing the actual wire prompt/schema requires updating both the version and
# these values, otherwise Teacher dispatch fails closed before an API call.
SYSTEM_PROMPT_SHA256 = (
    "cf62b619fa0304b5d6cc7901c1d0037a424511d4572362242aecf0ac8c34a3f9"
)
RESPONSE_SCHEMA_SHA256 = (
    "3686fa00ea8fe5dc31115dcfa4c39c411c1bb2121e246c63f85fbc14db043ea3"
)


def require_scorer_v12_local_timestamp(
    value: Any,
    *,
    field: str,
) -> float:
    """Validate one local Teacher coordinate on the explicit 10 ms wire grid."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a JSON number on the 10ms time grid")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a JSON number on the 10ms time grid") from error
    step = Decimal(str(SCORER_V12_LOCAL_TIMESTAMP_STEP_S))
    if decimal % step != 0:
        raise ValueError(f"{field} must align to the 10ms time grid")
    return number


def quantize_vocal_partition_boundary_frame(
    *,
    left_label: str,
    right_label: str,
    boundary_s: float,
    frame_count: int,
    frame_hop_s: float = SCORER_V12_FRAME_HOP_S,
) -> int:
    """Map one shared partition cut to the 20 ms frame grid.

    A boundary is quantized once and then shared by both adjacent spans.  Half-frame
    coordinates are biased toward retaining vocal evidence: vocal starts round down
    and vocal ends round up (e.g., left_label=vocal → ROUND_CEILING assigns the
    boundary frame to the right side, protecting the vocal end; right_label=vocal →
    ROUND_FLOOR assigns it to the left, protecting the vocal start).  Other
    transitions use the same deterministic ordering as the canonical Scorer v12
    compiler.
    """

    if frame_count < 0 or frame_hop_s <= 0.0:
        raise ValueError("Scorer v12 frame geometry must be non-negative")
    try:
        scaled = Decimal(str(boundary_s)) / Decimal(str(frame_hop_s))
    except (InvalidOperation, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("Scorer v12 boundary must be finite") from error
    if not scaled.is_finite():
        raise ValueError("Scorer v12 boundary must be finite")

    vocal_labels = {"vocal", "vocal_candidate"}
    nonvocal_labels = {"non_vocal", "non_vocal_candidate"}
    if left_label in vocal_labels:
        rounding = ROUND_CEILING
    elif right_label in vocal_labels:
        rounding = ROUND_FLOOR
    elif left_label in nonvocal_labels:
        rounding = ROUND_FLOOR
    elif right_label in nonvocal_labels:
        rounding = ROUND_CEILING
    else:
        rounding = ROUND_HALF_UP
    value = int(scaled.to_integral_value(rounding=rounding))
    return max(0, min(int(frame_count), value))


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash a JSON schema after provider-independent canonical serialization."""
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    """Hash the exact UTF-8 system-prompt text sent to the provider."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def teacher_contract_fingerprint_fields() -> dict[str, str]:
    return {
        "teacher_task_contract_id": TEACHER_TASK_CONTRACT_ID,
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
    }


def validate_teacher_contract_content(
    *, system_prompt: str, response_schema: Mapping[str, Any]
) -> dict[str, str]:
    """Fail closed when prompt content drifts behind unchanged contract metadata."""
    actual_prompt_sha = text_sha256(system_prompt)
    if actual_prompt_sha != SYSTEM_PROMPT_SHA256:
        raise ValueError(
            "Scorer v12 Teacher system prompt fingerprint mismatch; bump the "
            "prompt version and freeze the new system_prompt_sha256"
        )
    actual_schema_sha = canonical_json_sha256(response_schema)
    if actual_schema_sha != RESPONSE_SCHEMA_SHA256:
        raise ValueError(
            "Scorer v12 Teacher response schema fingerprint mismatch; bump the "
            "prompt version and freeze the new response_schema_sha256"
        )
    return teacher_contract_fingerprint_fields()
