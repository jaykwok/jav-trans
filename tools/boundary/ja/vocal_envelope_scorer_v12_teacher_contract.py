"""Frozen content identity for the Scorer v12 vocal-envelope Teacher."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


TEACHER_TASK_CONTRACT_ID = "human_voice_event_envelope_single_pass_tristate_v2"

# These hashes are deliberately frozen separately from the prompt version string.
# Changing the actual wire prompt/schema requires updating both the version and
# these values, otherwise Teacher dispatch fails closed before an API call.
SYSTEM_PROMPT_SHA256 = (
    "97ad032db7fbacc518f83afb5b528d3d02d62e727196c84fd3f9cc5b689d0a30"
)
RESPONSE_SCHEMA_SHA256 = (
    "975c2006b812fd0a3143d33c2d1216f5365d60e8d2ac12a67968624738f95ed8"
)


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
