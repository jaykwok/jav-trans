from __future__ import annotations


ACOUSTIC_SPLIT_TEACHER_SELECTION_SCHEMA = (
    "acoustic_split_canonical_candidate_selection_v1"
)
ACOUSTIC_SPLIT_TEACHER_LABEL_SCHEMA = "acoustic_split_canonical_teacher_label_v1"
ACOUSTIC_SPLIT_TEACHER_PROMPT_VERSION = (
    "acoustic_split_binary_event_teacher_centered_clip_v1"
)
ACOUSTIC_SPLIT_TEACHER_SUMMARY_SCHEMA = "acoustic_split_canonical_teacher_summary_v1"
ACOUSTIC_SPLIT_TEACHER_RETRY_SCHEMA = "acoustic_split_canonical_retry_candidate_v1"
ACOUSTIC_SPLIT_MANUAL_VERDICT_SCHEMA = (
    "acoustic_split_canonical_manual_verdict_v1"
)
ACOUSTIC_SPLIT_AUDIT_SUMMARY_SCHEMA = (
    "acoustic_split_canonical_candidate_audit_summary_v1"
)

# Historical labels remain approved data provenance. New tools never emit this
# identifier, and runtime does not inspect either prompt version.
HISTORICAL_APPROVED_SPLIT_TEACHER_PROMPT_VERSIONS = frozenset(
    {"semantic_split_v3_omni_plus_centered_clip_v3"}
)
APPROVED_SPLIT_TEACHER_PROMPT_VERSIONS = frozenset(
    {
        ACOUSTIC_SPLIT_TEACHER_PROMPT_VERSION,
        *HISTORICAL_APPROVED_SPLIT_TEACHER_PROMPT_VERSIONS,
    }
)
