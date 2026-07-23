#!/usr/bin/env python3
"""Compile calibrated Scorer v11 Protect/Remove evidence into train sources.

This compiler is deliberately separate from the human full-source compiler.
Gemini evidence is accepted only after the frozen held-out calibration gate is
bound and revalidated.  Protect-only frames become inside_candidate,
Remove-only frames become outside_candidate, and conflicts/unmarked frames stay
unsure (ignore=-100).
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import wave


PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402


AUDIT_SUMMARY_SCHEMA = "candidate_island_scorer_v11_train_teacher_review_summary_v1"
AUDIT_ITEM_SCHEMA = "candidate_island_scorer_v11_train_teacher_review_item_v1"
SOURCE_SCHEMA = "candidate_island_scorer_v11_train_teacher_source_v1"
TEACHER_SUMMARY_SCHEMA = "candidate_island_scorer_v11_dual_evidence_summary_v1"
TEACHER_SOURCE_SCHEMA = "candidate_island_scorer_v11_dual_evidence_preaudit_v1"
CALIBRATION_SUMMARY_SCHEMA = "candidate_island_dual_evidence_review_summary_v1"
CALIBRATION_ITEM_SCHEMA = "candidate_island_dual_evidence_review_item_v1"
GAP_VERDICT_SCHEMA = "candidate_island_scorer_v11_bridge_gap_manual_verdict_v3"
OUTPUT_SCHEMA = "candidate_island_scorer_v11_real_train_dual_evidence_source_v1"
SUMMARY_SCHEMA = (
    "candidate_island_scorer_v11_real_train_dual_evidence_compile_summary_v1"
)
FRAME_HOP_S = 0.02
FRAME_SAMPLES = 320
EXPECTED_PROMPT_PROFILE = "dual-evidence-protect-remove-v1"
EXPECTED_PROTECT_PROMPT = "candidate_island_scorer_v11_protect_evidence_v2_high_recall"
EXPECTED_REMOVE_PROMPT = "candidate_island_scorer_v11_remove_evidence_v1"
EXPECTED_PROMPT_VERSION = f"{EXPECTED_PROTECT_PROMPT}__{EXPECTED_REMOVE_PROMPT}"
EXPECTED_MODEL = "google/gemini-3.5-flash-lite"
EXPECTED_ENV_FILE = "gemini"
EXPECTED_AUDIO_MODE = "input_audio_raw"
EXPECTED_HOST = "openrouter.ai"
EXPECTED_MERGE_CONTRACT = (
    "protect_only=inside; remove_only=outside; overlap_or_neither=unsure"
)
LABELS = {"outside_candidate", "inside_candidate", "unsure"}
ALLOWED_GAP_VERDICTS = {
    "acceptable_nonsemantic_bridge",
    "teacher_overmerged_independent_background",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path, *, owner: Path | None = None) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [
        *((owner.parent / raw,) if owner is not None else ()),
        PROJECT_ROOT / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _index(
    rows: Sequence[dict[str, Any]], key: str, *, name: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get(key) or "")
        if not identity or identity in result:
            raise ValueError(f"{name} requires unique non-empty {key}: {identity!r}")
        result[identity] = row
    return result


def _bound_file(
    summary: Mapping[str, Any],
    field: str,
    *,
    owner: Path,
    sha_field: str | None = None,
) -> Path:
    path = _resolve(str(summary.get(field) or ""), owner=owner)
    if not path.is_file():
        raise ValueError(f"bound Scorer v11 evidence is missing: {field}")
    if sha_field is not None and str(summary.get(sha_field) or "") != _sha256(path):
        raise ValueError(f"Scorer v11 evidence SHA mismatch: {field}")
    return path


def _wav_geometry(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        sample_count = int(handle.getnframes())
    if sample_rate != 16000 or channels != 1 or sample_count <= 0:
        raise ValueError(f"Scorer v11 train audio must be non-empty mono 16k WAV: {path}")
    return sample_rate, sample_count, sample_count // FRAME_SAMPLES


def _teacher_identity(summary: Mapping[str, Any], *, name: str) -> None:
    expected = {
        "schema": TEACHER_SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "prompt_profile": EXPECTED_PROMPT_PROFILE,
        "prompt_version": EXPECTED_PROMPT_VERSION,
        "protect_prompt_version": EXPECTED_PROTECT_PROMPT,
        "remove_prompt_version": EXPECTED_REMOVE_PROMPT,
        "model": EXPECTED_MODEL,
        "env_file_name": EXPECTED_ENV_FILE,
        "audio_content_mode": EXPECTED_AUDIO_MODE,
        "base_url_host": EXPECTED_HOST,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"{name} has wrong {field}: {summary.get(field)!r}")
    if (
        summary.get("training_manifest_allowed") is not False
        or summary.get("manual_review_required") is not True
        or int(summary.get("failed_closed_count") or 0) != 0
    ):
        raise ValueError(f"{name} weakens fail-closed Teacher evidence")


def _validate_calibration(
    *,
    calibration_summary: Path,
    calibration_teacher_summary: Path,
    calibration_gap_verdicts: Path,
) -> dict[str, Any]:
    review = json.loads(calibration_summary.read_text(encoding="utf-8-sig"))
    teacher = json.loads(calibration_teacher_summary.read_text(encoding="utf-8-sig"))
    if review.get("schema") != CALIBRATION_SUMMARY_SCHEMA:
        raise ValueError("wrong Scorer v11 dual-evidence calibration summary schema")
    if review.get("boundary_serialization_contract_id") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError("wrong central Boundary contract in calibration summary")
    _teacher_identity(teacher, name="calibration Teacher summary")
    candidate = _bound_file(
        review, "candidate", owner=calibration_summary, sha_field="candidate_sha256"
    )
    teacher_labels = _bound_file(teacher, "labels", owner=calibration_teacher_summary)
    if candidate != teacher_labels:
        raise ValueError("calibration review and Teacher summary bind different labels")
    manifest = _bound_file(
        review, "manifest", owner=calibration_summary, sha_field="manifest_sha256"
    )
    teacher_manifest = _bound_file(
        teacher, "manifest", owner=calibration_teacher_summary, sha_field="manifest_sha256"
    )
    if manifest != teacher_manifest:
        raise ValueError("calibration review and Teacher summary bind different manifests")
    _bound_file(
        review,
        "human_verdicts",
        owner=calibration_summary,
        sha_field="human_verdicts_sha256",
    )
    per_source = _bound_file(review, "per_source", owner=calibration_summary)
    bound_gap_verdicts = _bound_file(
        review, "manual_verdicts", owner=calibration_summary
    )
    if bound_gap_verdicts != calibration_gap_verdicts:
        raise ValueError("calibration summary binds different bridge-gap verdicts")

    if (
        float(review.get("protect_recall") or 0.0) < 0.95
        or not math.isclose(
            float(review.get("final_outside_precision") or 0.0), 1.0, abs_tol=1e-12
        )
        or review.get("zero_true_speech_outside") is not True
        or int(review.get("unsafe_outside_frames") or 0) != 0
        or int(review.get("failed_closed_count") or 0) != 0
        or int(review.get("source_count") or 0) != int(teacher.get("source_count") or -1)
        or int(review.get("frame_count") or 0) != int(teacher.get("frame_count") or -1)
    ):
        raise ValueError("Scorer v11 dual-evidence held-out calibration gate failed")

    expected_gaps: dict[str, dict[str, Any]] = {}
    for row in _rows(per_source):
        if row.get("schema") != CALIBRATION_ITEM_SCHEMA:
            raise ValueError("wrong Scorer v11 calibration per-source schema")
        if row.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError("wrong central Boundary contract in calibration item")
        if int(row.get("failed_closed_count") or 0) != 0:
            raise ValueError("failed-closed calibration source cannot unlock training")
        for gap in row.get("bridged_background_gaps") or ():
            gap_id = str(gap.get("gap_id") or "")
            if not gap_id or gap_id in expected_gaps:
                raise ValueError(f"invalid duplicate calibration bridge gap: {gap_id!r}")
            expected_gaps[gap_id] = {
                "source_id": str(row.get("source_id") or ""),
                "partition": str(row.get("partition") or ""),
                "start_frame": int(gap.get("start_frame", -1)),
                "end_frame": int(gap.get("end_frame", -1)),
            }
    verdicts = _index(_rows(calibration_gap_verdicts), "gap_id", name="gap verdict")
    if (
        len(expected_gaps) != int(review.get("bridged_gap_count") or 0)
        or set(verdicts) != set(expected_gaps)
        or len(verdicts) != 23
        or review.get("manual_verdict_schema") != GAP_VERDICT_SCHEMA
    ):
        raise ValueError("Scorer v11 calibration bridge-gap review is incomplete")
    gap_counts: Counter[str] = Counter()
    for gap_id, verdict in verdicts.items():
        expected_gap = expected_gaps[gap_id]
        if (
            verdict.get("schema") != GAP_VERDICT_SCHEMA
            or verdict.get("boundary_serialization_contract_id")
            != ACOUSTIC_BINARY_V12_CONTRACT.contract_id
            or verdict.get("complete") is not True
            or str(verdict.get("verdict") or "") not in ALLOWED_GAP_VERDICTS
            or str(verdict.get("combined_verdict") or "")
            != str(verdict.get("verdict") or "")
            or str(verdict.get("content_verdict") or "") != "no_semantic_dialogue"
            or str(verdict.get("semantic_coverage_verdict") or "")
            != "not_applicable_no_semantic"
        ):
            raise ValueError(f"unsafe or incomplete calibration gap verdict: {gap_id}")
        for field, value in expected_gap.items():
            actual = int(verdict.get(field, -1)) if field.endswith("_frame") else str(
                verdict.get(field) or ""
            )
            if actual != value:
                raise ValueError(f"calibration gap identity mismatch: {gap_id}:{field}")
        gap_counts[str(verdict["verdict"])] += 1
    return {
        "summary": review,
        "teacher_summary": teacher,
        "candidate": candidate,
        "manifest": manifest,
        "per_source": per_source,
        "gap_counts": dict(sorted(gap_counts.items())),
    }


def _span_key(span: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(span.get("label") or ""),
        int(span.get("start_frame", -1)),
        int(span.get("end_frame", -1)),
    )


def _validate_evidence_spans(
    spans: Sequence[Mapping[str, Any]],
    *,
    source_id: str,
    frame_count: int,
    label: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    cursor = 0
    for span in spans:
        start = int(span.get("start_frame", -1))
        end = int(span.get("end_frame", -1))
        if (
            str(span.get("label") or "") != label
            or start < cursor
            or end <= start
            or end > frame_count
            or not math.isclose(float(span.get("start_s", -1.0)), start * FRAME_HOP_S)
            or not math.isclose(float(span.get("end_s", -1.0)), end * FRAME_HOP_S)
        ):
            raise ValueError(f"invalid dual-evidence span geometry: {source_id}:{label}")
        normalized.append({"label": label, "start_frame": start, "end_frame": end})
        cursor = end
    return normalized


def _runs(states: Sequence[str], state: str, label: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    start: int | None = None
    for index, value in enumerate([*states, "__end__"]):
        if value == state and start is None:
            start = index
        elif value != state and start is not None:
            result.append(
                {
                    "label": label,
                    "start_frame": start,
                    "end_frame": index,
                    "start_s": round(start * FRAME_HOP_S, 6),
                    "end_s": round(index * FRAME_HOP_S, 6),
                }
            )
            start = None
    return result


def _recompute_three_state(
    row: Mapping[str, Any], *, source_id: str, frame_count: int
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    protected = _validate_evidence_spans(
        list(row.get("protected_evidence_spans") or ()),
        source_id=source_id,
        frame_count=frame_count,
        label="inside_candidate",
    )
    removable = _validate_evidence_spans(
        list(row.get("remove_evidence_spans") or ()),
        source_id=source_id,
        frame_count=frame_count,
        label="outside_candidate",
    )
    protect_mask = [False] * frame_count
    remove_mask = [False] * frame_count
    for span in protected:
        protect_mask[span["start_frame"] : span["end_frame"]] = [True] * (
            span["end_frame"] - span["start_frame"]
        )
    for span in removable:
        remove_mask[span["start_frame"] : span["end_frame"]] = [True] * (
            span["end_frame"] - span["start_frame"]
        )
    states: list[str] = []
    for has_protect, has_remove in zip(protect_mask, remove_mask):
        if has_protect and not has_remove:
            states.append("inside")
        elif has_remove and not has_protect:
            states.append("outside")
        elif has_protect and has_remove:
            states.append("conflict")
        else:
            states.append("unresolved")
    inside = _runs(states, "inside", "inside_candidate")
    outside = _runs(states, "outside", "outside_candidate")
    conflict = _runs(states, "conflict", "unsure")
    unresolved = _runs(states, "unresolved", "unsure")
    unsure = sorted([*conflict, *unresolved], key=lambda span: span["start_frame"])
    expected_fields = {
        "islands": inside,
        "safe_outside_spans": outside,
        "conflict_spans": conflict,
        "unresolved_spans": unresolved,
        "unsure_spans": unsure,
    }
    for field, expected in expected_fields.items():
        actual = [_span_key(span) for span in row.get(field) or ()]
        if actual != [_span_key(span) for span in expected]:
            raise ValueError(f"dual-evidence three-state merge mismatch: {source_id}:{field}")
    canonical = sorted([*inside, *outside, *unsure], key=lambda span: span["start_frame"])
    cursor = 0
    counts: Counter[str] = Counter()
    for span in canonical:
        if span["start_frame"] != cursor or span["end_frame"] <= cursor:
            raise ValueError(f"dual-evidence canonical coverage is not contiguous: {source_id}")
        counts[span["label"]] += span["end_frame"] - span["start_frame"]
        cursor = span["end_frame"]
    if cursor != frame_count:
        raise ValueError(f"dual-evidence canonical does not cover source tail: {source_id}")
    conflict_frames = sum(span["end_frame"] - span["start_frame"] for span in conflict)
    return canonical, counts, conflict_frames


def compile_real_train_dual_evidence(
    *,
    audit_summary: Path,
    audit_manifest: Path,
    teacher_summary: Path,
    teacher_preaudit: Path,
    calibration_summary: Path,
    calibration_teacher_summary: Path,
    calibration_gap_verdicts: Path,
    output_dir: Path,
    verify_audio: bool = True,
) -> dict[str, Any]:
    paths = [
        audit_summary,
        audit_manifest,
        teacher_summary,
        teacher_preaudit,
        calibration_summary,
        calibration_teacher_summary,
        calibration_gap_verdicts,
    ]
    paths = [path.resolve() for path in paths]
    (
        audit_summary,
        audit_manifest,
        teacher_summary,
        teacher_preaudit,
        calibration_summary,
        calibration_teacher_summary,
        calibration_gap_verdicts,
    ) = paths
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    calibration = _validate_calibration(
        calibration_summary=calibration_summary,
        calibration_teacher_summary=calibration_teacher_summary,
        calibration_gap_verdicts=calibration_gap_verdicts,
    )
    audit = json.loads(audit_summary.read_text(encoding="utf-8-sig"))
    teacher = json.loads(teacher_summary.read_text(encoding="utf-8-sig"))
    if audit.get("schema") != AUDIT_SUMMARY_SCHEMA:
        raise ValueError("wrong Scorer v11 train selection summary schema")
    if audit.get("boundary_serialization_contract_id") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError("wrong central Boundary contract in train selection summary")
    if (
        audit.get("training_manifest_allowed") is not False
        or audit.get("teacher_output_used_as_truth") is not False
        or audit.get("unselected_source_label_inheritance") is not False
    ):
        raise ValueError("Scorer v11 train selection summary weakens source isolation")
    bound_manifest = _bound_file(
        audit,
        "audit_manifest",
        owner=audit_summary,
        sha_field="audit_manifest_sha256",
    )
    if bound_manifest != audit_manifest:
        raise ValueError("train selection summary binds a different audit manifest")
    source_manifest = _bound_file(
        audit, "source_manifest", owner=audit_summary, sha_field="source_manifest_sha256"
    )
    _bound_file(audit, "preaudit", owner=audit_summary, sha_field="preaudit_sha256")
    exclude_sources = _bound_file(
        audit, "exclude_sources", owner=audit_summary, sha_field="exclude_sources_sha256"
    )
    _teacher_identity(teacher, name="train Teacher summary")
    bound_teacher_manifest = _bound_file(
        teacher, "manifest", owner=teacher_summary, sha_field="manifest_sha256"
    )
    bound_teacher_labels = _bound_file(teacher, "labels", owner=teacher_summary)
    if bound_teacher_manifest != audit_manifest or bound_teacher_labels != teacher_preaudit:
        raise ValueError("train Teacher summary binds different source evidence")

    audit_rows = _index(_rows(audit_manifest), "source_id", name="train audit manifest")
    source_rows = _index(_rows(source_manifest), "source_id", name="frozen train source")
    teacher_rows = _index(_rows(teacher_preaudit), "source_id", name="dual-evidence source")
    selected_ids = [str(value) for value in audit.get("selected_source_ids") or ()]
    teacher_ids = [str(value) for value in teacher.get("source_ids") or ()]
    if (
        len(selected_ids) != len(set(selected_ids))
        or selected_ids != teacher_ids
        or set(selected_ids) != set(audit_rows)
        or set(selected_ids) != set(teacher_rows)
        or len(selected_ids) != int(audit.get("source_count") or 0)
        or len(selected_ids) != int(teacher.get("source_count") or 0)
        or int(audit.get("video_count") or 0) != len(selected_ids)
    ):
        raise ValueError("Scorer v11 dual-evidence train source set is not exactly frozen")
    excluded_rows = _rows(exclude_sources)
    excluded_source_ids = {str(row.get("source_id") or "") for row in excluded_rows}

    compiled: list[dict[str, Any]] = []
    total_counts: Counter[str] = Counter()
    seen_videos: set[str] = set()
    total_conflict_frames = 0
    for source_id in selected_ids:
        item = audit_rows[source_id]
        source = source_rows.get(source_id)
        evidence = teacher_rows[source_id]
        if source is None:
            raise ValueError(f"selected source is absent from frozen train scope: {source_id}")
        if (
            item.get("schema") != AUDIT_ITEM_SCHEMA
            or source.get("schema") != SOURCE_SCHEMA
            or evidence.get("schema") != TEACHER_SOURCE_SCHEMA
        ):
            raise ValueError(f"wrong Scorer v11 dual-evidence source schema: {source_id}")
        for row in (item, source, evidence):
            if row.get("boundary_serialization_contract_id") != (
                ACOUSTIC_BINARY_V12_CONTRACT.contract_id
            ):
                raise ValueError(f"wrong central Boundary contract: {source_id}")
        if (
            item.get("partition") != "train"
            or source.get("partition") != "train"
            or evidence.get("partition") != "train"
            or evidence.get("teacher_failed_closed") is True
            or evidence.get("reviewed_full_source") is not False
            or evidence.get("training_manifest_allowed") is not False
            or evidence.get("human_review_required") is not True
            or evidence.get("merge_contract") != EXPECTED_MERGE_CONTRACT
            or evidence.get("unmarked_semantics") != "unsure_ignore_minus_100"
        ):
            raise ValueError(f"invalid fail-closed dual-evidence source: {source_id}")
        for field, expected in {
            "model": EXPECTED_MODEL,
            "env_file_name": EXPECTED_ENV_FILE,
            "base_url_host": EXPECTED_HOST,
            "prompt_profile": EXPECTED_PROMPT_PROFILE,
            "prompt_version": EXPECTED_PROMPT_VERSION,
            "protect_prompt_version": EXPECTED_PROTECT_PROMPT,
            "remove_prompt_version": EXPECTED_REMOVE_PROMPT,
        }.items():
            if evidence.get(field) != expected:
                raise ValueError(f"dual-evidence source has wrong {field}: {source_id}")
        video_id = str(source.get("video_id") or "")
        if (
            not video_id
            or video_id != str(item.get("video_id") or "")
            or video_id in seen_videos
            or source_id in excluded_source_ids
        ):
            raise ValueError(f"dual-evidence source/video identity is not isolated: {source_id}")
        seen_videos.add(video_id)
        frame_count = int(source.get("frame_count") or 0)
        if (
            frame_count <= 0
            or int(item.get("frame_count") or 0) != frame_count
            or int(evidence.get("frame_count") or 0) != frame_count
            or float(source.get("frame_hop_s") or 0.0) != FRAME_HOP_S
            or float(item.get("frame_hop_s") or 0.0) != FRAME_HOP_S
            or float(evidence.get("frame_hop_s") or 0.0) != FRAME_HOP_S
        ):
            raise ValueError(f"dual-evidence frame geometry mismatch: {source_id}")
        spans, counts, conflict_frames = _recompute_three_state(
            evidence, source_id=source_id, frame_count=frame_count
        )
        total_counts.update(counts)
        total_conflict_frames += conflict_frames

        audio = _resolve(str(source.get("audio") or ""), owner=source_manifest)
        audit_audio = _resolve(str(item.get("audio") or ""), owner=audit_manifest)
        evidence_audio = _resolve(str(evidence.get("audio") or ""), owner=teacher_preaudit)
        audio_sha = str(source.get("audio_sha256") or "")
        if (
            not audio.is_file()
            or not audit_audio.is_file()
            or not evidence_audio.is_file()
            or len(audio_sha) != 64
            or str(item.get("audio_sha256") or "") != audio_sha
            or str(evidence.get("audio_sha256") or "") != audio_sha
        ):
            raise ValueError(f"dual-evidence audio identity mismatch: {source_id}")
        if verify_audio and any(
            _sha256(path) != audio_sha for path in (audio, audit_audio, evidence_audio)
        ):
            raise ValueError(f"dual-evidence audio SHA mismatch: {source_id}")
        sample_rate, sample_count, audio_frames = _wav_geometry(audio)
        if (
            sample_rate != int(source.get("sample_rate") or 0)
            or sample_count != int(source.get("sample_count") or 0)
            or audio_frames != frame_count
        ):
            raise ValueError(f"dual-evidence WAV/frame geometry mismatch: {source_id}")
        compiled.append(
            {
                "schema": OUTPUT_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "video_id": video_id,
                "partition": "train",
                "input_distribution": "real_workflow_source_window_calibrated_dual_evidence_v1",
                "source_kind": "real_train_full_source_calibrated_dual_evidence",
                "synthetic_composite": False,
                "audio": _display(audio),
                "audio_sha256": audio_sha,
                "sample_rate": sample_rate,
                "sample_count": sample_count,
                "duration_s": sample_count / sample_rate,
                "frame_count": frame_count,
                "frame_hop_s": FRAME_HOP_S,
                "core_ids": [f"real-train-dual-evidence-source::{source_id}"],
                "canonical_spans": spans,
                "annotation_provenance": "calibrated_gemini_independent_dual_evidence_v1",
                "teacher_output_used_as_truth": True,
                "teacher_evidence_used_as_training_supervision": True,
                "human_full_source_confirmed": False,
                "calibration_gate_passed": True,
                "unsure_training_label": -100,
                "unselected_source_label_inheritance": False,
                "training_manifest_allowed": True,
                "dual_evidence_summary": _display(teacher_summary),
                "dual_evidence_summary_sha256": _sha256(teacher_summary),
                "dual_evidence_preaudit": _display(teacher_preaudit),
                "dual_evidence_preaudit_sha256": _sha256(teacher_preaudit),
                "calibration_summary": _display(calibration_summary),
                "calibration_summary_sha256": _sha256(calibration_summary),
                "calibration_teacher_summary": _display(calibration_teacher_summary),
                "calibration_teacher_summary_sha256": _sha256(
                    calibration_teacher_summary
                ),
                "calibration_gap_verdicts": _display(calibration_gap_verdicts),
                "calibration_gap_verdicts_sha256": _sha256(calibration_gap_verdicts),
                "train_selection_summary": _display(audit_summary),
                "train_selection_summary_sha256": _sha256(audit_summary),
                "source_manifest": _display(source_manifest),
                "source_manifest_sha256": _sha256(source_manifest),
            }
        )

    if total_counts["inside_candidate"] <= 0 or total_counts["outside_candidate"] <= 0:
        raise ValueError("dual-evidence train sources require both supervised classes")
    summary_counts = {
        "inside_candidate": int(teacher.get("inside_frames") or -1),
        "outside_candidate": int(teacher.get("outside_frames") or -1),
        "unsure": int(teacher.get("unsure_frames") or -1),
    }
    if (
        dict(total_counts) != summary_counts
        or sum(total_counts.values()) != int(teacher.get("frame_count") or -1)
        or total_conflict_frames != int(teacher.get("conflict_frames") or -1)
    ):
        raise ValueError("dual-evidence Teacher summary frame totals do not match source rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "real_train_dual_evidence_sources.jsonl"
    _write_jsonl(output, compiled)
    result = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "real_train_dual_evidence_sources": _display(output),
        "real_train_dual_evidence_sources_sha256": _sha256(output),
        "source_count": len(compiled),
        "video_count": len(seen_videos),
        "canonical_frame_counts": dict(sorted(total_counts.items())),
        "conflict_frames_mapped_to_unsure": total_conflict_frames,
        "teacher_output_used_as_truth": True,
        "teacher_evidence_used_as_training_supervision": True,
        "human_full_source_confirmed": False,
        "calibration_gate_passed": True,
        "calibration_protect_recall": calibration["summary"]["protect_recall"],
        "calibration_final_outside_precision": calibration["summary"][
            "final_outside_precision"
        ],
        "calibration_zero_true_speech_outside": True,
        "calibration_gap_verdict_counts": calibration["gap_counts"],
        "calibration_gap_verdict_count": 23,
        "failed_closed_count": 0,
        "unsure_training_label": -100,
        "unselected_source_label_inheritance": False,
        "training_manifest_allowed": True,
        "dual_evidence_summary": _display(teacher_summary),
        "dual_evidence_summary_sha256": _sha256(teacher_summary),
        "dual_evidence_preaudit": _display(teacher_preaudit),
        "dual_evidence_preaudit_sha256": _sha256(teacher_preaudit),
        "calibration_summary": _display(calibration_summary),
        "calibration_summary_sha256": _sha256(calibration_summary),
        "calibration_teacher_summary": _display(calibration_teacher_summary),
        "calibration_teacher_summary_sha256": _sha256(calibration_teacher_summary),
        "calibration_gap_verdicts": _display(calibration_gap_verdicts),
        "calibration_gap_verdicts_sha256": _sha256(calibration_gap_verdicts),
        "train_selection_summary": _display(audit_summary),
        "train_selection_summary_sha256": _sha256(audit_summary),
        "audit_manifest": _display(audit_manifest),
        "audit_manifest_sha256": _sha256(audit_manifest),
        "source_manifest": _display(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument("--teacher-summary", required=True)
    parser.add_argument("--teacher-preaudit", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--calibration-teacher-summary", required=True)
    parser.add_argument("--calibration-gap-verdicts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-audio-content-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return compile_real_train_dual_evidence(
        audit_summary=Path(args.audit_summary),
        audit_manifest=Path(args.audit_manifest),
        teacher_summary=Path(args.teacher_summary),
        teacher_preaudit=Path(args.teacher_preaudit),
        calibration_summary=Path(args.calibration_summary),
        calibration_teacher_summary=Path(args.calibration_teacher_summary),
        calibration_gap_verdicts=Path(args.calibration_gap_verdicts),
        output_dir=Path(args.output_dir),
        verify_audio=not args.skip_audio_content_check,
    )


if __name__ == "__main__":
    main()
