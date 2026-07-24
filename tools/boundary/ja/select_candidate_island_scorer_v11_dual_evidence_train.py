#!/usr/bin/env python3
"""Freeze one calibrated dual-evidence Scorer v11 source per train video.

The input Teacher run may contain several windows for the same video because it
was run over the prior outside-only source set.  This adapter deterministically
selects one mixed source per video when possible, materializes a bound Teacher
subset, and removes exactly those source identities from the prior outside-only
manifest.  It never inherits labels between sources.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402


SOURCE_SCHEMA = "candidate_island_scorer_v11_train_teacher_source_v1"
OUTSIDE_SOURCE_SCHEMA = "candidate_island_scorer_v11_real_train_outside_source_v1"
TEACHER_SUMMARY_SCHEMA = "candidate_island_scorer_v11_dual_evidence_summary_v1"
TEACHER_SOURCE_SCHEMA = "candidate_island_scorer_v11_dual_evidence_preaudit_v1"
AUDIT_ITEM_SCHEMA = "candidate_island_scorer_v11_train_teacher_review_item_v1"
SELECTION_SUMMARY_SCHEMA = (
    "candidate_island_scorer_v11_dual_evidence_train_selection_summary_v1"
)
SELECTION_POLICY = (
    "one_per_video_mixed_then_balance_coverage_conflict_source_id_v1"
)
FRAME_HOP_S = 0.02


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


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


def _span_frames(row: Mapping[str, Any], field: str) -> int:
    return sum(
        int(span.get("end_frame", 0)) - int(span.get("start_frame", 0))
        for span in row.get(field) or ()
    )


def _counts(row: Mapping[str, Any]) -> dict[str, int]:
    frame_count = int(row.get("frame_count") or 0)
    inside = _span_frames(row, "islands")
    outside = _span_frames(row, "safe_outside_spans")
    unsure = _span_frames(row, "unsure_spans")
    conflict = _span_frames(row, "conflict_spans")
    if (
        frame_count <= 0
        or min(inside, outside, unsure, conflict) < 0
        or inside + outside + unsure != frame_count
        or conflict > unsure
    ):
        raise ValueError(f"invalid dual-evidence frame totals: {row.get('source_id')}")
    return {
        "frame_count": frame_count,
        "inside_candidate": inside,
        "outside_candidate": outside,
        "unsure": unsure,
        "conflict": conflict,
    }


def _selection_key(row: Mapping[str, Any]) -> tuple[float | int | str, ...]:
    counts = _counts(row)
    inside = counts["inside_candidate"]
    outside = counts["outside_candidate"]
    supervised = inside + outside
    mixed_rank = 0 if inside > 0 and outside > 0 else 1
    class_imbalance = abs(inside - outside) / max(1, supervised)
    unsupervised_ratio = counts["unsure"] / counts["frame_count"]
    conflict_ratio = counts["conflict"] / counts["frame_count"]
    return (
        mixed_rank,
        class_imbalance,
        unsupervised_ratio,
        conflict_ratio,
        str(row.get("source_id") or ""),
    )


def select_dual_evidence_train_sources(
    *,
    source_manifest: Path,
    outside_sources: Path,
    teacher_summary: Path,
    teacher_preaudit: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_manifest = source_manifest.resolve()
    outside_sources = outside_sources.resolve()
    teacher_summary = teacher_summary.resolve()
    teacher_preaudit = teacher_preaudit.resolve()
    for path in (source_manifest, outside_sources, teacher_summary, teacher_preaudit):
        if not path.is_file():
            raise FileNotFoundError(path)

    contract = ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    sources = _index(_rows(source_manifest), "source_id", name="train source manifest")
    outside = _index(_rows(outside_sources), "source_id", name="outside-only sources")
    evidence = _index(_rows(teacher_preaudit), "source_id", name="dual evidence")
    teacher = json.loads(teacher_summary.read_text(encoding="utf-8-sig"))
    if teacher.get("schema") != TEACHER_SUMMARY_SCHEMA:
        raise ValueError("wrong Scorer v11 dual-evidence Teacher summary schema")
    if teacher.get("boundary_serialization_contract_id") != contract:
        raise ValueError("wrong central Boundary contract in Teacher summary")
    bound_manifest = _resolve(str(teacher.get("manifest") or ""), owner=teacher_summary)
    bound_labels = _resolve(str(teacher.get("labels") or ""), owner=teacher_summary)
    if (
        bound_manifest != outside_sources
        or bound_labels != teacher_preaudit
        or str(teacher.get("manifest_sha256") or "") != _sha256(outside_sources)
    ):
        raise ValueError("Teacher summary does not bind the requested outside source set")
    teacher_ids = [str(value) for value in teacher.get("source_ids") or ()]
    if (
        teacher_ids != list(outside)
        or set(teacher_ids) != set(evidence)
        or len(teacher_ids) != int(teacher.get("source_count") or -1)
        or int(teacher.get("failed_closed_count") or 0) != 0
        or teacher.get("reasoning_contract_satisfied") is not True
    ):
        raise ValueError("Teacher source identities or fail-closed evidence are invalid")

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_id in teacher_ids:
        source = sources.get(source_id)
        old = outside[source_id]
        row = evidence[source_id]
        if source is None:
            raise ValueError(f"Teacher source is outside frozen train scope: {source_id}")
        if (
            source.get("schema") != SOURCE_SCHEMA
            or old.get("schema") != OUTSIDE_SOURCE_SCHEMA
            or row.get("schema") != TEACHER_SOURCE_SCHEMA
            or any(item.get("boundary_serialization_contract_id") != contract for item in (source, old, row))
            or source.get("partition") != "train"
            or old.get("partition") != "train"
            or row.get("partition") != "train"
        ):
            raise ValueError(f"invalid Scorer v11 train source contract: {source_id}")
        video_id = str(source.get("video_id") or "")
        if not video_id or video_id != str(old.get("video_id") or ""):
            raise ValueError(f"source/video identity mismatch: {source_id}")
        for field in ("audio_sha256", "frame_count", "frame_hop_s"):
            if source.get(field) != old.get(field) or source.get(field) != row.get(field):
                raise ValueError(f"source geometry mismatch: {source_id}:{field}")
        if float(source.get("frame_hop_s") or 0.0) != FRAME_HOP_S:
            raise ValueError(f"wrong frame hop: {source_id}")
        _counts(row)
        by_video[video_id].append(row)

    selected_ids: list[str] = []
    for video_id in sorted(by_video):
        candidates = sorted(by_video[video_id], key=_selection_key)
        selected_ids.append(str(candidates[0]["source_id"]))
    if len(selected_ids) != len(set(selected_ids)) or len(selected_ids) != len(by_video):
        raise ValueError("dual-evidence selection is not one source per video")
    selected_set = set(selected_ids)
    remaining_ids = [source_id for source_id in teacher_ids if source_id not in selected_set]

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_preaudit = output_dir / "selected_preaudit.jsonl"
    audit_manifest = output_dir / "audit_manifest.jsonl"
    remaining_outside = output_dir / "remaining_outside_sources.jsonl"
    selected_rows = [evidence[source_id] for source_id in selected_ids]
    _write_jsonl(selected_preaudit, selected_rows)
    _write_jsonl(remaining_outside, [outside[source_id] for source_id in remaining_ids])
    audit_rows: list[dict[str, Any]] = []
    for source_id in selected_ids:
        source = sources[source_id]
        audit_rows.append(
            {
                "schema": AUDIT_ITEM_SCHEMA,
                "boundary_serialization_contract_id": contract,
                "source_id": source_id,
                "video_id": str(source["video_id"]),
                "partition": "train",
                "frame_count": int(source["frame_count"]),
                "frame_hop_s": FRAME_HOP_S,
                "duration_s": float(source["duration_s"]),
                "audio": str(source["audio"]),
                "audio_sha256": str(source["audio_sha256"]),
            }
        )
    _write_jsonl(audit_manifest, audit_rows)

    totals: Counter[str] = Counter()
    protect_reasoning_tokens = 0
    remove_reasoning_tokens = 0
    for row in selected_rows:
        counts = _counts(row)
        totals.update(
            {
                "frame_count": counts["frame_count"],
                "inside_candidate": counts["inside_candidate"],
                "outside_candidate": counts["outside_candidate"],
                "unsure": counts["unsure"],
                "conflict": counts["conflict"],
            }
        )
        protect_reasoning_tokens += int(
            (row.get("protect_reasoning") or {}).get("reasoning_tokens") or 0
        )
        remove_reasoning_tokens += int(
            (row.get("remove_reasoning") or {}).get("reasoning_tokens") or 0
        )

    selected_teacher = dict(teacher)
    selected_teacher.update(
        {
            "manifest": _display(audit_manifest),
            "manifest_sha256": _sha256(audit_manifest),
            "labels": _display(selected_preaudit),
            "source_ids": selected_ids,
            "source_count": len(selected_ids),
            "frame_count": totals["frame_count"],
            "inside_frames": totals["inside_candidate"],
            "outside_frames": totals["outside_candidate"],
            "unsure_frames": totals["unsure"],
            "conflict_frames": totals["conflict"],
            "inside_ratio": totals["inside_candidate"] / totals["frame_count"],
            "outside_ratio": totals["outside_candidate"] / totals["frame_count"],
            "unsure_ratio": totals["unsure"] / totals["frame_count"],
            "conflict_ratio": totals["conflict"] / totals["frame_count"],
            "protect_reasoning_tokens": protect_reasoning_tokens,
            "remove_reasoning_tokens": remove_reasoning_tokens,
            "protect_reasoning_evidence_count": len(selected_ids),
            "remove_reasoning_evidence_count": len(selected_ids),
            "selection_derived": True,
            "selection_policy": SELECTION_POLICY,
            "selection_parent_teacher_summary": _display(teacher_summary),
            "selection_parent_teacher_summary_sha256": _sha256(teacher_summary),
            "selection_parent_teacher_preaudit": _display(teacher_preaudit),
            "selection_parent_teacher_preaudit_sha256": _sha256(teacher_preaudit),
        }
    )
    selected_teacher_summary = output_dir / "selected_teacher_summary.json"
    selected_teacher_summary.write_text(
        json.dumps(selected_teacher, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema": SELECTION_SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": contract,
        "selection_policy": SELECTION_POLICY,
        "source_manifest": _display(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "preaudit": _display(selected_preaudit),
        "preaudit_sha256": _sha256(selected_preaudit),
        "audit_manifest": _display(audit_manifest),
        "audit_manifest_sha256": _sha256(audit_manifest),
        "exclude_sources": _display(remaining_outside),
        "exclude_sources_sha256": _sha256(remaining_outside),
        "remaining_outside_sources": _display(remaining_outside),
        "remaining_outside_sources_sha256": _sha256(remaining_outside),
        "original_outside_sources": _display(outside_sources),
        "original_outside_sources_sha256": _sha256(outside_sources),
        "parent_teacher_summary": _display(teacher_summary),
        "parent_teacher_summary_sha256": _sha256(teacher_summary),
        "parent_teacher_preaudit": _display(teacher_preaudit),
        "parent_teacher_preaudit_sha256": _sha256(teacher_preaudit),
        "selected_teacher_summary": _display(selected_teacher_summary),
        "selected_teacher_summary_sha256": _sha256(selected_teacher_summary),
        "selected_source_ids": selected_ids,
        "replaced_outside_source_ids": selected_ids,
        "remaining_outside_source_ids": remaining_ids,
        "source_count": len(selected_ids),
        "video_count": len(by_video),
        "remaining_outside_source_count": len(remaining_ids),
        "canonical_frame_counts": {
            "inside_candidate": totals["inside_candidate"],
            "outside_candidate": totals["outside_candidate"],
            "unsure": totals["unsure"],
        },
        "conflict_frames": totals["conflict"],
        "teacher_output_used_as_truth": False,
        "teacher_output_used_as_calibrated_evidence": True,
        "unselected_source_label_inheritance": False,
        "training_manifest_allowed": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--outside-sources", required=True)
    parser.add_argument("--teacher-summary", required=True)
    parser.add_argument("--teacher-preaudit", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return select_dual_evidence_train_sources(
        source_manifest=Path(args.source_manifest),
        outside_sources=Path(args.outside_sources),
        teacher_summary=Path(args.teacher_summary),
        teacher_preaudit=Path(args.teacher_preaudit),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
