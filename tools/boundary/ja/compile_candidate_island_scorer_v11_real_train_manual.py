#!/usr/bin/env python3
"""Compile fully reviewed Scorer v11 train sources into signed canonical rows."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence
import wave


PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402


AUDIT_SUMMARY_SCHEMA = "candidate_island_scorer_v11_train_teacher_review_summary_v1"
AUDIT_ITEM_SCHEMA = "candidate_island_scorer_v11_train_teacher_review_item_v1"
SOURCE_SCHEMA = "candidate_island_scorer_v11_train_teacher_source_v1"
VERDICT_SCHEMA = "candidate_island_scorer_v11_train_manual_verdict_v1"
OUTPUT_SCHEMA = "candidate_island_scorer_v11_real_train_manual_source_v1"
SUMMARY_SCHEMA = "candidate_island_scorer_v11_real_train_manual_compile_summary_v1"
FRAME_HOP_S = 0.02
FRAME_SAMPLES = 320
LABELS = {"outside_candidate", "inside_candidate", "unsure"}


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
    summary: dict[str, Any], field: str, sha_field: str, *, owner: Path
) -> Path:
    path = _resolve(str(summary.get(field) or ""), owner=owner)
    if not path.is_file() or str(summary.get(sha_field) or "") != _sha256(path):
        raise ValueError(f"Scorer v11 train review evidence SHA mismatch: {field}")
    return path


def _validate_spans(
    spans: Sequence[dict[str, Any]], *, source_id: str, frame_count: int
) -> list[dict[str, Any]]:
    if not spans:
        raise ValueError(f"train manual verdict has no full-source spans: {source_id}")
    normalized: list[dict[str, Any]] = []
    cursor = 0
    for span in spans:
        label = str(span.get("label") or "")
        start = int(span.get("start_frame", -1))
        end = int(span.get("end_frame", -1))
        if label not in LABELS:
            raise ValueError(f"invalid Scorer v11 train label: {source_id}:{label}")
        if start != cursor or end <= start or end > frame_count:
            raise ValueError(
                "Scorer v11 train spans must be ordered contiguous full-source coverage: "
                f"{source_id}"
            )
        normalized.append(
            {
                "label": label,
                "start_frame": start,
                "end_frame": end,
                "start_s": round(start * FRAME_HOP_S, 6),
                "end_s": round(end * FRAME_HOP_S, 6),
            }
        )
        cursor = end
    if cursor != frame_count:
        raise ValueError(f"Scorer v11 train spans do not cover source tail: {source_id}")
    return normalized


def _wav_geometry(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        sample_count = int(handle.getnframes())
    if sample_rate != 16000 or channels != 1 or sample_count <= 0:
        raise ValueError(f"Scorer v11 train audio must be non-empty mono 16k WAV: {path}")
    return sample_rate, sample_count, sample_count // FRAME_SAMPLES


def _expected_verdict(spans: Sequence[dict[str, Any]]) -> str:
    labels = {str(span["label"]) for span in spans}
    if "inside_candidate" in labels:
        return "complete_with_target_inside_candidate"
    if "unsure" in labels:
        return "complete_with_unsure_only"
    return "complete_all_outside_candidate"


def compile_real_train_manual(
    *,
    audit_summary: Path,
    audit_manifest: Path,
    manual_verdicts: Path,
    output_dir: Path,
    verify_audio: bool = True,
) -> dict[str, Any]:
    audit_summary = audit_summary.resolve()
    audit_manifest = audit_manifest.resolve()
    manual_verdicts = manual_verdicts.resolve()
    for path in (audit_summary, audit_manifest, manual_verdicts):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(audit_summary.read_text(encoding="utf-8-sig"))
    if summary.get("schema") != AUDIT_SUMMARY_SCHEMA:
        raise ValueError("wrong Scorer v11 train review summary schema")
    if summary.get("boundary_serialization_contract_id") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError("wrong central Boundary contract in train review summary")
    if (
        summary.get("training_manifest_allowed") is not False
        or summary.get("teacher_output_used_as_truth") is not False
        or summary.get("human_full_source_confirmation_required") is not True
        or summary.get("unselected_source_label_inheritance") is not False
    ):
        raise ValueError("Scorer v11 train review summary weakens the human-truth contract")
    bound_manifest = _bound_file(
        summary, "audit_manifest", "audit_manifest_sha256", owner=audit_summary
    )
    if bound_manifest != audit_manifest:
        raise ValueError("Scorer v11 train review summary binds a different audit manifest")
    source_manifest = _bound_file(
        summary, "source_manifest", "source_manifest_sha256", owner=audit_summary
    )
    _bound_file(summary, "preaudit", "preaudit_sha256", owner=audit_summary)
    if summary.get("exclude_sources"):
        _bound_file(
            summary, "exclude_sources", "exclude_sources_sha256", owner=audit_summary
        )

    audit_rows = _index(_rows(audit_manifest), "source_id", name="train audit manifest")
    source_rows = _index(_rows(source_manifest), "source_id", name="train source manifest")
    verdict_rows = _index(_rows(manual_verdicts), "source_id", name="train manual verdicts")
    selected_ids = [str(value) for value in summary.get("selected_source_ids") or ()]
    if (
        len(selected_ids) != len(set(selected_ids))
        or set(selected_ids) != set(audit_rows)
        or set(verdict_rows) != set(audit_rows)
        or int(summary.get("source_count") or 0) != len(audit_rows)
        or int(summary.get("video_count") or 0) != len(audit_rows)
    ):
        raise ValueError("Scorer v11 train manual review is incomplete or has extra sources")

    compiled: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    seen_videos: set[str] = set()
    for source_id in selected_ids:
        item = audit_rows[source_id]
        source = source_rows.get(source_id)
        verdict = verdict_rows[source_id]
        if source is None:
            raise ValueError(f"selected train source is missing from frozen manifest: {source_id}")
        if item.get("schema") != AUDIT_ITEM_SCHEMA or source.get("schema") != SOURCE_SCHEMA:
            raise ValueError(f"wrong Scorer v11 train evidence schema: {source_id}")
        if verdict.get("schema") != VERDICT_SCHEMA:
            raise ValueError(f"wrong Scorer v11 train manual verdict schema: {source_id}")
        for row in (item, source, verdict):
            if row.get("boundary_serialization_contract_id") != (
                ACOUSTIC_BINARY_V12_CONTRACT.contract_id
            ):
                raise ValueError(f"wrong central Boundary contract: {source_id}")
        if (
            item.get("partition") != "train"
            or source.get("partition") != "train"
            or verdict.get("partition") != "train"
            or verdict.get("reviewed_full_source") is not True
        ):
            raise ValueError(f"train source was not fully reviewed: {source_id}")
        video_id = str(source.get("video_id") or "")
        if not video_id or video_id != str(item.get("video_id") or ""):
            raise ValueError(f"train source video identity mismatch: {source_id}")
        if video_id in seen_videos:
            raise ValueError(f"train review must use at most one source per video: {video_id}")
        seen_videos.add(video_id)
        frame_count = int(source.get("frame_count") or 0)
        if (
            frame_count <= 0
            or int(item.get("frame_count") or 0) != frame_count
            or int(verdict.get("frame_count") or 0) != frame_count
            or float(source.get("frame_hop_s") or 0.0) != FRAME_HOP_S
            or float(item.get("frame_hop_s") or 0.0) != FRAME_HOP_S
            or float(verdict.get("frame_hop_s") or 0.0) != FRAME_HOP_S
        ):
            raise ValueError(f"train source frame geometry mismatch: {source_id}")
        spans = _validate_spans(
            list(verdict.get("spans") or ()), source_id=source_id, frame_count=frame_count
        )
        if str(verdict.get("verdict") or "") != _expected_verdict(spans):
            raise ValueError(f"train source verdict/spans mismatch: {source_id}")

        audio = _resolve(str(source.get("audio") or ""), owner=source_manifest)
        if not audio.is_file():
            raise FileNotFoundError(audio)
        audio_sha = str(source.get("audio_sha256") or "")
        if len(audio_sha) != 64 or audio_sha != str(item.get("audio_sha256") or ""):
            raise ValueError(f"train source audio identity mismatch: {source_id}")
        if verify_audio and _sha256(audio) != audio_sha:
            raise ValueError(f"train source audio SHA mismatch: {source_id}")
        audit_audio = _resolve(str(item.get("audio") or ""), owner=audit_manifest)
        if not audit_audio.is_file() or (verify_audio and _sha256(audit_audio) != audio_sha):
            raise ValueError(f"train audit audio SHA mismatch: {source_id}")
        sample_rate, sample_count, audio_frames = _wav_geometry(audio)
        if (
            sample_rate != int(source.get("sample_rate") or 0)
            or sample_count != int(source.get("sample_count") or 0)
            or audio_frames != frame_count
        ):
            raise ValueError(f"train source WAV/frame geometry mismatch: {source_id}")
        for span in spans:
            label_counts[str(span["label"])] += int(span["end_frame"]) - int(
                span["start_frame"]
            )
        compiled.append(
            {
                "schema": OUTPUT_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "video_id": video_id,
                "partition": "train",
                "input_distribution": "real_workflow_source_window_human_full_source_v1",
                "source_kind": "real_train_full_source_manual",
                "synthetic_composite": False,
                "audio": _display(audio),
                "audio_sha256": audio_sha,
                "sample_rate": sample_rate,
                "sample_count": sample_count,
                "duration_s": sample_count / sample_rate,
                "frame_count": frame_count,
                "frame_hop_s": FRAME_HOP_S,
                "core_ids": [f"real-train-manual-source::{source_id}"],
                "canonical_spans": spans,
                "annotation_provenance": "human_full_source_review",
                "audit_summary": _display(audit_summary),
                "audit_summary_sha256": _sha256(audit_summary),
                "audit_manifest": _display(audit_manifest),
                "audit_manifest_sha256": _sha256(audit_manifest),
                "manual_verdicts": _display(manual_verdicts),
                "manual_verdicts_sha256": _sha256(manual_verdicts),
                "source_manifest": _display(source_manifest),
                "source_manifest_sha256": _sha256(source_manifest),
                "teacher_output_used_as_annotation_seed": bool(
                    summary.get("teacher_output_used_as_annotation_seed")
                ),
                "teacher_output_used_as_truth": False,
                "unselected_source_label_inheritance": False,
                "unsure_training_label": -100,
                "reviewed_full_source": True,
                "training_manifest_allowed": True,
            }
        )

    if label_counts["inside_candidate"] <= 0:
        raise ValueError("Scorer v11 manual train truth contains no inside_candidate frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "real_train_manual_sources.jsonl"
    _write_jsonl(output, compiled)
    result = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "real_train_manual_sources": _display(output),
        "real_train_manual_sources_sha256": _sha256(output),
        "audit_summary": _display(audit_summary),
        "audit_summary_sha256": _sha256(audit_summary),
        "audit_manifest": _display(audit_manifest),
        "audit_manifest_sha256": _sha256(audit_manifest),
        "manual_verdicts": _display(manual_verdicts),
        "manual_verdicts_sha256": _sha256(manual_verdicts),
        "source_manifest": _display(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "source_count": len(compiled),
        "video_count": len(seen_videos),
        "canonical_frame_counts": dict(sorted(label_counts.items())),
        "all_sources_human_confirmed": True,
        "teacher_output_used_as_truth": False,
        "unselected_source_label_inheritance": False,
        "unsure_training_label": -100,
        "training_manifest_allowed": True,
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
    parser.add_argument("--manual-verdicts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-audio-content-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return compile_real_train_manual(
        audit_summary=Path(args.audit_summary),
        audit_manifest=Path(args.audit_manifest),
        manual_verdicts=Path(args.manual_verdicts),
        output_dir=Path(args.output_dir),
        verify_audio=not args.skip_audio_content_check,
    )


if __name__ == "__main__":
    main()
