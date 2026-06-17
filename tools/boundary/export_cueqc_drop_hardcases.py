#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LABEL_SCHEMA = "cueqc_false_drop_audit_label_v1"
CANDIDATE_SCHEMA = "boundary_hardcase_candidate_from_cueqc_v1"
SAFETY_SCHEMA = "cueqc_drop_safety_holdout_v1"
SUMMARY_SCHEMA = "cueqc_drop_hardcase_export_summary_v1"

CONFIRMED_DROP = "drop_ok"
SAFETY_DECISIONS = {"false_drop_keep", "uncertain"}

FRAME_NEGATIVE_ROUTE = "speech_boundary_frame_negative_candidate"
PREFERENCE_ROUTE = "boundary_preference_candidate"

PUNCT_OR_NONLEXICAL_RE = re.compile(
    r"^[\s\u3000、。！？!?.,，．…・「」『』（）()【】\[\]~〜ー"
    r"ぁ-ぉゃゅょっァ-ォャュョッ"
    r"あいうえおアイウエオ"
    r"んンっッはぁハァふぅフゥへぇヘェほぉホォ"
    r"うぅウゥえぇエェおぉオォあぁアァ"
    r"くぅクゥぐぅグゥむぅムゥ"
    r"はっハッふっフッへっヘッほっホッ"
    r"ー]+$"
)


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("/", "\\")
    except ValueError:
        return str(path)


def local_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(dict(row))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def discover_label_paths() -> list[Path]:
    audit_root = PROJECT_ROOT / "agents" / "audits"
    if not audit_root.exists():
        return []
    return sorted(audit_root.glob("*/cueqc_false_drop_audit_labels.jsonl"))


def sample_key(row: Mapping[str, Any]) -> str:
    key = str(row.get("sample_id") or row.get("audit_id") or "").strip()
    if key:
        return key
    video_id = str(row.get("video_id") or "")
    chunk_index = str(row.get("chunk_index") or "")
    start = str(row.get("start") or "")
    end = str(row.get("end") or "")
    if video_id and chunk_index:
        return f"{video_id}:chunk{chunk_index}:{start}:{end}"
    raise ValueError(f"label row is missing sample_id/audit_id and video chunk identity: {row!r}")


def row_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalized_tags(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    tags: set[str] = set()
    for row in rows:
        for tag in row.get("reason_tags") or []:
            text = str(tag).strip()
            if text:
                tags.add(text)
    return sorted(tags)


def nonempty_notes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    notes: list[str] = []
    seen: set[str] = set()
    for row in rows:
        note = str(row.get("notes") or "").strip()
        if note and note not in seen:
            notes.append(note)
            seen.add(note)
    return notes


def latest_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return max(rows, key=lambda row: str(row.get("updated_at") or ""))


def is_nonlexical_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    compact = text.replace(" ", "").replace("\u3000", "")
    return bool(PUNCT_OR_NONLEXICAL_RE.fullmatch(compact))


def text_bucket(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "empty"
    if is_nonlexical_text(text):
        return "nonlexical"
    if len(text) <= 4:
        return "short_text"
    if len(text) <= 16:
        return "medium_text"
    return "long_text"


def duration_bucket(duration_s: float) -> str:
    if duration_s < 0.5:
        return "<0.5s"
    if duration_s < 1.0:
        return "0.5-1s"
    if duration_s < 2.0:
        return "1-2s"
    if duration_s < 4.0:
        return "2-4s"
    return ">=4s"


def route_confirmed_drop(*, tags: Sequence[str], duration_s: float, text: str) -> tuple[str, str]:
    tag_set = set(tags)
    lexical_or_mixed = bool(tag_set & {"dialogue", "overlap"})
    if lexical_or_mixed:
        return (
            PREFERENCE_ROUTE,
            "confirmed drop contains dialogue/overlap tag; needs boundary preference target before training",
        )
    if duration_s >= 2.0:
        return (
            PREFERENCE_ROUTE,
            "confirmed drop is long enough to need neighbor/boundary context before training",
        )
    if tag_set & {"environment", "breath", "short_fragment", "vocalization"}:
        return (
            FRAME_NEGATIVE_ROUTE,
            "confirmed short non-lexical/noise drop; candidate for SpeechBoundary-JA frame-negative conversion",
        )
    if is_nonlexical_text(text):
        return (
            FRAME_NEGATIVE_ROUTE,
            "confirmed short non-lexical drop without explicit tag; candidate for frame-negative conversion",
        )
    return (
        PREFERENCE_ROUTE,
        "confirmed drop has possible lexical signal; keep as boundary hard-case candidate",
    )


def evidence_summary(rows: Sequence[Mapping[str, Any]], source_paths: Sequence[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        source_index = int(row.get("_source_index") or 0)
        result.append(
            {
                "source_label_path": repo_display_path(source_paths[source_index]),
                "audit_id": str(row.get("audit_id") or ""),
                "dataset_id": str(row.get("dataset_id") or ""),
                "manual_decision": str(row.get("manual_decision") or ""),
                "reason_tags": list(row.get("reason_tags") or []),
                "updated_at": str(row.get("updated_at") or ""),
            }
        )
    return result


def build_candidate(
    *,
    key: str,
    rows: Sequence[Mapping[str, Any]],
    source_paths: Sequence[Path],
) -> dict[str, Any]:
    row = latest_row(rows)
    tags = normalized_tags(rows)
    start = row_float(row, "start")
    end = row_float(row, "end", start)
    duration_s = max(0.0, end - start)
    text = str(row.get("text") or "")
    route, route_reason = route_confirmed_drop(tags=tags, duration_s=duration_s, text=text)
    probs_drop = [row_float(item, "display_prob_drop", row_float(item, "confidence")) for item in rows]
    probs_keep = [row_float(item, "display_prob_keep") for item in rows]
    candidate_id = f"cueqc-drop-hardcase-{key}"
    source_label_paths = sorted(
        {
            repo_display_path(source_paths[int(item.get("_source_index") or 0)])
            for item in rows
        }
    )
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_id": candidate_id,
        "source": "cueqc_false_drop_audit",
        "source_label_paths": source_label_paths,
        "source_label_count": len(rows),
        "source_evidence": evidence_summary(rows, source_paths),
        "sample_id": key,
        "audit_id": str(row.get("audit_id") or key),
        "video_id": str(row.get("video_id") or ""),
        "video_label": str(row.get("video_label") or ""),
        "chunk_index": int(row.get("chunk_index") or 0),
        "start": round(start, 6),
        "end": round(end, 6),
        "duration_s": round(duration_s, 6),
        "duration_bucket": duration_bucket(duration_s),
        "text": text,
        "text_bucket": text_bucket(text),
        "manual_decision": CONFIRMED_DROP,
        "reason_tags": tags,
        "notes": nonempty_notes(rows),
        "display_prob_drop_min": round(min(probs_drop), 6) if probs_drop else 0.0,
        "display_prob_drop_max": round(max(probs_drop), 6) if probs_drop else 0.0,
        "display_prob_drop_mean": round(sum(probs_drop) / len(probs_drop), 6) if probs_drop else 0.0,
        "display_prob_keep_mean": round(sum(probs_keep) / len(probs_keep), 6) if probs_keep else 0.0,
        "candidate_route": route,
        "route_reason": route_reason,
        "v51_status": "candidate_only_not_direct_delta_label",
        "required_conversion": (
            "convert to SpeechBoundary-JA frame-negative labels"
            if route == FRAME_NEGATIVE_ROUTE
            else "convert to Boundary preference or start/end delta targets"
        ),
    }


def build_safety_holdout(
    *,
    key: str,
    rows: Sequence[Mapping[str, Any]],
    source_paths: Sequence[Path],
) -> dict[str, Any]:
    row = latest_row(rows)
    decisions = sorted({str(item.get("manual_decision") or "") for item in rows})
    start = row_float(row, "start")
    end = row_float(row, "end", start)
    return {
        "schema": SAFETY_SCHEMA,
        "sample_id": key,
        "audit_id": str(row.get("audit_id") or key),
        "source": "cueqc_false_drop_audit",
        "source_label_paths": sorted(
            {
                repo_display_path(source_paths[int(item.get("_source_index") or 0)])
                for item in rows
            }
        ),
        "source_label_count": len(rows),
        "source_evidence": evidence_summary(rows, source_paths),
        "manual_decisions": decisions,
        "holdout_reason": (
            "conflicting or non-drop manual decisions; use for CueQC false-drop safety, not Boundary training"
        ),
        "video_id": str(row.get("video_id") or ""),
        "video_label": str(row.get("video_label") or ""),
        "chunk_index": int(row.get("chunk_index") or 0),
        "start": round(start, 6),
        "end": round(end, 6),
        "duration_s": round(max(0.0, end - start), 6),
        "text": str(row.get("text") or ""),
        "reason_tags": normalized_tags(rows),
        "notes": nonempty_notes(rows),
    }


def summarize_candidates(
    *,
    source_paths: Sequence[Path],
    raw_rows: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    candidates: Sequence[Mapping[str, Any]],
    safety: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    raw_decisions = Counter(str(row.get("manual_decision") or "") for row in raw_rows)
    unique_decisions: Counter[str] = Counter()
    for rows in grouped.values():
        decisions = {str(row.get("manual_decision") or "") for row in rows}
        if decisions == {CONFIRMED_DROP}:
            unique_decisions[CONFIRMED_DROP] += 1
        elif decisions & SAFETY_DECISIONS:
            unique_decisions["safety_holdout"] += 1
        else:
            unique_decisions["other"] += 1
    route_counts = Counter(str(row.get("candidate_route") or "") for row in candidates)
    video_counts = Counter(str(row.get("video_id") or "") for row in candidates)
    reason_counts: Counter[str] = Counter()
    duration_counts = Counter(str(row.get("duration_bucket") or "") for row in candidates)
    for row in candidates:
        for tag in row.get("reason_tags") or []:
            reason_counts[str(tag)] += 1
    duplicate_items = sum(1 for rows in grouped.values() if len(rows) > 1)
    duplicate_rows = sum(max(0, len(rows) - 1) for rows in grouped.values())
    return {
        "schema": SUMMARY_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_label_paths": [repo_display_path(path) for path in source_paths],
        "output_dir": repo_display_path(output_dir),
        "outputs": {
            "confirmed_drop_candidates": repo_display_path(
                output_dir / "cueqc_confirmed_drop_candidates.jsonl"
            ),
            "safety_holdout": repo_display_path(output_dir / "cueqc_drop_safety_holdout.jsonl"),
            "summary_json": repo_display_path(output_dir / "summary.json"),
            "summary_md": repo_display_path(output_dir / "summary.md"),
        },
        "counts": {
            "raw_label_rows": len(raw_rows),
            "unique_label_items": len(grouped),
            "duplicate_label_items": duplicate_items,
            "duplicate_extra_rows": duplicate_rows,
            "confirmed_drop_candidates": len(candidates),
            "safety_holdout_items": len(safety),
        },
        "raw_manual_decision_counts": dict(raw_decisions),
        "unique_decision_counts": dict(unique_decisions),
        "candidate_route_counts": dict(route_counts),
        "candidate_reason_tag_counts": dict(reason_counts),
        "candidate_video_counts": dict(video_counts),
        "candidate_duration_bucket_counts": dict(duration_counts),
        "v51_boundary": {
            "direct_training_dataset_emitted": False,
            "direct_schema": None,
            "reason": "CueQC drop_ok is chunk-level display routing, not Boundary v5.1 start/end delta supervision.",
            "next_conversion": [
                "frame-negative candidates need audio/frame materialization before SpeechBoundary-JA finetune",
                "boundary preference candidates need neighbor context and A/B or delta targets before Boundary Refiner v5.1 training",
            ],
        },
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# CueQC Drop Hard-Case Export",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in summary["input_label_paths"])
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Raw label rows: `{counts['raw_label_rows']}`",
            f"- Unique label items: `{counts['unique_label_items']}`",
            f"- Duplicate extra rows: `{counts['duplicate_extra_rows']}`",
            f"- Confirmed drop candidates: `{counts['confirmed_drop_candidates']}`",
            f"- Safety holdout items: `{counts['safety_holdout_items']}`",
            f"- Routes: `{summary['candidate_route_counts']}`",
            "",
            "## Outputs",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in summary["outputs"].items())
    lines.extend(
        [
            "",
            "## Boundary v5.1 Status",
            "",
            "- This export is a candidate manifest only.",
            "- It does not emit `boundary_refiner_frame_sequence_dataset_v5`.",
            "- Confirmed CueQC drops must be converted into frame-negative labels or boundary preference/delta targets before v5.1 training.",
            "",
        ]
    )
    return "\n".join(lines)


def export_cueqc_drop_hardcases(
    *,
    label_paths: Sequence[Path],
    output_dir: Path,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    source_paths = [path.resolve() for path in label_paths]
    if not source_paths:
        raise ValueError("no CueQC false-drop label paths were provided or discovered")
    missing = [repo_display_path(path) for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing label files: {missing}")

    raw_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_index, path in enumerate(source_paths):
        for row in read_jsonl(path):
            schema = str(row.get("schema") or "")
            if schema and schema != LABEL_SCHEMA:
                raise ValueError(f"unsupported label schema {schema!r} in {path}")
            item = dict(row)
            item["_source_index"] = source_index
            key = sample_key(item)
            raw_rows.append(item)
            grouped[key].append(item)

    candidates: list[dict[str, Any]] = []
    safety: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        decisions = {str(row.get("manual_decision") or "") for row in rows}
        if decisions == {CONFIRMED_DROP}:
            candidates.append(build_candidate(key=key, rows=rows, source_paths=source_paths))
        else:
            safety.append(build_safety_holdout(key=key, rows=rows, source_paths=source_paths))

    if require_nonempty and not candidates:
        raise ValueError("no confirmed drop_ok rows were found")

    candidates.sort(
        key=lambda row: (
            str(row.get("candidate_route") or ""),
            str(row.get("video_id") or ""),
            float(row.get("start") or 0.0),
            str(row.get("sample_id") or ""),
        )
    )
    safety.sort(
        key=lambda row: (
            str(row.get("video_id") or ""),
            float(row.get("start") or 0.0),
            str(row.get("sample_id") or ""),
        )
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "cueqc_confirmed_drop_candidates.jsonl", candidates)
    write_jsonl(output_dir / "cueqc_drop_safety_holdout.jsonl", safety)
    summary = summarize_candidates(
        source_paths=source_paths,
        raw_rows=raw_rows,
        grouped=grouped,
        candidates=candidates,
        safety=safety,
        output_dir=output_dir,
    )
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export confirmed CueQC drop_ok audit labels as Boundary/SpeechBoundary hard-case "
            "candidates. This does not create a direct Boundary v5.1 training dataset."
        )
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help=(
            "CueQC false-drop label JSONL files. Defaults to "
            "agents/audits/*/cueqc_false_drop_audit_labels.jsonl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to agents/temp/YYYYMMDD_HHMMSS_boundary-hardcase-candidates-from-cueqc.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write empty manifests instead of failing when no confirmed drop candidates exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    label_paths = (
        [project_path(value) for value in args.labels]
        if args.labels is not None
        else discover_label_paths()
    )
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT
        / "agents"
        / "temp"
        / f"{local_timestamp()}_boundary-hardcase-candidates-from-cueqc"
    )
    summary = export_cueqc_drop_hardcases(
        label_paths=label_paths,
        output_dir=output_dir,
        require_nonempty=not args.allow_empty,
    )
    print(f"output_dir={summary['output_dir']}")
    print(f"candidates={summary['outputs']['confirmed_drop_candidates']}")
    print(f"safety_holdout={summary['outputs']['safety_holdout']}")
    print(
        "raw_rows={raw} unique_items={unique} confirmed_candidates={confirmed} safety_holdout={safety}".format(
            raw=summary["counts"]["raw_label_rows"],
            unique=summary["counts"]["unique_label_items"],
            confirmed=summary["counts"]["confirmed_drop_candidates"],
            safety=summary["counts"]["safety_holdout_items"],
        )
    )
    print(f"routes={json.dumps(summary['candidate_route_counts'], ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
