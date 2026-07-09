#!/usr/bin/env python3
"""Evaluate manual verdicts for the Pre-ASR v12 long false-drop gate."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUMMARY_SCHEMA = "pre_asr_v12_false_drop_audit_gate_summary_v1"
VALID_VERDICTS = {"drop", "keep", "unsure"}


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(dict(payload))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def candidate_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("candidate_id") or row.get("sample_id") or row.get("id") or "").strip()
    if value:
        return value
    audio_id = str(row.get("audio_id") or row.get("window_id") or "").strip()
    chunk_index = int(float(row.get("chunk_index", 0) or 0))
    return f"preasr-{audio_id}-chunk{chunk_index:05d}"


def _is_long_false_drop_row(row: Mapping[str, Any]) -> bool:
    audit_kind = str(row.get("audit_kind") or "").strip()
    source_pool = str(row.get("source_pool") or "").strip()
    if audit_kind or source_pool:
        return audit_kind == "long_false_drop" or "long_false_drop" in source_pool
    return True


def _compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_id",
        "window_id",
        "audio_id",
        "chunk_index",
        "start",
        "end",
        "duration_s",
        "v12_prob_drop",
        "model_prob_drop",
        "truth",
        "v12_prediction",
        "verdict",
        "note",
    )
    compact = {key: row.get(key) for key in keys if key in row}
    compact["candidate_id"] = candidate_id(row)
    return compact


def evaluate_false_drop_audit(
    *,
    manifest_rows: Iterable[Mapping[str, Any]],
    verdict_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    targets = [dict(row) for row in manifest_rows if _is_long_false_drop_row(row)]
    target_by_id = {candidate_id(row): row for row in targets}
    if len(target_by_id) != len(targets):
        counts = Counter(candidate_id(row) for row in targets)
        duplicates = sorted(candidate for candidate, count in counts.items() if count > 1)
        raise ValueError(f"duplicate manifest candidates: {duplicates[:10]}")

    verdict_by_id: dict[str, dict[str, Any]] = {}
    duplicate_verdicts: list[str] = []
    unknown_verdicts: list[dict[str, Any]] = []
    for row in verdict_rows:
        item = dict(row)
        cid = candidate_id(item)
        verdict = str(item.get("verdict") or item.get("manual_label") or "").strip().lower()
        item["candidate_id"] = cid
        item["verdict"] = verdict
        if verdict not in VALID_VERDICTS:
            unknown_verdicts.append(_compact_row(item))
            continue
        if cid in verdict_by_id:
            duplicate_verdicts.append(cid)
            continue
        verdict_by_id[cid] = item

    target_ids = set(target_by_id)
    reviewed_ids = target_ids & set(verdict_by_id)
    unexpected_ids = sorted(set(verdict_by_id) - target_ids)
    missing_ids = sorted(target_ids - set(verdict_by_id))
    reviewed = [verdict_by_id[cid] for cid in sorted(reviewed_ids)]
    verdict_counts = Counter(str(row["verdict"]) for row in reviewed)
    true_semantic_keep_deletions = [
        {**_compact_row(target_by_id[candidate_id(row)]), "verdict": row["verdict"], "note": row.get("note", "")}
        for row in reviewed
        if str(row["verdict"]) == "keep"
    ]
    uncertain = [
        {**_compact_row(target_by_id[candidate_id(row)]), "verdict": row["verdict"], "note": row.get("note", "")}
        for row in reviewed
        if str(row["verdict"]) == "unsure"
    ]
    complete = (
        not missing_ids
        and not unexpected_ids
        and not duplicate_verdicts
        and not unknown_verdicts
        and len(reviewed_ids) == len(target_ids)
    )
    gate_pass = complete and not true_semantic_keep_deletions
    return {
        "schema": SUMMARY_SCHEMA,
        "target_manifest_count": len(targets),
        "manual_verdict_count": len(list(verdict_by_id.values())),
        "reviewed_target_count": len(reviewed_ids),
        "missing_count": len(missing_ids),
        "missing_candidates": missing_ids,
        "unexpected_count": len(unexpected_ids),
        "unexpected_candidates": unexpected_ids,
        "duplicate_verdict_count": len(duplicate_verdicts),
        "duplicate_verdict_candidates": sorted(duplicate_verdicts),
        "invalid_verdict_count": len(unknown_verdicts),
        "invalid_verdict_rows": unknown_verdicts,
        "manual_verdict_counts": dict(sorted(verdict_counts.items())),
        "true_semantic_keep_deletion_count": len(true_semantic_keep_deletions),
        "true_semantic_keep_deletions": true_semantic_keep_deletions,
        "uncertain_count": len(uncertain),
        "uncertain_candidates": uncertain,
        "complete": complete,
        "gate_pass": gate_pass,
        "promote_allowed": gate_pass,
        "decision_note": (
            "Pass means every long false-drop target was reviewed and no manual verdict is keep. "
            "Unsure is recorded as residual audit risk but is not a confirmed semantic keep deletion."
        ),
    }


def evaluate_paths(*, manifest: Path, verdicts: Path | Iterable[Path], output: Path | None) -> dict[str, Any]:
    verdict_paths = [verdicts] if isinstance(verdicts, Path) else list(verdicts)
    verdict_rows: list[dict[str, Any]] = []
    for path in verdict_paths:
        verdict_rows.extend(read_jsonl(path))
    summary = evaluate_false_drop_audit(
        manifest_rows=read_jsonl(manifest),
        verdict_rows=verdict_rows,
    )
    summary.update(
        {
            "manifest": repo_rel(manifest),
            "verdicts": [repo_rel(path) for path in verdict_paths],
        }
    )
    if output:
        summary["output"] = repo_rel(output)
        write_json(output, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        default="agents/audits/20260709_122906_pre-asr-v12-v3-train-long-false-drop-audit",
    )
    parser.add_argument("--manifest", default="")
    parser.add_argument(
        "--verdicts",
        action="append",
        default=[],
        help="Manual verdict JSONL. May be repeated; defaults to <audit-dir>/manual_verdicts.jsonl.",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit_dir = project_path(args.audit_dir)
    manifest = project_path(args.manifest) if args.manifest else audit_dir / "manifest.jsonl"
    verdicts = [project_path(path) for path in args.verdicts] if args.verdicts else [audit_dir / "manual_verdicts.jsonl"]
    output = project_path(args.output) if args.output else audit_dir / "gate_summary.json"
    summary = evaluate_paths(manifest=manifest, verdicts=verdicts, output=output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
