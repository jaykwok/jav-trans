#!/usr/bin/env python3
"""Select unreviewed low-confidence Split v4 events into a compact audit page."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _stable_key(row: dict) -> tuple:
    if row["category"] == "unmatched_predicted_cut_event":
        return (
            row["category"],
            row["audio_id"],
            round(float(row["time_s"]), 3),
        )
    return (
        row["category"],
        row["audio_id"],
        round(float(row["start_s"]), 3),
        round(float(row["end_s"]), 3),
    )


def _reviewed_by_key(audit_dir: Path) -> dict[tuple, dict]:
    manifest = {row["audit_id"]: row for row in _rows(audit_dir / "audit_manifest.jsonl")}
    reviewed: dict[tuple, dict] = {}
    for verdict in _rows(audit_dir / "manual_verdicts.jsonl"):
        if verdict.get("verdict") in {None, "", "unreviewed"}:
            continue
        source = manifest.get(verdict["audit_id"])
        if source is not None:
            reviewed[_stable_key(source)] = {"source": source, "verdict": verdict}
    return reviewed


def build(
    *,
    source_dir: Path,
    output_dir: Path,
    max_p_cut: float,
    prior_audit_dirs: tuple[Path, ...] = (),
    title: str = "Acoustic Split v4 Low-confidence Hard Audit",
) -> dict:
    manifest = _rows(source_dir / "audit_manifest.jsonl")
    verdicts = {row["audit_id"]: row for row in _rows(source_dir / "manual_verdicts.jsonl")}
    reused_by_key: dict[tuple, dict] = {}
    for prior_dir in prior_audit_dirs:
        reused_by_key.update(_reviewed_by_key(prior_dir))
    reused = []
    for row in manifest:
        prior = reused_by_key.get(_stable_key(row))
        if prior is None or verdicts.get(row["audit_id"], {}).get("verdict") not in {None, "", "unreviewed"}:
            continue
        reused.append({
            "schema": "split_v4_binary_gate_reused_verdict_v1",
            "audit_id": row["audit_id"],
            "category": row["category"],
            "audio_id": row["audio_id"],
            "verdict": prior["verdict"]["verdict"],
            "note": prior["verdict"].get("note", ""),
            "source_audit_id": prior["source"]["audit_id"],
            "source_audio_id": prior["source"]["audio_id"],
            "match_key": list(_stable_key(row)),
        })
    reused_keys = {_stable_key(row) for row in manifest if any(item["audit_id"] == row["audit_id"] for item in reused)}
    unreviewed = [
        row
        for row in manifest
        if verdicts.get(row["audit_id"], {}).get("verdict") in {None, "", "unreviewed"}
        and _stable_key(row) not in reused_keys
    ]
    low_confidence = sorted(
        (
            row
            for row in unreviewed
            if row["category"] == "unmatched_predicted_cut_event"
            and float(row.get("p_cut") or 1.0) <= max_p_cut
        ),
        key=lambda row: (float(row["p_cut"]), str(row["audit_id"])),
    )
    residuals = sorted(
        (row for row in unreviewed if row["category"] == "long_residual"),
        key=lambda row: (-float(row["duration_s"]), str(row["audit_id"])),
    )
    selected = [*low_confidence, *residuals]
    if not selected:
        raise ValueError("no unreviewed low-confidence or long-residual items selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    copied: set[str] = set()
    for row in selected:
        filename = Path(str(row["audio_src"])).name
        if filename not in copied:
            shutil.copyfile(source_dir / "audio" / filename, audio_dir / filename)
            copied.add(filename)
        row["selection_reason"] = (
            f"unreviewed_p_cut_le_{max_p_cut:.2f}"
            if row["category"] == "unmatched_predicted_cut_event"
            else "unreviewed_long_residual"
        )

    (output_dir / "audit_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        "utf-8",
    )
    (output_dir / "reused_manual_verdicts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reused),
        "utf-8",
    )
    html = (source_dir / "index.html").read_text("utf-8")
    prefix = "const rows="
    marker = ",key='split-v4-binary-gate-audit-v1:'+location.pathname"
    start = html.index(prefix) + len(prefix)
    end = html.index(marker, start)
    payload = json.dumps(selected, ensure_ascii=False).replace("</", "<\\/")
    html = html[:start] + payload + html[end:]
    html = html.replace(
        "Acoustic Split v4 · 二分类晋升人工 gate",
        title,
    ).replace(
        "<title>Split v4 binary gate audit</title>",
        f"<title>{title}</title>",
    )
    (output_dir / "index.html").write_text(html, "utf-8")
    summary = {
        "schema": "split_v4_binary_hard_audit_selection_v1",
        "title": title,
        "source_audit_dir": str(source_dir),
        "prior_audit_dirs": [str(path) for path in prior_audit_dirs],
        "reused_manual_verdict_count": len(reused),
        "max_p_cut": max_p_cut,
        "low_confidence_event_count": len(low_confidence),
        "long_residual_count": len(residuals),
        "item_count": len(selected),
        "p_cut_range": [
            min(float(row["p_cut"]) for row in low_confidence),
            max(float(row["p_cut"]) for row in low_confidence),
        ] if low_confidence else [],
        "manual_verdicts": str(output_dir / "manual_verdicts.jsonl"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    update_audit_entrypoints(
        latest_html=output_dir / "index.html",
        title=title,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-p-cut", type=float, default=0.65)
    parser.add_argument("--prior-audit-dir", action="append", default=[])
    parser.add_argument(
        "--title",
        default="Acoustic Split v4 Low-confidence Hard Audit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build(
        source_dir=Path(args.source_audit_dir),
        output_dir=Path(args.output_dir),
        max_p_cut=args.max_p_cut,
        prior_audit_dirs=tuple(Path(path) for path in args.prior_audit_dir),
        title=str(args.title),
    ), ensure_ascii=False))
