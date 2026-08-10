#!/usr/bin/env python3
"""Build a blinded real-JAV audit from observation-only CTC shadow runs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment_shadow import SHADOW_RUN_SCHEMA  # noqa: E402
from tools.audits.audit_nav import audit_generated_at, update_audit_entrypoints  # noqa: E402
from tools.audits.generate_ctc_alignment_ab_audit import (  # noqa: E402
    ANSWER_SCHEMA,
    SUMMARY_SCHEMA,
    materialize,
    render_page,
    select_trials,
)


def _json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def observation_files(values: list[str]) -> list[Path]:
    files: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            files.extend(sorted(item for item in path.glob("*.json") if item.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted({item.resolve() for item in files}, key=lambda item: item.as_posix())


def _candidate_id(payload: dict[str, Any], row: dict[str, Any]) -> str:
    identity = "|".join(
        (
            str(payload.get("audio_cache_key") or ""),
            str(payload.get("source_video_path") or ""),
            str(row.get("chunk_index") or ""),
            str(row.get("text") or ""),
        )
    )
    return "jav-shadow:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def candidates_from_observations(paths: list[Path]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("schema") != SHADOW_RUN_SCHEMA:
            continue
        audio = Path(str(payload.get("source_video_path") or "")).expanduser()
        if not audio.is_file():
            continue
        try:
            duration = float(payload["source_video_duration_s"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            minimum_delta_ms = max(
                0.0,
                float(payload.get("minimum_disagreement_ms") or 0.0),
            )
        except (TypeError, ValueError):
            minimum_delta_ms = 0.0
        for row in payload.get("comparisons") or []:
            if not isinstance(row, dict) or row.get("status") != "ok":
                continue
            required = (
                "primary_start_abs_s",
                "primary_end_abs_s",
                "shadow_start_abs_s",
                "shadow_end_abs_s",
            )
            try:
                primary_start, primary_end, shadow_start, shadow_end = (
                    float(row[key]) for key in required
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not (
                0.0 <= primary_start < primary_end <= duration + 1e-6
                and 0.0 <= shadow_start < shadow_end <= duration + 1e-6
            ):
                continue
            candidate_id = _candidate_id(payload, row)
            by_id[candidate_id] = {
                "candidate_id": candidate_id,
                "domain": "jav",
                "source_id": f"{payload.get('job_id', '')}:{row.get('chunk_index', '')}",
                "audio": str(audio.resolve()),
                "audio_duration_s": duration,
                "text": str(row.get("text") or ""),
                "model_a_start_s": primary_start,
                "model_a_end_s": primary_end,
                "model_b_start_s": shadow_start,
                "model_b_end_s": shadow_end,
                "minimum_delta_ms": minimum_delta_ms,
                "observation_path": str(path),
                "job_id": str(payload.get("job_id") or ""),
                "chunk_index": int(row.get("chunk_index") or 0),
            }
    return [by_id[key] for key in sorted(by_id)]


def _excluded_pairs(paths: list[str]) -> set[tuple[str, str]]:
    return {
        (str(row["candidate_id"]), str(row["boundary"]))
        for value in paths
        for row in _json_rows(Path(value))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        action="append",
        required=True,
        help="Shadow-run JSON file or directory; repeat to combine roots.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-answers", action="append", default=[])
    parser.add_argument("--per-boundary", type=int, default=25)
    parser.add_argument("--clip-seconds", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--no-update-latest", action="store_true")
    args = parser.parse_args()

    inputs = observation_files(args.observations)
    candidates = candidates_from_observations(inputs)
    excluded = _excluded_pairs(args.exclude_answers)
    trials = select_trials(
        candidates,
        per_boundary=int(args.per_boundary),
        clip_s=float(args.clip_seconds),
        seed=int(args.seed),
        domains=("jav",),
        exclude_pairs=excluded,
        minimum_delta_ms=0.0,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "shadow_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    page_rows, answers = materialize(
        trials,
        output_dir=output_dir,
        clip_s=float(args.clip_seconds),
        seed=int(args.seed),
    )
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in page_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for row in answers:
            if row.get("schema") != ANSWER_SCHEMA:
                raise AssertionError("unexpected answer schema")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    page = output_dir / "index.html"
    rendered = render_page(page_rows)
    for forbidden in ("model_a", "model_b", "ctc_aligner.pt", "delta_ms"):
        if forbidden in rendered:
            raise AssertionError(f"blind page leaked {forbidden}")
    page.write_text(rendered, encoding="utf-8")

    summary = {
        "schema": SUMMARY_SCHEMA,
        "generated_at": audit_generated_at(),
        "page": str(page.resolve()),
        "observation_files": len(inputs),
        "unique_candidates": len(candidates),
        "review_items": len(answers),
        "counts": dict(Counter(f"{row['domain']}:{row['boundary']}" for row in answers)),
        "excluded_prior_pairs": len(excluded),
        "minimum_selected_delta_ms": min(
            (float(row["delta_ms"]) for row in answers), default=None
        ),
        "maximum_selected_delta_ms": max(
            (float(row["delta_ms"]) for row in answers), default=None
        ),
        "clip_seconds": float(args.clip_seconds),
        "blind": True,
        "domain": "jav",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.no_update_latest:
        update_audit_entrypoints(latest_html=page, title="CTC 影子分歧 · 真实 JAV A/B")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
