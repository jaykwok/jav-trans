#!/usr/bin/env python3
"""Build pre-ASR label overrides from the manual disagreement audit.

Inputs are the manual verdicts (``manual_verdicts.jsonl`` from the Omni-v3
disagreement audit) and the route-replay ``checkpoint_evaluation.json`` whose
``pre_asr.<checkpoint>.false_decisions`` rows enumerate the full disagreement
population. Policy (2026-07-07):

- audited rows (100): verdict overrides the Omni label
  (drop -> definite_drop, keep -> definite_keep, unsure -> ambiguous_ignore);
- unaudited A-direction rows (model drop / Omni keep): ambiguous_ignore —
  the audit showed Omni is ~96% wrong here, but bulk-flipping without a human
  verdict is not allowed, so they are excluded from training instead;
- unaudited B-direction rows (model keep / Omni drop): no override — the Omni
  drop label stays (~20% noise absorbed by the keep-class weight).

Every override row carries both the ``candidate_id`` join key (exact-match
compile overlays) and the ``(video_id, start, end)`` span key so stage-D chunk
re-exports can project the same decisions onto new chunk boundaries.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OVERRIDE_SCHEMA = "pre_asr_label_override_v1"
VERDICT_TO_LABEL = {
    "drop": "definite_drop",
    "keep": "definite_keep",
    "unsure": "ambiguous_ignore",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(dict(row))
    return rows


def load_false_decisions(replay_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    pre_asr = payload.get("pre_asr")
    if not isinstance(pre_asr, Mapping):
        raise ValueError(f"replay JSON has no pre_asr section: {replay_path}")
    carriers = [
        (str(key), value)
        for key, value in pre_asr.items()
        if isinstance(value, Mapping) and isinstance(value.get("false_decisions"), list)
    ]
    if len(carriers) != 1:
        raise ValueError(
            "expected exactly one pre_asr checkpoint with false_decisions, got "
            f"{[key for key, _ in carriers]}"
        )
    return [dict(row) for row in carriers[0][1]["false_decisions"]]


def replay_candidate_id(row: Mapping[str, Any]) -> str:
    return f"preasr-{row['window_id']}-chunk{int(row['chunk_index']):05d}"


def replay_direction(row: Mapping[str, Any]) -> str:
    truth = str(row.get("truth") or "")
    prediction = str(row.get("prediction") or "")
    if truth == "keep" and prediction == "drop":
        return "A"
    if truth == "drop" and prediction == "keep":
        return "B"
    raise ValueError(f"unexpected truth/prediction pair: {truth}/{prediction}")


def build_overrides(
    verdict_rows: list[dict[str, Any]],
    false_decisions: list[dict[str, Any]],
    *,
    audit_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replay_by_id: dict[str, dict[str, Any]] = {}
    for row in false_decisions:
        candidate_id = replay_candidate_id(row)
        if candidate_id in replay_by_id:
            raise ValueError(f"duplicate replay candidate: {candidate_id}")
        replay_by_id[candidate_id] = row

    verdict_by_id: dict[str, dict[str, Any]] = {}
    for row in verdict_rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ValueError(f"verdict row without candidate_id: {row}")
        if candidate_id in verdict_by_id:
            raise ValueError(f"duplicate verdict candidate: {candidate_id}")
        replay_row = replay_by_id.get(candidate_id)
        if replay_row is None:
            raise ValueError(
                f"verdict candidate not in replay false_decisions: {candidate_id}"
            )
        expected_direction = replay_direction(replay_row)
        direction = str(row.get("direction") or "").strip()
        if direction != expected_direction:
            raise ValueError(
                f"verdict direction {direction!r} disagrees with replay "
                f"{expected_direction!r} for {candidate_id}"
            )
        verdict = str(row.get("verdict") or "").strip().lower()
        if verdict not in VERDICT_TO_LABEL:
            raise ValueError(f"unknown verdict {verdict!r} for {candidate_id}")
        verdict_by_id[candidate_id] = row

    def _base_row(candidate_id: str, replay_row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema": OVERRIDE_SCHEMA,
            "candidate_id": candidate_id,
            "sample_id": candidate_id,
            "window_id": str(replay_row["window_id"]),
            "video_id": str(replay_row["video_id"]),
            "chunk_index": int(replay_row["chunk_index"]),
            "start": float(replay_row["start"]),
            "end": float(replay_row["end"]),
            "duration_s": float(replay_row["duration_s"]),
            "direction": replay_direction(replay_row),
            "model_prob_drop": float(replay_row.get("prob_drop", 0.0)),
            "partition": str(replay_row.get("partition") or ""),
        }

    overrides: list[dict[str, Any]] = []
    for row in verdict_rows:
        candidate_id = str(row["candidate_id"]).strip()
        verdict = str(row["verdict"]).strip().lower()
        item = _base_row(candidate_id, replay_by_id[candidate_id])
        item.update(
            {
                "label": VERDICT_TO_LABEL[verdict],
                "label_source": f"manual_audit:{audit_id}",
                "override_source": "manual_audit",
                "override_reason": f"manual_verdict:{verdict}",
                "verdict": verdict,
                "omni_label": str(row.get("omni_label") or ""),
                "omni_confidence": row.get("omni_confidence"),
                "note": str(row.get("note") or ""),
            }
        )
        overrides.append(item)

    unaudited_a_ids = sorted(
        candidate_id
        for candidate_id, row in replay_by_id.items()
        if replay_direction(row) == "A" and candidate_id not in verdict_by_id
    )
    for candidate_id in unaudited_a_ids:
        item = _base_row(candidate_id, replay_by_id[candidate_id])
        item.update(
            {
                "label": "ambiguous_ignore",
                "label_source": f"audit_policy:{audit_id}:unaudited_a",
                "override_source": "audit_policy_unaudited_a",
                "override_reason": "omni_keep_vs_model_drop_unaudited",
            }
        )
        overrides.append(item)

    direction_counts = Counter(replay_direction(row) for row in replay_by_id.values())
    verdict_counts = Counter(
        str(row["verdict"]).strip().lower() for row in verdict_rows
    )
    unaudited_b = int(direction_counts["B"]) - sum(
        1 for row in verdict_by_id.values() if str(row.get("direction")) == "B"
    )
    summary = {
        "schema": "pre_asr_label_override_summary_v1",
        "audit_id": audit_id,
        "replay_false_decisions": len(replay_by_id),
        "direction_counts": dict(direction_counts),
        "audited": len(verdict_rows),
        "audited_verdicts": dict(verdict_counts),
        "unaudited_a_ambiguous_ignore": len(unaudited_a_ids),
        "unaudited_b_noop": unaudited_b,
        "override_rows": len(overrides),
        "override_labels": dict(Counter(row["label"] for row in overrides)),
    }
    return overrides, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pre-ASR label overrides from manual audit verdicts."
    )
    parser.add_argument(
        "--verdicts",
        default="agents/audits/20260706_113811_omni-v3-disagreement/manual_verdicts.jsonl",
    )
    parser.add_argument(
        "--replay",
        default="agents/temp/20260706_112302_v3-route-replay/checkpoint_evaluation.json",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to agents/temp/<timestamp>_pre-asr-label-overrides/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verdicts_path = Path(args.verdicts)
    if not verdicts_path.is_absolute():
        verdicts_path = PROJECT_ROOT / verdicts_path
    replay_path = Path(args.replay)
    if not replay_path.is_absolute():
        replay_path = PROJECT_ROOT / replay_path
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "agents" / "temp" / f"{stamp}_pre-asr-label-overrides"
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_id = verdicts_path.parent.name
    overrides, summary = build_overrides(
        _read_jsonl(verdicts_path),
        load_false_decisions(replay_path),
        audit_id=audit_id,
    )
    summary["verdicts"] = str(verdicts_path)
    summary["replay"] = str(replay_path)

    output_path = output_dir / "pre_asr_label_overrides.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in overrides:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["output"] = str(output_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
