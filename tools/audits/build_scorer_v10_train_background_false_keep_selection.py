#!/usr/bin/env python3
"""Select every train all-background false keep for manual speech discovery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCORE_SUMMARY_SCHEMA = "speech_scorer_v10_checkpoint_audit_summary_v2"
SUMMARY_SCHEMA = "speech_scorer_v10_train_background_false_keep_selection_v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_selection(
    *, score_summary_path: Path, predictions_path: Path, output_dir: Path
) -> dict[str, Any]:
    score = _json(score_summary_path)
    if score.get("schema") != SCORE_SUMMARY_SCHEMA:
        raise ValueError("train background selection requires a checkpoint audit summary")
    configured = Path(str(score.get("predictions") or ""))
    if configured.resolve() != predictions_path.resolve():
        raise ValueError("checkpoint summary references another prediction manifest")
    rows = _rows(predictions_path)
    selected = [
        row
        for row in rows
        if row.get("partition") == "train"
        and row.get("row_role") == "all_background"
        and row.get("category") == "background_false_keep"
        and int(row.get("false_positive_frames") or 0) > 0
    ]
    if not selected:
        raise ValueError("checkpoint has no train all-background false keeps")
    for row in selected:
        if int(row.get("truth_speech_frames") or 0) != 0:
            raise ValueError("train background selection contains canonical speech")
        if int(row.get("false_negative_frames") or 0) != 0:
            raise ValueError("train background selection contains false-negative frames")
    selected.sort(
        key=lambda row: (
            -int(row["false_positive_frames"]),
            -float(row.get("max_predicted_speech_run_s") or 0.0),
            str(row["source_id"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selection.jsonl"
    _write_rows(selection_path, selected)
    result = {
        "schema": SUMMARY_SCHEMA,
        "score_summary": str(score_summary_path),
        "score_summary_sha256": _sha256(score_summary_path),
        "predictions": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "checkpoint": str(score.get("checkpoint") or ""),
        "selection": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "selection_count": len(selected),
        "selection_contract": "all_train_all_background_argmax_false_keeps_v1",
        "duration_or_probability_filter_applied": False,
        "diagnostic_only": True,
        "training_manifest_allowed": False,
        "manual_gate_status": "pending",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-summary", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build_selection(
                score_summary_path=Path(args.score_summary),
                predictions_path=Path(args.predictions),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
