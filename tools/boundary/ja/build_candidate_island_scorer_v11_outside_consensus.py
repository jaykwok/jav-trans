#!/usr/bin/env python3
"""Compile conservative Scorer v11 clear-outside evidence.

The legacy inventory is Omni/CueQC-derived and is not sufficient training
truth for Scorer.  A source is admitted as ``clear_outside`` only when both
known audio teachers return no candidate/unsure spans and the 1.7B ASR probe
returns neither text nor an error.  Every disagreement remains ``unsure`` and
is forbidden from training labels, normalization, metrics, and gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402


SCHEMA = "candidate_island_scorer_v11_outside_consensus_v1"
SUMMARY_SCHEMA = "candidate_island_scorer_v11_outside_consensus_summary_v1"
SELECTION_SCHEMA = "candidate_island_scorer_v11_outside_asr_selection_v1"
TEACHER_SCHEMA = "candidate_island_scorer_v11_omni_preaudit_v2"
INVENTORY_SCHEMA = "speech_scorer_v10_canonical_source_v1"
REQUIRED_TEACHERS = ("qwen", "gemini")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


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


def _index(rows: Iterable[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in result:
            raise ValueError(f"{name} has missing or duplicate source_id: {source_id!r}")
        result[source_id] = row
    return result


def _teacher_spec(spec: str) -> tuple[str, Path]:
    name, separator, value = spec.partition("=")
    if not separator or name not in REQUIRED_TEACHERS:
        raise ValueError("--teacher must use qwen=PATH or gemini=PATH")
    return name, Path(value).resolve()


def build(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection).resolve()
    inventory_path = Path(args.background_inventory).resolve()
    asr_path = Path(args.asr_enriched).resolve()
    teacher_paths = dict(_teacher_spec(spec) for spec in args.teacher)
    if set(teacher_paths) != set(REQUIRED_TEACHERS):
        raise ValueError("outside consensus requires exactly qwen and gemini teacher files")
    for path in (selection_path, inventory_path, asr_path, *teacher_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    selection = _index(_rows(selection_path), name="selection")
    inventory = _index(_rows(inventory_path), name="background inventory")
    asr = _index(_rows(asr_path), name="ASR evidence")
    teachers = {
        name: _index(_rows(path), name=f"{name} teacher")
        for name, path in teacher_paths.items()
    }
    output_rows: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for source_id, selected in selection.items():
        if selected.get("schema") != SELECTION_SCHEMA:
            raise ValueError(f"wrong outside selection schema: {source_id}")
        inventory_row = inventory.get(source_id)
        if inventory_row is None:
            raise ValueError(f"outside selection is absent from inventory: {source_id}")
        if inventory_row.get("schema") != INVENTORY_SCHEMA:
            raise ValueError(f"wrong background inventory schema: {source_id}")
        if inventory_row.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError(f"wrong central Boundary contract: {source_id}")
        if inventory_row.get("partition") != "train" or inventory_row.get("row_role") != "all_background":
            raise ValueError(f"outside consensus source is not a train all-background row: {source_id}")
        selected_sha = str(selected.get("audio_sha256") or "")
        if not selected_sha:
            raise ValueError(f"outside selection lacks audio SHA: {source_id}")

        evidence_reasons: list[str] = []
        teacher_evidence: dict[str, Any] = {}
        all_teachers_clear = True
        for name in REQUIRED_TEACHERS:
            teacher = teachers[name].get(source_id)
            if teacher is None:
                all_teachers_clear = False
                evidence_reasons.append(f"{name}_missing")
                continue
            if teacher.get("schema") != TEACHER_SCHEMA:
                raise ValueError(f"wrong {name} teacher schema: {source_id}")
            if str(teacher.get("audio_sha256") or "") != selected_sha:
                raise ValueError(f"{name} teacher audio SHA mismatch: {source_id}")
            islands = list(teacher.get("islands") or ())
            unsure_spans = list(teacher.get("unsure_spans") or ())
            clear = not islands and not unsure_spans
            all_teachers_clear = all_teachers_clear and clear
            if islands:
                evidence_reasons.append(f"{name}_inside")
            if unsure_spans:
                evidence_reasons.append(f"{name}_unsure")
            teacher_evidence[name] = {
                "model": str(teacher.get("model") or ""),
                "prompt_version": str(teacher.get("prompt_version") or ""),
                "island_count": len(islands),
                "unsure_span_count": len(unsure_spans),
                "clear_outside_vote": clear,
            }

        asr_row = asr.get(source_id)
        if asr_row is None:
            asr_clear = False
            asr_summary: dict[str, Any] = {}
            evidence_reasons.append("asr_missing")
        else:
            if str(asr_row.get("audio_sha256") or "") != selected_sha:
                raise ValueError(f"ASR evidence audio SHA mismatch: {source_id}")
            asr_summary = dict(asr_row.get("asr_probe_summary") or {})
            nonempty = int(asr_summary.get("nonempty_text_span_count") or 0)
            errors = int(asr_summary.get("error_span_count") or 0)
            span_count = int(asr_summary.get("span_count") or 0)
            asr_clear = span_count > 0 and nonempty == 0 and errors == 0
            if span_count <= 0:
                evidence_reasons.append("asr_no_span")
            if nonempty:
                evidence_reasons.append("asr_text")
            if errors:
                evidence_reasons.append("asr_error")

        clear_outside = all_teachers_clear and asr_clear
        decision = "clear_outside" if clear_outside else "unsure"
        decisions[decision] += 1
        if not evidence_reasons:
            evidence_reasons.append("three_way_clear_outside_consensus")
        reasons.update(evidence_reasons)
        output_rows.append(
            {
                "schema": SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "partition": "train",
                "audio": str(selected["audio"]),
                "audio_sha256": selected_sha,
                "duration_s": float(selected["duration_s"]),
                "decision": decision,
                "canonical_label": "outside_candidate" if clear_outside else "unsure",
                "training_label": 0 if clear_outside else -100,
                "training_manifest_allowed": clear_outside,
                "teacher_evidence": teacher_evidence,
                "asr_evidence": {
                    "model_id": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
                    "span_count": int(asr_summary.get("span_count") or 0),
                    "nonempty_text_span_count": int(asr_summary.get("nonempty_text_span_count") or 0),
                    "error_span_count": int(asr_summary.get("error_span_count") or 0),
                    "texts_in_workflow_order": list(asr_summary.get("texts_in_workflow_order") or ()),
                    "clear_outside_vote": asr_clear,
                },
                "decision_reasons": evidence_reasons,
                "source_inventory_provenance": {
                    "background_type": str(inventory_row.get("background_type") or ""),
                    "omni_flags": list(inventory_row.get("omni_flags") or ()),
                    "omni_only_truth_forbidden": True,
                },
            }
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "outside_consensus.jsonl"
    _write_jsonl(manifest_path, output_rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "selection": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "background_inventory": str(inventory_path),
        "background_inventory_sha256": _sha256(inventory_path),
        "asr_enriched": str(asr_path),
        "asr_enriched_sha256": _sha256(asr_path),
        "teacher_files": {name: str(path) for name, path in teacher_paths.items()},
        "teacher_files_sha256": {name: _sha256(path) for name, path in teacher_paths.items()},
        "source_count": len(output_rows),
        "decision_counts": dict(sorted(decisions.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "outside_consensus": str(manifest_path),
        "outside_consensus_sha256": _sha256(manifest_path),
        "unsure_training_label": -100,
        "omni_only_truth_allowed": False,
        "training_manifest_allowed": decisions["clear_outside"] > 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--background-inventory", required=True)
    parser.add_argument("--asr-enriched", required=True)
    parser.add_argument("--teacher", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
