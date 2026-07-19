#!/usr/bin/env python3
"""Compile the complete Scorer v10 corrected-canonical evidence chain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


INITIAL_GATE_SCHEMA = "speech_scorer_v10_canonical_manual_gate_v1"
SPAN_GATE_SCHEMA = "speech_scorer_v10_canonical_span_repair_gate_v1"
CORRECTED_SUMMARY_SCHEMA = "speech_scorer_v10_corrected_canonical_summary_v1"
RESULT_SCHEMA = "speech_scorer_v10_corrected_canonical_manual_gate_v1"
BOUNDARY_CONTRACT_ID = "boundary_acoustic_binary_v12"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonl_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [
            str(json.loads(line)["source_id"])
            for line in handle
            if line.strip()
        ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_gate(
    path: Path, *, schema: str, canonical_sha256: str, complete: bool = True
) -> dict[str, Any]:
    gate = _json(path)
    if gate.get("schema") != schema:
        raise ValueError(f"unexpected Scorer v10 gate schema: {path}")
    if complete and gate.get("complete") is not True:
        raise ValueError(f"Scorer v10 gate is incomplete: {path}")
    if gate.get("canonical_sources_sha256") != canonical_sha256:
        raise ValueError(f"Scorer v10 gate is bound to another canonical manifest: {path}")
    return gate


def compile_gate(
    *,
    initial_gate: Path,
    span_repair_gate: Path,
    corrected_r1_summary: Path,
    replacement7_gate: Path,
    corrected_r2_summary: Path,
    replacement2_gate: Path,
    output: Path,
) -> dict[str, Any]:
    r1 = _json(corrected_r1_summary)
    r2 = _json(corrected_r2_summary)
    if (
        r1.get("schema") != CORRECTED_SUMMARY_SCHEMA
        or r2.get("schema") != CORRECTED_SUMMARY_SCHEMA
    ):
        raise ValueError("invalid corrected-canonical summary schema")
    if any(
        summary.get("boundary_serialization_contract_id") != BOUNDARY_CONTRACT_ID
        for summary in (r1, r2)
    ):
        raise ValueError("corrected canonical summary uses another boundary contract")

    original_sha = str(r1.get("input_canonical_sources_sha256") or "")
    initial = _require_gate(
        initial_gate, schema=INITIAL_GATE_SCHEMA, canonical_sha256=original_sha
    )
    span = _require_gate(
        span_repair_gate, schema=SPAN_GATE_SCHEMA, canonical_sha256=original_sha
    )
    if (
        initial.get("manual_gate_pass") is not False
        or initial.get("training_manifest_allowed") is not False
        or int(initial.get("risk_count") or 0) <= 0
    ):
        raise ValueError("initial fixed24 evidence must retain its failed risk findings")
    if span.get("canonical_recompile_ready") is not True:
        raise ValueError("span-repair evidence is not ready for canonical recompilation")
    if r1.get("repair_gate_sha256") != _sha256(span_repair_gate):
        raise ValueError("corrected r1 is not bound to the span-repair gate")

    r1_canonical = Path(str(r1["canonical_sources"]))
    r1_sha = _sha256(r1_canonical)
    replacement7 = _require_gate(
        replacement7_gate,
        schema=INITIAL_GATE_SCHEMA,
        canonical_sha256=r1_sha,
    )
    if (
        replacement7.get("manual_gate_pass") is not False
        or replacement7.get("canonical_recompile_ready") is not True
        or replacement7.get("training_manifest_allowed") is not False
        or int(replacement7.get("risk_count") or 0) <= 0
        or not replacement7.get("quarantined_background_ids")
    ):
        raise ValueError("replacement7 must retain its source-quarantine finding")
    replacement7_ids = _jsonl_ids(Path(str(replacement7["audit_manifest"])))
    if (
        int(replacement7.get("target_count") or 0) != len(replacement7_ids)
        or int(replacement7.get("verdict_count") or 0) != len(replacement7_ids)
    ):
        raise ValueError("replacement7 target/verdict counts do not match its manifest")
    if replacement7_ids != list(r1.get("replacement_audit_source_ids") or ()):
        raise ValueError("replacement7 targets do not match corrected r1")

    if r2.get("input_canonical_sources_sha256") != r1_sha:
        raise ValueError("corrected r2 is not based on corrected r1")
    if r2.get("repair_gate_sha256") != _sha256(replacement7_gate):
        raise ValueError("corrected r2 is not bound to replacement7 quarantine")
    if Path(str(r2.get("previous_summary") or "")).resolve() != corrected_r1_summary.resolve():
        raise ValueError("corrected r2 does not reference corrected r1 summary")
    if not set(r1.get("quarantined_background_ids") or ()).issubset(
        set(r2.get("quarantined_background_ids") or ())
    ):
        raise ValueError("corrected r2 lost a prior background quarantine")
    if not set(r1.get("ignored_core_ids") or ()).issubset(
        set(r2.get("ignored_core_ids") or ())
    ):
        raise ValueError("corrected r2 lost a prior ignored-core identity")

    r2_canonical = Path(str(r2["canonical_sources"]))
    r2_sha = _sha256(r2_canonical)
    replacement2 = _require_gate(
        replacement2_gate,
        schema=INITIAL_GATE_SCHEMA,
        canonical_sha256=r2_sha,
    )
    if (
        replacement2.get("manual_gate_pass") is not True
        or replacement2.get("training_manifest_allowed") is not True
        or int(replacement2.get("risk_count") or 0) != 0
    ):
        raise ValueError("replacement2 manual gate did not pass")
    replacement2_ids = _jsonl_ids(Path(str(replacement2["audit_manifest"])))
    if (
        int(replacement2.get("target_count") or 0) != len(replacement2_ids)
        or int(replacement2.get("verdict_count") or 0) != len(replacement2_ids)
    ):
        raise ValueError("replacement2 target/verdict counts do not match its manifest")
    if replacement2_ids != list(r2.get("replacement_audit_source_ids") or ()):
        raise ValueError("replacement2 targets do not match corrected r2")

    evidence = {
        "initial_gate": initial_gate,
        "span_repair_gate": span_repair_gate,
        "corrected_r1_summary": corrected_r1_summary,
        "replacement7_gate": replacement7_gate,
        "corrected_r2_summary": corrected_r2_summary,
        "replacement2_gate": replacement2_gate,
    }
    result = {
        "schema": RESULT_SCHEMA,
        "boundary_serialization_contract_id": BOUNDARY_CONTRACT_ID,
        "canonical_sources": str(r2_canonical),
        "canonical_sources_sha256": r2_sha,
        "dataset": r2["dataset"],
        "canonical_frame_counts": r2["canonical_frame_counts"],
        "quarantined_background_ids": r2["quarantined_background_ids"],
        "ignored_core_ids": r2["ignored_core_ids"],
        "initial_fixed24_correct_count": int(
            (initial.get("verdict_counts") or {}).get("correct") or 0
        ),
        "span_repair_item_count": int(span.get("target_count") or 0),
        "replacement7_correct_count": int(
            (replacement7.get("verdict_counts") or {}).get("correct") or 0
        ),
        "replacement2_correct_count": int(
            (replacement2.get("verdict_counts") or {}).get("correct") or 0
        ),
        "evidence": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in evidence.items()
        },
        "complete": True,
        "manual_gate_pass": True,
        "feature_cache_allowed": True,
        "training_manifest_allowed": True,
        "promotion_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-gate", required=True)
    parser.add_argument("--span-repair-gate", required=True)
    parser.add_argument("--corrected-r1-summary", required=True)
    parser.add_argument("--replacement7-gate", required=True)
    parser.add_argument("--corrected-r2-summary", required=True)
    parser.add_argument("--replacement2-gate", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            compile_gate(
                initial_gate=Path(args.initial_gate),
                span_repair_gate=Path(args.span_repair_gate),
                corrected_r1_summary=Path(args.corrected_r1_summary),
                replacement7_gate=Path(args.replacement7_gate),
                corrected_r2_summary=Path(args.corrected_r2_summary),
                replacement2_gate=Path(args.replacement2_gate),
                output=Path(args.output),
            ),
            ensure_ascii=False,
        )
    )
