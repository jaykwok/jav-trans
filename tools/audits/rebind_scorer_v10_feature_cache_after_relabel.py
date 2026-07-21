#!/usr/bin/env python3
"""Rebind an unchanged signed Scorer cache to a label-only canonical revision."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    SPEECH_ISLAND_SCORER_V10_FEATURE_CACHE_GATE_SCHEMA,
    SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
    SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA,
)
from pipeline.memory_safety import runtime_memory_snapshot  # noqa: E402
from tools.boundary.ja.apply_speech_island_scorer_v10_repair_event_unsure import (  # noqa: E402
    SUMMARY_SCHEMA as RELABEL_SUMMARY_SCHEMA,
)


SUMMARY_SCHEMA = "speech_scorer_v10_label_only_feature_cache_rebind_summary_v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _index(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        identity = str(row.get(key) or "")
        if not identity or identity in result:
            raise ValueError(f"label-only cache rebind requires unique {key}")
        result[identity] = row
    return result


def _validate_memory(snapshot: Mapping[str, Any]) -> None:
    if float(snapshot.get("physical_ram_used_mb") or 0.0) > float(
        snapshot.get("physical_ram_budget_mb") or 0.0
    ):
        raise MemoryError("label-only cache rebind exceeded the 0.95 RAM budget")


def rebind(
    *, relabel_summary_path: Path, base_feature_gate_path: Path, output_dir: Path
) -> dict[str, Any]:
    memory_before = runtime_memory_snapshot(require_shared_vram=False)
    _validate_memory(memory_before)
    relabel = _json(relabel_summary_path)
    if relabel.get("schema") != RELABEL_SUMMARY_SCHEMA:
        raise ValueError("cache rebind requires a repair-event unsure summary")
    if relabel.get("boundary_serialization_contract_id") != ACOUSTIC_BINARY_V12_CONTRACT.contract_id:
        raise ValueError("label-only canonical uses another Boundary contract")
    if (
        relabel.get("audio_bytes_changed") is not False
        or relabel.get("source_identity_changed") is not False
        or relabel.get("partition_identity_changed") is not False
    ):
        raise ValueError("cache rebind only accepts label-only canonical changes")
    canonical_path = _resolve(str(relabel.get("canonical_sources") or ""))
    audio_manifest_path = _resolve(str(relabel.get("audio_manifest") or ""))
    labels_path = _resolve(str(relabel.get("feature_cache_labels") or ""))
    for path, field in (
        (canonical_path, "canonical_sources_sha256"),
        (audio_manifest_path, "audio_manifest_sha256"),
        (labels_path, "feature_cache_labels_sha256"),
    ):
        if _sha256(path) != str(relabel.get(field) or ""):
            raise ValueError(f"label-only canonical {field} mismatch")
    sources = _index(_rows(canonical_path), "source_id")
    audio_manifest = _index(_json(audio_manifest_path), "audio_id")
    if set(sources) != set(audio_manifest):
        raise ValueError("label-only canonical/audio identities differ")

    base = _json(base_feature_gate_path)
    if base.get("schema") != SPEECH_ISLAND_SCORER_V10_FEATURE_CACHE_GATE_SCHEMA:
        raise ValueError("base feature cache gate schema changed")
    if base.get("boundary_serialization_contract_id") != ACOUSTIC_BINARY_V12_CONTRACT.contract_id:
        raise ValueError("base feature cache uses another Boundary contract")
    if base.get("feature_extractor_schema") != SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA:
        raise ValueError("base feature extractor schema changed")
    signed_path = _resolve(str(base.get("signed_feature_manifest") or ""))
    if _sha256(signed_path) != str(base.get("signed_feature_manifest_sha256") or ""):
        raise ValueError("base signed feature manifest SHA256 mismatch")
    signed = _index(_rows(signed_path), "source_id")
    if set(signed) != set(sources):
        raise ValueError("label-only canonical/signed feature identities differ")
    feature_config_sha256 = str(base.get("feature_config_sha256") or "")
    for source_id, source in sources.items():
        feature = signed[source_id]
        if feature.get("schema") != SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA:
            raise ValueError("base signed feature manifest contains a legacy row")
        if feature.get("feature_config_sha256") != feature_config_sha256:
            raise ValueError("base signed feature manifest mixes configurations")
        if _resolve(str(feature.get("audio_path") or "")) != _resolve(str(source["audio"])):
            raise ValueError(f"label-only canonical changed audio path: {source_id}")
        if int(feature.get("audio_sample_count") or 0) != int(source["sample_count"]):
            raise ValueError(f"label-only canonical changed sample count: {source_id}")
        if int(feature.get("audio_sample_rate") or 0) != int(source["sample_rate"]):
            raise ValueError(f"label-only canonical changed sample rate: {source_id}")

    gate = {
        "schema": SPEECH_ISLAND_SCORER_V10_FEATURE_CACHE_GATE_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": str(relabel["canonical_label_schema"]),
        "feature_extractor_schema": SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
        "relabel_summary": _display(relabel_summary_path),
        "relabel_summary_sha256": _sha256(relabel_summary_path),
        "canonical_sources": _display(canonical_path),
        "canonical_sources_sha256": _sha256(canonical_path),
        "audio_manifest": _display(audio_manifest_path),
        "audio_manifest_sha256": _sha256(audio_manifest_path),
        "feature_cache_labels": _display(labels_path),
        "feature_cache_labels_sha256": _sha256(labels_path),
        "base_feature_gate": _display(base_feature_gate_path),
        "base_feature_gate_sha256": _sha256(base_feature_gate_path),
        "signed_feature_manifest": _display(signed_path),
        "signed_feature_manifest_sha256": _sha256(signed_path),
        "feature_config": dict(base.get("feature_config") or {}),
        "feature_config_sha256": feature_config_sha256,
        "audio_content_signature": str(base.get("audio_content_signature") or ""),
        "feature_content_signature": str(base.get("feature_content_signature") or ""),
        "cache_binding_signature": str(base.get("cache_binding_signature") or ""),
        "source_count": len(sources),
        "label_only_changed_source_count": int(relabel["changed_source_count"]),
        "reused_signed_source_count": len(sources),
        "cache_reuse_basis": "unchanged_audio_identity_plus_prior_signed_gate_v1",
        "feature_cache_reuse_allowed": True,
        "training_manifest_allowed": True,
        "checkpoint_promotion_authorized": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = output_dir / "feature_cache_gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    memory_after = runtime_memory_snapshot(require_shared_vram=False)
    _validate_memory(memory_after)
    result = {
        **gate,
        "schema": SUMMARY_SCHEMA,
        "feature_cache_gate": _display(gate_path),
        "feature_cache_gate_sha256": _sha256(gate_path),
        "memory_before": memory_before,
        "memory_after": memory_after,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relabel-summary", required=True)
    parser.add_argument("--base-feature-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            rebind(
                relabel_summary_path=Path(args.relabel_summary),
                base_feature_gate_path=Path(args.base_feature_gate),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
