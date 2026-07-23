#!/usr/bin/env python3
"""Reuse unchanged raw PTM2048 features and isolate sources needing extraction."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA,
)
from tools.boundary.ja.compile_candidate_island_scorer_v11_features import (  # noqa: E402
    _validate_raw_feature,
)


SUMMARY_SCHEMA = "candidate_island_scorer_v11_raw_feature_rebind_summary_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _index(
    rows: Sequence[dict[str, Any]], *, name: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in result:
            raise ValueError(f"{name} requires unique non-empty source_id: {source_id!r}")
        result[source_id] = row
    return result


def _rebound_row(
    row: dict[str, Any],
    *,
    canonical_sha: str,
    origin_manifest: Path,
    origin_manifest_sha: str,
) -> dict[str, Any]:
    reused_from_audit_preextract = bool(row.get("audit_preextract_only"))
    return {
        **row,
        "canonical_sources_sha256": canonical_sha,
        "feature_source_manifest_schema": (
            CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA
        ),
        "feature_source_manifest_kind": "canonical",
        "audit_preextract_only": False,
        "reused_from_audit_preextract": reused_from_audit_preextract,
        "reused_feature_bytes_unchanged": True,
        "reused_from_raw_feature_manifest": _display(origin_manifest),
        "reused_from_raw_feature_manifest_sha256": origin_manifest_sha,
    }


def rebind_raw_features(
    *,
    canonical_sources: Path,
    prior_raw_feature_manifest: Path,
    output_dir: Path,
    new_raw_feature_manifest: Path | None = None,
) -> dict[str, Any]:
    canonical_sources = canonical_sources.resolve()
    prior_raw_feature_manifest = prior_raw_feature_manifest.resolve()
    if new_raw_feature_manifest is not None:
        new_raw_feature_manifest = new_raw_feature_manifest.resolve()
    for path in (
        canonical_sources,
        prior_raw_feature_manifest,
        *((new_raw_feature_manifest,) if new_raw_feature_manifest is not None else ()),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    canonical = _index(_rows(canonical_sources), name="canonical")
    prior = _index(_rows(prior_raw_feature_manifest), name="prior raw feature")
    canonical_sha = _sha256(canonical_sources)
    prior_sha = _sha256(prior_raw_feature_manifest)
    for source_id, row in canonical.items():
        if row.get("schema") != CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA:
            raise ValueError(f"wrong Scorer v11 canonical schema: {source_id}")
        if row.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError(f"wrong central Boundary contract: {source_id}")

    reusable: dict[str, dict[str, Any]] = {}
    for source_id in sorted(set(canonical) & set(prior)):
        row = prior[source_id]
        _validate_raw_feature(row, canonical=canonical[source_id])
        reusable[source_id] = _rebound_row(
            row,
            canonical_sha=canonical_sha,
            origin_manifest=prior_raw_feature_manifest,
            origin_manifest_sha=prior_sha,
        )
    missing_ids = sorted(set(canonical) - set(reusable))
    missing_rows = [canonical[source_id] for source_id in missing_ids]

    new_rows: dict[str, dict[str, Any]] = {}
    if new_raw_feature_manifest is not None:
        new_rows = _index(_rows(new_raw_feature_manifest), name="new raw feature")
        if set(new_rows) != set(missing_ids):
            raise ValueError(
                "new raw feature manifest must cover exactly the missing canonical sources: "
                f"missing={sorted(set(missing_ids)-set(new_rows))[:8]} "
                f"extra={sorted(set(new_rows)-set(missing_ids))[:8]}"
            )
        new_sha = _sha256(new_raw_feature_manifest)
        for source_id, row in new_rows.items():
            _validate_raw_feature(row, canonical=canonical[source_id])
            new_rows[source_id] = _rebound_row(
                row,
                canonical_sha=canonical_sha,
                origin_manifest=new_raw_feature_manifest,
                origin_manifest_sha=new_sha,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    reusable_path = output_dir / "reusable_raw_feature_manifest.jsonl"
    missing_path = output_dir / "missing_canonical_sources.jsonl"
    _write(reusable_path, [reusable[source_id] for source_id in sorted(reusable)])
    _write(missing_path, missing_rows)
    final_rows = {**reusable, **new_rows}
    complete = set(final_rows) == set(canonical)
    final_path = output_dir / "raw_feature_manifest.jsonl"
    if complete:
        _write(final_path, [final_rows[source_id] for source_id in sorted(final_rows)])
    elif final_path.exists():
        raise FileExistsError(
            "incomplete rebind refuses to leave a stale final raw feature manifest"
        )

    partition_counts = Counter(
        str(canonical[source_id].get("partition") or "") for source_id in missing_ids
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_sources": _display(canonical_sources),
        "canonical_sources_sha256": canonical_sha,
        "prior_raw_feature_manifest": _display(prior_raw_feature_manifest),
        "prior_raw_feature_manifest_sha256": prior_sha,
        "new_raw_feature_manifest": (
            _display(new_raw_feature_manifest)
            if new_raw_feature_manifest is not None
            else None
        ),
        "new_raw_feature_manifest_sha256": (
            _sha256(new_raw_feature_manifest)
            if new_raw_feature_manifest is not None
            else None
        ),
        "canonical_source_count": len(canonical),
        "reused_source_count": len(reusable),
        "missing_source_count": len(missing_ids),
        "missing_partition_counts": dict(sorted(partition_counts.items())),
        "missing_source_ids": missing_ids,
        "reusable_raw_feature_manifest": _display(reusable_path),
        "reusable_raw_feature_manifest_sha256": _sha256(reusable_path),
        "missing_canonical_sources": _display(missing_path),
        "missing_canonical_sources_sha256": _sha256(missing_path),
        "raw_feature_manifest": _display(final_path) if complete else None,
        "raw_feature_manifest_sha256": _sha256(final_path) if complete else None,
        "feature_bytes_recomputed_for_reused_sources": False,
        "complete": complete,
        "training_manifest_allowed": complete,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-sources", required=True)
    parser.add_argument("--prior-raw-feature-manifest", required=True)
    parser.add_argument("--new-raw-feature-manifest")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return rebind_raw_features(
        canonical_sources=Path(args.canonical_sources),
        prior_raw_feature_manifest=Path(args.prior_raw_feature_manifest),
        new_raw_feature_manifest=(
            Path(args.new_raw_feature_manifest) if args.new_raw_feature_manifest else None
        ),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
