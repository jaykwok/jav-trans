#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import current_qwen_asr_backend, qwen_asr_repo_id  # noqa: E402
from asr.pre_asr_cueqc import (  # noqa: E402
    PRE_ASR_CUEQC_FEATURE_NAMES,
    PRE_ASR_CUEQC_FEATURE_SCHEMA,
    PRE_ASR_CUEQC_IGNORE_LABEL,
    PRE_ASR_CUEQC_MODEL_PTM_TOKENS,
    PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES,
    PRE_ASR_CUEQC_PTM_BINS,
    PRE_ASR_CUEQC_PTM_DIM,
    PRE_ASR_CUEQC_RUNTIME_ADAPTER,
    PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES,
    candidate_from_span,
    ptm_bin_matrix,
    scalar_vector,
)


FEATURE_BUNDLE_SCHEMA = "cueqc_pre_asr_semantic_chunk_v11_features"


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_display_path(path: str | Path | None) -> str:
    if not path:
        return ""
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(PROJECT_ROOT)).replace("/", "\\")
    except ValueError:
        return str(raw)


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"JSON payload must be a list: {path}")
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(dict(row))
    return rows


def extract_chunks(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("pre_asr_candidates", "processing_spans", "transcript_chunks", "chunks", "chunk_infos"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, Mapping)]
    details = payload.get("details")
    if isinstance(details, Mapping):
        return extract_chunks(details)
    return []


def infer_audio_id(path: Path, payload: Mapping[str, Any] | None = None) -> str:
    if payload is not None:
        for key in ("video_id", "audio_id"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
    name = path.name
    for suffix in (
        ".pre_asr_candidates.json",
        ".transcript.json",
        ".timings.json",
        ".aligned_segments.json",
        ".jsonl",
        ".json",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def read_chunk_document(path: Path) -> tuple[str, list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8-sig")
    payload: Any
    if text.lstrip().startswith("["):
        payload = json.loads(text)
    elif text.lstrip().startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = read_json_or_jsonl(path)
    else:
        payload = read_json_or_jsonl(path)
    chunks = extract_chunks(payload)
    audio_id = infer_audio_id(path, payload if isinstance(payload, Mapping) else None)
    if chunks:
        embedded_audio_id = str(
            chunks[0].get("audio_id") or chunks[0].get("video_id") or ""
        ).strip()
        if embedded_audio_id:
            audio_id = embedded_audio_id
    return audio_id, chunks


def label_keys(row: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in ("sample_id", "candidate_id", "id"):
        value = str(row.get(key) or "").strip()
        if value:
            keys.append(value)
            match = re.match(r"^cueqc-(.+)-chunk(\d+)$", value)
            if match:
                keys.append(f"{match.group(1)}#{int(match.group(2))}")
            match = re.match(r"^preasr-(.+)-chunk(\d+)$", value)
            if match:
                keys.append(f"{match.group(1)}#{int(match.group(2))}")
    audio_id = str(row.get("audio_id") or row.get("video_id") or "").strip()
    chunk_index = row.get("chunk_index", row.get("index", ""))
    if audio_id != "" and str(chunk_index).strip() != "":
        try:
            keys.append(f"{audio_id}#{int(chunk_index)}")
        except (TypeError, ValueError):
            keys.append(f"{audio_id}#{chunk_index}")
    return list(dict.fromkeys(key for key in keys if key))


def label_items(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = dict(row)
    cluster_id = str(row.get("cluster_id") or "").strip()
    base.setdefault("cluster_label_source", cluster_id)
    base.setdefault("label_source", cluster_id or str(row.get("label_source") or row.get("source") or ""))
    items: list[dict[str, Any]] = []

    examples = row.get("examples")
    if isinstance(examples, list) and examples:
        for example in examples:
            if isinstance(example, Mapping):
                items.append({**base, **dict(example)})

    for field in ("sample_ids", "samples", "candidate_ids", "ids"):
        values = row.get(field)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, Mapping):
                items.append({**base, **dict(value)})
            else:
                key = "candidate_id" if field == "candidate_ids" else "sample_id"
                items.append({**base, key: str(value)})

    if not items:
        items.append(base)
    return items


def normalize_label(row: Mapping[str, Any]) -> int | None:
    raw = str(
        row.get("label")
        or row.get("route")
        or row.get("decision")
        or row.get("display_decision")
        or ""
    ).strip().lower()
    if raw in {"keep", "keep_for_asr", "1", "positive", "definite_keep"}:
        return 1
    if raw in {"drop", "drop_before_asr", "0", "negative", "definite_drop"}:
        return 0
    if raw in {"ignore", "skip", "ambiguous", "ambiguous_ignore", str(PRE_ASR_CUEQC_IGNORE_LABEL)}:
        return PRE_ASR_CUEQC_IGNORE_LABEL
    return None


def read_labels(paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = project_path(raw_path)
        for row in read_json_or_jsonl(path):
            value = normalize_label(row)
            if value is None:
                continue
            if row.get("training_label_included") is False:
                value = PRE_ASR_CUEQC_IGNORE_LABEL
            for source_item in label_items(row):
                item = dict(source_item)
                item["label_index"] = value
                cluster_id = str(item.get("cluster_id") or "").strip()
                item_keys = label_keys(item)
                if cluster_id and not item_keys:
                    labels[f"cluster:{cluster_id}"] = item
                for key in item_keys:
                    labels[key] = item
    return labels


def row_id(audio_id: str, chunk: Mapping[str, Any], index: int) -> str:
    explicit = str(chunk.get("sample_id") or chunk.get("candidate_id") or chunk.get("id") or "").strip()
    if explicit:
        return explicit
    chunk_index = chunk.get("chunk_index", chunk.get("index", index))
    return f"{audio_id}#{chunk_index}"


def label_for_chunk(
    labels: Mapping[str, dict[str, Any]],
    *,
    audio_id: str,
    chunk: Mapping[str, Any],
    index: int,
) -> dict[str, Any] | None:
    keys = [row_id(audio_id, chunk, index)]
    keys.extend(label_keys(chunk))
    keys.append(f"{audio_id}#{chunk.get('chunk_index', chunk.get('index', index))}")
    cluster_id = str(chunk.get("cluster_id") or "").strip()
    if cluster_id:
        keys.append(f"cluster:{cluster_id}")
    for key in dict.fromkeys(key for key in keys if key):
        label = labels.get(key)
        if label is not None:
            return label
    return None


def _has_required_ptm_pooling(candidate: Mapping[str, Any]) -> bool:
    values = candidate.get("pre_asr_ptm_pooled_features")
    return (
        bool(candidate.get("ptm_pooling_available"))
        and isinstance(values, list)
        and len(values) == len(PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES)
    )


def candidate_for_chunk(chunks: list[dict[str, Any]], index: int) -> dict[str, Any]:
    chunk = chunks[index]
    features = chunk.get("features")
    feature_names = tuple(str(item) for item in chunk.get("feature_names") or ())
    if isinstance(features, Mapping) and feature_names == PRE_ASR_CUEQC_FEATURE_NAMES:
        candidate = dict(chunk)
    else:
        candidate = candidate_from_span(chunks, index, require_ptm_pooling=True)
    if not _has_required_ptm_pooling(candidate):
        raise ValueError(
            "Pre-ASR CueQC v11 feature compilation requires chunk-level pooled PTM features"
        )
    return candidate


def _group_key(source: str, audio_id: str, candidate: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        source,
        str(candidate.get("audio_id") or candidate.get("video_id") or audio_id),
        str(candidate.get("planned_island_id") or "sequence"),
    )


def _make_tensor_bundle(rows: list[dict[str, Any]], groups: list[list[int]]) -> dict[str, Any]:
    import torch

    group_count = len(groups)
    max_chunks = max((len(group) for group in groups), default=0)
    scalar = np.zeros(
        (group_count, max_chunks, len(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES)),
        dtype=np.float32,
    )
    ptm_bins = np.zeros(
        (group_count, max_chunks, PRE_ASR_CUEQC_MODEL_PTM_TOKENS, PRE_ASR_CUEQC_PTM_DIM),
        dtype=np.float32,
    )
    bin_mask = np.zeros((group_count, max_chunks, PRE_ASR_CUEQC_MODEL_PTM_TOKENS), dtype=np.float32)
    chunk_mask = np.zeros((group_count, max_chunks), dtype=np.float32)
    labels = np.full((group_count, max_chunks), PRE_ASR_CUEQC_IGNORE_LABEL, dtype=np.int64)
    for group_index, row_indexes in enumerate(groups):
        for chunk_position, row_index in enumerate(row_indexes):
            row = rows[row_index]
            candidate = row["candidate"]
            scalar[group_index, chunk_position] = scalar_vector(candidate)
            bins, mask = ptm_bin_matrix(candidate)
            ptm_bins[group_index, chunk_position] = bins
            bin_mask[group_index, chunk_position] = mask
            chunk_mask[group_index, chunk_position] = 1.0
            labels[group_index, chunk_position] = int(row["label_index"])
    return {
        "scalar_features": torch.from_numpy(scalar),
        "ptm_bins": torch.from_numpy(ptm_bins),
        "bin_mask": torch.from_numpy(bin_mask),
        "chunk_mask": torch.from_numpy(chunk_mask),
        "labels": torch.from_numpy(labels),
    }


def compile_features(
    *,
    chunk_paths: list[str],
    label_paths: list[str],
    output: Path,
    asr_repo_id: str,
) -> dict[str, Any]:
    import torch

    labels = read_labels(label_paths)
    rows: list[dict[str, Any]] = []
    group_map: dict[tuple[str, str, str], list[int]] = {}
    for raw_path in chunk_paths:
        path = project_path(raw_path)
        source = repo_display_path(path)
        audio_id, chunks = read_chunk_document(path)
        for index, chunk in enumerate(chunks):
            candidate = candidate_for_chunk(chunks, index)
            rid = row_id(audio_id, chunk, index)
            label = label_for_chunk(labels, audio_id=audio_id, chunk=chunk, index=index)
            label_index = (
                PRE_ASR_CUEQC_IGNORE_LABEL if label is None else int(label["label_index"])
            )
            group_key = _group_key(source, audio_id, candidate)
            row_index = len(rows)
            group_map.setdefault(group_key, []).append(row_index)
            rows.append(
                {
                    "id": rid,
                    "source": source,
                    "audio_id": audio_id,
                    "planned_island_id": str(candidate.get("planned_island_id") or "sequence"),
                    "chunk_index": int(candidate["index"]),
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "label_index": label_index,
                    "label": (
                        "keep_for_asr"
                        if label_index == 1
                        else "drop_before_asr"
                        if label_index == 0
                        else "ambiguous_ignore"
                    ),
                    "label_source": ""
                    if label is None
                    else str(
                        label.get("label_source")
                        or label.get("cluster_label_source")
                        or label.get("source")
                        or ""
                    ),
                    "candidate": candidate,
                }
            )
    groups: list[list[int]] = []
    for row_indexes in group_map.values():
        if any(int(rows[row_index]["label_index"]) in (0, 1) for row_index in row_indexes):
            groups.append(row_indexes)
    if not groups:
        raise ValueError("no definite labeled Pre-ASR CueQC examples were compiled")
    bundle_tensors = _make_tensor_bundle(rows, groups)
    y = bundle_tensors["labels"].numpy()
    row_payload = [
        {key: value for key, value in row.items() if key != "candidate"}
        for group in groups
        for row in (rows[row_index] for row_index in group)
    ]
    group_payload = [
        {
            "group_index": group_index,
            "row_ids": [rows[row_index]["id"] for row_index in group],
            "audio_id": rows[group[0]]["audio_id"],
            "planned_island_id": rows[group[0]]["planned_island_id"],
        }
        for group_index, group in enumerate(groups)
    ]
    bundle = {
        "schema": FEATURE_BUNDLE_SCHEMA,
        "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
        "feature_names": list(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),
        "all_feature_names": list(PRE_ASR_CUEQC_FEATURE_NAMES),
        "ptm_bin_count": PRE_ASR_CUEQC_MODEL_PTM_TOKENS,
        "ptm_dim": PRE_ASR_CUEQC_PTM_DIM,
        "asr_repo_id": qwen_asr_repo_id(asr_repo_id),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": row_payload,
        "groups": group_payload,
        "source_files": [repo_display_path(project_path(path)) for path in chunk_paths],
        "label_files": [repo_display_path(project_path(path)) for path in label_paths],
        **bundle_tensors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    summary = {
        "schema": "cueqc_pre_asr_semantic_chunk_v11_feature_summary",
        "feature_bundle": repo_display_path(output),
        "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
        "feature_names": list(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),
        "asr_repo_id": qwen_asr_repo_id(asr_repo_id),
        "group_count": int(len(groups)),
        "chunk_count": int(np.sum(y != PRE_ASR_CUEQC_IGNORE_LABEL)),
        "keep": int(np.sum(y == 1)),
        "drop": int(np.sum(y == 0)),
        "ambiguous_ignore": int(np.sum((y == PRE_ASR_CUEQC_IGNORE_LABEL) & (bundle_tensors["chunk_mask"].numpy() > 0))),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Pre-ASR CueQC v11 semantic-chunk features.")
    parser.add_argument("--chunks", action="append", required=True, help="Workflow details/chunk JSON or JSONL.")
    parser.add_argument("--labels", action="append", required=True, help="JSON/JSONL labels with keep/drop/ignore.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--asr-repo-id", default=current_qwen_asr_backend())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = compile_features(
        chunk_paths=list(args.chunks),
        label_paths=list(args.labels),
        output=project_path(args.output),
        asr_repo_id=str(args.asr_repo_id),
    )
    print(
        "features={feature_bundle} groups={group_count} keep={keep} drop={drop} ignore={ignore}".format(
            feature_bundle=summary["feature_bundle"],
            group_count=summary["group_count"],
            keep=summary["keep"],
            drop=summary["drop"],
            ignore=summary["ambiguous_ignore"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
