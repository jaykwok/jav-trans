#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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

from asr.backends.qwen import (  # noqa: E402
    QWEN_ASR_17B_REPO_ID,
    current_qwen_asr_backend,
    qwen_asr_repo_id,
)
from asr.pre_asr_cueqc import (  # noqa: E402
    PRE_ASR_CUEQC_FEATURE_NAMES,
    PRE_ASR_CUEQC_FEATURE_SCHEMA,
    PRE_ASR_CUEQC_IGNORE_LABEL,
    PRE_ASR_CUEQC_MODEL_PTM_TOKENS,
    PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES,
    PRE_ASR_CUEQC_PTM_DIM,
    PRE_ASR_CUEQC_RUNTIME_ADAPTER,
    PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES,
    candidate_from_span,
    ptm_bin_matrix,
    scalar_vector,
)
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402


FEATURE_BUNDLE_SCHEMA = "cueqc_pre_asr_semantic_chunk_v13_features"
RUNTIME_CHUNK_SCHEMA = "runtime_v12_provisional_subisland_v2"
CANONICAL_LABEL_SCHEMA = "cueqc_v13_canonical_label_v2"
CURRENT_INPUT_DISTRIBUTION = "runtime_v12_provisional_subisland_v2_pre_cueqc"


def _required_sha256(value: Any, *, field: str, row_id: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"CueQC chunk {row_id!r} is missing exact {field}")
    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_save_atomic(payload: Any, output: Path) -> None:
    import torch

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _training_chunk_provenance(
    chunk: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    row_id: str,
    strict_current: bool = True,
) -> tuple[str, str, list[str]]:
    if strict_current:
        if chunk.get("schema") != RUNTIME_CHUNK_SCHEMA:
            raise ValueError(
                f"CueQC chunk {row_id!r} is not a current Runtime v12 provisional row"
            )
        if chunk.get("inner_execution_status") != "deferred_until_cueqc_keep":
            raise ValueError(
                f"CueQC chunk {row_id!r} contains pre-CueQC Inner state"
            )
    if chunk.get("training_manifest_allowed") is not True:
        raise ValueError(
            f"CueQC chunk {row_id!r} is not an approved training manifest row"
        )
    contract_id = str(
        chunk.get("boundary_serialization_contract_id")
        or candidate.get("boundary_serialization_contract_id")
        or ""
    )
    if not ACOUSTIC_BINARY_V12_CONTRACT.matches(contract_id):
        raise ValueError(f"CueQC chunk {row_id!r} uses a stale Boundary contract")
    if not ACOUSTIC_BINARY_V12_CONTRACT.matches(
        candidate.get("boundary_contract_id")
    ):
        raise ValueError(
            f"CueQC chunk {row_id!r} has stale candidate boundary features"
        )
    split_sha256 = _required_sha256(
        chunk.get("semantic_split_weights_sha256")
        or candidate.get("semantic_split_weights_sha256"),
        field="semantic_split_weights_sha256",
        row_id=row_id,
    )
    inner_sha256 = _required_sha256(
        chunk.get("inner_edge_refiner_weights_sha256")
        or candidate.get("inner_edge_refiner_weights_sha256"),
        field="inner_edge_refiner_weights_sha256",
        row_id=row_id,
    )
    raw_core_ids = chunk.get("source_core_ids", candidate.get("source_core_ids"))
    if not isinstance(raw_core_ids, list):
        raise ValueError(f"CueQC chunk {row_id!r} is missing source_core_ids")
    core_ids = [str(value).strip() for value in raw_core_ids]
    if any(not value for value in core_ids) or len(core_ids) != len(set(core_ids)):
        raise ValueError(f"CueQC chunk {row_id!r} has invalid source_core_ids")
    return split_sha256, inner_sha256, core_ids


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
    if (
        ("candidate_id" in payload or "sample_id" in payload or "id" in payload)
        and "start" in payload
        and "end" in payload
    ):
        return [dict(payload)]
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


def expand_chunk_paths(paths: Iterable[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = project_path(raw_path)
        try:
            rows = read_json_or_jsonl(path)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            expanded.append(path)
            continue
        manifest_paths = [
            str(row.get("pre_asr_candidates") or "").strip()
            for row in rows
            if isinstance(row, Mapping)
            and isinstance(row.get("pre_asr_candidates"), (str, Path))
        ]
        manifest_paths = [value for value in manifest_paths if value]
        if manifest_paths and len(manifest_paths) == len(rows):
            expanded.extend(project_path(value) for value in manifest_paths)
        else:
            expanded.append(path)
    return expanded


def label_keys(row: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in ("subisland_id", "sample_id", "candidate_id", "id"):
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


def raw_label(row: Mapping[str, Any]) -> str:
    return str(
        row.get("label")
        or row.get("route")
        or row.get("decision")
        or row.get("display_decision")
        or ""
    ).strip().lower()


def normalize_label(row: Mapping[str, Any]) -> int | None:
    raw = raw_label(row)
    if raw in {"keep", "keep_for_asr", "1", "positive", "definite_keep"}:
        return 1
    if raw in {"drop", "drop_before_asr", "0", "negative", "definite_drop"}:
        return 0
    if raw in {"unsure", "unsure_for_asr", "abstain", "uncertain", "2"}:
        return PRE_ASR_CUEQC_IGNORE_LABEL
    if raw in {"ignore", "skip", "ambiguous", "ambiguous_ignore", str(PRE_ASR_CUEQC_IGNORE_LABEL)}:
        return PRE_ASR_CUEQC_IGNORE_LABEL
    return None


def read_labels(
    paths: Iterable[str], *, strict_current: bool = False
) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}

    def register(key: str, item: dict[str, Any], *, source_path: Path) -> None:
        previous = labels.get(key)
        if previous is None or not strict_current:
            labels[key] = item
            return
        comparable = (
            int(previous.get("label_index", PRE_ASR_CUEQC_IGNORE_LABEL)),
            str(previous.get("teacher_label") or ""),
            str(previous.get("source_id") or ""),
            str(previous.get("source_partition") or ""),
        )
        current = (
            int(item.get("label_index", PRE_ASR_CUEQC_IGNORE_LABEL)),
            str(item.get("teacher_label") or ""),
            str(item.get("source_id") or ""),
            str(item.get("source_partition") or ""),
        )
        if comparable != current:
            raise ValueError(
                f"conflicting CueQC labels for key {key!r} in {source_path}"
            )
        if strict_current and previous.get("subisland_id") != item.get("subisland_id"):
            raise ValueError(f"duplicate CueQC label identity for key {key!r}")

    for raw_path in paths:
        path = project_path(raw_path)
        for row in read_json_or_jsonl(path):
            if strict_current and row.get("schema") != CANONICAL_LABEL_SCHEMA:
                raise ValueError(
                    f"CueQC training labels require {CANONICAL_LABEL_SCHEMA}: {path}"
                )
            if strict_current and row.get("training_manifest_allowed") is not True:
                raise ValueError(f"CueQC training label is not manifest-approved: {path}")
            canonical_label = raw_label(row)
            teacher_label = str(row.get("teacher_label") or canonical_label).strip().lower()
            value = normalize_label(row)
            if value is None:
                continue
            ignore_reason = ""
            if canonical_label in {"unsure", "unsure_for_asr", "abstain", "uncertain", "2"}:
                ignore_reason = "teacher_unsure"
            if row.get("training_label_included") is False:
                value = PRE_ASR_CUEQC_IGNORE_LABEL
                ignore_reason = ignore_reason or "training_label_excluded"
            for source_item in label_items(row):
                item = dict(source_item)
                item["label_index"] = value
                item["teacher_label"] = teacher_label
                item["training_label_included"] = value != PRE_ASR_CUEQC_IGNORE_LABEL
                item["training_ignore_reason"] = ignore_reason
                cluster_id = str(item.get("cluster_id") or "").strip()
                item_keys = label_keys(item)
                if cluster_id and not item_keys:
                    register(f"cluster:{cluster_id}", item, source_path=path)
                for key in item_keys:
                    register(key, item, source_path=path)
    return labels


def row_id(audio_id: str, chunk: Mapping[str, Any], index: int) -> str:
    explicit = str(
        chunk.get("subisland_id")
        or chunk.get("sample_id")
        or chunk.get("candidate_id")
        or chunk.get("id")
        or ""
    ).strip()
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


def _has_required_ptm_pooling(
    candidate: Mapping[str, Any], *, strict_current: bool = True
) -> bool:
    values = candidate.get("pre_asr_ptm_pooled_features")
    valid = (
        bool(candidate.get("ptm_pooling_available"))
        and isinstance(values, list)
        and len(values) == len(PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES)
    )
    if not valid:
        return False
    try:
        numeric = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(numeric).all():
        return False
    if strict_current:
        if str(candidate.get("ptm_pooling_schema") or "").strip() == "":
            return False
        if int(candidate.get("ptm_pooling_bins") or 0) != PRE_ASR_CUEQC_MODEL_PTM_TOKENS:
            return False
        if int(candidate.get("ptm_pooling_dim") or 0) != len(
            PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES
        ):
            return False
        digest = str(candidate.get("ptm_projection_digest") or "").lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            return False
    return True


def candidate_for_chunk(
    chunks: list[dict[str, Any]], index: int, *, strict_current: bool = True
) -> dict[str, Any]:
    chunk = chunks[index]
    embedded = chunk.get("pre_asr_candidate")
    if strict_current and not isinstance(embedded, Mapping):
        raise ValueError(
            f"CueQC chunk {row_id('', chunk, index)!r} lacks an embedded current Pre-ASR candidate"
        )
    source = dict(embedded) if isinstance(embedded, Mapping) else chunk
    features = source.get("features")
    feature_names = tuple(str(item) for item in source.get("feature_names") or ())
    if strict_current and source.get("schema") != PRE_ASR_CUEQC_FEATURE_SCHEMA:
        raise ValueError("CueQC current Runtime row contains a stale Pre-ASR feature schema")
    if isinstance(features, Mapping) and feature_names == PRE_ASR_CUEQC_FEATURE_NAMES:
        candidate = dict(source)
    else:
        if strict_current:
            raise ValueError("CueQC current Runtime row has incomplete embedded features")
        candidate = candidate_from_span(chunks, index, require_ptm_pooling=True)
    candidate.setdefault("sample_id", str(chunk.get("subisland_id") or ""))
    candidate.setdefault("source_sample_id", str(chunk.get("sample_id") or ""))
    candidate.setdefault(
        "semantic_split_weights_sha256",
        str(chunk.get("semantic_split_weights_sha256") or ""),
    )
    candidate.setdefault(
        "inner_edge_refiner_weights_sha256",
        str(chunk.get("inner_edge_refiner_weights_sha256") or ""),
    )
    if not _has_required_ptm_pooling(candidate, strict_current=strict_current):
        raise ValueError(
            "Pre-ASR CueQC v13 feature compilation requires chunk-level pooled PTM features"
        )
    return candidate


def _group_key(source: str, audio_id: str, candidate: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        source,
        str(candidate.get("audio_id") or candidate.get("video_id") or audio_id),
        str(candidate.get("planned_island_id") or "sequence"),
    )


def _source_video_id(candidate: Mapping[str, Any], audio_id: str) -> str:
    explicit = str(candidate.get("source_video_id") or "").strip()
    if explicit:
        return explicit
    value = str(candidate.get("video_id") or audio_id)
    return re.sub(r"-w\d+$", "", value)


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
    legacy_audit_only: bool = False,
) -> dict[str, Any]:
    import torch

    selected_repo = qwen_asr_repo_id(asr_repo_id)
    if selected_repo != QWEN_ASR_17B_REPO_ID:
        raise ValueError("CueQC v13 feature compilation is restricted to the 1.7B repo")
    strict_current = not legacy_audit_only
    labels = read_labels(label_paths, strict_current=strict_current)
    expanded_chunk_paths = expand_chunk_paths(chunk_paths)
    rows: list[dict[str, Any]] = []
    row_ids: set[str] = set()
    core_owners: dict[str, str] = {}
    group_map: dict[tuple[str, str, str], list[int]] = {}
    source_partitions: dict[str, set[str]] = {}
    canonical_label_ids: set[str] = set()
    for path in expanded_chunk_paths:
        source = repo_display_path(path)
        audio_id, chunks = read_chunk_document(path)
        for index, chunk in enumerate(chunks):
            candidate = candidate_for_chunk(
                chunks, index, strict_current=strict_current
            )
            rid = row_id(audio_id, chunk, index)
            if rid in row_ids:
                raise ValueError(f"duplicate CueQC provisional subisland identity: {rid}")
            row_ids.add(rid)
            split_sha256, inner_sha256, source_core_ids = _training_chunk_provenance(
                chunk,
                candidate,
                row_id=rid,
                strict_current=strict_current,
            )
            for core_id in source_core_ids:
                previous = core_owners.get(core_id)
                if previous is not None and previous != rid:
                    raise ValueError(
                        f"CueQC core {core_id!r} is reused by provisional "
                        f"subislands {previous!r} and {rid!r}"
                    )
                core_owners[core_id] = rid
            label = label_for_chunk(labels, audio_id=audio_id, chunk=chunk, index=index)
            if strict_current:
                if label is None:
                    raise ValueError(f"CueQC current Runtime row {rid!r} has no canonical label")
                if str(label.get("schema") or "") != CANONICAL_LABEL_SCHEMA:
                    raise ValueError(f"CueQC label for {rid!r} is not canonical v2")
                if label.get("training_manifest_allowed") is not True:
                    raise ValueError(f"CueQC label for {rid!r} is not training-approved")
                if str(label.get("subisland_id") or "") != rid:
                    raise ValueError(f"CueQC label identity mismatch for {rid!r}")
                for key in ("sample_id", "source_id", "source_partition", "audio"):
                    if str(label.get(key) or "") != str(chunk.get(key) or ""):
                        raise ValueError(f"CueQC chunk/label {key} mismatch for {rid!r}")
                for key in ("start_s", "end_s"):
                    if not math.isclose(float(label.get(key)), float(chunk.get(key)), abs_tol=1e-6):
                        raise ValueError(f"CueQC chunk/label {key} mismatch for {rid!r}")
                canonical_label_ids.add(rid)
            label_index = (
                PRE_ASR_CUEQC_IGNORE_LABEL if label is None else int(label["label_index"])
            )
            group_key = _group_key(source, audio_id, candidate)
            candidate_audio_id = str(
                candidate.get("audio_id")
                or candidate.get("video_id")
                or chunk.get("sample_id")
                or audio_id
            )
            source_id = str(
                chunk.get("source_id") or (label or {}).get("source_id") or ""
            ).strip()
            chunk_source_id = str(chunk.get("source_id") or "").strip()
            label_source_id = str((label or {}).get("source_id") or "").strip()
            if chunk_source_id and label_source_id and chunk_source_id != label_source_id:
                raise ValueError(
                    f"CueQC chunk/label source mismatch for {rid!r}: "
                    f"{chunk_source_id!r} != {label_source_id!r}"
                )
            if not source_id:
                raise ValueError(f"CueQC chunk {rid!r} has no frozen source_id")
            dataset_role = str(
                chunk.get("source_partition")
                or (label or {}).get("source_partition")
                or ""
            ).strip()
            chunk_partition = str(chunk.get("source_partition") or "").strip()
            label_partition = str((label or {}).get("source_partition") or "").strip()
            if chunk_partition and label_partition and chunk_partition != label_partition:
                raise ValueError(
                    f"CueQC chunk/label partition mismatch for {rid!r}: "
                    f"{chunk_partition!r} != {label_partition!r}"
                )
            if dataset_role not in {"train", "val", "test"}:
                raise ValueError(
                    f"CueQC chunk {rid!r} has no frozen train/val/test source partition"
                )
            source_partitions.setdefault(source_id, set()).add(dataset_role)
            row_index = len(rows)
            group_map.setdefault(group_key, []).append(row_index)
            rows.append(
                {
                    "id": rid,
                    "source": source,
                    "audio_id": candidate_audio_id,
                    "audio": str(chunk.get("audio") or candidate.get("audio") or ""),
                    "video_id": _source_video_id(candidate, candidate_audio_id),
                    "source_id": source_id,
                    "source_core_ids": source_core_ids,
                    "dataset_role": dataset_role,
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
                        else "teacher_unsure_ignored"
                        if (label or {}).get("training_ignore_reason") == "teacher_unsure"
                        else "ambiguous_ignore"
                    ),
                    "teacher_label": "" if label is None else str(label.get("teacher_label") or ""),
                    "canonical_label": "" if label is None else raw_label(label),
                    "exact_core_label": ""
                    if label is None
                    else str(label.get("exact_core_label") or ""),
                    "training_ignore_reason": ""
                    if label is None
                    else str(label.get("training_ignore_reason") or ""),
                    "label_source": ""
                    if label is None
                    else str(
                        label.get("label_source")
                        or label.get("cluster_label_source")
                        or label.get("source")
                        or ""
                    ),
                    "candidate": candidate,
                    "semantic_split_weights_sha256": split_sha256,
                    "inner_edge_refiner_weights_sha256": inner_sha256,
                }
            )
    groups: list[list[int]] = []
    for row_indexes in group_map.values():
        group_roles = {str(rows[row_index]["dataset_role"]) for row_index in row_indexes}
        group_sources = {str(rows[row_index]["source_id"]) for row_index in row_indexes}
        if len(group_roles) != 1 or len(group_sources) != 1:
            raise ValueError("CueQC sequence group crosses source identity or partition")
        if any(int(rows[row_index]["label_index"]) in (0, 1) for row_index in row_indexes):
            groups.append(row_indexes)
    if not groups:
        raise ValueError("no definite labeled Pre-ASR CueQC examples were compiled")
    if strict_current:
        leaked_sources = [
            source_id for source_id, roles in source_partitions.items() if len(roles) != 1
        ]
        if leaked_sources:
            raise ValueError(
                f"CueQC source identity crosses dataset partitions: {sorted(leaked_sources)[:3]}"
            )
        if set(source_partitions) and set().union(*source_partitions.values()) != {
            "train",
            "val",
            "test",
        }:
            raise ValueError("CueQC current training compilation requires train/val/test sources")
        if canonical_label_ids != row_ids:
            raise ValueError("CueQC canonical labels do not cover every Runtime row")
    projection_digests = {
        str(rows[row_index]["candidate"].get("ptm_projection_digest") or "")
        for group in groups
        for row_index in group
        if str(rows[row_index]["candidate"].get("ptm_projection_digest") or "")
    }
    if len(projection_digests) > 1:
        raise ValueError(f"multiple PTM projection digests in feature bundle: {sorted(projection_digests)}")
    ptm_projection_digest = next(iter(projection_digests), "")
    split_checkpoint_shas = {
        str(rows[row_index]["semantic_split_weights_sha256"])
        for group in groups
        for row_index in group
    }
    inner_checkpoint_shas = {
        str(rows[row_index]["inner_edge_refiner_weights_sha256"])
        for group in groups
        for row_index in group
    }
    if len(split_checkpoint_shas) > 1 or len(inner_checkpoint_shas) > 1:
        raise ValueError("CueQC feature bundle mixes upstream checkpoint identities")
    pooling_schemas = sorted(
        {
            str(rows[row_index]["candidate"].get("ptm_pooling_schema") or "")
            for group in groups
            for row_index in group
            if str(rows[row_index]["candidate"].get("ptm_pooling_schema") or "")
        }
    )
    bundle_tensors = _make_tensor_bundle(rows, groups)
    y = bundle_tensors["labels"].numpy()
    selected_rows = [rows[row_index] for group in groups for row_index in group]
    teacher_unsure_ignored = sum(
        row.get("training_ignore_reason") == "teacher_unsure"
        for row in selected_rows
    )
    row_payload = [
        {key: value for key, value in row.items() if key != "candidate"}
        for row in selected_rows
    ]
    group_payload = [
        {
            "group_index": group_index,
            "row_ids": [rows[row_index]["id"] for row_index in group],
            "audio_id": rows[group[0]]["audio_id"],
            "video_id": rows[group[0]]["video_id"],
            "source_id": rows[group[0]]["source_id"],
            "source_core_ids": sorted(
                {
                    core_id
                    for row_index in group
                    for core_id in rows[row_index]["source_core_ids"]
                }
            ),
            "planned_island_id": rows[group[0]]["planned_island_id"],
            "dataset_role": rows[group[0]]["dataset_role"],
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
        "asr_repo_id": selected_repo,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "training_manifest_allowed": bool(strict_current),
        "input_distribution": CURRENT_INPUT_DISTRIBUTION if strict_current else "legacy_audit_only",
        "runtime_chunk_schema": RUNTIME_CHUNK_SCHEMA if strict_current else "legacy",
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA if strict_current else "legacy",
        "all_partitions_present": bool(
            set().union(*source_partitions.values()) == {"train", "val", "test"}
            if source_partitions
            else False
        ),
        "ptm_pooling_schemas": pooling_schemas,
        "ptm_projection_digest": ptm_projection_digest,
        "semantic_split_weights_sha256": next(iter(split_checkpoint_shas), ""),
        "inner_edge_refiner_weights_sha256": next(iter(inner_checkpoint_shas), ""),
        "teacher_unsure_ignored": teacher_unsure_ignored,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": row_payload,
        "groups": group_payload,
        "source_files": [repo_display_path(path) for path in expanded_chunk_paths],
        "source_file_sha256": {
            repo_display_path(path): _file_sha256(path) for path in expanded_chunk_paths
        },
        "label_files": [repo_display_path(project_path(path)) for path in label_paths],
        "label_file_sha256": {
            repo_display_path(project_path(path)): _file_sha256(project_path(path))
            for path in label_paths
        },
        **bundle_tensors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _torch_save_atomic(bundle, output)
    summary = {
        "schema": "cueqc_pre_asr_semantic_chunk_v13_feature_summary",
        "feature_bundle": repo_display_path(output),
        "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
        "feature_names": list(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),
        "asr_repo_id": selected_repo,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "training_manifest_allowed": bool(strict_current),
        "input_distribution": CURRENT_INPUT_DISTRIBUTION if strict_current else "legacy_audit_only",
        "runtime_chunk_schema": RUNTIME_CHUNK_SCHEMA if strict_current else "legacy",
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA if strict_current else "legacy",
        "all_partitions_present": bool(
            set().union(*source_partitions.values()) == {"train", "val", "test"}
            if source_partitions
            else False
        ),
        "source_file_sha256": {
            repo_display_path(path): _file_sha256(path) for path in expanded_chunk_paths
        },
        "label_file_sha256": {
            repo_display_path(project_path(path)): _file_sha256(project_path(path))
            for path in label_paths
        },
        "ptm_pooling_schemas": pooling_schemas,
        "ptm_projection_digest": ptm_projection_digest,
        "semantic_split_weights_sha256": next(iter(split_checkpoint_shas), ""),
        "inner_edge_refiner_weights_sha256": next(iter(inner_checkpoint_shas), ""),
        "group_count": int(len(groups)),
        "chunk_count": int(np.sum(y != PRE_ASR_CUEQC_IGNORE_LABEL)),
        "keep": int(np.sum(y == 1)),
        "drop": int(np.sum(y == 0)),
        "teacher_unsure_ignored": teacher_unsure_ignored,
        "ambiguous_ignore": int(
            np.sum(
                (y == PRE_ASR_CUEQC_IGNORE_LABEL)
                & (bundle_tensors["chunk_mask"].numpy() > 0)
            )
            - teacher_unsure_ignored
        ),
        "output_sha256": _file_sha256(output),
    }
    _write_text_atomic(
        output.with_suffix(".summary.json"),
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Pre-ASR CueQC v13 provisional-subisland features.")
    parser.add_argument("--chunks", action="append", required=True, help="Workflow details/chunk JSON or JSONL.")
    parser.add_argument("--labels", action="append", required=True, help="JSON/JSONL labels with keep/drop/ignore.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--asr-repo-id", default=current_qwen_asr_backend())
    parser.add_argument(
        "--legacy-audit-only",
        action="store_true",
        help=(
            "Accept retired candidate/label formats only for reproducing audits; "
            "the resulting bundle is marked training_manifest_allowed=false."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = compile_features(
        chunk_paths=list(args.chunks),
        label_paths=list(args.labels),
        output=project_path(args.output),
        asr_repo_id=str(args.asr_repo_id),
        legacy_audit_only=bool(args.legacy_audit_only),
    )
    print(
        "features={feature_bundle} groups={group_count} keep={keep} drop={drop} "
        "teacher_unsure_ignored={teacher_unsure_ignored} ambiguous_ignore={ambiguous_ignore}".format(
            feature_bundle=summary["feature_bundle"],
            group_count=summary["group_count"],
            keep=summary["keep"],
            drop=summary["drop"],
            teacher_unsure_ignored=summary["teacher_unsure_ignored"],
            ambiguous_ignore=summary["ambiguous_ignore"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
