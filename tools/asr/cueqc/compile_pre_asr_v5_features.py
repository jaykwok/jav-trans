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
    candidate_from_span,
    feature_vector,
)


FEATURE_BUNDLE_SCHEMA = "cueqc_pre_asr_mamba_v5_features"


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


def read_chunks(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        return extract_chunks(json.loads(text))
    if text.lstrip().startswith("{"):
        return extract_chunks(json.loads(text))
    return read_json_or_jsonl(path)


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
        payload = json.loads(text)
    else:
        payload = read_json_or_jsonl(path)
    audio_id = infer_audio_id(path, payload if isinstance(payload, Mapping) else None)
    return audio_id, extract_chunks(payload)


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
    if raw in {"keep", "keep_for_asr", "1", "positive"}:
        return 1
    if raw in {"drop", "drop_before_asr", "0", "negative"}:
        return 0
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
                continue
            for source_item in label_items(row):
                item = dict(source_item)
                item["label_index"] = value
                cluster_id = str(item.get("cluster_id") or "").strip()
                if cluster_id:
                    labels[f"cluster:{cluster_id}"] = item
                for key in label_keys(item):
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


def candidate_for_chunk(chunks: list[dict[str, Any]], index: int) -> dict[str, Any]:
    chunk = chunks[index]
    features = chunk.get("features")
    feature_names = tuple(str(item) for item in chunk.get("feature_names") or ())
    if isinstance(features, Mapping) and feature_names == PRE_ASR_CUEQC_FEATURE_NAMES:
        return dict(chunk)
    return candidate_from_span(chunks, index)


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
    features: list[np.ndarray] = []
    targets: list[int] = []
    for raw_path in chunk_paths:
        path = project_path(raw_path)
        audio_id, chunks = read_chunk_document(path)
        for index, chunk in enumerate(chunks):
            rid = row_id(audio_id, chunk, index)
            label = label_for_chunk(labels, audio_id=audio_id, chunk=chunk, index=index)
            if label is None:
                continue
            candidate = candidate_for_chunk(chunks, index)
            vector = feature_vector(candidate)
            features.append(vector)
            targets.append(int(label["label_index"]))
            rows.append(
                {
                    "id": rid,
                    "source": repo_display_path(path),
                    "chunk_index": int(candidate["index"]),
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "label": "keep_for_asr" if int(label["label_index"]) == 1 else "drop_before_asr",
                    "label_source": str(
                        label.get("label_source")
                        or label.get("cluster_label_source")
                        or label.get("source")
                        or ""
                    ),
                }
            )
    if not features:
        raise ValueError("no labeled Pre-ASR CueQC examples were compiled")
    x = np.stack(features).astype(np.float32)
    y = np.asarray(targets, dtype=np.int64)
    bundle = {
        "schema": FEATURE_BUNDLE_SCHEMA,
        "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "feature_names": list(PRE_ASR_CUEQC_FEATURE_NAMES),
        "asr_repo_id": qwen_asr_repo_id(asr_repo_id),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "rows": rows,
        "source_files": [repo_display_path(project_path(path)) for path in chunk_paths],
        "label_files": [repo_display_path(project_path(path)) for path in label_paths],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    summary = {
        "schema": "cueqc_pre_asr_mamba_v5_feature_summary",
        "feature_bundle": repo_display_path(output),
        "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "feature_names": list(PRE_ASR_CUEQC_FEATURE_NAMES),
        "asr_repo_id": qwen_asr_repo_id(asr_repo_id),
        "count": int(y.shape[0]),
        "keep": int(np.sum(y == 1)),
        "drop": int(np.sum(y == 0)),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Pre-ASR CueQC v5 numeric features.")
    parser.add_argument("--chunks", action="append", required=True, help="Workflow details/chunk JSON or JSONL.")
    parser.add_argument("--labels", action="append", required=True, help="JSON/JSONL labels with keep/drop.")
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
    print(f"features={summary['feature_bundle']} count={summary['count']} keep={summary['keep']} drop={summary['drop']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
