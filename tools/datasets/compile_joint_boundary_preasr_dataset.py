#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SRC_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from asr.pre_asr_cueqc import PRE_ASR_CUEQC_IGNORE_LABEL  # noqa: E402
from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.sequence_features import SPLIT_CANDIDATE_SCALAR_NAMES  # noqa: E402
from boundary.sequence_store import (  # noqa: E402
    StreamingFrameWriter,
    frames_sidecar_path,
    save_sequence_dataset,
)
from boundary.split_model import SEMANTIC_SPLIT_FEATURE_SCHEMA  # noqa: E402
from tools.asr.cueqc.pre_asr_feature_compiler import (  # noqa: E402
    compile_features,
    normalize_label,
)
from tools.boundary.ja.acoustic_split_teacher_contracts import (  # noqa: E402
    APPROVED_SPLIT_TEACHER_PROMPT_VERSIONS,
)
from tools.boundary.ja.acoustic_split_v4_dataset import (  # noqa: E402
    SPLIT_V4_DATASET_SUMMARY_SCHEMA,
    SPLIT_V4_INPUT_DISTRIBUTION,
    SPLIT_V4_MFCC_DIM,
    SPLIT_V4_PTM_DIM,
    SPLIT_V4_UPSTREAM_SHA_FIELDS,
    file_sha256,
)


SPLIT_LABEL_IDS = {"cut": 0, "continue": 1, "unsure": 2}
IGNORE_ID = -100
# Acoustic Split trains only on approved per-candidate centered-clip labels.
# Other teacher geometries hard-fail rather than silently entering the binary
# model's canonical data layer.


def _reject_foreign_split_labels(labels: list[dict[str, Any]]) -> None:
    foreign: Counter[str] = Counter(
        str(row.get("prompt_version") or "<missing>")
        for row in labels
        if str(row.get("prompt_version") or "")
        not in APPROVED_SPLIT_TEACHER_PROMPT_VERSIONS
    )
    if foreign:
        raise ValueError(
            "semantic_split/labels.jsonl contains split labels from a retired "
            "teacher contract; only approved centered-candidate provenance "
            f"{sorted(APPROVED_SPLIT_TEACHER_PROMPT_VERSIONS)} may be compiled "
            "into Acoustic Split training data. Offending prompt_version "
            f"counts: {dict(foreign)}"
        )


def _variant_npz_path(path: Path, variant: str) -> Path:
    if not variant:
        return path
    return path.with_name(f"{path.stem}.{variant}{path.suffix}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _torch_save_atomic(payload: object, path: Path) -> None:
    import torch

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _required_sha256(value: Any, *, field: str, window_id: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"source window {window_id!r} has invalid {field}")
    return normalized


def _bundle_scalar(bundle: dict[str, np.ndarray], key: str) -> Any:
    if key not in bundle:
        raise ValueError(f"semantic split feature bundle is missing {key!r}")
    values = np.asarray(bundle[key]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"semantic split feature bundle {key!r} must be scalar")
    return values[0]


def _bundle_text(bundle: dict[str, np.ndarray], key: str) -> str:
    return str(_bundle_scalar(bundle, key)).strip()


def _window_split_provenance(window: dict[str, Any]) -> dict[str, str]:
    window_id = str(window.get("window_id") or "").strip()
    if window.get("semantic_split_training_manifest_allowed") is not True:
        raise ValueError(
            f"source window {window_id!r} is not approved for Split v4 "
            "training; rebuild candidates from the audited Scorer v11 -> "
            "Proposal -> Outer v3 chain and set an explicit training gate"
        )
    expected = {
        "feature_schema": SEMANTIC_SPLIT_FEATURE_SCHEMA,
        "input_distribution": SPLIT_V4_INPUT_DISTRIBUTION,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
    }
    window_fields = {
        "feature_schema": "semantic_split_feature_schema",
        "input_distribution": "semantic_split_input_distribution",
        "boundary_serialization_contract_id": (
            "boundary_serialization_contract_id"
        ),
        "ptm_repo_id": "ptm_repo_id",
    }
    result: dict[str, str] = {}
    for canonical, field in window_fields.items():
        actual = str(window.get(field) or "").strip()
        if actual != expected[canonical]:
            raise ValueError(
                f"source window {window_id!r} has stale {field}: {actual!r}"
            )
        result[canonical] = actual
    result["audio_wav_sha256"] = _required_sha256(
        window.get("audio_wav_sha256"),
        field="audio_wav_sha256",
        window_id=window_id,
    )
    audio_path = Path(str(window.get("audio_wav") or ""))
    if not audio_path.is_file():
        raise FileNotFoundError(
            f"source window {window_id!r} audio_wav not found: {audio_path}"
        )
    if file_sha256(audio_path) != result["audio_wav_sha256"]:
        raise ValueError(f"source window {window_id!r} audio SHA binding mismatch")
    result["audio_wav"] = str(audio_path.resolve())
    for field in SPLIT_V4_UPSTREAM_SHA_FIELDS:
        result[field] = _required_sha256(
            window.get(field), field=field, window_id=window_id
        )
    return result


PARTITIONS = {"train", "val", "test"}


def _window_identity(window: dict[str, Any]) -> tuple[str, str]:
    source_id = str(window.get("source_id") or "").strip()
    partition = str(window.get("source_partition") or "").strip()
    if not source_id:
        raise ValueError(
            f"source window {window.get('window_id')!r} is missing frozen source_id"
        )
    if partition not in PARTITIONS:
        raise ValueError(
            f"source window {window.get('window_id')!r} has invalid frozen "
            f"source_partition: {partition!r}"
        )
    return source_id, partition


def _core_identity(
    window: dict[str, Any], *, core_start_s: float, core_end_s: float
) -> str:
    source_id, _partition_name = _window_identity(window)
    source_start_s = float(window.get("source_start_s") or 0.0)
    start_sample = int(round((source_start_s + core_start_s) * 16000.0))
    end_sample = int(round((source_start_s + core_end_s) * 16000.0))
    if end_sample <= start_sample:
        raise ValueError(
            f"source window {window.get('window_id')!r} has invalid core extent"
        )
    return f"{source_id}:samples:{start_sample}-{end_sample}"


def _omni_aux_row(row: dict[str, Any]) -> list[float]:
    return [
        1.0 if bool(row.get("left_complete")) else 0.0,
        1.0 if bool(row.get("right_complete")) else 0.0,
        1.0 if bool(row.get("merged_better")) else 0.0,
    ]


def _compile_split(
    *,
    dataset: Path,
    windows: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    load_workers: int = 6,
    feature_variant: str = "",
    output_variant: str = "",
) -> dict[str, Any]:
    """Emit whole-island candidate sequences.

    Every candidate of an island that carries at least one Omni label is
    included so the island-sequence model sees the same candidate context as
    runtime; candidates without a label use ignore id ``-100`` and omni aux
    ``-1`` so per-candidate losses can mask them.
    """

    window_by_id: dict[str, dict[str, Any]] = {}
    source_partitions: dict[str, str] = {}
    for window in windows:
        window_id = str(window.get("window_id") or "").strip()
        if not window_id:
            raise ValueError("source window is missing window_id")
        if window_id in window_by_id:
            raise ValueError(f"duplicate source window_id: {window_id}")
        source_id, partition = _window_identity(window)
        previous_partition = source_partitions.setdefault(source_id, partition)
        if previous_partition != partition:
            raise ValueError(
                f"source {source_id!r} crosses frozen source partitions"
            )
        window_by_id[window_id] = window
    _reject_foreign_split_labels(labels)
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for row in labels:
        label = str(row.get("label") or "")
        if label not in SPLIT_LABEL_IDS:
            raise ValueError(f"unsupported Semantic Split label: {label!r}")
        window_id = str(row.get("window_id") or "").strip()
        if not window_id:
            raise ValueError("Semantic Split label is missing window_id")
        try:
            feature_index = int(row["feature_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Semantic Split label for {window_id!r} has invalid feature_index"
            ) from exc
        try:
            candidate_time_s = float(row["time_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Semantic Split label for {window_id!r} feature_index="
                f"{feature_index} is missing a numeric time_s binding"
            ) from exc
        if not math.isfinite(candidate_time_s):
            raise ValueError(
                f"Semantic Split label for {window_id!r} feature_index="
                f"{feature_index} has non-finite time_s"
            )
        window_rows = grouped.setdefault(window_id, {})
        if feature_index in window_rows:
            raise ValueError(
                f"duplicate Semantic Split label for {window_id!r} "
                f"feature_index={feature_index}"
            )
        window_rows[feature_index] = {**row, "time_s": candidate_time_s}
    output_name = f"features.{output_variant}.npz" if output_variant else "features.npz"
    output = dataset / "semantic_split" / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    # Frames stream to the sidecar .frames.npy: a full-dim (2048-PTM) compile
    # cannot hold an np.stack of every row on a 16GB box.
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.stem}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary_output = Path(handle.name)
    temporary_output.unlink(missing_ok=True)
    temporary_sidecar = frames_sidecar_path(temporary_output)
    frame_writer = StreamingFrameWriter(temporary_output)
    scalar_parts: list[np.ndarray] = []
    label_parts: list[int] = []
    partition_parts: list[str] = []
    window_parts: list[str] = []
    video_parts: list[str] = []
    source_parts: list[str] = []
    core_parts: list[str] = []
    feature_index_parts: list[int] = []
    time_parts: list[float] = []
    group_parts: list[str] = []
    omni_parts: list[list[float]] = []
    applied_label_counts: Counter[str] = Counter()
    prompt_version_counts: Counter[str] = Counter()
    source_bindings: dict[str, dict[str, str]] = {}
    labeled_count = 0
    core_origins: dict[str, tuple[str, tuple[float, float]]] = {}
    ordered: list[tuple[str, dict[int, dict[str, Any]], dict[str, Any]]] = []
    common_upstream: dict[str, str] = {}
    for window_id, labeled_rows in sorted(grouped.items()):
        window = window_by_id.get(window_id)
        if window is None:
            raise ValueError(f"Semantic Split labels reference unknown window {window_id!r}")
        provenance = _window_split_provenance(window)
        for field in SPLIT_V4_UPSTREAM_SHA_FIELDS:
            previous = common_upstream.setdefault(field, provenance[field])
            if previous != provenance[field]:
                raise ValueError(
                    f"Split v4 dataset mixes {field} identities: "
                    f"{previous} != {provenance[field]}"
                )
        ordered.append((window_id, labeled_rows, {**window, "_split_provenance": provenance}))

    def _load_window_arrays(window: dict[str, Any]) -> dict[str, Any]:
        # Per-window npz are zlib-compressed; decompressing in worker threads
        # (zlib releases the GIL) keeps the single writer thread fed.
        feature_path = _variant_npz_path(
            Path(str(window["semantic_split_features"])), feature_variant
        )
        if not feature_path.exists():
            raise FileNotFoundError(
                f"semantic split feature variant {feature_variant!r} missing: "
                f"{feature_path}"
            )
        with np.load(feature_path, allow_pickle=False) as handle:
            bundle = {key: np.asarray(handle[key]) for key in handle.files}
        gate = _bundle_scalar(bundle, "training_manifest_allowed")
        if not isinstance(gate, (bool, np.bool_)) or not bool(gate):
            distribution = str(bundle.get("input_distribution", "unknown"))
            raise ValueError(
                "semantic split features are audit-only, not training-ready: "
                f"{distribution}"
            )
        provenance = dict(window["_split_provenance"])
        expected_bundle = {
            "feature_schema": provenance["feature_schema"],
            "input_distribution": provenance["input_distribution"],
            "boundary_serialization_contract_id": provenance[
                "boundary_serialization_contract_id"
            ],
            "ptm_repo_id": provenance["ptm_repo_id"],
            "window_id": str(window["window_id"]),
            "source_id": str(window["source_id"]),
            "source_partition": str(window["source_partition"]),
            "audio_wav_sha256": provenance["audio_wav_sha256"],
            **{field: provenance[field] for field in SPLIT_V4_UPSTREAM_SHA_FIELDS},
        }
        for field, expected in expected_bundle.items():
            actual = _bundle_text(bundle, field)
            if actual != expected:
                raise ValueError(
                    f"semantic split feature bundle {feature_path} has stale "
                    f"{field}: {actual!r}"
                )
        required_arrays = (
            "frame_features",
            "scalar_features",
            "proposal_times_s",
            "core_starts_s",
            "core_ends_s",
        )
        missing = [key for key in required_arrays if key not in bundle]
        if missing:
            raise ValueError(
                f"semantic split feature bundle {feature_path} is missing {missing}"
            )
        frames = np.asarray(bundle["frame_features"])
        scalars = np.asarray(bundle["scalar_features"])
        if frames.ndim != 3 or frames.shape[0] <= 0:
            raise ValueError("semantic split frame features must be non-empty [rows,bins,dim]")
        if int(frames.shape[2]) != SPLIT_V4_PTM_DIM + SPLIT_V4_MFCC_DIM:
            raise ValueError("semantic split frame width must be raw PTM2048 + MFCC40")
        if scalars.shape != (frames.shape[0], len(SPLIT_CANDIDATE_SCALAR_NAMES)):
            raise ValueError("semantic split scalar feature shape is stale")
        if not np.isfinite(frames).all() or not np.isfinite(scalars).all():
            raise ValueError("semantic split feature bundle contains non-finite values")
        total = int(frames.shape[0])
        for key in ("proposal_times_s", "core_starts_s", "core_ends_s"):
            values = np.asarray(bundle[key])
            if values.shape != (total,) or not np.isfinite(values).all():
                raise ValueError(f"semantic split feature bundle has invalid {key}")
        starts = np.asarray(bundle["core_starts_s"], dtype=np.float64)
        ends = np.asarray(bundle["core_ends_s"], dtype=np.float64)
        times = np.asarray(bundle["proposal_times_s"], dtype=np.float64)
        if np.any(ends <= starts) or np.any(times <= starts) or np.any(times >= ends):
            raise ValueError("semantic split feature coordinates are invalid")
        return {
            "bundle": bundle,
            "feature_path": feature_path,
            "feature_sha256": file_sha256(feature_path),
        }

    try:
        with ThreadPoolExecutor(max_workers=max(1, load_workers)) as pool:
            pending: deque = deque()
            next_index = 0

            def _submit_next() -> None:
                nonlocal next_index
                if next_index < len(ordered):
                    entry = ordered[next_index]
                    pending.append((entry, pool.submit(_load_window_arrays, entry[2])))
                    next_index += 1

            for _slot in range(max(1, load_workers)):
                _submit_next()
            while pending:
                (window_id, labeled_rows, window), future = pending.popleft()
                loaded = future.result()
                bundle = loaded["bundle"]
                _submit_next()
                video_id = str(window["video_id"])
                source_id, partition = _window_identity(window)
                provenance = dict(window["_split_provenance"])
                source_bindings[window_id] = {
                    "feature_path": str(Path(loaded["feature_path"]).resolve()),
                    "feature_sha256": str(loaded["feature_sha256"]),
                    "audio_wav": provenance["audio_wav"],
                    "audio_wav_sha256": provenance["audio_wav_sha256"],
                }
                total = int(bundle["frame_features"].shape[0])
                for index in labeled_rows:
                    if index < 0 or index >= total:
                        raise IndexError(
                            f"semantic split feature index {index} out of range for {window_id}"
                        )
                core_starts = np.asarray(bundle["core_starts_s"], dtype=np.float64)
                core_ends = np.asarray(bundle["core_ends_s"], dtype=np.float64)
                times = np.asarray(bundle["proposal_times_s"], dtype=np.float64)
                for index, labeled in labeled_rows.items():
                    label_time_s = float(labeled["time_s"])
                    feature_time_s = float(times[index])
                    if not math.isclose(
                        label_time_s,
                        feature_time_s,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    ):
                        raise ValueError(
                            "Semantic Split label time does not match the bound "
                            f"feature candidate for {window_id!r} feature_index={index}: "
                            f"{label_time_s} != {feature_time_s}"
                        )
                island_members: dict[tuple[float, float], list[int]] = {}
                for index in range(total):
                    key = (
                        round(float(core_starts[index]), 6),
                        round(float(core_ends[index]), 6),
                    )
                    island_members.setdefault(key, []).append(index)
                for key, members in sorted(island_members.items()):
                    if not any(index in labeled_rows for index in members):
                        continue
                    core_id = _core_identity(
                        window, core_start_s=float(key[0]), core_end_s=float(key[1])
                    )
                    origin = (window_id, key)
                    previous_origin = core_origins.setdefault(core_id, origin)
                    if previous_origin != origin:
                        raise ValueError(
                            f"core {core_id!r} is duplicated by overlapping source windows: "
                            f"{previous_origin[0]!r} and {window_id!r}"
                        )
                    group_id = f"{source_id}|island|{core_id}"
                    ordered_members = sorted(members, key=lambda item: float(times[item]))
                    ordered_times = [float(times[index]) for index in ordered_members]
                    if any(
                        left >= right
                        for left, right in zip(ordered_times, ordered_times[1:])
                    ):
                        raise ValueError(
                            f"semantic split candidates are not strictly ordered for {core_id}"
                        )
                    for index in ordered_members:
                        labeled = labeled_rows.get(index)
                        frame_writer.append(
                            bundle["frame_features"][index].astype(np.float32)
                        )
                        scalar_parts.append(
                            bundle["scalar_features"][index].astype(np.float32)
                        )
                        if labeled is None:
                            label_parts.append(IGNORE_ID)
                            omni_parts.append([-1.0, -1.0, -1.0])
                        else:
                            label = str(labeled["label"])
                            label_parts.append(SPLIT_LABEL_IDS[label])
                            omni_parts.append(_omni_aux_row(labeled))
                            applied_label_counts[label] += 1
                            prompt_version_counts[str(labeled["prompt_version"])] += 1
                            labeled_count += 1
                        partition_parts.append(partition)
                        window_parts.append(window_id)
                        video_parts.append(video_id)
                        source_parts.append(source_id)
                        core_parts.append(core_id)
                        feature_index_parts.append(index)
                        time_parts.append(float(times[index]))
                        group_parts.append(group_id)
        if labeled_count <= 0:
            raise ValueError("no joint semantic split labels found")
        frame_writer.finalize()
        save_sequence_dataset(
            temporary_output,
            frames_finalized=True,
            compress=True,
            scalar_features=np.stack(scalar_parts),
            labels=np.asarray(label_parts, dtype=np.int64),
            partitions=np.asarray(partition_parts),
            window_ids=np.asarray(window_parts),
            video_ids=np.asarray(video_parts),
            source_ids=np.asarray(source_parts),
            core_ids=np.asarray(core_parts),
            feature_indexes=np.asarray(feature_index_parts, dtype=np.int64),
            times_s=np.asarray(time_parts, dtype=np.float32),
            group_ids=np.asarray(group_parts),
            structural_roles=np.full(len(label_parts), IGNORE_ID, dtype=np.int64),
            pair_ids=np.full(len(label_parts), -1, dtype=np.int64),
            omni_aux=np.asarray(omni_parts, dtype=np.float32),
        )
        os.replace(temporary_sidecar, frames_sidecar_path(output))
        os.replace(temporary_output, output)
    finally:
        frame_writer.abort()
        temporary_output.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
    partitions = Counter(partition_parts)
    frame_shape = (len(label_parts), *frame_writer.shape[1:])
    summary = {
        "schema": SPLIT_V4_DATASET_SUMMARY_SCHEMA,
        "training_manifest_allowed": True,
        "output": str(output.resolve()),
        "dataset_sha256": file_sha256(output),
        "frame_sidecar_sha256": file_sha256(frames_sidecar_path(output)),
        "input_distribution": SPLIT_V4_INPUT_DISTRIBUTION,
        "feature_schema": SEMANTIC_SPLIT_FEATURE_SCHEMA,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
        "ptm_dim": SPLIT_V4_PTM_DIM,
        "mfcc_dim": SPLIT_V4_MFCC_DIM,
        "scalar_names": list(SPLIT_CANDIDATE_SCALAR_NAMES),
        **common_upstream,
        "count": len(label_parts),
        "frame_bins": int(frame_shape[1]),
        "frame_dim": int(frame_shape[2]),
        "scalar_dim": len(SPLIT_CANDIDATE_SCALAR_NAMES),
        "labeled_count": labeled_count,
        "context_only_count": len(label_parts) - labeled_count,
        "group_count": len(set(group_parts)),
        "labels": dict(applied_label_counts),
        "label_prompt_versions": dict(prompt_version_counts),
        "partitions": dict(partitions),
        "video_count": len(set(video_parts)),
        "window_count": len(set(window_parts)),
        "partition_unit": "frozen_source_id_and_core_id",
        "source_count": len(set(source_parts)),
        "core_count": len(set(core_parts)),
        "source_feature_audio_sha256": source_bindings,
    }
    summary_name = f"summary.{output_variant}.json" if output_variant else "summary.json"
    _write_json(output.with_suffix(".summary.json"), summary)
    alias_path = dataset / "semantic_split" / summary_name
    if alias_path != output.with_suffix(".summary.json"):
        _write_json(alias_path, summary)
    return summary


def _normalized_label(row: dict[str, Any]) -> int | None:
    value = normalize_label(row)
    if value is not None and row.get("training_label_included") is False:
        value = PRE_ASR_CUEQC_IGNORE_LABEL
    return value


def _pre_asr_override_summary(
    base_labels: list[dict[str, Any]],
    override_paths: list[Path],
) -> dict[str, Any]:
    """Validate override files against the base labels before compiling.

    ``read_labels`` applies later files over earlier ones, so overrides only
    need to be appended to ``label_paths`` — this helper exists to fail fast on
    override rows whose candidate never appears in the base labels (a typo'd or
    stale id would otherwise be silently ignored) and to report counts.
    """

    base_by_id: dict[str, int | None] = {}
    for row in base_labels:
        candidate_id = str(row.get("candidate_id") or row.get("sample_id") or "").strip()
        if candidate_id:
            base_by_id[candidate_id] = _normalized_label(row)
    counts: Counter[str] = Counter()
    changed = 0
    unmatched: list[str] = []
    total = 0
    for path in override_paths:
        for row in _read_jsonl(path):
            candidate_id = str(
                row.get("candidate_id") or row.get("sample_id") or ""
            ).strip()
            value = _normalized_label(row)
            if not candidate_id or value is None:
                raise ValueError(
                    f"override row needs candidate_id and a keep/drop/ignore label: "
                    f"{path}: {row.get('candidate_id')!r}/{row.get('label')!r}"
                )
            total += 1
            counts[str(row.get("label"))] += 1
            if candidate_id not in base_by_id:
                unmatched.append(candidate_id)
            elif base_by_id[candidate_id] != value:
                changed += 1
    if unmatched:
        preview = ", ".join(unmatched[:5])
        raise ValueError(
            f"{len(unmatched)} override candidate ids not present in base "
            f"pre_asr labels (e.g. {preview})"
        )
    return {
        "files": [str(path) for path in override_paths],
        "count": total,
        "by_label": dict(counts),
        "changed_from_base": changed,
    }


def _compile_pre_asr(
    *,
    dataset: Path,
    windows: list[dict[str, Any]],
    labels_path: Path,
    asr_repo_id: str,
    override_paths: list[Path] | None = None,
) -> dict[str, Any]:
    chunk_paths = [
        str(Path(row["pre_asr_candidates"]))
        for row in windows
        if row.get("pre_asr_candidates")
        and Path(row["pre_asr_candidates"]).exists()
    ]
    if not chunk_paths:
        raise ValueError("no Pre-ASR candidate files found")
    override_summary: dict[str, Any] | None = None
    label_paths = [str(labels_path)]
    if override_paths:
        override_summary = _pre_asr_override_summary(
            _read_jsonl(labels_path), override_paths
        )
        # read_labels 后读覆盖先读：override 文件必须排在基础标签之后。
        label_paths.extend(str(path) for path in override_paths)
    output = dataset / "pre_asr" / "features.pt"
    summary = compile_features(
        chunk_paths=chunk_paths,
        label_paths=label_paths,
        output=output,
        asr_repo_id=asr_repo_id,
    )
    if override_summary is not None:
        summary["label_overrides"] = override_summary
    import torch

    payload = torch.load(output, map_location="cpu", weights_only=False)
    role_by_audio_id = {
        str(row["window_id"]): _window_identity(row)[1] for row in windows
    }
    for group in payload["groups"]:
        group["dataset_role"] = role_by_audio_id.get(
            str(group.get("audio_id") or ""),
            "",
        )
        if group["dataset_role"] not in PARTITIONS:
            raise ValueError(
                f"CueQC group {group.get('audio_id')!r} has no frozen source partition"
            )
    _torch_save_atomic(payload, output)
    summary["output_sha256"] = file_sha256(output)
    role_counts = Counter(
        str(group.get("dataset_role") or "")
        for group in payload["groups"]
    )
    summary["dataset_roles"] = dict(role_counts)
    summary["partition_unit"] = "frozen_source_id"
    _write_json(dataset / "pre_asr" / "summary.json", summary)
    return summary


def _write_dataset_card(
    *,
    dataset: Path,
    split_summary: dict[str, Any],
    pre_asr_summary: dict[str, Any],
) -> None:
    lines = [
        "# Omni joint Boundary / Pre-ASR dataset",
        "",
        "同一 32 kbps MP3 请求同时标注 Semantic Split 候选切点与当前边界链输出的 Pre-ASR chunk。",
        "模型训练和复核使用 16 kHz 单声道 PCM WAV；MP3 只保留为 Omni 请求审计载体。",
        "",
        "## Layout",
        "",
        "- `audio_wav/`: 运行时同滤镜的随机源窗口 WAV。",
        "- `omni_mp3_32k/`: 实际提交给 Omni 的 32 kbps MP3，每个窗口一次请求。",
        "- `annotations/omni_joint/`: 请求、原始响应与逐窗口联合标签。",
        "- `semantic_split/`: 切点标签与可直接训练的 `features.npz`。",
        "- `pre_asr/`: keep/drop/unsure 标签、分类 WAV 切片与 v12 `features.pt`。",
        "- `features/<window_id>/`: 每个窗口的原始 Split / PTM / Pre-ASR 特征。",
        "",
        "## Counts",
        "",
        f"- Semantic Split: `{split_summary['count']}` labels / "
        f"`{split_summary['video_count']}` videos / `{split_summary['window_count']}` windows.",
        f"- Pre-ASR: `{int(pre_asr_summary['keep']) + int(pre_asr_summary['drop'])}` "
        f"definite labels and `{pre_asr_summary['ambiguous_ignore']}` unsure labels "
        f"across `{pre_asr_summary['group_count']}` windows.",
        "",
        "train/val/test 使用预先冻结的 `source_id` 分区；同一 source/core 不得跨集合或重复使用。",
    ]
    (dataset / "DATASET.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset_dir)
    windows = _read_jsonl(dataset / "source_windows.jsonl")
    split_labels_path = dataset / "semantic_split" / "labels.jsonl"
    pre_asr_labels_path = dataset / "pre_asr" / "labels.jsonl"
    split_labels = _read_jsonl(split_labels_path)
    if not windows:
        raise ValueError("source_windows.jsonl is empty")
    if not split_labels:
        raise ValueError("semantic_split/labels.jsonl is empty")
    if not args.split_only and not pre_asr_labels_path.exists():
        raise ValueError("pre_asr/labels.jsonl is missing")
    split_summary = _compile_split(
        dataset=dataset,
        windows=windows,
        labels=split_labels,
        load_workers=args.load_workers,
        feature_variant=args.semantic_split_feature_variant,
        output_variant=args.semantic_split_output_variant,
    )
    if args.split_only:
        print(json.dumps({"semantic_split": split_summary["count"]}, ensure_ascii=False))
        return
    override_paths: list[Path] = []
    for raw in args.pre_asr_label_overrides or []:
        path = Path(raw)
        if not path.exists():
            raise ValueError(f"--pre-asr-label-overrides file not found: {raw}")
        override_paths.append(path)
    pre_asr_summary = _compile_pre_asr(
        dataset=dataset,
        windows=windows,
        labels_path=pre_asr_labels_path,
        asr_repo_id=args.asr_repo_id,
        override_paths=override_paths,
    )
    _write_dataset_card(
        dataset=dataset,
        split_summary=split_summary,
        pre_asr_summary=pre_asr_summary,
    )
    _write_json(
        dataset / "compiled_summary.json",
        {
            "schema": "joint_boundary_preasr_compiled_dataset_v1",
            "semantic_split": split_summary,
            "pre_asr": pre_asr_summary,
        },
    )
    print(
        json.dumps(
            {
                "semantic_split": split_summary["count"],
                "pre_asr": pre_asr_summary["chunk_count"],
            },
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="datasets/train/omni-joint-boundary-preasr-v1",
    )
    parser.add_argument(
        "--asr-repo-id",
        default="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    parser.add_argument(
        "--semantic-split-feature-variant",
        default="",
        help="Read per-window semantic_split_features.<variant>.npz artifacts.",
    )
    parser.add_argument(
        "--semantic-split-output-variant",
        default="",
        help="Write semantic_split/features.<variant>.npz instead of the canonical output.",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Compile only Semantic Split and leave Pre-ASR artifacts untouched.",
    )
    parser.add_argument(
        "--pre-asr-label-overrides",
        action="append",
        default=None,
        help=(
            "JSONL label override file(s) applied over pre_asr/labels.jsonl by "
            "candidate_id (later files win; unmatched ids are an error). The "
            "original labels.jsonl is never modified."
        ),
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=6,
        help="Window-npz decompression threads feeding the frame writer.",
    )
    args = parser.parse_args()
    if args.semantic_split_feature_variant and not args.semantic_split_output_variant:
        args.semantic_split_output_variant = args.semantic_split_feature_variant
    for value in (
        args.semantic_split_feature_variant,
        args.semantic_split_output_variant,
    ):
        if value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
            parser.error(f"invalid semantic split variant: {value!r}")
    return args


if __name__ == "__main__":
    run(parse_args())
