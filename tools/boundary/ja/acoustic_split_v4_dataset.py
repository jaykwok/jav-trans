from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.sequence_features import SPLIT_CANDIDATE_SCALAR_NAMES
from boundary.sequence_store import frames_sidecar_path, load_sequence_arrays
from boundary.split_model import SEMANTIC_SPLIT_FEATURE_SCHEMA


IGNORE_ID = -100
SPLIT_V4_DATASET_SUMMARY_SCHEMA = "acoustic_split_v4_sequence_dataset_summary_v1"
SPLIT_V4_INPUT_DISTRIBUTION = (
    "runtime_v12_post_scorer_v11_proposal_v1_outer_v3_split_candidates"
)
SPLIT_V4_PTM_DIM = 2048
SPLIT_V4_MFCC_DIM = 40
SPLIT_V4_UPSTREAM_SHA_FIELDS = (
    "scorer_checkpoint_sha256",
    "proposal_checkpoint_sha256",
    "outer_checkpoint_sha256",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"Split v4 dataset summary has invalid {field}")
    return normalized


def dataset_summary_path(dataset_path: Path) -> Path:
    """Resolve the current summary while accepting the compiler's stable alias."""

    dataset_path = Path(dataset_path)
    companion = dataset_path.with_suffix(".summary.json")
    if companion.is_file():
        return companion
    name = dataset_path.name
    if name == "features.npz":
        alias = dataset_path.parent / "summary.json"
    elif name.startswith("features.") and name.endswith(".npz"):
        variant = name[len("features.") : -len(".npz")]
        alias = dataset_path.parent / f"summary.{variant}.json"
    else:
        alias = companion
    if alias.is_file():
        return alias
    raise FileNotFoundError(f"Split v4 dataset summary not found for {dataset_path}")


def load_training_summary(
    dataset_path: Path,
    *,
    expected_ptm_repo_id: str | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    summary_path = dataset_summary_path(dataset_path)
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Split v4 dataset summary: {summary_path}") from exc
    if not isinstance(summary, dict):
        raise ValueError("Split v4 dataset summary must be a JSON object")
    if summary.get("schema") != SPLIT_V4_DATASET_SUMMARY_SCHEMA:
        raise ValueError("Split v4 dataset summary schema is stale or unsupported")
    if summary.get("training_manifest_allowed") is not True:
        raise ValueError("Split v4 dataset is not approved for training")
    if summary.get("boundary_serialization_contract_id") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError("Split v4 dataset uses a stale Boundary contract")
    if summary.get("feature_schema") != SEMANTIC_SPLIT_FEATURE_SCHEMA:
        raise ValueError("Split v4 dataset feature schema mismatch")
    if summary.get("input_distribution") != SPLIT_V4_INPUT_DISTRIBUTION:
        raise ValueError("Split v4 dataset input distribution is not current")
    ptm_repo_id = str(summary.get("ptm_repo_id") or "").strip()
    if not ptm_repo_id:
        raise ValueError("Split v4 dataset summary is missing ptm_repo_id")
    if expected_ptm_repo_id is not None and ptm_repo_id != expected_ptm_repo_id:
        raise ValueError(
            f"Split v4 dataset ptm_repo_id={ptm_repo_id!r} does not match "
            f"{expected_ptm_repo_id!r}"
        )
    if int(summary.get("ptm_dim") or 0) != SPLIT_V4_PTM_DIM:
        raise ValueError("Split v4 dataset must use raw PTM width 2048")
    if int(summary.get("mfcc_dim") or 0) != SPLIT_V4_MFCC_DIM:
        raise ValueError("Split v4 dataset must use MFCC width 40")
    if tuple(summary.get("scalar_names") or ()) != SPLIT_CANDIDATE_SCALAR_NAMES:
        raise ValueError("Split v4 dataset scalar feature names mismatch")
    if int(summary.get("count") or 0) <= 0:
        raise ValueError("Split v4 dataset summary must have a positive row count")
    if int(summary.get("frame_bins") or 0) <= 0:
        raise ValueError("Split v4 dataset summary must have positive frame bins")
    if int(summary.get("frame_dim") or 0) != SPLIT_V4_PTM_DIM + SPLIT_V4_MFCC_DIM:
        raise ValueError("Split v4 dataset summary frame width is stale")
    if int(summary.get("scalar_dim") or 0) != len(SPLIT_CANDIDATE_SCALAR_NAMES):
        raise ValueError("Split v4 dataset summary scalar width is stale")
    source_bindings = summary.get("source_feature_audio_sha256")
    if not isinstance(source_bindings, dict) or not source_bindings:
        raise ValueError("Split v4 dataset summary lacks source feature/audio bindings")
    for field in SPLIT_V4_UPSTREAM_SHA_FIELDS:
        _required_sha256(summary.get(field), field=field)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Split v4 dataset not found: {dataset_path}")
    if _required_sha256(summary.get("dataset_sha256"), field="dataset_sha256") != (
        file_sha256(dataset_path)
    ):
        raise ValueError("Split v4 dataset SHA binding mismatch")
    sidecar = frames_sidecar_path(dataset_path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"Split v4 frame sidecar not found: {sidecar}")
    if _required_sha256(
        summary.get("frame_sidecar_sha256"), field="frame_sidecar_sha256"
    ) != file_sha256(sidecar):
        raise ValueError("Split v4 frame sidecar SHA binding mismatch")
    return {**summary, "summary_path": str(summary_path)}


def validate_source_partition_isolation(data: dict[str, Any]) -> None:
    """Require every island from one source to stay in exactly one partition."""

    if "source_ids" not in data or "core_ids" not in data:
        raise ValueError(
            "Split v4 dataset requires explicit frozen source_ids/core_ids; "
            "legacy group-name inference is rejected"
        )
    source_partitions: dict[str, set[str]] = defaultdict(set)
    core_partitions: dict[str, set[str]] = defaultdict(set)
    core_groups: dict[str, set[str]] = defaultdict(set)
    for group_id, indexes in data["groups"].items():
        partitions = set(data["partitions"][indexes].astype(str).tolist())
        if len(partitions) != 1:
            raise ValueError(f"group {group_id!r} crosses dataset partitions")
        sources = set(data["source_ids"][indexes].astype(str).tolist())
        cores = set(data["core_ids"][indexes].astype(str).tolist())
        if len(sources) != 1 or "" in sources:
            raise ValueError(f"group {group_id!r} has inconsistent source identity")
        if len(cores) != 1 or "" in cores:
            raise ValueError(f"group {group_id!r} has inconsistent core identity")
        source_id = next(iter(sources))
        core_id = next(iter(cores))
        source_partitions[source_id].update(partitions)
        core_partitions[core_id].update(partitions)
        core_groups[core_id].add(str(group_id))
    leaked = {
        source_id: sorted(partitions)
        for source_id, partitions in source_partitions.items()
        if len(partitions) != 1
    }
    if leaked:
        source_id = sorted(leaked)[0]
        raise ValueError(
            f"source {source_id!r} crosses dataset partitions: {leaked[source_id]}"
        )
    leaked_cores = {
        core_id: sorted(partitions)
        for core_id, partitions in core_partitions.items()
        if len(partitions) != 1
    }
    if leaked_cores:
        core_id = sorted(leaked_cores)[0]
        raise ValueError(
            f"core {core_id!r} crosses dataset partitions: {leaked_cores[core_id]}"
        )
    reused = {core_id: groups for core_id, groups in core_groups.items() if len(groups) > 1}
    if reused:
        core_id = sorted(reused)[0]
        raise ValueError(f"core {core_id!r} is reused by multiple Split groups")


def load_island_dataset(
    path: Path,
    *,
    require_training_summary: bool = False,
    expected_ptm_repo_id: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    bundle = load_sequence_arrays(path)
    required = (
        "frame_features",
        "scalar_features",
        "labels",
        "partitions",
        "group_ids",
        "source_ids",
        "core_ids",
    )
    for key in required:
        if key not in bundle:
            raise ValueError(f"sequence dataset missing {key!r}: {path}")
    frames = bundle["frame_features"]
    scalars = np.asarray(bundle["scalar_features"])
    labels = np.asarray(bundle["labels"])
    if frames.ndim != 3 or int(frames.shape[0]) <= 0:
        raise ValueError("Split v4 frame_features must have shape [rows,bins,dim]")
    if scalars.ndim != 2:
        raise ValueError("Split v4 scalar_features must have shape [rows,dim]")
    if labels.ndim != 1:
        raise ValueError("Split v4 labels must have shape [rows]")
    count = int(labels.shape[0])
    if int(frames.shape[0]) != count or int(scalars.shape[0]) != count:
        raise ValueError("Split v4 feature row counts do not match labels")
    for key in ("partitions", "group_ids", "source_ids", "core_ids"):
        values = np.asarray(bundle[key])
        if values.ndim != 1 or int(values.shape[0]) != count:
            raise ValueError(f"Split v4 {key} row count does not match labels")
    try:
        integer_labels = labels.astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Split v4 labels must be integers") from exc
    if not np.array_equal(labels, integer_labels):
        raise ValueError("Split v4 labels contain non-integer values")
    observed_labels = set(integer_labels.tolist())
    if not observed_labels.issubset({0, 1, 2, IGNORE_ID}):
        raise ValueError(f"Split v4 dataset has invalid labels: {sorted(observed_labels)}")
    if not np.isfinite(scalars.astype(np.float32, copy=False)).all():
        raise ValueError("Split v4 scalar features contain non-finite values")
    if int(frames.shape[2]) != SPLIT_V4_PTM_DIM + SPLIT_V4_MFCC_DIM:
        raise ValueError("Split v4 frame width must be raw PTM2048 + MFCC40")
    if int(scalars.shape[1]) != len(SPLIT_CANDIDATE_SCALAR_NAMES):
        raise ValueError("Split v4 scalar feature width is stale")
    for start in range(0, count, 1024):
        frame_block = np.asarray(frames[start : start + 1024], dtype=np.float32)
        if not np.isfinite(frame_block).all():
            raise ValueError("Split v4 frame features contain non-finite values")
    partitions = bundle["partitions"].astype(str)
    if not set(partitions.tolist()).issubset({"train", "val", "test"}):
        raise ValueError("Split v4 dataset has an invalid frozen partition")
    if any(not str(value).strip() for value in bundle["group_ids"].astype(str)):
        raise ValueError("Split v4 dataset has an empty group identity")
    if any(not str(value).strip() for value in bundle["source_ids"].astype(str)):
        raise ValueError("Split v4 dataset has an empty source identity")
    if any(not str(value).strip() for value in bundle["core_ids"].astype(str)):
        raise ValueError("Split v4 dataset has an empty core identity")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(bundle["group_ids"].astype(str).tolist()):
        groups[group_id].append(index)
    if not isinstance(frames, np.memmap):
        frames = frames.astype(np.float32, copy=False)
    result = {
        "frames": frames,
        "scalars": scalars.astype(np.float32, copy=False),
        "labels": integer_labels,
        "partitions": partitions,
        "source_ids": bundle["source_ids"].astype(str),
        "core_ids": bundle["core_ids"].astype(str),
        "dataset_roles": (
            bundle["dataset_roles"].astype(str)
            if "dataset_roles" in bundle
            else np.asarray(["default"] * count)
        ),
        "structural_roles": (
            bundle["structural_roles"].astype(np.int64)
            if "structural_roles" in bundle
            else np.full(count, IGNORE_ID, dtype=np.int64)
        ),
        "pair_ids": (
            bundle["pair_ids"].astype(np.int64)
            if "pair_ids" in bundle
            else np.full(count, -1, dtype=np.int64)
        ),
        "groups": {
            name: np.asarray(indexes, dtype=np.int64)
            for name, indexes in groups.items()
        },
    }
    for key, default in (("structural_roles", IGNORE_ID), ("pair_ids", -1)):
        values = np.asarray(bundle.get(key, np.full(count, default, dtype=np.int64)))
        if values.ndim != 1 or int(values.shape[0]) != count:
            raise ValueError(f"Split v4 {key} shape does not match labels")
        integer_values = values.astype(np.int64, copy=False)
        if not np.array_equal(values, integer_values):
            raise ValueError(f"Split v4 {key} contains non-integer values")
        if key == "structural_roles" and not set(integer_values.tolist()).issubset(
            {IGNORE_ID, 0, 1, 2, 3}
        ):
            raise ValueError("Split v4 structural_roles contains an invalid role")
        if key == "pair_ids" and np.any(integer_values < -1):
            raise ValueError("Split v4 pair_ids must be -1 or non-negative")
    if "dataset_roles" in bundle:
        roles = np.asarray(bundle["dataset_roles"])
        if roles.ndim != 1 or int(roles.shape[0]) != count:
            raise ValueError("Split v4 dataset_roles shape does not match labels")
        if any(not str(value).strip() for value in roles.astype(str)):
            raise ValueError("Split v4 dataset_roles contains an empty role")
    if "times_s" in bundle:
        times = np.asarray(bundle["times_s"], dtype=np.float64)
        if times.shape != (count,) or not np.isfinite(times).all():
            raise ValueError("Split v4 times_s must be finite and match labels")
        for indexes in result["groups"].values():
            group_times = times[indexes]
            if np.any(group_times[1:] <= group_times[:-1]):
                raise ValueError("Split v4 candidate times must increase within groups")
    if require_training_summary:
        summary = load_training_summary(
            path,
            expected_ptm_repo_id=expected_ptm_repo_id,
        )
        if int(summary.get("count") or -1) != count:
            raise ValueError("Split v4 dataset summary row count mismatch")
        if int(summary.get("frame_bins") or -1) != int(frames.shape[1]):
            raise ValueError("Split v4 dataset summary frame-bin mismatch")
        if int(summary.get("frame_dim") or -1) != int(frames.shape[2]):
            raise ValueError("Split v4 dataset summary frame-width mismatch")
        if int(summary.get("scalar_dim") or -1) != int(scalars.shape[1]):
            raise ValueError("Split v4 dataset summary scalar-width mismatch")
        if int(summary.get("group_count") or -1) != len(groups):
            raise ValueError("Split v4 dataset summary group-count mismatch")
        if int(summary.get("source_count") or -1) != len(
            set(result["source_ids"].tolist())
        ):
            raise ValueError("Split v4 dataset summary source-count mismatch")
        if int(summary.get("core_count") or -1) != len(
            set(result["core_ids"].tolist())
        ):
            raise ValueError("Split v4 dataset summary core-count mismatch")
        result["training_summary"] = summary
    return result


def island_batches(
    names: list[str],
    groups: dict[str, np.ndarray],
    *,
    batch_islands: int,
    max_batch_candidates: int,
) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    candidates = 0
    for name in names:
        count = int(groups[name].size)
        if current and (
            len(current) >= batch_islands
            or candidates + count > max_batch_candidates
        ):
            batches.append(current)
            current = []
            candidates = 0
        current.append(name)
        candidates += count
    if current:
        batches.append(current)
    return batches


def pad_batch(
    data: dict[str, Any],
    names: list[str],
    *,
    frame_mean: np.ndarray,
    frame_std: np.ndarray,
    scalar_mean: np.ndarray,
    scalar_std: np.ndarray,
):
    import torch

    groups = data["groups"]
    counts = [int(groups[name].size) for name in names]
    max_count = max(counts)
    bins = int(data["frames"].shape[1])
    frame_dim = int(data["frames"].shape[2])
    scalar_dim = int(data["scalars"].shape[1])
    frames = np.zeros((len(names), max_count, bins, frame_dim), dtype=np.float32)
    scalars = np.zeros((len(names), max_count, scalar_dim), dtype=np.float32)
    mask = np.zeros((len(names), max_count), dtype=np.int64)
    labels = np.full((len(names), max_count), IGNORE_ID, dtype=np.int64)
    roles = np.full((len(names), max_count), IGNORE_ID, dtype=np.int64)
    pairs = np.full((len(names), max_count), -1, dtype=np.int64)
    for row, name in enumerate(names):
        indexes = groups[name]
        count = int(indexes.size)
        frames[row, :count] = (data["frames"][indexes] - frame_mean) / frame_std
        scalars[row, :count] = (data["scalars"][indexes] - scalar_mean) / scalar_std
        mask[row, :count] = 1
        labels[row, :count] = data["labels"][indexes]
        roles[row, :count] = data["structural_roles"][indexes]
        pairs[row, :count] = data["pair_ids"][indexes]
    return tuple(
        torch.from_numpy(value)
        for value in (frames, scalars, mask, labels, roles, pairs)
    )


def pair_loss(gate_probabilities, labels, pairs):
    """Soft-AND objective on the two cuts isolating one background run."""

    import torch

    losses = []
    for row in range(pairs.shape[0]):
        by_pair: dict[int, list[int]] = defaultdict(list)
        for position in range(pairs.shape[1]):
            pair_id = int(pairs[row, position])
            if pair_id >= 0 and int(labels[row, position]) == 0:
                by_pair[pair_id].append(position)
        for positions in by_pair.values():
            if len(positions) < 2:
                continue
            first, second = positions[:2]
            losses.append(
                1.0
                - gate_probabilities[row, first]
                * gate_probabilities[row, second]
            )
    return torch.stack(losses).mean() if losses else None
