#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from boundary.sequence_store import (  # noqa: E402
    StreamingFrameWriter,
    save_sequence_dataset,
)
from tools.asr.cueqc.compile_pre_asr_v12_features import (  # noqa: E402
    compile_features,
    normalize_label,
)


SPLIT_LABEL_IDS = {"cut": 0, "continue": 1, "unsure": 2}
IGNORE_ID = -100


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _partition(video_id: str, val_percent: int) -> str:
    bucket = int(hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "val" if bucket < val_percent else "train"


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
    val_percent: int,
    load_workers: int = 6,
) -> dict[str, Any]:
    """Emit whole-island candidate sequences.

    Every candidate of an island that carries at least one Omni label is
    included so the island-sequence model sees the same candidate context as
    runtime; candidates without a label use ignore id ``-100`` and omni aux
    ``-1`` so per-candidate losses can mask them.
    """

    window_by_id = {str(row["window_id"]): row for row in windows}
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for row in labels:
        if str(row.get("label") or "") in SPLIT_LABEL_IDS:
            grouped.setdefault(str(row["window_id"]), {})[
                int(row["feature_index"])
            ] = row
    output = dataset / "semantic_split" / "features.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Frames stream to the sidecar .frames.npy: a full-dim (2048-PTM) compile
    # cannot hold an np.stack of every row on a 16GB box.
    frame_writer = StreamingFrameWriter(output)
    scalar_parts: list[np.ndarray] = []
    label_parts: list[int] = []
    partition_parts: list[str] = []
    window_parts: list[str] = []
    video_parts: list[str] = []
    feature_index_parts: list[int] = []
    time_parts: list[float] = []
    group_parts: list[str] = []
    omni_parts: list[list[float]] = []
    labeled_count = 0
    ordered: list[tuple[str, dict[int, dict[str, Any]], dict[str, Any]]] = []
    for window_id, labeled_rows in sorted(grouped.items()):
        window = window_by_id.get(window_id)
        if window is None:
            continue
        ordered.append((window_id, labeled_rows, window))

    def _load_window_arrays(window: dict[str, Any]) -> dict[str, np.ndarray]:
        # Per-window npz are zlib-compressed; decompressing in worker threads
        # (zlib releases the GIL) keeps the single writer thread fed.
        with np.load(window["semantic_split_features"]) as handle:
            return {key: np.asarray(handle[key]) for key in handle.files}

    pool = ThreadPoolExecutor(max_workers=load_workers)
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
        bundle = future.result()
        _submit_next()
        video_id = str(window["video_id"])
        partition = _partition(video_id, val_percent)
        total = int(bundle["frame_features"].shape[0])
        for index in labeled_rows:
            if index < 0 or index >= total:
                raise IndexError(
                    f"semantic split feature index {index} out of range for {window_id}"
                )
        core_starts = np.asarray(bundle["core_starts_s"], dtype=np.float64)
        core_ends = np.asarray(bundle["core_ends_s"], dtype=np.float64)
        times = np.asarray(bundle["proposal_times_s"], dtype=np.float64)
        island_members: dict[tuple[float, float], list[int]] = {}
        for index in range(total):
            key = (round(float(core_starts[index]), 6), round(float(core_ends[index]), 6))
            island_members.setdefault(key, []).append(index)
        for key, members in sorted(island_members.items()):
            if not any(index in labeled_rows for index in members):
                continue
            group_id = f"{window_id}|core{key[0]:.3f}-{key[1]:.3f}"
            for index in sorted(members, key=lambda item: float(times[item])):
                labeled = labeled_rows.get(index)
                frame_writer.append(bundle["frame_features"][index].astype(np.float32))
                scalar_parts.append(bundle["scalar_features"][index].astype(np.float32))
                if labeled is None:
                    label_parts.append(IGNORE_ID)
                    omni_parts.append([-1.0, -1.0, -1.0])
                else:
                    label_parts.append(SPLIT_LABEL_IDS[str(labeled["label"])])
                    omni_parts.append(_omni_aux_row(labeled))
                    labeled_count += 1
                partition_parts.append(partition)
                window_parts.append(window_id)
                video_parts.append(video_id)
                feature_index_parts.append(index)
                time_parts.append(float(times[index]))
                group_parts.append(group_id)
    pool.shutdown()
    if labeled_count <= 0:
        raise ValueError("no joint semantic split labels found")
    frame_writer.finalize()
    save_sequence_dataset(
        output,
        frames_finalized=True,
        compress=True,
        scalar_features=np.stack(scalar_parts),
        labels=np.asarray(label_parts, dtype=np.int64),
        partitions=np.asarray(partition_parts),
        window_ids=np.asarray(window_parts),
        video_ids=np.asarray(video_parts),
        feature_indexes=np.asarray(feature_index_parts, dtype=np.int64),
        times_s=np.asarray(time_parts, dtype=np.float32),
        group_ids=np.asarray(group_parts),
        structural_roles=np.full(len(label_parts), IGNORE_ID, dtype=np.int64),
        pair_ids=np.full(len(label_parts), -1, dtype=np.int64),
        omni_aux=np.asarray(omni_parts, dtype=np.float32),
    )
    counts = Counter(str(row["label"]) for row in labels)
    partitions = Counter(partition_parts)
    summary = {
        "schema": "joint_semantic_split_dataset_v2",
        "output": str(output.resolve()),
        "count": len(label_parts),
        "labeled_count": labeled_count,
        "context_only_count": len(label_parts) - labeled_count,
        "group_count": len(set(group_parts)),
        "labels": dict(counts),
        "partitions": dict(partitions),
        "video_count": len(set(video_parts)),
        "window_count": len(set(window_parts)),
        "partition_unit": "video_id",
        "val_percent": val_percent,
    }
    _write_json(dataset / "semantic_split" / "summary.json", summary)
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
    val_percent: int,
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
        str(row["window_id"]): _partition(str(row["video_id"]), val_percent)
        for row in windows
    }
    for group in payload["groups"]:
        group["dataset_role"] = role_by_audio_id.get(
            str(group.get("audio_id") or ""),
            "train",
        )
    torch.save(payload, output)
    role_counts = Counter(
        str(group.get("dataset_role") or "")
        for group in payload["groups"]
    )
    summary["dataset_roles"] = dict(role_counts)
    summary["partition_unit"] = "video_id"
    summary["val_percent"] = val_percent
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
        "训练/验证划分按 `video_id` 进行，避免同一视频的多个随机窗口跨集合泄漏。",
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
    if not pre_asr_labels_path.exists():
        raise ValueError("pre_asr/labels.jsonl is missing")
    split_summary = _compile_split(
        dataset=dataset,
        windows=windows,
        labels=split_labels,
        val_percent=args.val_percent,
        load_workers=args.load_workers,
    )
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
        val_percent=args.val_percent,
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
    parser.add_argument("--val-percent", type=int, default=20)
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
    if not 1 <= args.val_percent <= 50:
        parser.error("--val-percent must be between 1 and 50")
    return args


if __name__ == "__main__":
    run(parse_args())
