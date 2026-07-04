#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.asr.cueqc.compile_pre_asr_v11_features import (  # noqa: E402
    compile_features,
)


SPLIT_LABEL_IDS = {"cut": 0, "continue": 1, "unsure": 2}


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


def _compile_split(
    *,
    dataset: Path,
    windows: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    val_percent: int,
) -> dict[str, Any]:
    window_by_id = {str(row["window_id"]): row for row in windows}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in labels:
        if str(row.get("label") or "") in SPLIT_LABEL_IDS:
            grouped.setdefault(str(row["window_id"]), []).append(row)
    frame_parts: list[np.ndarray] = []
    scalar_parts: list[np.ndarray] = []
    label_parts: list[int] = []
    partition_parts: list[str] = []
    window_parts: list[str] = []
    video_parts: list[str] = []
    feature_index_parts: list[int] = []
    time_parts: list[float] = []
    for window_id, rows in sorted(grouped.items()):
        window = window_by_id.get(window_id)
        if window is None:
            continue
        bundle = np.load(window["semantic_split_features"])
        video_id = str(window["video_id"])
        partition = _partition(video_id, val_percent)
        for row in rows:
            index = int(row["feature_index"])
            if index < 0 or index >= int(bundle["frame_features"].shape[0]):
                raise IndexError(
                    f"semantic split feature index {index} out of range for {window_id}"
                )
            frame_parts.append(bundle["frame_features"][index].astype(np.float32))
            scalar_parts.append(bundle["scalar_features"][index].astype(np.float32))
            label_parts.append(SPLIT_LABEL_IDS[str(row["label"])])
            partition_parts.append(partition)
            window_parts.append(window_id)
            video_parts.append(video_id)
            feature_index_parts.append(index)
            time_parts.append(float(row["time_s"]))
    if not label_parts:
        raise ValueError("no joint semantic split labels found")
    output = dataset / "semantic_split" / "features.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frame_features=np.stack(frame_parts),
        scalar_features=np.stack(scalar_parts),
        labels=np.asarray(label_parts, dtype=np.int64),
        partitions=np.asarray(partition_parts),
        window_ids=np.asarray(window_parts),
        video_ids=np.asarray(video_parts),
        feature_indexes=np.asarray(feature_index_parts, dtype=np.int64),
        times_s=np.asarray(time_parts, dtype=np.float32),
    )
    counts = Counter(str(row["label"]) for row in labels)
    partitions = Counter(partition_parts)
    summary = {
        "schema": "joint_semantic_split_dataset_v1",
        "output": str(output.resolve()),
        "count": len(label_parts),
        "labels": dict(counts),
        "partitions": dict(partitions),
        "video_count": len(set(video_parts)),
        "window_count": len(set(window_parts)),
        "partition_unit": "video_id",
        "val_percent": val_percent,
    }
    _write_json(dataset / "semantic_split" / "summary.json", summary)
    return summary


def _compile_pre_asr(
    *,
    dataset: Path,
    windows: list[dict[str, Any]],
    labels_path: Path,
    asr_repo_id: str,
    val_percent: int,
) -> dict[str, Any]:
    chunk_paths = [
        str(Path(row["pre_asr_candidates"]))
        for row in windows
        if row.get("pre_asr_candidates")
        and Path(row["pre_asr_candidates"]).exists()
    ]
    if not chunk_paths:
        raise ValueError("no Pre-ASR candidate files found")
    output = dataset / "pre_asr" / "features.pt"
    summary = compile_features(
        chunk_paths=chunk_paths,
        label_paths=[str(labels_path)],
        output=output,
        asr_repo_id=asr_repo_id,
    )
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
        "- `pre_asr/`: keep/drop/unsure 标签、分类 WAV 切片与 v11 `features.pt`。",
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
    )
    pre_asr_summary = _compile_pre_asr(
        dataset=dataset,
        windows=windows,
        labels_path=pre_asr_labels_path,
        asr_repo_id=args.asr_repo_id,
        val_percent=args.val_percent,
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
    args = parser.parse_args()
    if not 1 <= args.val_percent <= 50:
        parser.error("--val-percent must be between 1 and 50")
    return args


if __name__ == "__main__":
    run(parse_args())
