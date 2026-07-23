#!/usr/bin/env python3
"""Generate a model-independent held-out candidate-membership source audit."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import wave
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.ja.dataset import frame_count  # noqa: E402
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.audits.generate_scorer_v10_full_source_span_audit_html import (  # noqa: E402
    MANUAL_VERDICT_SCHEMA as LEGACY_MANUAL_VERDICT_SCHEMA,
    _render_page as _render_legacy_full_source_page,
)


FRAME_HOP_S = 0.02
SUMMARY_SCHEMA = "candidate_island_scorer_v11_heldout_audit_summary_v1"
ITEM_SCHEMA = "candidate_island_scorer_v11_heldout_audit_item_v1"
MANUAL_VERDICT_SCHEMA = (
    "candidate_island_scorer_v11_heldout_manual_verdict_v1"
)
PARTITION_SCHEMA = "candidate_island_scorer_v11_partition_manifest_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wav_duration(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        samples = int(handle.getnframes())
    if sample_rate != 16000 or channels != 1 or samples <= 0:
        raise ValueError(f"held-out audit audio must be non-empty 16k mono WAV: {path}")
    duration_s = samples / sample_rate
    return duration_s, frame_count(duration_s, FRAME_HOP_S)


def _candidate_page(rows: list[dict[str, Any]]) -> str:
    page = _render_legacy_full_source_page(rows)
    page = page.replace(
        LEGACY_MANUAL_VERDICT_SCHEMA,
        MANUAL_VERDICT_SCHEMA,
    )
    page = page.replace(
        "complete_with_target_speech",
        "complete_with_target_inside_candidate",
    )
    page = page.replace(
        "complete_all_background",
        "complete_all_outside_candidate",
    )
    # Replace label tokens only. A raw substring replacement would also mutate
    # unrelated schema/verdict identifiers when the legacy template evolves.
    page = re.sub(r"(?<![A-Za-z0-9_])speech(?![A-Za-z0-9_])", "inside_candidate", page)
    page = re.sub(
        r"(?<![A-Za-z0-9_])background(?![A-Za-z0-9_])",
        "outside_candidate",
        page,
    )
    # ``background`` is both a legacy label token and a CSS property.  The
    # label rewrite above intentionally updates class/variable names, but must
    # not turn declarations such as ``background:#fff`` into the invalid
    # ``outside_candidate:#fff``.  Repair property names only inside the style
    # block while preserving the renamed ``--outside_candidate`` variable.
    style, separator, remainder = page.partition("</style>")
    if not separator:
        raise ValueError("candidate audit template is missing its style block")
    style = re.sub(r"(?<!-)outside_candidate:", "background:", style)
    page = style + separator + remainder
    page = page.replace(
        "Scorer v10 full-source truth repair",
        "Scorer v11 held-out candidate membership",
    )
    page = page.replace(
        "1.7B Scorer v10 · 完整 source 真值修复",
        "1.7B Scorer v11 · held-out candidate membership",
    )
    page = page.replace(
        "添加所有目标语音区间",
        "添加所有应继续送往 Proposal/Outer/Split/CueQC 的 inside_candidate 区间",
    )
    page = page.replace(
        "没有显式区间即表示人工复核为全段 outside_candidate",
        "没有显式区间即表示人工复核为全段可安全删除的 outside_candidate",
    )
    page = page.replace(
        "只允许显式 inside_candidate/unsure 区间",
        "只允许显式 inside_candidate/unsure 区间；同一 ASR 单元内部停顿和尾音也属于 inside_candidate",
    )
    outside_marker = (
        "<div><b>outside_candidate 合同：</b>未标出的差集只有在勾选“已从头听到尾”后才成为 "
        "outside_candidate。unsure 会保留在 canonical，并在 normalization、split、loss、metrics "
        "和 gate 中映射为 ignore=-100。</div>"
    )
    if outside_marker not in page:
        raise ValueError("candidate audit template outside-candidate guidance changed")
    page = page.replace(
        outside_marker,
        outside_marker
        + "\n  <div><b>黄色 outside_candidate 检查：</b>点击黄色条即可精确播放该区间。"
        "黄色只有在确认不含词语或对白、并且能独立于同一轮对白波形安全删除时才成立；"
        "若疑似存在短词、尾音、含混对白，或删除会把同一轮对白割开，应改为 inside_candidate 或 unsure。"
        "独立的纯呻吟、喘息可以标 outside；夹在同一对白包络内的声音应随 inside_candidate 保留。"
        "CueQC 可以丢弃已切出的整段非语义 sub-island，Inner 主要负责已保留 sub-island 的首尾 acoustic core，"
        "不是内部噪声分离器；不要把下游能力当作提前删除 Scorer 语音的理由，也不要为了提前清理而牺牲 Scorer 的 inside recall。</div>",
        1,
    )
    return page


def build_audit(
    *,
    source_windows: Path,
    feature_bundle: Path,
    val_video_ids: list[str],
    test_video_ids: list[str],
    output_dir: Path,
) -> Path:
    import torch

    windows = _rows(source_windows)
    by_window = {str(row.get("window_id") or ""): row for row in windows}
    if len(by_window) != len(windows) or "" in by_window:
        raise ValueError("source windows require unique non-empty window_id")
    bundle = torch.load(feature_bundle, map_location="cpu", weights_only=False)
    groups = list(bundle.get("groups") or ())
    if not groups:
        raise ValueError("pre-ASR feature bundle has no groups")

    val_ids = set(val_video_ids)
    test_ids = set(test_video_ids)
    if not val_ids or not test_ids or val_ids & test_ids:
        raise ValueError("val/test video ids must be non-empty and disjoint")
    heldout_videos: set[str] = set()
    partition_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for group in groups:
        window_id = str(group.get("audio_id") or "")
        source = by_window.get(window_id)
        if source is None:
            raise ValueError(f"feature group source window is missing: {window_id}")
        video_id = str(source.get("video_id") or "")
        original_role = str(group.get("dataset_role") or "")
        if original_role == "train":
            partition = "train"
        elif original_role == "val" and video_id in val_ids:
            partition = "val"
            heldout_videos.add(video_id)
        elif original_role == "val" and video_id in test_ids:
            partition = "test"
            heldout_videos.add(video_id)
        else:
            raise ValueError(
                f"held-out video partition is not frozen: {window_id}:{video_id}"
            )
        partition_rows.append(
            {
                "schema": PARTITION_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "source_id": window_id,
                "video_id": video_id,
                "partition": partition,
                "original_dataset_role": original_role,
            }
        )
        if partition in {"val", "test"}:
            selected.append({**source, "partition": partition})
    expected_heldout = val_ids | test_ids
    if heldout_videos != expected_heldout:
        raise ValueError(
            f"held-out video ids mismatch: missing={sorted(expected_heldout-heldout_videos)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    partition_manifest = output_dir / "partition_manifest.jsonl"
    partition_manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in sorted(partition_rows, key=lambda row: row["source_id"])
        ),
        encoding="utf-8",
    )

    payload: list[dict[str, Any]] = []
    selected.sort(key=lambda row: (row["partition"], str(row["window_id"])))
    for index, row in enumerate(selected):
        source = Path(str(row.get("audio_wav") or ""))
        if not source.is_file():
            raise ValueError(f"held-out source audio is missing: {source}")
        destination = audio_dir / f"source-{index:03d}.wav"
        shutil.copy2(source, destination)
        duration_s, total_frames = _wav_duration(source)
        payload.append(
            {
                "schema": ITEM_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "source_id": str(row["window_id"]),
                "video_id": str(row["video_id"]),
                "partition": str(row["partition"]),
                "frame_count": total_frames,
                "frame_hop_s": FRAME_HOP_S,
                "duration_s": duration_s,
                "audio": destination.relative_to(output_dir).as_posix(),
                "audio_size_bytes": destination.stat().st_size,
                "audio_sha256": _sha256(destination),
            }
        )

    manifest = output_dir / "audit_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload),
        encoding="utf-8",
    )
    index = output_dir / "index.html"
    index.write_text(_candidate_page(payload), encoding="utf-8")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "title": "Scorer v11 held-out candidate membership",
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "source_windows": str(source_windows),
        "source_windows_sha256": _sha256(source_windows),
        "feature_bundle": str(feature_bundle),
        "feature_bundle_sha256": _sha256(feature_bundle),
        "partition_manifest": str(partition_manifest),
        "partition_manifest_sha256": _sha256(partition_manifest),
        "source_count": len(payload),
        "partition_counts": dict(Counter(row["partition"] for row in payload)),
        "val_video_ids": sorted(val_ids),
        "test_video_ids": sorted(test_ids),
        "frame_hop_s": FRAME_HOP_S,
        "audit_manifest": str(manifest),
        "audit_manifest_sha256": _sha256(manifest),
        "manual_verdict_schema": MANUAL_VERDICT_SCHEMA,
        "manual_gate_status": "pending",
        "model_output_used_as_annotation_seed": False,
        "asr_output_used_as_annotation_seed": False,
        "cueqc_output_used_as_annotation_seed": False,
        "unmarked_complement_becomes_outside_candidate_only_after_full_source_confirmation": True,
        "unsure_training_label": -100,
        "training_manifest_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_audit_entrypoints(latest_html=index, title=summary["title"])
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-windows", required=True)
    parser.add_argument("--feature-bundle", required=True)
    parser.add_argument("--val-video-id", action="append", default=[])
    parser.add_argument("--test-video-id", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        build_audit(
            source_windows=Path(args.source_windows),
            feature_bundle=Path(args.feature_bundle),
            val_video_ids=list(args.val_video_id),
            test_video_ids=list(args.test_video_id),
            output_dir=Path(args.output_dir),
        )
    )
