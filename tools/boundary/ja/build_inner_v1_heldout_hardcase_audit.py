#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.boundary.ja.build_inner_subisland_edge_audit import (  # noqa: E402
    ITEM_SCHEMA,
    build_page,
)
from tools.boundary.ja.build_outer_v2_noisy_edge_fixed5 import (  # noqa: E402
    canonical_negative_categories,
)


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_hardcases(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()

    def add(role: str, ordered: list[dict[str, Any]]) -> None:
        for row in ordered:
            audio_id = str(row["audio_id"])
            if audio_id not in used:
                used.add(audio_id)
                selected.append((role, row))
                return

    add(
        "worst_start_inward",
        sorted(rows, key=lambda row: float(row["start_inward_s"]), reverse=True),
    )
    add(
        "worst_end_inward",
        sorted(rows, key=lambda row: float(row["end_inward_s"]), reverse=True),
    )
    add(
        "next_worst_inward",
        sorted(
            rows,
            key=lambda row: max(
                float(row["start_inward_s"]), float(row["end_inward_s"])
            ),
            reverse=True,
        ),
    )
    add(
        "worst_start_outward",
        sorted(rows, key=lambda row: float(row["start_outward_s"]), reverse=True),
    )
    abstain = [row for row in rows if str(row.get("abstain_reason") or "")]
    add(
        "model_abstain" if abstain else "worst_end_outward",
        abstain
        or sorted(rows, key=lambda row: float(row["end_outward_s"]), reverse=True),
    )
    if len(selected) != 5:
        raise ValueError(
            f"held-out hard-case audit requires 5 unique rows; got {len(selected)}"
        )
    return selected


def _edge_noise(
    detail: dict[str, Any], negative_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    sources = list(detail.get("sources") or [])
    speech_indexes = [
        index for index, source in enumerate(sources) if source.get("source_audio_id")
    ]
    if not speech_indexes:
        return {}
    first, last = min(speech_indexes), max(speech_indexes)

    def side_categories(items: list[dict[str, Any]]) -> str:
        categories: set[str] = set()
        for source in items:
            negative = negative_by_id.get(str(source.get("audio_id") or ""))
            if negative is not None:
                categories.update(canonical_negative_categories(negative))
        return ",".join(sorted(categories)) or "other"

    return {
        "leading": {"background_type": side_categories(sources[:first])},
        "trailing": {"background_type": side_categories(sources[last + 1 :])},
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    selected = select_hardcases(_rows(Path(args.directional_details)))
    labels_by_id = {str(row["audio_id"]): row for row in _rows(Path(args.labels))}
    details_by_id = {
        str(row["audio_id"]): row for row in _rows(Path(args.synthetic_details))
    }
    negative_by_id = {
        str(row["audio_id"]): row for row in _rows(Path(args.negative_manifest))
    }
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for role, row in selected:
        audio_id = str(row["audio_id"])
        label = labels_by_id[audio_id]
        source_audio = Path(str(row["source_audio"]))
        target_audio = audio_dir / source_audio.name
        shutil.copyfile(source_audio, target_audio)
        prediction = {
            "raw_start_s": 0.0,
            "raw_end_s": float(row["raw_end_s"]),
            "start_s": float(row["predicted_start_s"]),
            "end_s": float(row["predicted_end_s"]),
            "start_action": str(row["start_action"]),
            "end_action": str(row["end_action"]),
            "abstain_reason": str(row.get("abstain_reason") or ""),
        }
        items.append(
            {
                "schema": ITEM_SCHEMA,
                "audit_mode": "formal_inner_v1_heldout_hardcase_v1",
                "selection_role": role,
                "sample_id": audio_id,
                "subisland_id": f"{audio_id}__inner",
                "audio": target_audio.relative_to(output_dir).as_posix(),
                "source_duration_s": float(row["raw_end_s"]),
                "raw_start_s": 0.0,
                "raw_end_s": float(row["raw_end_s"]),
                "refined_start_s": float(row["predicted_start_s"]),
                "refined_end_s": float(row["predicted_end_s"]),
                "start_requires_inner": True,
                "end_requires_inner": True,
                "reference_text": str(label.get("text") or ""),
                "known_semantic_core_span": {
                    "start_s": float(row["truth_start_s"]),
                    "end_s": float(row["truth_end_s"]),
                },
                "edge_noise": _edge_noise(details_by_id[audio_id], negative_by_id),
                "model_prediction": prediction,
                "checkpoint_sha256": str(args.checkpoint_sha256),
                "teacher_usage": "formal_inner_model_heldout_evaluation",
            }
        )
    items_path = output_dir / "inner_items.jsonl"
    items_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items),
        encoding="utf-8",
    )
    page = build_page(
        rows=items,
        output_dir=output_dir,
        update_latest=not args.no_update_latest,
        noisy_edge_mode=True,
        formal_inner_mode=True,
    )
    summary = {
        "schema": "inner_v1_heldout_hardcase_audit_summary_v1",
        "sample_count": len(items),
        "selection_roles": [row["selection_role"] for row in items],
        "checkpoint_sha256": str(args.checkpoint_sha256),
        "items": str(items_path),
        "page": str(page),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build formal Inner v1 held-out worst-5 audit."
    )
    parser.add_argument("--directional-details", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--synthetic-details", required=True)
    parser.add_argument("--negative-manifest", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False))
