#!/usr/bin/env python3
"""Generate a one-source-per-video editable Scorer v11 train audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.audits.generate_candidate_island_v11_heldout_audit_html import (  # noqa: E402
    MANUAL_VERDICT_SCHEMA as HELDOUT_VERDICT_SCHEMA,
    _candidate_page,
)


SOURCE_SCHEMA = "candidate_island_scorer_v11_train_teacher_source_v1"
PREAUDIT_SCHEMA = "candidate_island_scorer_v11_omni_preaudit_v2"
ITEM_SCHEMA = "candidate_island_scorer_v11_train_teacher_review_item_v1"
SUMMARY_SCHEMA = "candidate_island_scorer_v11_train_teacher_review_summary_v1"
MANUAL_VERDICT_SCHEMA = "candidate_island_scorer_v11_train_manual_verdict_v1"
FRAME_HOP_S = 0.02


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index(rows: list[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in result:
            raise ValueError(f"{name} requires unique non-empty source_id")
        result[source_id] = row
    return result


def _preaudit_ranges(row: dict[str, Any]) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    frame_count = int(row["frame_count"])
    for label, field in (("inside_candidate", "islands"), ("unsure", "unsure_spans")):
        for index, span in enumerate(row.get(field) or ()):
            start_frame = max(0, min(frame_count, int(span["start_frame"])))
            end_frame = max(0, min(frame_count, int(span["end_frame"])))
            if end_frame <= start_frame:
                continue
            ranges.append(
                {
                    "id": f"gemini-{label}-{index}",
                    "label": label,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )
    ranges.sort(key=lambda span: (span["start_frame"], span["end_frame"], span["label"]))
    previous = 0
    for span in ranges:
        if (
            span["start_frame"] < previous
            or span["end_frame"] <= span["start_frame"]
            or span["end_frame"] > frame_count
        ):
            raise ValueError(f"Gemini preaudit has invalid/overlapping ranges: {row['source_id']}")
        previous = span["end_frame"]
    return ranges


def _inside_ratio(row: dict[str, Any]) -> float:
    frame_count = max(1, int(row["frame_count"]))
    inside = sum(
        max(
            0,
            min(frame_count, int(span["end_frame"]))
            - max(0, min(frame_count, int(span["start_frame"]))),
        )
        for span in row.get("islands") or ()
    )
    return inside / frame_count


def _page(payload: list[dict[str, Any]], initial: dict[str, Any]) -> str:
    page = _candidate_page(payload)
    page = page.replace(HELDOUT_VERDICT_SCHEMA, MANUAL_VERDICT_SCHEMA)
    page = page.replace(
        "Scorer v11 held-out candidate membership",
        "Scorer v11 train full-source human review",
    )
    page = page.replace(
        "1.7B Scorer v11 · held-out candidate membership",
        "1.7B Scorer v11 · train full-source human review",
    )
    page = page.replace(
        "<style>",
        "<style>\n.preaudit-note{margin:8px 0;padding:8px 10px;border-radius:8px;"
        "background:#fff3cd;border:1px solid #d6aa2f}",
        1,
    )
    page = page.replace(
        "<section>\n  <div><b>本页不显示模型输出：</b>",
        "<section>\n  <div><b>Gemini 仅是可编辑底稿：</b>每个 train video 只选一条 source；"
        "必须人工从头听到尾并确认，未确认结果禁止进入 canonical/training truth。</div>"
        "\n  <div><b>本页不显示模型输出：</b>",
        1,
    )
    old_ann = (
        "let ann={};try{ann=JSON.parse(localStorage.getItem(key)||'{}');}"
        "catch(_error){ann={};}"
    )
    encoded_initial = (
        json.dumps(initial, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    new_ann = (
        f"const initialAnn={encoded_initial};let ann={{...initialAnn}};"
        "try{ann={...initialAnn,...JSON.parse(localStorage.getItem(key)||'{}')};}"
        "catch(_error){ann={...initialAnn};}"
    )
    if old_ann not in page:
        raise ValueError("candidate audit template annotation state changed")
    page = page.replace(old_ann, new_ann, 1)
    old_header = (
        '<small>${esc(row.partition)} / ${row.frame_count} frames / '
        '${Number(row.duration_s).toFixed(2)}s / 当前=${esc(currentVerdict)}</small>'
        '<audio controls preload="none" src="${esc(row.audio)}"></audio>'
    )
    new_header = (
        '<small>${esc(row.partition)} / ${esc(row.video_id)} / ${row.frame_count} frames / '
        '${Number(row.duration_s).toFixed(2)}s / 当前=${esc(currentVerdict)}</small>'
        '<div class="preaudit-note"><b>Gemini 可编辑底稿（未确认）</b> / '
        'inside=${(100*Number(row.preaudit_inside_ratio)).toFixed(1)}% / '
        'confidence=${Number(row.preaudit_confidence).toFixed(2)}<br>'
        '${esc(row.preaudit_reason||"无说明")}</div>'
        '<audio controls preload="none" src="${esc(row.audio)}"></audio>'
    )
    if old_header not in page:
        raise ValueError("candidate audit template card header changed")
    return page.replace(old_header, new_header, 1)


def build(
    *,
    source_manifest: Path,
    preaudit: Path,
    output_dir: Path,
    exclude_sources: Path | None = None,
    target_inside_ratio: float = 0.35,
) -> dict[str, Any]:
    source_manifest = source_manifest.resolve()
    preaudit = preaudit.resolve()
    sources = _index(_rows(source_manifest), name="train teacher manifest")
    drafts = _index(_rows(preaudit), name="Gemini preaudit")
    if set(sources) != set(drafts):
        raise ValueError("Gemini preaudit must cover the exact train teacher manifest")
    excluded: set[str] = set()
    if exclude_sources is not None:
        exclude_sources = exclude_sources.resolve()
        excluded = {
            str(row.get("source_id") or "")
            for row in _rows(exclude_sources)
            if str(row.get("source_id") or "")
        }
    by_video: dict[str, list[str]] = defaultdict(list)
    for source_id, source in sources.items():
        if source.get("schema") != SOURCE_SCHEMA or source.get("partition") != "train":
            raise ValueError(f"invalid train teacher source: {source_id}")
        if source.get("boundary_serialization_contract_id") != ACOUSTIC_BINARY_V12_CONTRACT.contract_id:
            raise ValueError(f"wrong central boundary contract: {source_id}")
        video_id = str(source.get("video_id") or "")
        if not video_id:
            raise ValueError(f"train teacher source lacks video identity: {source_id}")
        if source_id not in excluded:
            by_video[video_id].append(source_id)
    all_videos = {str(source.get("video_id") or "") for source in sources.values()}
    if set(by_video) != all_videos:
        raise ValueError("source exclusion removed every candidate from a train video")

    selected: list[str] = []
    for video_id in sorted(by_video):
        candidates = by_video[video_id]
        candidates.sort(
            key=lambda source_id: (
                abs(_inside_ratio(drafts[source_id]) - target_inside_ratio),
                _inside_ratio(drafts[source_id]),
                source_id,
            )
        )
        selected.append(candidates[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    initial: dict[str, Any] = {}
    for index, source_id in enumerate(selected):
        source = sources[source_id]
        draft = drafts[source_id]
        if draft.get("schema") != PREAUDIT_SCHEMA or draft.get("partition") != "train":
            raise ValueError(f"invalid Gemini preaudit: {source_id}")
        if str(draft.get("audio_sha256") or "") != str(source.get("audio_sha256") or ""):
            raise ValueError(f"Gemini/source audio identity mismatch: {source_id}")
        if int(draft.get("frame_count") or 0) != int(source.get("frame_count") or 0):
            raise ValueError(f"Gemini/source frame geometry mismatch: {source_id}")
        audio = Path(str(source.get("audio") or ""))
        if not audio.is_absolute():
            audio = (PROJECT_ROOT / audio).resolve()
        if not audio.is_file() or _sha256(audio) != str(source.get("audio_sha256") or ""):
            raise ValueError(f"train teacher audio SHA mismatch: {source_id}")
        target = audio_dir / f"source-{index:03d}.wav"
        shutil.copy2(audio, target)
        ranges = _preaudit_ranges(draft)
        initial[source_id] = {
            "ranges": ranges,
            "reviewed_full_source": False,
            "updated_at": "",
        }
        payload.append(
            {
                "schema": ITEM_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "video_id": str(source["video_id"]),
                "partition": "train",
                "frame_count": int(source["frame_count"]),
                "frame_hop_s": FRAME_HOP_S,
                "duration_s": float(source["duration_s"]),
                "audio": target.relative_to(output_dir).as_posix(),
                "audio_sha256": str(source["audio_sha256"]),
                "preaudit_model": str(draft.get("model") or ""),
                "preaudit_confidence": float(draft.get("overall_confidence") or 0.0),
                "preaudit_inside_ratio": _inside_ratio(draft),
                "preaudit_reason": str(draft.get("overall_reason") or ""),
                "materialize_unreviewed_ranges": True,
            }
        )

    manifest = output_dir / "audit_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload),
        encoding="utf-8",
    )
    index = output_dir / "index.html"
    index.write_text(_page(payload, initial), encoding="utf-8")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "preaudit": str(preaudit),
        "preaudit_sha256": _sha256(preaudit),
        "exclude_sources": str(exclude_sources) if exclude_sources else None,
        "exclude_sources_sha256": _sha256(exclude_sources) if exclude_sources else None,
        "selection_policy": "one_per_video_closest_to_target_gemini_inside_ratio_v1",
        "target_inside_ratio": target_inside_ratio,
        "source_count": len(payload),
        "video_count": len({row["video_id"] for row in payload}),
        "selected_source_ids": selected,
        "preaudit_inside_ratio_bins": dict(
            Counter(
                "full" if row["preaudit_inside_ratio"] >= 0.999
                else "mixed" if row["preaudit_inside_ratio"] > 0.0
                else "empty"
                for row in payload
            )
        ),
        "audit_manifest": str(manifest),
        "audit_manifest_sha256": _sha256(manifest),
        "manual_verdict_schema": MANUAL_VERDICT_SCHEMA,
        "manual_gate_status": "pending",
        "teacher_output_used_as_annotation_seed": True,
        "teacher_output_used_as_truth": False,
        "human_full_source_confirmation_required": True,
        "unselected_source_label_inheritance": False,
        "training_manifest_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_audit_entrypoints(
        latest_html=index,
        title="Scorer v11 train full-source human review",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--preaudit", required=True)
    parser.add_argument("--exclude-sources")
    parser.add_argument("--target-inside-ratio", type=float, default=0.35)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if not 0.0 <= args.target_inside_ratio <= 1.0:
        raise ValueError("target inside ratio must be within [0, 1]")
    summary = build(
        source_manifest=Path(args.source_manifest),
        preaudit=Path(args.preaudit),
        exclude_sources=Path(args.exclude_sources) if args.exclude_sources else None,
        target_inside_ratio=float(args.target_inside_ratio),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
