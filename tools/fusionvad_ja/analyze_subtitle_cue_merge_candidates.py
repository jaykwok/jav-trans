#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from subtitles.options import SubtitleOptions
from subtitles.qc import compute_quality_report
from subtitles import writer as subtitle_writer


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_blocks(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError(f"missing blocks list in {path}")
    return [dict(block) for block in blocks if isinstance(block, dict)]


def load_asr_qc(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    details = payload.get("asr_details") if isinstance(payload.get("asr_details"), dict) else {}
    qc = details.get("asr_qc") if isinstance(details.get("asr_qc"), dict) else {}
    if qc:
        return dict(qc)
    direct = payload.get("asr_qc")
    return dict(direct) if isinstance(direct, dict) else {}


def estimate_duration(blocks: list[dict[str, Any]]) -> float:
    max_end = 0.0
    for block in blocks:
        try:
            max_end = max(max_end, float(block.get("end", 0.0)))
        except (TypeError, ValueError):
            continue
    return max(max_end, 0.001)


def as_quality_segments(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in blocks:
        text = str(block.get("text") or block.get("ja_text") or block.get("ja") or "")
        rows.append(
            {
                **block,
                "text": text,
                "ja": str(block.get("ja") or block.get("ja_text") or text),
                "zh": str(block.get("zh") or block.get("zh_text") or ""),
            }
        )
    return rows


def _float(block: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(block.get(key, default))
    except (TypeError, ValueError):
        return default


def _duration(block: dict[str, Any]) -> float:
    start = _float(block, "start")
    return max(0.0, _float(block, "end", start) - start)


def _gap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return _float(right, "start") - _float(left, "end", _float(left, "start"))


def _text_units(block: dict[str, Any]) -> float:
    return float(subtitle_writer._block_text_units(block))


def _same_speaker_or_unknown(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(subtitle_writer._same_speaker_or_unknown(left, right))


def _ends_sentence(block: dict[str, Any]) -> bool:
    ja = str(block.get("ja_text") or block.get("text") or "").strip()
    zh = str(block.get("zh_text") or block.get("zh") or "").strip()
    return ja.endswith(tuple(subtitle_writer._SENTENCE_END_PUNCTUATION)) or zh.endswith(
        tuple(subtitle_writer._ZH_SOFT_SPLIT_PUNCTUATION)
    )


def _frames(seconds: float, options: SubtitleOptions) -> float:
    return seconds / max(0.001, options.frame_duration_s)


def dense_merge_blockers(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    options: SubtitleOptions,
) -> list[str]:
    blockers: list[str] = []
    gap_s = _gap(left, right)
    left_duration = _duration(left)
    right_duration = _duration(right)
    combined_duration = max(_float(left, "start"), _float(right, "end")) - _float(left, "start")
    combined_units = _text_units(left) + _text_units(right)

    if not _same_speaker_or_unknown(left, right):
        blockers.append("speaker_change")
    if gap_s < -options.frame_gap_s:
        blockers.append("negative_overlap")
    if gap_s > options.dense_cue_merge_max_gap_frames * options.frame_duration_s:
        blockers.append("gap_too_large")
    if max(left_duration, right_duration) > (
        options.dense_cue_merge_max_single_frames * options.frame_duration_s
    ):
        blockers.append("single_duration_too_long")
    if combined_duration > options.dense_cue_merge_max_combined_frames * options.frame_duration_s:
        blockers.append("combined_duration_too_long")
    if combined_units > options.dense_cue_merge_max_text_units:
        blockers.append("text_units_too_large")
    if _ends_sentence(left):
        blockers.append("sentence_boundary")
    return blockers


def planner_merge_score(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    options: SubtitleOptions,
    max_gap_s: float,
    max_combined_s: float,
    max_text_units: float,
) -> tuple[float, list[str]]:
    gap_s = _gap(left, right)
    left_duration = _duration(left)
    right_duration = _duration(right)
    combined_duration = max(_float(left, "end"), _float(right, "end")) - _float(left, "start")
    combined_units = _text_units(left) + _text_units(right)
    reasons: list[str] = []

    if not _same_speaker_or_unknown(left, right):
        return 0.0, ["speaker_change"]
    if gap_s < -options.frame_gap_s:
        return 0.0, ["negative_overlap"]
    if gap_s > max_gap_s:
        return 0.0, ["gap_too_large"]
    if combined_duration > max_combined_s:
        return 0.0, ["combined_duration_too_long"]
    if combined_units > max_text_units:
        return 0.0, ["text_units_too_large"]
    if _ends_sentence(left):
        return 0.0, ["sentence_boundary"]

    if left_duration < 0.8:
        reasons.append("left_short")
    if right_duration < 0.8:
        reasons.append("right_short")
    if max(left_duration, right_duration) < 1.4:
        reasons.append("both_compact")
    if gap_s <= max(0.12, options.frame_gap_s * 2):
        reasons.append("tight_gap")
    if combined_units <= max_text_units * 0.7:
        reasons.append("low_text_units")

    score = 0.0
    score += max(0.0, 1.0 - gap_s / max(max_gap_s, 0.001)) * 0.35
    score += max(0.0, 1.0 - combined_duration / max(max_combined_s, 0.001)) * 0.25
    score += max(0.0, 1.0 - combined_units / max(max_text_units, 0.001)) * 0.25
    score += (1.0 if min(left_duration, right_duration) < 0.8 else 0.0) * 0.15
    return round(min(score, 1.0), 6), reasons


def merge_blocks(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = subtitle_writer._merge_overlapping_blocks(left, right)
    merged["planner_merge_count"] = int(left.get("planner_merge_count") or 0) + 1
    source_ids: list[Any] = []
    for block in (left, right):
        for source_id in block.get("source_segment_ids") or []:
            if source_id not in source_ids:
                source_ids.append(source_id)
    if source_ids:
        merged["source_segment_ids"] = source_ids
    return merged


def apply_planner_candidates(
    blocks: list[dict[str, Any]],
    *,
    options: SubtitleOptions,
    min_score: float,
    max_gap_s: float,
    max_combined_s: float,
    max_text_units: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    index = 0
    while index < len(blocks):
        current = dict(blocks[index])
        while index + 1 < len(blocks):
            nxt = blocks[index + 1]
            score, reasons = planner_merge_score(
                current,
                nxt,
                options=options,
                max_gap_s=max_gap_s,
                max_combined_s=max_combined_s,
                max_text_units=max_text_units,
            )
            if score < min_score:
                break
            actions.append(
                {
                    "left_index": index,
                    "right_index": index + 1,
                    "score": score,
                    "reasons": reasons,
                    "gap_s": round(_gap(current, nxt), 6),
                    "combined_duration_s": round(
                        max(_float(current, "end"), _float(nxt, "end"))
                        - _float(current, "start"),
                        6,
                    ),
                    "combined_text_units": round(_text_units(current) + _text_units(nxt), 3),
                }
            )
            current = merge_blocks(current, nxt)
            index += 1
        merged.append(current)
        index += 1
    return merged, actions


def summarize_pairs(
    blocks: list[dict[str, Any]],
    *,
    options: SubtitleOptions,
    min_score: float,
    max_gap_s: float,
    max_combined_s: float,
    max_text_units: float,
) -> dict[str, Any]:
    blocker_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    candidate_examples: list[dict[str, Any]] = []
    pair_count = max(0, len(blocks) - 1)

    for index, (left, right) in enumerate(zip(blocks, blocks[1:])):
        blockers = dense_merge_blockers(left, right, options=options)
        if blockers:
            blocker_counts.update(blockers)
        else:
            blocker_counts.update(["dense_merge_allowed"])
        score, reasons = planner_merge_score(
            left,
            right,
            options=options,
            max_gap_s=max_gap_s,
            max_combined_s=max_combined_s,
            max_text_units=max_text_units,
        )
        if score >= min_score:
            candidate_counts.update(reasons or ["candidate"])
            if len(candidate_examples) < 25:
                candidate_examples.append(
                    {
                        "index": index,
                        "score": score,
                        "reasons": reasons,
                        "gap_s": round(_gap(left, right), 6),
                        "gap_frames": round(_frames(_gap(left, right), options), 2),
                        "left_duration_s": round(_duration(left), 6),
                        "right_duration_s": round(_duration(right), 6),
                        "combined_duration_s": round(
                            max(_float(left, "end"), _float(right, "end"))
                            - _float(left, "start"),
                            6,
                        ),
                        "combined_text_units": round(_text_units(left) + _text_units(right), 3),
                        "left_text": str(left.get("ja_text") or left.get("text") or "")[:80],
                        "right_text": str(right.get("ja_text") or right.get("text") or "")[:80],
                    }
                )

    return {
        "pair_count": pair_count,
        "dense_blocker_counts": dict(blocker_counts.most_common()),
        "planner_candidate_reason_counts": dict(candidate_counts.most_common()),
        "planner_candidate_examples": candidate_examples,
    }


def quality(blocks: list[dict[str, Any]], *, video_duration_s: float, asr_qc: dict[str, Any]) -> dict:
    return compute_quality_report(
        as_quality_segments(blocks),
        video_duration_s,
        [],
        int(asr_qc.get("alignment_fallback_count") or 0),
        len(blocks),
        asr_qc=asr_qc,
    )


def build_markdown(summary: dict[str, Any]) -> str:
    before = summary["before"]
    after = summary["after"]
    delta = summary["delta"]
    pair = summary["pair_analysis"]
    lines = [
        "# Subtitle Cue Merge Candidate Analysis",
        "",
        f"- source: `{summary['source_bilingual']}`",
        f"- video_duration_s: {summary['video_duration_s']:.3f}",
        f"- video_fps: {summary['video_fps']:.6f}",
        f"- planner_min_score: {summary['planner']['min_score']}",
        "",
        "| metric | before | planner | delta |",
        "|---|---:|---:|---:|",
        f"| blocks | {before['block_count']} | {after['block_count']} | {delta['block_count']} |",
        f"| planner merges | 0 | {after['planner_merge_count']} | {after['planner_merge_count']} |",
        f"| short_segment_ratio | {before['quality']['short_segment_ratio']:.6f} | {after['quality']['short_segment_ratio']:.6f} | {delta['short_segment_ratio']:.6f} |",
        f"| per_min_subtitle_count | {before['quality']['per_min_subtitle_count']:.2f} | {after['quality']['per_min_subtitle_count']:.2f} | {delta['per_min_subtitle_count']:.2f} |",
        f"| kana_only_ratio | {before['quality']['kana_only_ratio']:.6f} | {after['quality']['kana_only_ratio']:.6f} | {delta['kana_only_ratio']:.6f} |",
        f"| repetition_ratio | {before['quality']['repetition_ratio']:.6f} | {after['quality']['repetition_ratio']:.6f} | {delta['repetition_ratio']:.6f} |",
        f"| subtitle_overlap_count | {before['quality']['subtitle_overlap_count']} | {after['quality']['subtitle_overlap_count']} | {delta['subtitle_overlap_count']} |",
        "",
        "## Dense Merge Blockers",
        "",
    ]
    lines.extend(
        f"- {key}: {value}" for key, value in pair["dense_blocker_counts"].items()
    )
    lines.extend(["", "## Planner Candidate Reasons", ""])
    if pair["planner_candidate_reason_counts"]:
        lines.extend(
            f"- {key}: {value}"
            for key, value in pair["planner_candidate_reason_counts"].items()
        )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def build_summary(
    *,
    bilingual_path: Path,
    timings_path: Path | None,
    output_dir: Path,
    video_fps: float,
    min_score: float,
    max_gap_s: float,
    max_combined_s: float,
    max_text_units: float,
) -> dict[str, Any]:
    source_blocks = load_blocks(bilingual_path)
    options = SubtitleOptions.from_env().with_video_fps(video_fps)
    prepared = subtitle_writer.prepare_srt_blocks(source_blocks, options=options, mode="bilingual")
    asr_qc = load_asr_qc(timings_path)
    video_duration_s = estimate_duration(prepared)
    pair_analysis = summarize_pairs(
        prepared,
        options=options,
        min_score=min_score,
        max_gap_s=max_gap_s,
        max_combined_s=max_combined_s,
        max_text_units=max_text_units,
    )
    planner_blocks, actions = apply_planner_candidates(
        prepared,
        options=options,
        min_score=min_score,
        max_gap_s=max_gap_s,
        max_combined_s=max_combined_s,
        max_text_units=max_text_units,
    )
    planner_blocks = subtitle_writer.prepare_srt_blocks(
        planner_blocks,
        options=options,
        mode="bilingual",
    )
    before_quality = quality(prepared, video_duration_s=video_duration_s, asr_qc=asr_qc)
    after_quality = quality(planner_blocks, video_duration_s=video_duration_s, asr_qc=asr_qc)
    before = {
        "block_count": len(prepared),
        "quality": before_quality,
    }
    after = {
        "block_count": len(planner_blocks),
        "planner_merge_count": len(actions),
        "planner_merge_marker_count": sum(
            int(block.get("planner_merge_count") or 0) for block in planner_blocks
        ),
        "quality": after_quality,
    }
    summary = {
        "source_bilingual": project_rel(bilingual_path),
        "source_timings": project_rel(timings_path),
        "video_duration_s": video_duration_s,
        "video_fps": video_fps,
        "planner": {
            "min_score": min_score,
            "max_gap_s": max_gap_s,
            "max_combined_s": max_combined_s,
            "max_text_units": max_text_units,
        },
        "pair_analysis": pair_analysis,
        "before": before,
        "after": after,
        "delta": {
            "block_count": after["block_count"] - before["block_count"],
            "short_segment_ratio": round(
                after_quality["short_segment_ratio"] - before_quality["short_segment_ratio"],
                6,
            ),
            "per_min_subtitle_count": round(
                after_quality["per_min_subtitle_count"] - before_quality["per_min_subtitle_count"],
                6,
            ),
            "kana_only_ratio": round(
                after_quality["kana_only_ratio"] - before_quality["kana_only_ratio"],
                6,
            ),
            "repetition_ratio": round(
                after_quality["repetition_ratio"] - before_quality["repetition_ratio"],
                6,
            ),
            "subtitle_overlap_count": (
                after_quality["subtitle_overlap_count"]
                - before_quality["subtitle_overlap_count"]
            ),
        },
    }
    write_json(output_dir / "before_blocks.json", prepared)
    write_json(output_dir / "planner_blocks.json", planner_blocks)
    write_json(output_dir / "planner_actions.json", actions)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze subtitle cue merge candidates. This is an offline diagnostic "
            "tool; it does not change the runtime subtitle writer defaults."
        )
    )
    parser.add_argument("--bilingual", required=True, help="bilingual.json path")
    parser.add_argument("--timings", default="", help="timings.json path with asr_qc")
    parser.add_argument(
        "--output-dir",
        default="agents/temp/fusionvad-ja/subtitle-cue-merge-candidates",
    )
    parser.add_argument("--video-fps", type=float, default=30000 / 1001)
    parser.add_argument("--min-score", type=float, default=0.72)
    parser.add_argument("--max-gap-s", type=float, default=0.45)
    parser.add_argument("--max-combined-s", type=float, default=4.8)
    parser.add_argument("--max-text-units", type=float, default=34.0)
    args = parser.parse_args(argv)

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(
        bilingual_path=project_path(args.bilingual),
        timings_path=project_path(args.timings) if args.timings else None,
        output_dir=output_dir,
        video_fps=float(args.video_fps),
        min_score=float(args.min_score),
        max_gap_s=float(args.max_gap_s),
        max_combined_s=float(args.max_combined_s),
        max_text_units=float(args.max_text_units),
    )
    print(
        "summary={path} merges={merges} blocks={before}->{after}".format(
            path=project_rel(output_dir / "summary.json"),
            merges=summary["after"]["planner_merge_count"],
            before=summary["before"]["block_count"],
            after=summary["after"]["block_count"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
