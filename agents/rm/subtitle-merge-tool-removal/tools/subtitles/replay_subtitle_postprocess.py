#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from subtitles.options import SubtitleOptions
from subtitles.qc import compute_quality_report
from subtitles.writer import prepare_srt_blocks
from asr.qc import evaluate_asr_chunk_qc


_COMPACT_RE = re.compile(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u9fff]+")


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


@contextmanager
def temp_env(overrides: dict[str, str]):
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def temp_logger_level(logger_name: str, level: int):
    logger = logging.getLogger(logger_name)
    previous = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous)


def load_blocks(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError(f"missing blocks list in {path}")
    return [dict(block) for block in blocks if isinstance(block, dict)]


def load_segments(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = read_json(path)
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return []
    return [dict(segment) for segment in segments if isinstance(segment, dict)]


def blocks_from_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(segment.get("text") or segment.get("ja_text") or "").strip()
        block = {
            "start": start,
            "end": max(start, end),
            "ja_text": text,
            "zh_text": text,
            "text": text,
            "words": list(segment.get("words") or []),
            "source_segment_ids": list(segment.get("source_segment_ids") or [index]),
        }
        blocks.append(block)
    return blocks


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
    segments: list[dict[str, Any]] = []
    for block in blocks:
        segment = dict(block)
        segment["text"] = str(block.get("text") or block.get("ja_text") or block.get("ja") or "")
        segment["ja"] = str(block.get("ja") or block.get("ja_text") or segment["text"])
        segment["zh"] = str(block.get("zh") or block.get("zh_text") or "")
        segments.append(segment)
    return segments


def summarize_blocks(
    blocks: list[dict[str, Any]],
    *,
    video_duration_s: float,
    total_segments: int,
    asr_qc: dict[str, Any],
) -> dict[str, Any]:
    quality = compute_quality_report(
        as_quality_segments(blocks),
        video_duration_s,
        [],
        int(asr_qc.get("alignment_fallback_count") or 0),
        total_segments,
        asr_qc=asr_qc,
    )
    durations = sorted(
        max(0.0, float(block.get("end", 0.0)) - float(block.get("start", 0.0)))
        for block in blocks
    )
    dense_merge_count = sum(int(block.get("dense_cue_merge_count") or 0) for block in blocks)
    return {
        "block_count": len(blocks),
        "dense_cue_merge_count": dense_merge_count,
        "duration_p50": percentile(durations, 0.50),
        "duration_p90": percentile(durations, 0.90),
        "duration_max": percentile(durations, 1.00),
        "quality": quality,
        "nonlexical_repetition": count_nonlexical_repetition(blocks),
    }


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * ratio))))
    return round(values[index], 6)


def count_nonlexical_repetition(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    examples: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        text = str(block.get("ja_text") or block.get("text") or "")
        compact = _COMPACT_RE.sub("", text)
        if not compact:
            continue
        qc = evaluate_asr_chunk_qc(
            {
                "index": index,
                "start": block.get("start", 0.0),
                "end": block.get("end", 0.0),
            },
            {
                "text": text,
                "raw_text": text,
                "duration": max(
                    0.0,
                    float(block.get("end", 0.0)) - float(block.get("start", 0.0)),
                ),
            },
        )
        if "repeated_nonlexical_vocalization" not in set(qc.get("reasons") or []):
            continue
        count += 1
        if len(examples) < 10:
            metrics = qc.get("metrics") if isinstance(qc.get("metrics"), dict) else {}
            examples.append(
                {
                    "index": index,
                    "start": round(float(block.get("start", 0.0)), 3),
                    "end": round(float(block.get("end", 0.0)), 3),
                    "text": text[:80],
                    "repeat": metrics.get("max_repeat", {}),
                    "policy": (
                        metrics.get("vocalization_repetition", {})
                        if isinstance(metrics.get("vocalization_repetition"), dict)
                        else {}
                    ).get("policy", ""),
                }
            )
    return {"count": count, "examples": examples}


def replay_blocks(
    source_blocks: list[dict[str, Any]],
    *,
    dense_enabled: bool,
    video_fps: float,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with temp_env({"SUBTITLE_DENSE_CUE_MERGE_ENABLED": "1" if dense_enabled else "0"}):
        options = SubtitleOptions.from_env().with_video_fps(video_fps)
        prepared = prepare_srt_blocks(source_blocks, options=options, mode=mode)
        return prepared, options.signature()


def build_markdown(summary: dict[str, Any]) -> str:
    before = summary["before"]
    after = summary["after"]
    delta = summary["delta"]
    lines = [
        "# Subtitle Postprocess Replay",
        "",
        f"- source: `{summary['source_bilingual']}`",
        f"- aligned: `{summary.get('source_aligned', '')}`",
        f"- video_duration_s: {summary['video_duration_s']:.3f}",
        f"- video_fps: {summary['video_fps']:.6f}",
        "",
        "| metric | before | after | delta |",
        "|---|---:|---:|---:|",
        f"| blocks | {before['block_count']} | {after['block_count']} | {delta['block_count']} |",
        f"| dense merges | {before['dense_cue_merge_count']} | {after['dense_cue_merge_count']} | {delta['dense_cue_merge_count']} |",
        f"| short_segment_ratio | {before['quality']['short_segment_ratio']:.6f} | {after['quality']['short_segment_ratio']:.6f} | {delta['short_segment_ratio']:.6f} |",
        f"| per_min_subtitle_count | {before['quality']['per_min_subtitle_count']:.2f} | {after['quality']['per_min_subtitle_count']:.2f} | {delta['per_min_subtitle_count']:.2f} |",
        f"| kana_only_ratio | {before['quality']['kana_only_ratio']:.6f} | {after['quality']['kana_only_ratio']:.6f} | {delta['kana_only_ratio']:.6f} |",
        f"| repetition_ratio | {before['quality']['repetition_ratio']:.6f} | {after['quality']['repetition_ratio']:.6f} | {delta['repetition_ratio']:.6f} |",
        f"| subtitle_overlap_count | {before['quality']['subtitle_overlap_count']} | {after['quality']['subtitle_overlap_count']} | {delta['subtitle_overlap_count']} |",
        f"| nonlexical repetition count | {before['nonlexical_repetition']['count']} | {after['nonlexical_repetition']['count']} | {delta['nonlexical_repetition_count']} |",
        "",
        "## After Warnings",
        "",
    ]
    warnings = list(after["quality"].get("warnings") or [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def build_summary(
    *,
    bilingual_path: Path,
    aligned_path: Path | None,
    timings_path: Path | None,
    video_fps: float,
    output_dir: Path,
    mode: str = "bilingual",
    source: str = "blocks",
) -> dict[str, Any]:
    aligned_segments = load_segments(aligned_path)
    if source == "aligned":
        if not aligned_segments:
            raise ValueError("--source aligned requires --aligned with segments")
        source_blocks = blocks_from_segments(aligned_segments)
    else:
        source_blocks = load_blocks(bilingual_path)
    asr_qc = load_asr_qc(timings_path)
    video_duration_s = estimate_duration(source_blocks)
    total_segments = len(aligned_segments) or len(source_blocks)

    before_blocks, before_options = replay_blocks(
        source_blocks,
        dense_enabled=False,
        video_fps=video_fps,
        mode=mode,
    )
    after_blocks, after_options = replay_blocks(
        source_blocks,
        dense_enabled=True,
        video_fps=video_fps,
        mode=mode,
    )
    before = summarize_blocks(
        before_blocks,
        video_duration_s=video_duration_s,
        total_segments=total_segments,
        asr_qc=asr_qc,
    )
    after = summarize_blocks(
        after_blocks,
        video_duration_s=video_duration_s,
        total_segments=total_segments,
        asr_qc=asr_qc,
    )
    summary = {
        "source_bilingual": project_rel(bilingual_path),
        "source_aligned": project_rel(aligned_path),
        "source_timings": project_rel(timings_path),
        "mode": mode,
        "source": source,
        "video_duration_s": video_duration_s,
        "video_fps": video_fps,
        "before_options": before_options,
        "after_options": after_options,
        "before": before,
        "after": after,
        "delta": {
            "block_count": after["block_count"] - before["block_count"],
            "dense_cue_merge_count": after["dense_cue_merge_count"]
            - before["dense_cue_merge_count"],
            "short_segment_ratio": round(
                after["quality"]["short_segment_ratio"]
                - before["quality"]["short_segment_ratio"],
                6,
            ),
            "per_min_subtitle_count": round(
                after["quality"]["per_min_subtitle_count"]
                - before["quality"]["per_min_subtitle_count"],
                6,
            ),
            "kana_only_ratio": round(
                after["quality"]["kana_only_ratio"] - before["quality"]["kana_only_ratio"],
                6,
            ),
            "repetition_ratio": round(
                after["quality"]["repetition_ratio"]
                - before["quality"]["repetition_ratio"],
                6,
            ),
            "subtitle_overlap_count": after["quality"]["subtitle_overlap_count"]
            - before["quality"]["subtitle_overlap_count"],
            "nonlexical_repetition_count": after["nonlexical_repetition"]["count"]
            - before["nonlexical_repetition"]["count"],
        },
    }
    write_json(output_dir / "before_blocks.json", before_blocks)
    write_json(output_dir / "after_blocks.json", after_blocks)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay subtitle postprocessing with dense cue merge A/B."
    )
    parser.add_argument("--bilingual", required=True, help="bilingual.json path")
    parser.add_argument("--aligned", default="", help="aligned_segments.json path")
    parser.add_argument("--timings", default="", help="timings.json path with asr_qc")
    parser.add_argument(
        "--output-dir",
        default="agents/temp/speech-boundary-ja/subtitle-postprocess-replay",
    )
    parser.add_argument("--video-fps", type=float, default=30000 / 1001)
    parser.add_argument("--mode", choices=("srt", "bilingual"), default="bilingual")
    parser.add_argument("--source", choices=("blocks", "aligned"), default="blocks")
    args = parser.parse_args(argv)

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with temp_logger_level("asr.qc", logging.ERROR):
        summary = build_summary(
            bilingual_path=project_path(args.bilingual),
            aligned_path=project_path(args.aligned) if args.aligned else None,
            timings_path=project_path(args.timings) if args.timings else None,
            video_fps=float(args.video_fps),
            output_dir=output_dir,
            mode=args.mode,
            source=args.source,
        )
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    print(f"summary={project_rel(output_dir / 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
