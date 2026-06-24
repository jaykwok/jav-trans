from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from rich.console import Console

from llm.glossary import parse_glossary_pairs
from subtitles.qc import compute_quality_report


def parse_glossary_pairs_from_text(text: str) -> list[tuple[str, str]]:
    return parse_glossary_pairs(text)


def load_global_glossary_pairs(
    job_temp_dir: str,
    video_stem: str,
    *,
    console: Console,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    root = Path(job_temp_dir)
    candidates = [
        root / f"{video_stem}.translation_global_glossary.json",
        root / "translation_global_glossary.json",
    ]
    candidates.extend(sorted(root.glob("translation_global_glossary.*.json")))
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            terms = payload.get("terms") if isinstance(payload, dict) else payload
            if not isinstance(terms, list):
                continue
            for item in terms:
                if not isinstance(item, dict):
                    continue
                ja = str(item.get("ja", "")).strip()
                zh = str(item.get("zh", "")).strip()
                if ja and zh:
                    pairs.append((ja, zh))
        except Exception as exc:
            console.print(f"[yellow]WARNING: 读取全局术语表失败 {path}: {exc}[/yellow]")
    return pairs


def collect_glossary_pairs(
    job_temp_dir: str,
    video_stem: str,
    *,
    glossary: str | None = None,
    console: Console,
) -> list[tuple[str, str]]:
    source_glossary = os.getenv("TRANSLATION_GLOSSARY", "") if glossary is None else glossary
    pairs = parse_glossary_pairs_from_text(source_glossary)
    pairs.extend(
        load_global_glossary_pairs(
            job_temp_dir,
            video_stem,
            console=console,
        )
    )

    merged: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ja, zh in pairs:
        key = (ja, zh)
        if key in seen:
            continue
        seen.add(key)
        merged.append(key)
    return merged


def quality_segments_from_blocks(blocks: list[dict]) -> list[dict]:
    quality_segments: list[dict] = []
    for block in blocks:
        segment = {
            "start": float(block.get("start", 0.0)),
            "end": float(block.get("end", 0.0)),
            "text": str(block.get("ja_text") or block.get("text") or block.get("ja") or ""),
            "ja": str(block.get("ja_text") or block.get("text") or block.get("ja") or ""),
            "zh": str(block.get("zh_text") or block.get("zh") or ""),
        }
        quality_segments.append(segment)
    return quality_segments


def _format_report_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _quality_report_markdown(video_stem: str, report: dict) -> str:
    metric_keys = [
        "empty_zh_ratio",
        "repetition_ratio",
        "kana_only_ratio",
        "short_segment_ratio",
        "short_segment_count",
        "micro_segment_count",
        "long_segment_count",
        "subtitle_duration_p50_s",
        "subtitle_duration_p90_s",
        "subtitle_duration_p95_s",
        "subtitle_duration_max_s",
        "subtitle_density_cps_threshold",
        "subtitle_density_over_4cps_count",
        "subtitle_density_over_4cps_ratio",
        "subtitle_density_max_ja_cps",
        "subtitle_density_p90_ja_cps",
        "subtitle_density_p95_ja_cps",
        "subtitle_density_window_10s_max_cue_count",
        "subtitle_density_window_10s_max_active_ratio",
        "subtitle_density_window_10s_max_cps",
        "subtitle_density_window_10s_min_gap_s",
        "subtitle_density_window_10s_median_gap_s",
        "subtitle_density_window_30s_max_cue_count",
        "subtitle_density_window_30s_max_active_ratio",
        "subtitle_density_window_30s_max_cps",
        "subtitle_density_window_30s_min_gap_s",
        "subtitle_density_window_30s_median_gap_s",
        "per_min_subtitle_count",
        "glossary_hit_rate",
        "alignment_issue_count",
        "alignment_issue_total",
        "alignment_issue_ratio",
        "subtitle_overlap_count",
        "subtitle_overlap_total_s",
        "subtitle_overlap_max_s",
        "asr_generation_error_count",
        "asr_generation_overflow_count",
        "asr_timeout_count",
        "asr_quarantined_count",
    ]
    lines = [
        f"# Quality Report: {video_stem}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in metric_keys:
        if key in report:
            lines.append(f"| `{key}` | {_format_report_value(report.get(key))} |")

    warnings = list(report.get("warnings") or [])
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")

    examples = list(report.get("subtitle_overlap_examples") or [])
    if examples:
        lines.extend(["", "## Overlap Examples", ""])
        for item in examples:
            lines.append(
                "- "
                f"{_format_report_value(item.get('previous_start'))}-"
                f"{_format_report_value(item.get('previous_end'))} overlaps "
                f"{_format_report_value(item.get('current_start'))}-"
                f"{_format_report_value(item.get('current_end'))} by "
                f"{_format_report_value(item.get('overlap_s'))}s"
            )

    density_examples = list(report.get("subtitle_density_review_examples") or [])
    if density_examples:
        lines.extend(["", "## Density Review Examples", ""])
        for item in density_examples[:20]:
            lines.append(
                "- "
                f"#{item.get('index')} "
                f"{_format_report_value(item.get('start'))}-"
                f"{_format_report_value(item.get('end'))}: "
                f"{_format_report_value(item.get('ja_cps'))} cps, "
                f"{item.get('ja_units')} units"
            )

    lines.append("")
    return "\n".join(lines)


def _default_quality_report_dir(project_root: Path, video_stem: str) -> Path:
    return project_root / "video" / video_stem


def write_quality_report(
    *,
    video_stem: str,
    job_temp_dir: str,
    aligned_segments: list[dict],
    asr_details: dict,
    project_root: Path,
    console: Console,
    write_json_atomic: Callable[[Path, dict], None],
    env_flag: Callable[[str], bool],
    video_duration_s: float | None = None,
    enabled: bool | None = None,
    glossary: str | None = None,
    report_dir: Path | str | None = None,
    hard_fail: bool | None = None,
) -> str | None:
    if enabled is None:
        enabled = env_flag("QUALITY_REPORT_ENABLED")
    if not enabled:
        return None

    try:
        glossary_pairs = collect_glossary_pairs(
            job_temp_dir,
            video_stem,
            glossary=glossary,
            console=console,
        )
        alignment_issue_count = int((asr_details or {}).get("alignment_issue_count", 0))
        if video_duration_s is None:
            video_duration_s = float(
                (asr_details or {}).get("audio_duration_s", len(aligned_segments) * 2.0)
            )
        report = compute_quality_report(
            aligned_segments,
            video_duration_s,
            glossary_pairs,
            alignment_issue_count,
            int((asr_details or {}).get("chunk_count") or len(aligned_segments)),
            asr_generation={},
        )
        explicit_report_dir = str(report_dir).strip() if report_dir is not None else ""
        env_report_dir = os.getenv("QUALITY_REPORT_DIR", "").strip()
        effective_report_dir = (
            Path(explicit_report_dir or env_report_dir).expanduser()
            if explicit_report_dir or env_report_dir
            else _default_quality_report_dir(project_root, video_stem)
        )
        if not effective_report_dir.is_absolute():
            effective_report_dir = project_root / effective_report_dir
        report_dir_path = effective_report_dir.resolve()
        json_path = report_dir_path / f"{video_stem}.quality_report.json"
        markdown_path = report_dir_path / f"{video_stem}.quality_report.md"
        write_json_atomic(json_path, report)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            _quality_report_markdown(video_stem, report),
            encoding="utf-8",
        )

        warnings_list = report.get("warnings") or []
        if warnings_list:
            console.print("[yellow]WARNING: 字幕质量报告发现以下问题：[/yellow]")
            for warning in warnings_list:
                console.print(f"[yellow]- {warning}[/yellow]")
        effective_hard_fail = (
            bool(hard_fail)
            if hard_fail is not None
            else os.getenv("QC_HARD_FAIL", "0").strip() == "1"
        )
        if effective_hard_fail and warnings_list:
            raise RuntimeError(
                f"QC_HARD_FAIL=1 and quality warnings present: {warnings_list}"
            )
        return str(markdown_path)
    except RuntimeError:
        raise
    except Exception as exc:
        console.print(f"[yellow]WARNING: 生成字幕质量报告失败：{exc}[/yellow]")
        return None
