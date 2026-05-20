from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

from rich.console import Console

from subtitles.qc import compute_quality_report


def parse_glossary_pairs_from_text(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for part in re.split(r"[,，\n]+", text or ""):
        item = part.strip()
        if not item:
            continue
        if "→" in item:
            ja, zh = item.split("→", 1)
        elif "->" in item:
            ja, zh = item.split("->", 1)
        else:
            continue
        ja = ja.strip()
        zh = zh.strip()
        if ja and zh:
            pairs.append((ja, zh))
    return pairs


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
        if "gender" in block:
            segment["gender"] = block.get("gender")
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
        "per_min_subtitle_count",
        "glossary_hit_rate",
        "alignment_fallback_ratio",
        "subtitle_overlap_count",
        "subtitle_overlap_total_s",
        "subtitle_overlap_max_s",
        "f0_filtered_count",
        "f0_failure",
        "asr_generation_error_count",
        "asr_generation_overflow_count",
        "asr_timeout_count",
        "asr_quarantined_count",
        "asr_empty_text_for_speech_count",
        "asr_dropped_uncertain_count",
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

    dropped = report.get("asr_dropped_uncertain_items")
    if isinstance(dropped, list) and dropped:
        lines.extend(["", "## ASR Dropped Uncertain Items", ""])
        for item in dropped[:20]:
            lines.append(f"- `{json.dumps(item, ensure_ascii=False)}`")

    lines.append("")
    return "\n".join(lines)


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
    f0_filtered_count: int = 0,
    f0_failure: bool = False,
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
        fallback_count = int((asr_details or {}).get("fallback_count", 0))
        if video_duration_s is None:
            video_duration_s = float(
                (asr_details or {}).get("audio_duration_s", len(aligned_segments) * 2.0)
            )
        report = compute_quality_report(
            aligned_segments,
            video_duration_s,
            glossary_pairs,
            fallback_count,
            len(aligned_segments),
            f0_filtered_count=f0_filtered_count,
            f0_failure=f0_failure,
            asr_qc=(asr_details or {}).get("asr_qc") or {},
        )
        effective_report_dir = (
            Path(report_dir)
            if report_dir is not None
            else Path(os.getenv("QUALITY_REPORT_DIR", "./reports"))
        ).expanduser()
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
