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
        )
        report_dir = Path(os.getenv("QUALITY_REPORT_DIR", "./reports")).expanduser()
        if not report_dir.is_absolute():
            report_dir = project_root / report_dir
        report_path = report_dir.resolve() / f"{video_stem}.quality_report.json"
        write_json_atomic(report_path, report)

        warnings_list = report.get("warnings") or []
        if warnings_list:
            console.print("[yellow]WARNING: 字幕质量报告发现以下问题：[/yellow]")
            for warning in warnings_list:
                console.print(f"[yellow]- {warning}[/yellow]")
        if os.getenv("QC_HARD_FAIL", "0").strip() == "1" and warnings_list:
            raise RuntimeError(
                f"QC_HARD_FAIL=1 and quality warnings present: {warnings_list}"
            )
        return str(report_path)
    except RuntimeError:
        raise
    except Exception as exc:
        console.print(f"[yellow]WARNING: 生成字幕质量报告失败：{exc}[/yellow]")
        return None
