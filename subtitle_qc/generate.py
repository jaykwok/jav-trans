#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_MODES = (
    "whisperseg_adaptive",
    "fusion_lite",
    "fusion_lite_boost",
    "fusion_lite_sigmoid",
)
DEFAULT_BASE_MODE = "whisperseg_adaptive"
LABEL_STATUSES = ("unreviewed", "ok", "needs_fix", "unsure")
LABEL_ISSUES = (
    "asr_hallucination",
    "transcription_error",
    "translation_error",
    "timing_error",
    "vad_boundary_error",
    "missing_speech",
    "extra_speech",
)
SRT_RANGE_RE = re.compile(
    r"(?P<start>\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})"
)
TIMECODE_RE = re.compile(r"(?:(?P<hours>\d+):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{2})(?:[,.](?P<millis>\d{1,3}))?")
KANA_RE = re.compile(r"[ぁ-ゖァ-ヺー]")
JSON_CUE_LIST_KEYS = ("blocks", "segments", "items", "cues", "subtitles")
JSON_TEXT_KEYS = ("ja_text", "text", "transcript", "content", "line", "subtitle")
JSON_TRANSLATION_KEYS = ("zh_text", "translation", "zh", "cn_text")
JSON_START_KEYS = ("start", "start_time", "begin", "from")
JSON_END_KEYS = ("end", "end_time", "stop", "to")


@dataclass(frozen=True)
class Cue:
    index: int
    mode: str
    start: float
    end: float
    ja_text: str
    zh_text: str
    gender: str
    words: list[dict[str, Any]]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Match:
    mode: str
    cue: Cue | None
    score: float
    text_similarity: float
    translation_similarity: float
    time_delta_start: float
    time_delta_end: float


@dataclass
class ReviewItem:
    item_id: str
    start: float
    end: float
    severity: str
    labels: list[str]
    reason: str
    base_index: int
    base_ja: str
    base_zh: str
    variants: dict[str, dict[str, Any]]


def project_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def subtitle_qc_output_dir(video_path: Path) -> Path:
    return video_path.resolve().parent / video_path.stem / "subtitle_qc"


def video_artifact_path(video_stem: str, filename: str) -> Path:
    nested = PROJECT_ROOT / "video" / video_stem / filename
    if nested.exists():
        return nested
    return PROJECT_ROOT / "video" / filename


def iter_video_artifacts(video_stem: str, pattern: str) -> list[Path]:
    nested_dir = PROJECT_ROOT / "video" / video_stem
    paths = list(nested_dir.glob(pattern)) if nested_dir.exists() else []
    paths.extend((PROJECT_ROOT / "video").glob(pattern))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def fmt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def normalize_text(value: str) -> str:
    return "".join(str(value or "").split())


def text_similarity(left: str, right: str) -> float:
    left_n = normalize_text(left)
    right_n = normalize_text(right)
    if not left_n and not right_n:
        return 1.0
    if not left_n or not right_n:
        return 0.0
    return SequenceMatcher(None, left_n, right_n).ratio()


def overlap(left: Cue, right: Cue) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_timecode(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    match = TIMECODE_RE.fullmatch(text)
    if not match:
        return optional_float(text)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    millis = (match.group("millis") or "0").ljust(3, "0")[:3]
    return hours * 3600 + minutes * 60 + seconds + int(millis) / 1000.0


def normalize_words(raw_words: Any) -> list[dict[str, Any]]:
    words = []
    if not isinstance(raw_words, list):
        return words
    for raw in raw_words:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("word") or raw.get("text") or "").strip()
        if not text:
            continue
        words.append(
            {
                "word": text,
                "start": optional_float(raw.get("start")),
                "end": optional_float(raw.get("end")),
                "gender": str(raw.get("gender") or ""),
            }
        )
    return words


def first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def choose_json_text(item: dict[str, Any]) -> tuple[str, str]:
    ja_text = str(first_present(item, JSON_TEXT_KEYS) or "").strip()
    zh_text = str(first_present(item, JSON_TRANSLATION_KEYS) or "").strip()
    if not ja_text and isinstance(item.get("lines"), list):
        lines = [str(line).strip() for line in item["lines"] if str(line).strip()]
        ja_text, zh_text = split_subtitle_lines(lines)
    return ja_text, zh_text


def cue_from_mapping(index: int, source_name: str, item: dict[str, Any]) -> Cue | None:
    start = parse_timecode(first_present(item, JSON_START_KEYS))
    end = parse_timecode(first_present(item, JSON_END_KEYS))
    if start is None or end is None:
        return None
    ja_text, zh_text = choose_json_text(item)
    return Cue(
        index=index,
        mode=source_name,
        start=start,
        end=end,
        ja_text=ja_text,
        zh_text=zh_text,
        gender=str(item.get("gender") or ""),
        words=normalize_words(item.get("words")),
    )


def load_cues(video_stem: str, mode: str) -> list[Cue]:
    path = video_artifact_path(video_stem, f"{video_stem}.{mode}.bilingual.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cues: list[Cue] = []
    for index, block in enumerate(payload.get("blocks") or []):
        cues.append(
            Cue(
                index=index,
                mode=mode,
                start=float(block.get("start") or 0.0),
                end=float(block.get("end") or 0.0),
                ja_text=str(block.get("ja_text") or "").strip(),
                zh_text=str(block.get("zh_text") or "").strip(),
                gender=str(block.get("gender") or ""),
                words=normalize_words(block.get("words")),
            )
        )
    return cues


def split_subtitle_lines(lines: list[str]) -> tuple[str, str]:
    visible = [line.strip() for line in lines if line.strip()]
    japanese = [line for line in visible if KANA_RE.search(line)]
    if japanese:
        ja_text = "\n".join(japanese)
        zh_text = "\n".join(line for line in visible if line not in japanese)
        return ja_text, zh_text
    if not visible:
        return "", ""
    return visible[0], "\n".join(visible[1:])


def parse_srt_time(value: str) -> float:
    parsed = parse_timecode(value.replace(",", "."))
    if parsed is None:
        raise ValueError(f"invalid SRT timecode: {value}")
    return parsed


def load_srt_file(path: Path, source_name: str) -> list[Cue]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n").strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_index = next((i for i, line in enumerate(lines) if SRT_RANGE_RE.search(line)), None)
        if time_index is None:
            continue
        match = SRT_RANGE_RE.search(lines[time_index])
        if match is None:
            continue
        ja_text, zh_text = split_subtitle_lines(lines[time_index + 1 :])
        cues.append(
            Cue(
                index=len(cues),
                mode=source_name,
                start=parse_srt_time(match.group("start")),
                end=parse_srt_time(match.group("end")),
                ja_text=ja_text,
                zh_text=zh_text,
                gender="",
                words=[],
            )
        )
    return cues


def load_json_file(path: Path, source_name: str) -> list[Cue]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = []
        for key in JSON_CUE_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                raw_items = value
                break
        if not raw_items and all(key in payload for key in ("start", "end")):
            raw_items = [payload]
    else:
        raw_items = []

    cues: list[Cue] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        cue = cue_from_mapping(len(cues), source_name, raw)
        if cue is not None:
            cues.append(cue)
    return cues


def load_subtitle_file(path: Path) -> list[Cue]:
    source_name = path.stem
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return load_srt_file(path, source_name)
    if suffix == ".json":
        return load_json_file(path, source_name)
    raise ValueError(f"unsupported subtitle file type: {path}")


def discover_modes(video_stem: str, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    found = []
    for mode in DEFAULT_MODES:
        if video_artifact_path(video_stem, f"{video_stem}.{mode}.bilingual.json").exists():
            found.append(mode)
    if not found:
        for path in iter_video_artifacts(video_stem, f"{video_stem}.*.bilingual.json"):
            suffix = path.name.removeprefix(f"{video_stem}.").removesuffix(".bilingual.json")
            found.append(suffix)
    return found


def find_best_match(base: Cue, candidates: list[Cue], cursor: int) -> tuple[Cue | None, int, float, float, float]:
    best: Cue | None = None
    best_index = cursor
    best_score = 0.0
    best_text = 0.0
    best_translation = 0.0
    window_start = max(0, cursor - 10)
    window_end = min(len(candidates), cursor + 28)

    for index in range(window_start, window_end):
        candidate = candidates[index]
        ov = overlap(base, candidate)
        if ov <= 0:
            if candidate.start > base.end + 10.0:
                break
            continue
        overlap_ratio = ov / max(0.001, max(base.duration, candidate.duration))
        ja_ratio = text_similarity(base.ja_text, candidate.ja_text)
        zh_ratio = text_similarity(base.zh_text, candidate.zh_text)
        score = overlap_ratio * 0.64 + ja_ratio * 0.24 + zh_ratio * 0.12
        if score > best_score:
            best = candidate
            best_index = index
            best_score = score
            best_text = ja_ratio
            best_translation = zh_ratio

    if best is None or best_score < 0.12:
        return None, cursor, 0.0, 0.0, 0.0
    return best, best_index + 1, best_score, best_text, best_translation


def align_modes(cues_by_mode: dict[str, list[Cue]], base_mode: str) -> list[tuple[Cue, dict[str, Match]]]:
    base_cues = cues_by_mode[base_mode]
    cursors = {mode: 0 for mode in cues_by_mode if mode != base_mode}
    rows: list[tuple[Cue, dict[str, Match]]] = []
    for base in base_cues:
        matches: dict[str, Match] = {}
        for mode, cues in cues_by_mode.items():
            if mode == base_mode:
                matches[mode] = Match(mode, base, 1.0, 1.0, 1.0, 0.0, 0.0)
                continue
            cue, next_cursor, score, ja_ratio, zh_ratio = find_best_match(base, cues, cursors[mode])
            cursors[mode] = next_cursor
            matches[mode] = Match(
                mode=mode,
                cue=cue,
                score=score,
                text_similarity=ja_ratio,
                translation_similarity=zh_ratio,
                time_delta_start=(cue.start - base.start) if cue else math.nan,
                time_delta_end=(cue.end - base.end) if cue else math.nan,
            )
        rows.append((base, matches))
    return rows


def html_page(title: str, body: str, extra_head: str = "") -> str:
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #17202b; }}
header {{ padding: 22px 28px 14px; background: #fff; border-bottom: 1px solid #d8dee8; position: sticky; top: 0; z-index: 3; }}
h1 {{ margin: 0 0 8px; font-size: 22px; }}
p {{ margin: 4px 0; color: #536174; }}
main {{ padding: 18px 22px 40px; }}
table {{ width: 100%; border-collapse: collapse; table-layout: fixed; background: #fff; border: 1px solid #d8dee8; }}
th, td {{ border: 1px solid #d8dee8; vertical-align: top; padding: 8px; }}
th {{ position: sticky; top: 94px; z-index: 2; background: #edf2f7; text-align: left; font-size: 13px; }}
.time {{ width: 116px; color: #5f6b7b; font-size: 12px; white-space: nowrap; background: #fbfcfe; }}
.jp {{ font-size: 15px; line-height: 1.45; word-break: break-word; }}
.zh {{ margin-top: 6px; font-size: 14px; color: #384658; line-height: 1.45; word-break: break-word; }}
.meta {{ margin-top: 6px; color: #6b7584; font-size: 11px; line-height: 1.35; }}
.wordline {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 7px; }}
.word {{ display: inline-flex; flex-direction: column; gap: 1px; max-width: 160px; border: 1px solid #cfd7e3; background: #fbfcfe; border-radius: 4px; padding: 3px 5px; }}
.word-text {{ font-size: 13px; line-height: 1.2; word-break: break-word; }}
.word-time {{ color: #6b7584; font-size: 10px; white-space: nowrap; }}
.same {{ background: #f2fbf5; }}
.close {{ background: #fff9e8; }}
.diff {{ background: #fff0f0; }}
.missing {{ background: #f1f2f4; color: #8b95a3; text-align: center; }}
.tag {{ display: inline-block; margin: 0 4px 4px 0; padding: 2px 6px; border-radius: 4px; background: #e7edf5; color: #334256; font-size: 12px; }}
.sev-high {{ background: #ffe5e5; }}
.sev-medium {{ background: #fff2cc; }}
.sev-low {{ background: #eef7ff; }}
button {{ border: 1px solid #aeb8c6; background: #fff; border-radius: 4px; padding: 5px 8px; cursor: pointer; }}
button:hover {{ background: #eef3f8; }}
video {{ width: 100%; max-height: 56vh; background: #000; }}
.review-layout {{ display: grid; grid-template-columns: minmax(420px, 0.9fr) minmax(520px, 1.1fr); gap: 16px; align-items: start; min-height: calc(100vh - 124px); }}
.review-layout .panel {{ position: sticky; top: 112px; max-height: calc(100vh - 128px); overflow: auto; }}
.panel {{ background: #fff; border: 1px solid #d8dee8; padding: 12px; }}
.items {{ display: grid; gap: 10px; position: relative; max-height: calc(100vh - 128px); overflow-y: auto; padding-right: 6px; scroll-padding-top: 10px; }}
.item {{ border: 1px solid #d8dee8; background: #fff; padding: 10px; }}
.item.active {{ outline: 2px solid #2f6fed; }}
.qc-controls {{ margin-top: 10px; display: grid; gap: 8px; border-top: 1px solid #d8dee8; padding-top: 8px; }}
.qc-row {{ display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; }}
.qc-row label {{ font-size: 12px; color: #334256; }}
.qc-controls textarea {{ width: 100%; min-height: 46px; box-sizing: border-box; border: 1px solid #aeb8c6; border-radius: 4px; padding: 6px; font-family: inherit; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
.missing-row {{ background: #f6f7f9; }}
pre {{ white-space: pre-wrap; word-break: break-word; }}
@media (max-width: 980px) {{
  .review-layout {{ display: block; min-height: 0; }}
  .review-layout .panel {{ position: sticky; top: 84px; z-index: 2; max-height: none; margin-bottom: 12px; }}
  .items {{ max-height: none; overflow: visible; padding-right: 0; }}
  video {{ max-height: 42vh; }}
}}
</style>
{extra_head}
</head>
<body>
{body}
</body>
</html>
"""


def class_for_similarity(value: float) -> str:
    if value >= 0.96:
        return "same"
    if value >= 0.82:
        return "close"
    return "diff"


def safe_delta(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:+.2f}s"


def match_cell(base: Cue, match: Match, *, include_translation: bool) -> str:
    if match.cue is None:
        return '<td class="missing">-</td>'
    cue = match.cue
    cls = class_for_similarity(match.translation_similarity if include_translation else match.text_similarity)
    meta = (
        f"#{cue.index + 1} {fmt_time(cue.start)}-{fmt_time(cue.end)} "
        f"ja={match.text_similarity:.2f} zh={match.translation_similarity:.2f} "
        f"ds={safe_delta(match.time_delta_start)} de={safe_delta(match.time_delta_end)}"
    )
    zh = f'<div class="zh">{html.escape(cue.zh_text)}</div>' if include_translation else ""
    return (
        f'<td class="{cls}">'
        f'<div class="jp">{html.escape(cue.ja_text)}</div>'
        f"{zh}"
        f'<div class="meta">{html.escape(meta)}</div>'
        "</td>"
    )


def build_compare_report(
    *,
    video_stem: str,
    modes: list[str],
    base_mode: str,
    rows: list[tuple[Cue, dict[str, Match]]],
    include_translation: bool,
) -> str:
    title = (
        f"{video_stem} translation compare"
        if include_translation
        else f"{video_stem} Japanese transcript compare"
    )
    counts = {"same": 0, "close": 0, "diff": 0, "missing": 0}
    table_rows = []
    for base, matches in rows:
        base_zh = f'<div class="zh">{html.escape(base.zh_text)}</div>' if include_translation else ""
        cells = [
            "<td>"
            f'<div class="jp">{html.escape(base.ja_text)}</div>'
            f"{base_zh}"
            f'<div class="meta">#{base.index + 1} {html.escape(base.gender)}</div>'
            "</td>"
        ]
        for mode in modes:
            if mode == base_mode:
                continue
            match = matches[mode]
            if match.cue is None:
                counts["missing"] += 1
            else:
                value = match.translation_similarity if include_translation else match.text_similarity
                counts[class_for_similarity(value)] += 1
            cells.append(match_cell(base, match, include_translation=include_translation))
        table_rows.append(
            "<tr>"
            f'<td class="time"><div>{fmt_time(base.start)}</div><div>{fmt_time(base.end)}</div><div>{base.duration:.2f}s</div></td>'
            + "".join(cells)
            + "</tr>"
        )

    headers = "".join(f"<th>{html.escape(mode)}</th>" for mode in modes)
    stat = ", ".join(f"{key}={value}" for key, value in counts.items())
    body = f"""
<header>
<h1>{html.escape(title)}</h1>
<p>Base mode: {html.escape(base_mode)}. Rows are aligned by cue time overlap, then text similarity.</p>
<p>Stats: {html.escape(stat)}</p>
</header>
<main>
<table>
<thead><tr><th class="time">time</th>{headers}</tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
</main>
"""
    return html_page(title, body)


def build_history_compare_report(
    *,
    video_stem: str,
    source_paths: list[Path],
    sources: list[str],
    base_source: str,
    rows: list[tuple[Cue, dict[str, Match]]],
) -> str:
    title = f"{video_stem} historical Japanese subtitle compare"
    counts = {"same": 0, "close": 0, "diff": 0, "missing": 0}
    table_rows = []
    for base, matches in rows:
        cells = [
            "<td>"
            f'<div class="jp">{html.escape(base.ja_text)}</div>'
            f'<div class="meta">#{base.index + 1}</div>'
            "</td>"
        ]
        row_classes = []
        for source in sources:
            if source == base_source:
                continue
            match = matches[source]
            if match.cue is None:
                counts["missing"] += 1
                row_classes.append("missing-row")
            else:
                counts[class_for_similarity(match.text_similarity)] += 1
                if match.text_similarity < 0.82:
                    row_classes.append("diff")
            cells.append(match_cell(base, match, include_translation=False))
        table_rows.append(
            f'<tr class="{" ".join(row_classes)}">'
            f'<td class="time"><div>{fmt_time(base.start)}</div><div>{fmt_time(base.end)}</div><div>{base.duration:.2f}s</div></td>'
            + "".join(cells)
            + "</tr>"
        )

    headers = "".join(f"<th>{html.escape(source)}</th>" for source in sources)
    stat = ", ".join(f"{key}={value}" for key, value in counts.items())
    inputs = "".join(f"<li>{html.escape(project_rel(path) if path.is_relative_to(PROJECT_ROOT) else str(path))}</li>" for path in source_paths)
    body = f"""
<header>
<h1>{html.escape(title)}</h1>
<p>Base source: {html.escape(base_source)}. Rows are aligned by cue time overlap, then Japanese text similarity.</p>
<p>Stats: {html.escape(stat)}</p>
</header>
<main>
<section class="panel">
<p>Inputs:</p>
<ul>{inputs}</ul>
</section>
<table>
<thead><tr><th class="time">time</th>{headers}</tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
</main>
"""
    return html_page(title, body)


def build_history_review_items(
    rows: list[tuple[Cue, dict[str, Match]]],
    sources: list[str],
    base_source: str,
    limit: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for base, matches in rows:
        missing = []
        variants: dict[str, dict[str, Any]] = {}
        worst_similarity = 1.0
        max_shift = 0.0
        for source in sources:
            match = matches[source]
            cue = match.cue
            variants[source] = {
                "matched": cue is not None,
                "index": cue.index if cue else None,
                "start": cue.start if cue else None,
                "end": cue.end if cue else None,
                "ja_text": cue.ja_text if cue else "",
                "text_similarity": match.text_similarity,
                "time_delta_start": None if math.isnan(match.time_delta_start) else match.time_delta_start,
                "time_delta_end": None if math.isnan(match.time_delta_end) else match.time_delta_end,
            }
            if source == base_source:
                continue
            if cue is None:
                missing.append(source)
                continue
            worst_similarity = min(worst_similarity, match.text_similarity)
            max_shift = max(max_shift, abs(match.time_delta_start), abs(match.time_delta_end))
        labels = []
        reasons = []
        if missing:
            labels.append("missing-match")
            reasons.append(f"missing in {', '.join(missing)}")
        if worst_similarity < 0.68:
            labels.append("large-japanese-diff")
            reasons.append(f"worst JA similarity {worst_similarity:.2f}")
        elif worst_similarity < 0.84:
            labels.append("moderate-japanese-diff")
            reasons.append(f"worst JA similarity {worst_similarity:.2f}")
        if max_shift > 2.0:
            labels.append("time-shift")
            reasons.append(f"max boundary shift {max_shift:.2f}s")
        if not labels:
            continue
        severity = "high" if missing or worst_similarity < 0.68 else "medium" if max_shift > 2.0 else "low"
        items.append(
            {
                "item_id": f"{base.index + 1:05d}",
                "start": max(0.0, base.start - 1.0),
                "end": base.end + 1.25,
                "severity": severity,
                "labels": labels,
                "reason": "; ".join(reasons),
                "base_index": base.index,
                "base_ja": base.ja_text,
                "variants": variants,
            }
        )
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda item: (severity_rank.get(str(item["severity"]), 9), float(item["start"])))
    return items[:limit]


def words_html(cue: Cue | None) -> str:
    if cue is None:
        return '<div class="missing">missing</div>'
    if not cue.words:
        return '<div class="meta">No word timestamps</div>'
    chips = []
    for word in cue.words:
        start = word.get("start")
        end = word.get("end")
        if isinstance(start, float) and isinstance(end, float):
            time_bits = f"{fmt_time(start)}-{fmt_time(end)}"
        else:
            time_bits = "no time"
        gender = f" {word['gender']}" if word.get("gender") else ""
        chips.append(
            '<span class="word">'
            f'<span class="word-text">{html.escape(str(word["word"]))}</span>'
            f'<span class="word-time">{html.escape(time_bits + gender)}</span>'
            "</span>"
        )
    return '<div class="wordline">' + "".join(chips) + "</div>"


def build_wordline_report(
    *,
    video_stem: str,
    modes: list[str],
    base_mode: str,
    rows: list[tuple[Cue, dict[str, Match]]],
) -> str:
    title = f"{video_stem} Japanese word-by-word report"
    table_rows = []
    for base, matches in rows:
        cells = []
        for mode in modes:
            match = matches[mode]
            cue = match.cue
            if cue is None:
                cells.append('<td class="missing">-</td>')
                continue
            cls = class_for_similarity(match.text_similarity)
            meta = (
                f"#{cue.index + 1} {fmt_time(cue.start)}-{fmt_time(cue.end)} "
                f"words={len(cue.words)} ja={match.text_similarity:.2f} "
                f"ds={match.time_delta_start:+.2f}s de={match.time_delta_end:+.2f}s"
            )
            cells.append(
                f'<td class="{cls}">'
                f'<div class="jp">{html.escape(cue.ja_text)}</div>'
                f"{words_html(cue)}"
                f'<div class="meta">{html.escape(meta)}</div>'
                "</td>"
            )
        table_rows.append(
            "<tr>"
            f'<td class="time"><div>{fmt_time(base.start)}</div><div>{fmt_time(base.end)}</div><div>{base.duration:.2f}s</div></td>'
            + "".join(cells)
            + "</tr>"
        )

    headers = "".join(f"<th>{html.escape(mode)}</th>" for mode in modes)
    body = f"""
<header>
<h1>{html.escape(title)}</h1>
<p>Base mode: {html.escape(base_mode)}. Each row keeps the cue-level Japanese text and available word timestamps.</p>
</header>
<main>
<table>
<thead><tr><th class="time">time</th>{headers}</tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
</main>
"""
    return html_page(title, body)


def build_review_items(rows: list[tuple[Cue, dict[str, Match]]], modes: list[str], base_mode: str, limit: int) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    for base, matches in rows:
        labels: list[str] = []
        reasons: list[str] = []
        variant_payload: dict[str, dict[str, Any]] = {}
        worst_similarity = 1.0
        max_shift = 0.0
        missing_count = 0
        for mode in modes:
            match = matches[mode]
            cue = match.cue
            variant_payload[mode] = {
                "matched": cue is not None,
                "index": cue.index if cue else None,
                "start": cue.start if cue else None,
                "end": cue.end if cue else None,
                "ja_text": cue.ja_text if cue else "",
                "zh_text": cue.zh_text if cue else "",
                "word_count": len(cue.words) if cue else 0,
                "text_similarity": match.text_similarity,
                "translation_similarity": match.translation_similarity,
                "time_delta_start": None if math.isnan(match.time_delta_start) else match.time_delta_start,
                "time_delta_end": None if math.isnan(match.time_delta_end) else match.time_delta_end,
            }
            if mode == base_mode:
                continue
            if cue is None:
                missing_count += 1
                continue
            worst_similarity = min(worst_similarity, match.text_similarity, match.translation_similarity)
            max_shift = max(max_shift, abs(match.time_delta_start), abs(match.time_delta_end))

        if missing_count:
            labels.append("missing-match")
            reasons.append(f"{missing_count} VAD mode(s) have no time-overlap match")
        if worst_similarity < 0.68:
            labels.append("large-text-or-translation-diff")
            reasons.append(f"worst similarity {worst_similarity:.2f}")
        elif worst_similarity < 0.84:
            labels.append("moderate-diff")
            reasons.append(f"worst similarity {worst_similarity:.2f}")
        if max_shift > 2.0:
            labels.append("time-shift")
            reasons.append(f"max boundary shift {max_shift:.2f}s")
        if base.duration < 0.75:
            labels.append("very-short-cue")
            reasons.append(f"base duration {base.duration:.2f}s")
        if not base.ja_text:
            labels.append("empty-ja")
            reasons.append("base Japanese text is empty")
        if not base.zh_text:
            labels.append("empty-translation")
            reasons.append("base translation is empty")

        if not labels:
            continue
        severity = "high" if missing_count or worst_similarity < 0.68 else "medium" if max_shift > 2.0 or worst_similarity < 0.84 else "low"
        start = max(0.0, base.start - 1.0)
        end = base.end + 1.25
        items.append(
            ReviewItem(
                item_id=f"{base.index + 1:05d}",
                start=start,
                end=end,
                severity=severity,
                labels=labels,
                reason="; ".join(reasons),
                base_index=base.index,
                base_ja=base.ja_text,
                base_zh=base.zh_text,
                variants=variant_payload,
            )
        )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda item: (severity_rank.get(item.severity, 9), item.start))
    return items[:limit]


def build_review_index(video_path: Path, video_stem: str, modes: list[str], items: list[ReviewItem]) -> str:
    video_href = Path("../../..") / video_path.relative_to(PROJECT_ROOT)
    cards = []
    for item in items:
        tags = "".join(f'<span class="tag">{html.escape(label)}</span>' for label in item.labels)
        status_controls = "".join(
            f'<label><input type="radio" name="status-{html.escape(item.item_id)}" value="{html.escape(status)}"> {html.escape(status)}</label>'
            for status in LABEL_STATUSES
        )
        issue_controls = "".join(
            f'<label><input type="checkbox" data-issue="{html.escape(issue)}"> {html.escape(issue)}</label>'
            for issue in LABEL_ISSUES
        )
        variant_bits = []
        for mode in modes:
            variant = item.variants[mode]
            if not variant["matched"]:
                variant_bits.append(f"<h4>{html.escape(mode)}</h4><p class=\"missing\">missing</p>")
                continue
            variant_bits.append(
                f"<h4>{html.escape(mode)}</h4>"
                f"<p><strong>JA</strong>: {html.escape(variant['ja_text'])}</p>"
                f"<p><strong>ZH</strong>: {html.escape(variant['zh_text'])}</p>"
                f"<p class=\"meta\">"
                f"{fmt_time(float(variant['start']))}-{fmt_time(float(variant['end']))} "
                f"ja={variant['text_similarity']:.2f} zh={variant['translation_similarity']:.2f}"
                "</p>"
            )
        cards.append(
            f"""
<article class="item sev-{html.escape(item.severity)}" id="item-{html.escape(item.item_id)}" data-start="{item.start:.3f}" data-end="{item.end:.3f}">
<p><button type="button" data-play="{html.escape(item.item_id)}">Play segment</button> <strong>{html.escape(item.item_id)}</strong> {fmt_time(item.start)}-{fmt_time(item.end)} {tags}</p>
<p class="meta">{html.escape(item.reason)}</p>
<p><strong>Base JA</strong>: {html.escape(item.base_ja)}</p>
<p><strong>Base ZH</strong>: {html.escape(item.base_zh)}</p>
<details><summary>All variants</summary>{''.join(variant_bits)}</details>
<div class="qc-controls" data-qc-item="{html.escape(item.item_id)}">
<div class="qc-row"><strong>Review</strong> {status_controls}</div>
<div class="qc-row"><strong>Issue</strong> {issue_controls}</div>
<textarea data-note placeholder="review note"></textarea>
</div>
</article>
"""
        )

    script = """
<script>
let stopAt = null;
const storageKey = "subtitle-qc:__VIDEO_STEM__:labels";
const player = () => document.getElementById("review-video");
function scrollItemIntoList(item) {
  const list = document.querySelector(".items");
  if (!list) return;
  if (getComputedStyle(list).overflowY === "visible") {
    item.scrollIntoView({block: "center", behavior: "smooth"});
    return;
  }
  const targetTop = item.offsetTop - list.offsetTop - Math.max(8, (list.clientHeight - item.clientHeight) / 2);
  list.scrollTo({top: Math.max(0, targetTop), behavior: "smooth"});
}
function readLabels() {
  try {
    return JSON.parse(localStorage.getItem(storageKey) || "{}");
  } catch (error) {
    return {};
  }
}
function writeLabels(labels) {
  localStorage.setItem(storageKey, JSON.stringify(labels));
  updateLabelCount(labels);
}
function updateLabelCount(labels) {
  const count = Object.values(labels).filter(item => item.status && item.status !== "unreviewed").length;
  const target = document.getElementById("label-count");
  if (target) target.textContent = String(count);
}
function collectControls(controls) {
  const status = controls.querySelector("input[type=radio]:checked")?.value || "unreviewed";
  const issues = Array.from(controls.querySelectorAll("input[type=checkbox]:checked")).map(input => input.dataset.issue);
  const note = controls.querySelector("[data-note]")?.value || "";
  return {status, issues, note};
}
function applyControls(controls, state) {
  const status = state.status || "unreviewed";
  const radio = controls.querySelector(`input[type=radio][value="${CSS.escape(status)}"]`);
  if (radio) radio.checked = true;
  controls.querySelectorAll("input[type=checkbox]").forEach(input => {
    input.checked = Array.isArray(state.issues) && state.issues.includes(input.dataset.issue);
  });
  const note = controls.querySelector("[data-note]");
  if (note) note.value = state.note || "";
}
function activate(id) {
  document.querySelectorAll(".item.active").forEach(el => el.classList.remove("active"));
  const item = document.getElementById("item-" + id);
  if (!item) return;
  item.classList.add("active");
  const start = Number(item.dataset.start || "0");
  const end = Number(item.dataset.end || String(start + 3));
  stopAt = end;
  const video = player();
  video.currentTime = start;
  video.play();
  scrollItemIntoList(item);
}
document.addEventListener("click", event => {
  const target = event.target.closest("[data-play]");
  if (target) {
    activate(target.dataset.play);
    return;
  }
  if (event.target.closest("[data-export-labels]")) {
    const payload = {
      video: "__VIDEO_PATH__",
      exported_at: new Date().toISOString(),
      labels: readLabels()
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "__VIDEO_STEM__.subtitle_qc_labels.json";
    link.click();
    URL.revokeObjectURL(link.href);
  }
});
document.addEventListener("input", event => {
  const controls = event.target.closest("[data-qc-item]");
  if (!controls) return;
  const labels = readLabels();
  labels[controls.dataset.qcItem] = collectControls(controls);
  writeLabels(labels);
});
document.addEventListener("DOMContentLoaded", () => {
  const labels = readLabels();
  document.querySelectorAll("[data-qc-item]").forEach(controls => {
    applyControls(controls, labels[controls.dataset.qcItem] || {});
  });
  updateLabelCount(labels);
  const video = player();
  video.addEventListener("timeupdate", () => {
    if (stopAt !== null && video.currentTime >= stopAt) {
      video.pause();
      stopAt = null;
    }
  });
});
</script>
""".replace("__VIDEO_STEM__", html.escape(video_stem)).replace("__VIDEO_PATH__", html.escape(project_rel(video_path)))
    body = f"""
<header>
<h1>{html.escape(video_stem)} human review index</h1>
<p>Fast mode: the page seeks inside the original video. No clips were generated.</p>
<div class="toolbar">
<span>Review candidates: {len(items)}</span>
<span>Reviewed: <strong id="label-count">0</strong></span>
<button type="button" data-export-labels>Export labels JSON</button>
</div>
</header>
<main class="review-layout">
<section class="panel">
<video id="review-video" controls preload="metadata" src="{html.escape(video_href.as_posix())}"></video>
<p class="meta">Video: {html.escape(project_rel(video_path))}</p>
</section>
<section class="items">
{''.join(cards)}
</section>
</main>
{script}
"""
    return html_page(f"{video_stem} review index", body)


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run(args, check=True)


def make_clips(video_path: Path, output_dir: Path, items: list[ReviewItem], *, compilation: bool) -> None:
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []
    for item in items:
        out = clips_dir / f"{item.item_id}_{fmt_time(item.start).replace(':', '').replace('.', '')}.mp4"
        duration = max(0.5, item.end - item.start)
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{item.start:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{duration:.3f}",
                "-c",
                "copy",
                str(out),
            ]
        )
        clip_paths.append(out)
    if compilation and clip_paths:
        list_path = output_dir / "clips.txt"
        list_path.write_text(
            "".join(f"file '{path.resolve().as_posix()}'\n" for path in clip_paths),
            encoding="utf-8",
        )
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_dir / "review_compilation.mp4"),
            ]
        )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def unique_source_name(path: Path, used: set[str]) -> str:
    name = path.stem
    if name not in used:
        used.add(name)
        return name
    parent_name = path.parent.name
    candidate = f"{parent_name}-{name}" if parent_name else name
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = 2
    while f"{candidate}-{suffix}" in used:
        suffix += 1
    final = f"{candidate}-{suffix}"
    used.add(final)
    return final


def load_history_sources(paths: list[str]) -> tuple[list[Path], dict[str, list[Cue]]]:
    resolved_paths = [resolve_project_path(value).resolve() for value in paths]
    used_names: set[str] = set()
    sources: dict[str, list[Cue]] = {}
    for path in resolved_paths:
        if not path.exists():
            raise SystemExit(f"history subtitle not found: {path}")
        source_name = unique_source_name(path, used_names)
        suffix = path.suffix.lower()
        if suffix == ".srt":
            cues = load_srt_file(path, source_name)
        elif suffix == ".json":
            cues = load_json_file(path, source_name)
        else:
            raise SystemExit(f"unsupported history subtitle file type: {path}")
        sources[source_name] = [
            Cue(
                index=cue.index,
                mode=source_name,
                start=cue.start,
                end=cue.end,
                ja_text=cue.ja_text,
                zh_text=cue.zh_text,
                gender=cue.gender,
                words=cue.words,
            )
            for cue in cues
        ]
    return resolved_paths, sources


def write_history_reports(
    *,
    output_dir: Path,
    video_stem: str,
    source_paths: list[Path],
    sources_by_name: dict[str, list[Cue]],
    base_source: str | None,
    limit: int,
) -> dict[str, Any]:
    if len(sources_by_name) < 2:
        raise SystemExit("at least two --history-subtitles files are required")
    sources = list(sources_by_name)
    selected_base = base_source or sources[0]
    if selected_base not in sources_by_name:
        raise SystemExit(f"history base {selected_base!r} not in sources: {sources}")
    rows = align_modes(sources_by_name, selected_base)
    items = build_history_review_items(rows, sources, selected_base, limit)
    (output_dir / "history_japanese_compare.html").write_text(
        build_history_compare_report(
            video_stem=video_stem,
            source_paths=source_paths,
            sources=sources,
            base_source=selected_base,
            rows=rows,
        ),
        encoding="utf-8",
    )
    write_json(output_dir / "history_review_items.json", items)
    return {
        "base_source": selected_base,
        "sources": sources,
        "counts": {source: len(cues) for source, cues in sources_by_name.items()},
        "review_item_count": len(items),
        "outputs": {
            "history_japanese_compare": "history_japanese_compare.html",
            "history_review_items": "history_review_items.json",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local subtitle QA reports.")
    parser.add_argument("--video", required=True, help="Path to source video, e.g. video/MKMP-577.mp4")
    parser.add_argument("--modes", nargs="*", help="VAD/result suffixes to compare")
    parser.add_argument("--base-mode", default=DEFAULT_BASE_MODE)
    parser.add_argument("--history-subtitles", nargs="*", help="SRT/JSON subtitle files for same-video historical JA comparison")
    parser.add_argument("--history-base", help="Source name to use as historical subtitle base. Defaults to first file stem.")
    parser.add_argument("--history-only", action="store_true", help="Only generate history subtitle comparison reports")
    parser.add_argument("--review-limit", type=int, default=160)
    parser.add_argument("--make-clips", action="store_true", help="Cut one mp4 per review item")
    parser.add_argument("--make-compilation", action="store_true", help="Cut clips and concatenate them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = (PROJECT_ROOT / args.video).resolve() if not Path(args.video).is_absolute() else Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")
    video_stem = video_path.stem
    if args.history_only and not args.history_subtitles:
        raise SystemExit("--history-only requires --history-subtitles")
    modes = [] if args.history_only else discover_modes(video_stem, args.modes)
    run_variant_compare = bool(modes) and not args.history_only
    if run_variant_compare and args.base_mode not in modes:
        raise SystemExit(f"base mode {args.base_mode!r} not in modes: {modes}")
    missing = [
        mode
        for mode in modes
        if not video_artifact_path(video_stem, f"{video_stem}.{mode}.bilingual.json").exists()
    ]
    if missing:
        raise SystemExit(f"missing bilingual json for modes: {missing}")
    if not run_variant_compare and not args.history_subtitles:
        raise SystemExit(f"no bilingual json found for {video_stem}; pass --history-subtitles for history-only comparison")

    output_dir = subtitle_qc_output_dir(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    cues_by_mode: dict[str, list[Cue]] = {}
    rows: list[tuple[Cue, dict[str, Match]]] = []
    items: list[ReviewItem] = []
    history_summary = None
    entry_path: Path | None = None
    summary_payload: dict[str, Any] = {
        "video": project_rel(video_path),
        "outputs": {},
        "clip_generation": {
            "make_clips": bool(args.make_clips or args.make_compilation),
            "make_compilation": bool(args.make_compilation),
            "ffmpeg_available": shutil.which("ffmpeg") is not None,
        },
    }

    if run_variant_compare:
        cues_by_mode = {mode: load_cues(video_stem, mode) for mode in modes}
        rows = align_modes(cues_by_mode, args.base_mode)
        items = build_review_items(rows, modes, args.base_mode, args.review_limit)
        (output_dir / "japanese_transcript_compare.html").write_text(
            build_compare_report(
                video_stem=video_stem,
                modes=modes,
                base_mode=args.base_mode,
                rows=rows,
                include_translation=False,
            ),
            encoding="utf-8",
        )
        (output_dir / "japanese_wordline_report.html").write_text(
            build_wordline_report(
                video_stem=video_stem,
                modes=modes,
                base_mode=args.base_mode,
                rows=rows,
            ),
            encoding="utf-8",
        )
        (output_dir / "translation_compare.html").write_text(
            build_compare_report(
                video_stem=video_stem,
                modes=modes,
                base_mode=args.base_mode,
                rows=rows,
                include_translation=True,
            ),
            encoding="utf-8",
        )
        (output_dir / "review_index.html").write_text(
            build_review_index(video_path, video_stem, modes, items),
            encoding="utf-8",
        )
        entry_path = output_dir / "review_index.html"
        write_json(output_dir / "review_items.json", [asdict(item) for item in items])
        summary_payload.update(
            {
                "base_mode": args.base_mode,
                "modes": modes,
                "counts": {mode: len(cues) for mode, cues in cues_by_mode.items()},
                "review_item_count": len(items),
                "outputs": {
                    "japanese_transcript_compare": "japanese_transcript_compare.html",
                    "japanese_wordline_report": "japanese_wordline_report.html",
                    "translation_compare": "translation_compare.html",
                    "review_index": "review_index.html",
                    "review_items": "review_items.json",
                },
            }
        )
    if args.history_subtitles:
        history_paths, history_sources = load_history_sources(args.history_subtitles)
        history_summary = write_history_reports(
            output_dir=output_dir,
            video_stem=video_stem,
            source_paths=history_paths,
            sources_by_name=history_sources,
            base_source=args.history_base,
            limit=args.review_limit,
        )

    if history_summary is not None:
        summary_payload["history_subtitle_compare"] = history_summary
        if not run_variant_compare:
            entry_path = output_dir / "history_japanese_compare.html"
    write_json(output_dir / "summary.json", summary_payload)
    if args.make_clips or args.make_compilation:
        make_clips(video_path, output_dir, items, compilation=bool(args.make_compilation))

    print(output_dir.relative_to(PROJECT_ROOT))
    if entry_path is not None:
        print(entry_path)


if __name__ == "__main__":
    main()
