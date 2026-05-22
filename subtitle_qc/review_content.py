#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate import (  # noqa: E402
    DEFAULT_MODES,
    PROJECT_ROOT,
    Cue,
    canonical_mode_label,
    fmt_time,
    load_json_file,
    load_srt_file,
    normalize_text,
    project_rel,
    resolve_project_path,
    subtitle_qc_output_dir,
    write_json,
)


VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm")
DISCOVERY_LABELS = DEFAULT_MODES
ASCII_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b")
JA_CHAR_RE = re.compile(r"[ぁ-ゖァ-ヺー一-龯]")
NOISE_RE = re.compile(
    r"("
    r"\bDJ\b|Music|Hook|Scene|Play|Thank\s+you|"
    r"おはようございます|ありがとうございました|ありがとうございます|"
    r"クリスマス|猫|大画像|"
    r"ボンジュール"
    r")",
    re.IGNORECASE,
)
REPEAT_RE = re.compile(r"(.{2,12})\1{2,}")
FRAGMENT_ENDINGS = (
    "を",
    "に",
    "が",
    "は",
    "も",
    "と",
    "で",
    "から",
    "この",
    "その",
    "し",
    "く",
)
OK_FRAGMENT_ENDINGS = (
    "して",
    "見て",
    "触って",
    "入れて",
    "出して",
    "舐めて",
    "近づいて",
    "続けて",
    "来て",
    "任せたいよ",
)
FLAG_WEIGHTS = {
    "likely_noise": 4.0,
    "ascii_words": 3.0,
    "repeat": 3.0,
    "long_text": 2.0,
    "fast_cps": 2.0,
    "packed": 1.5,
    "fragment": 1.5,
    "long_duration": 1.0,
    "too_short": 1.0,
}


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class ReviewConfig:
    long_duration_s: float
    long_text_chars: int
    fast_cps: float
    noise_patterns: tuple[re.Pattern[str], ...]


@dataclass
class CueIssue:
    label: str
    index: int
    start: float
    end: float
    duration_s: float
    flags: list[str]
    issue_score: float
    text: str


@dataclass
class CandidateMetrics:
    label: str
    path: str
    cue_count: int
    first: str
    last: str
    total_subtitle_s: float
    median_duration_s: float
    median_gap_s: float
    gap_gt8: int
    gap_gt20: int
    total_text_chars: int
    median_text_chars: float
    max_text_chars: int
    issue_cues: int
    issue_score: float
    long_duration: int
    long_text: int
    fast_cps: int
    ascii_words: int
    likely_noise: int
    repeat: int
    fragment: int
    packed: int
    too_short: int


@dataclass
class CandidateReview:
    spec: CandidateSpec
    cues: list[Cue]
    metrics: CandidateMetrics
    issues: list[CueIssue]


def display_path(path: Path) -> str:
    try:
        return project_rel(path)
    except ValueError:
        return str(path)


def safe_label(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    clean = clean.replace("-", "_")
    return clean.strip("_") or "candidate"


def subtitle_artifact_stem(path: Path) -> str:
    name = path.name
    lowered = name.lower()
    if lowered.endswith(".bilingual.json"):
        return name[: -len(".bilingual.json")]
    if lowered.endswith(".srt"):
        return name[: -len(".srt")]
    if lowered.endswith(".json"):
        return name[: -len(".json")]
    return path.stem


def normalized_candidate_suffix(video_stem: str, path: Path) -> str:
    label = subtitle_artifact_stem(path)
    prefix = f"{video_stem}."
    if label.lower().startswith(prefix.lower()):
        label = label[len(prefix) :]
    return safe_label(label).lower()


def unique_label(label: str, used: set[str]) -> str:
    clean = safe_label(label)
    if clean not in used:
        used.add(clean)
        return clean
    suffix = 2
    while f"{clean}_{suffix}" in used:
        suffix += 1
    final = f"{clean}_{suffix}"
    used.add(final)
    return final


def resolve_video(value: str) -> Path:
    raw = resolve_project_path(value).resolve()
    if raw.exists() or raw.suffix:
        return raw
    video_dir = PROJECT_ROOT / "video"
    for ext in VIDEO_EXTS:
        candidate = video_dir / f"{value}{ext}"
        if candidate.exists():
            return candidate.resolve()
    # The content-review tool can operate from existing subtitle artifacts even
    # when the original video is not available locally.
    return (video_dir / f"{value}.mp4").resolve()


def candidate_label_from_path(video_stem: str, path: Path) -> str:
    suffix = normalized_candidate_suffix(video_stem, path)
    try:
        return canonical_mode_label(suffix)
    except SystemExit as exc:
        allowed = ", ".join(DISCOVERY_LABELS)
        raise SystemExit(f"unrecognized candidate VAD label for {path.name!r}; allowed: {allowed}") from exc


def candidate_sort_key(spec: CandidateSpec) -> tuple[int, str]:
    try:
        index = DISCOVERY_LABELS.index(spec.label)
    except ValueError:
        index = len(DISCOVERY_LABELS)
    return index, spec.label


def candidate_path_priority(path: Path) -> int:
    if path.suffix.lower() == ".srt":
        return 0
    if path.name.endswith(".bilingual.json"):
        return 1
    return 2


def iter_subtitle_artifacts(video: Path) -> list[Path]:
    roots = [video.parent / video.stem, video.parent]
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_file():
                continue
            lowered = path.name.lower()
            if path.suffix.lower() != ".srt" and not lowered.endswith(".bilingual.json"):
                continue
            if path.name.endswith(".subtitle_qc_labels.json"):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return sorted(paths)


def discover_candidates(video: Path, *, all_candidates: bool, modes: list[str] | None) -> list[CandidateSpec]:
    requested = {canonical_mode_label(mode) for mode in modes or []}
    used: set[str] = set()
    specs: list[tuple[str, Path]] = []
    for path in iter_subtitle_artifacts(video):
        label = candidate_label_from_path(video.stem, path)
        if requested and label not in requested:
            continue
        if not all_candidates and not requested and label not in DEFAULT_MODES:
            continue
        specs.append((label, path.resolve()))
    if not all_candidates:
        best_by_label: dict[str, Path] = {}
        for label, path in specs:
            current = best_by_label.get(label)
            if current is None or candidate_path_priority(path) < candidate_path_priority(current):
                best_by_label[label] = path
        specs = sorted(best_by_label.items())
    candidates = [CandidateSpec(unique_label(label, used), path) for label, path in specs]
    return sorted(candidates, key=candidate_sort_key)


def parse_candidate_spec(spec: str, video_stem: str, used: set[str]) -> CandidateSpec:
    if "=" in spec:
        raw_label, raw_path = spec.split("=", 1)
        path = resolve_project_path(raw_path.strip()).resolve()
        label = safe_label(raw_label.strip())
    else:
        path = resolve_project_path(spec.strip()).resolve()
        label = candidate_label_from_path(video_stem, path)
    if not path.exists():
        raise SystemExit(f"candidate subtitle not found: {path}")
    return CandidateSpec(unique_label(label, used), path)


def load_candidate(spec: CandidateSpec) -> list[Cue]:
    suffix = spec.path.suffix.lower()
    if suffix == ".srt":
        cues = load_srt_file(spec.path, spec.label)
    elif suffix == ".json":
        cues = load_json_file(spec.path, spec.label)
    else:
        raise SystemExit(f"unsupported subtitle file type: {spec.path}")
    if not cues:
        raise SystemExit(f"no subtitle cues parsed from: {spec.path}")
    return cues


def cue_text(cue: Cue) -> str:
    return (cue.ja_text or cue.zh_text or "").strip()


def display_text(text: str) -> str:
    return " / ".join(line.strip() for line in text.splitlines() if line.strip())


def text_char_count(text: str) -> int:
    normalized = normalize_text(text)
    ja_count = len(JA_CHAR_RE.findall(normalized))
    return ja_count or len(normalized)


def has_repetition(text: str) -> bool:
    compact = normalize_text(text)
    if len(compact) < 6:
        return False
    if REPEAT_RE.search(compact):
        return True
    tokens = re.findall(r"[ぁ-ゖァ-ヺー一-龯A-Za-z0-9]{1,10}", text)
    if len(tokens) < 4:
        return False
    for index in range(len(tokens) - 3):
        if tokens[index] and tokens[index] == tokens[index + 1] == tokens[index + 2] == tokens[index + 3]:
            return True
    return False


def is_fragment(text: str) -> bool:
    compact = normalize_text(text).rstrip("、，,.!?！？…")
    if not compact:
        return False
    return compact.endswith(FRAGMENT_ENDINGS) and not compact.endswith(OK_FRAGMENT_ENDINGS)


def detect_issue_flags(cue: Cue, config: ReviewConfig) -> list[str]:
    text = cue_text(cue)
    if not text:
        return ["too_short"]
    flags: list[str] = []
    chars = text_char_count(text)
    duration = max(0.001, cue.duration)
    if cue.duration > config.long_duration_s:
        flags.append("long_duration")
    if chars >= config.long_text_chars:
        flags.append("long_text")
    if chars / duration > config.fast_cps:
        flags.append("fast_cps")
    if ASCII_WORD_RE.search(text):
        flags.append("ascii_words")
    if any(pattern.search(text) for pattern in config.noise_patterns):
        flags.append("likely_noise")
    if has_repetition(text):
        flags.append("repeat")
    if is_fragment(text):
        flags.append("fragment")
    if "\n" in text and chars > max(24, int(config.long_text_chars * 0.75)):
        flags.append("packed")
    if chars <= 2 or len(normalize_text(text)) <= 3:
        flags.append("too_short")
    return flags


def issue_score(flags: list[str]) -> float:
    return round(sum(FLAG_WEIGHTS.get(flag, 1.0) for flag in flags), 2)


def analyze_candidate(spec: CandidateSpec, cues: list[Cue], config: ReviewConfig) -> CandidateReview:
    issues: list[CueIssue] = []
    flag_counts = {flag: 0 for flag in FLAG_WEIGHTS}
    for cue in cues:
        flags = detect_issue_flags(cue, config)
        if not flags:
            continue
        for flag in flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        issues.append(
            CueIssue(
                label=spec.label,
                index=int(cue.index) + 1,
                start=cue.start,
                end=cue.end,
                duration_s=round(cue.duration, 3),
                flags=flags,
                issue_score=issue_score(flags),
                text=display_text(cue_text(cue)),
            )
        )

    durations = [cue.duration for cue in cues]
    gaps = [cues[index].start - cues[index - 1].end for index in range(1, len(cues))]
    char_counts = [text_char_count(cue_text(cue)) for cue in cues]
    metrics = CandidateMetrics(
        label=spec.label,
        path=display_path(spec.path),
        cue_count=len(cues),
        first=fmt_time(cues[0].start) if cues else "",
        last=fmt_time(cues[-1].end) if cues else "",
        total_subtitle_s=round(sum(durations), 3),
        median_duration_s=round(median(durations), 3) if durations else 0.0,
        median_gap_s=round(median(gaps), 3) if gaps else 0.0,
        gap_gt8=sum(1 for gap in gaps if gap > 8.0),
        gap_gt20=sum(1 for gap in gaps if gap > 20.0),
        total_text_chars=sum(char_counts),
        median_text_chars=round(median(char_counts), 3) if char_counts else 0.0,
        max_text_chars=max(char_counts) if char_counts else 0,
        issue_cues=len(issues),
        issue_score=round(sum(item.issue_score for item in issues), 2),
        long_duration=flag_counts.get("long_duration", 0),
        long_text=flag_counts.get("long_text", 0),
        fast_cps=flag_counts.get("fast_cps", 0),
        ascii_words=flag_counts.get("ascii_words", 0),
        likely_noise=flag_counts.get("likely_noise", 0),
        repeat=flag_counts.get("repeat", 0),
        fragment=flag_counts.get("fragment", 0),
        packed=flag_counts.get("packed", 0),
        too_short=flag_counts.get("too_short", 0),
    )
    return CandidateReview(spec=spec, cues=cues, metrics=metrics, issues=issues)


def build_review_config(args: argparse.Namespace) -> ReviewConfig:
    patterns = [NOISE_RE]
    for raw in args.noise_pattern or []:
        patterns.append(re.compile(raw, re.IGNORECASE))
    return ReviewConfig(
        long_duration_s=args.long_duration_s,
        long_text_chars=args.long_text_chars,
        fast_cps=args.fast_cps,
        noise_patterns=tuple(patterns),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(path: Path, reviews: list[CandidateReview]) -> None:
    rows = [asdict(review.metrics) for review in reviews]
    write_csv(path, rows, list(CandidateMetrics.__dataclass_fields__.keys()))


def write_issues_csv(path: Path, reviews: list[CandidateReview]) -> None:
    rows: list[dict[str, Any]] = []
    for review in reviews:
        for issue in review.issues:
            row = asdict(issue)
            row["start"] = fmt_time(issue.start)
            row["end"] = fmt_time(issue.end)
            row["flags"] = ";".join(issue.flags)
            rows.append(row)
    fieldnames = ["label", "index", "start", "end", "duration_s", "flags", "issue_score", "text"]
    write_csv(path, rows, fieldnames)


def sorted_reviews(reviews: list[CandidateReview]) -> list[CandidateReview]:
    return sorted(reviews, key=lambda item: (item.metrics.issue_score, item.metrics.issue_cues, -item.metrics.cue_count))


def write_summary_md(path: Path, video: Path, reviews: list[CandidateReview]) -> None:
    ranked = sorted_reviews(reviews)
    lines = [
        f"# {video.stem} Full Subtitle Content Review",
        "",
        f"- Video: `{display_path(video)}`",
        "- Ranking is heuristic. Use the issue rows and full text for the actual human review.",
        "",
        "## Candidate Ranking",
        "",
        "| rank | candidate | issue_score | issue_cues | cues | chars | first | last | noise | repeat | fragment | packed |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, review in enumerate(ranked, start=1):
        metrics = review.metrics
        lines.append(
            "| "
            f"{rank} | `{metrics.label}` | {metrics.issue_score:.2f} | {metrics.issue_cues} | "
            f"{metrics.cue_count} | {metrics.total_text_chars} | {metrics.first} | {metrics.last} | "
            f"{metrics.likely_noise} | {metrics.repeat} | {metrics.fragment} | {metrics.packed} |"
        )
    lines.extend(["", "## Paths", ""])
    for review in reviews:
        lines.append(f"- `{review.metrics.label}`: `{review.metrics.path}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `likely_noise` catches common non-dialogue or hallucination markers such as DJ/music/generic greetings.",
            "- `packed` means a cue contains multiple subtitle lines and is long enough to be hard to review at a glance.",
            "- `fragment` means the cue likely ends on an unfinished Japanese particle or clause.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_full_text_md(path: Path, reviews: list[CandidateReview]) -> None:
    lines = ["# Full Subtitle Text By Candidate", ""]
    for review in reviews:
        lines.extend([f"## {review.metrics.label}", ""])
        for cue in review.cues:
            lines.append(f"- {int(cue.index) + 1:04d} {fmt_time(cue.start)}-{fmt_time(cue.end)} {display_text(cue_text(cue))}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def review_windows(reviews: list[CandidateReview], window_minutes: float) -> list[tuple[str, float, float]]:
    max_end = max((cue.end for review in reviews for cue in review.cues), default=0.0)
    window_s = max(60.0, window_minutes * 60.0)
    count = max(1, math.ceil(max_end / window_s))
    return [(f"part_{index + 1}", index * window_s, min(max_end, (index + 1) * window_s)) for index in range(count)]


def write_window_compare_md(path: Path, reviews: list[CandidateReview], window_minutes: float) -> None:
    lines = ["# Full Window Comparison", ""]
    for title, start, end in review_windows(reviews, window_minutes):
        lines.extend([f"## {title} {fmt_time(start)}-{fmt_time(end)}", ""])
        for review in reviews:
            selected = [cue for cue in review.cues if cue.end >= start and cue.start <= end]
            lines.extend([f"### {review.metrics.label} ({len(selected)} cues)", ""])
            for cue in selected:
                lines.append(
                    f"- {int(cue.index) + 1:04d} {fmt_time(cue.start)}-{fmt_time(cue.end)} "
                    f"{display_text(cue_text(cue))}"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def review_payload(video: Path, reviews: list[CandidateReview], window_minutes: float) -> dict[str, Any]:
    return {
        "video": display_path(video),
        "metrics": [asdict(review.metrics) for review in reviews],
        "issues": [asdict(issue) for review in reviews for issue in review.issues],
        "windows": [
            {"title": title, "start": start, "end": end}
            for title, start, end in review_windows(reviews, window_minutes)
        ],
    }


def output_dir_for_video(video: Path, args: argparse.Namespace, video_count: int) -> Path:
    if args.output_dir:
        base = resolve_project_path(args.output_dir)
        return (base / video.stem) if video_count > 1 else base
    return subtitle_qc_output_dir(video) / "content_review"


def collect_specs(video: Path, args: argparse.Namespace) -> list[CandidateSpec]:
    if args.candidate:
        used: set[str] = set()
        return [parse_candidate_spec(spec, video.stem, used) for spec in args.candidate]
    specs = discover_candidates(video, all_candidates=args.all_candidates, modes=args.mode)
    if not specs:
        raise SystemExit(
            f"no candidate subtitles found for {video.stem}; pass --candidate label=path "
            "or --all-candidates to include every subtitle in the video artifact directory"
        )
    return specs


def run_for_video(video: Path, args: argparse.Namespace, config: ReviewConfig, video_count: int) -> dict[str, Path]:
    specs = collect_specs(video, args)
    reviews = [analyze_candidate(spec, load_candidate(spec), config) for spec in specs]
    out_dir = output_dir_for_video(video, args, video_count)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": out_dir / "content_review_summary.md",
        "metrics": out_dir / "content_review_metrics.csv",
        "issues": out_dir / "content_review_issues.csv",
        "full_text": out_dir / "content_review_full_text.md",
        "window_compare": out_dir / "content_review_window_compare.md",
        "json": out_dir / "content_review.json",
    }
    write_summary_md(paths["summary"], video, reviews)
    write_metrics_csv(paths["metrics"], reviews)
    write_issues_csv(paths["issues"], reviews)
    write_full_text_md(paths["full_text"], reviews)
    write_window_compare_md(paths["window_compare"], reviews, args.window_minutes)
    write_json(paths["json"], review_payload(video, reviews, args.window_minutes))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full-content subtitle review reports.")
    parser.add_argument("--video", action="append", required=True, help="Video path or video stem. Repeat for batch review.")
    parser.add_argument("--candidate", action="append", help="Candidate subtitle as label=path or path. Repeatable.")
    parser.add_argument("--mode", action="append", help="Limit auto-discovery to a candidate label such as fusion_lite.")
    parser.add_argument("--all-candidates", action="store_true", help="Include every SRT/JSON in the video artifact directory.")
    parser.add_argument("--output-dir", help="Output directory. With multiple videos, a per-video subdirectory is created.")
    parser.add_argument("--window-minutes", type=float, default=30.0, help="Minutes per full-text comparison window.")
    parser.add_argument("--long-duration-s", type=float, default=6.5, help="Cue duration threshold for long_duration.")
    parser.add_argument("--long-text-chars", type=int, default=60, help="Text length threshold for long_text.")
    parser.add_argument("--fast-cps", type=float, default=16.0, help="Character-per-second threshold for fast_cps.")
    parser.add_argument("--noise-pattern", action="append", help="Additional regex pattern treated as likely_noise.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = [resolve_video(value) for value in args.video]
    config = build_review_config(args)
    for video in videos:
        paths = run_for_video(video, args, config, len(videos))
        print(f"{video.stem}:")
        for name, path in paths.items():
            print(f"  {name}: {display_path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
