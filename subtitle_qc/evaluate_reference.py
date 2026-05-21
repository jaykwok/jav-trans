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
    fmt_time,
    load_json_file,
    load_srt_file,
    normalize_text,
    overlap,
    parse_timecode,
    project_rel,
    split_subtitle_lines,
    subtitle_qc_output_dir,
    text_similarity,
    write_json,
)


REFERENCE_DIR = PROJECT_ROOT / "video" / "reference"
JA_CHAR_RE = re.compile(r"[ぁ-ゖァ-ヺー一-龯]")
ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    path: Path


@dataclass
class ReferenceCueResult:
    index: int
    start: float
    end: float
    reference_text: str
    candidate_text: str
    candidate_indexes: list[int]
    similarity: float
    matched: bool


@dataclass
class CandidateMetrics:
    label: str
    path: str
    cue_count: int
    reference_cue_count: int
    matched_reference_count: int
    missing_reference_count: int
    low_similarity_count: int
    good_similarity_count: int
    extra_candidate_count: int
    covered_reference_ratio: float
    good_reference_ratio: float
    mean_similarity: float
    median_similarity: float
    weighted_similarity: float
    p10_similarity: float


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return project_rel(path)
    except ValueError:
        return str(path)


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


def safe_label(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean.strip("_") or "candidate"


def unique_label(label: str, used: set[str]) -> str:
    clean = safe_label(label)
    if clean not in used:
        used.add(clean)
        return clean
    suffix = 2
    while f"{clean}-{suffix}" in used:
        suffix += 1
    final = f"{clean}-{suffix}"
    used.add(final)
    return final


def parse_input_spec(spec: str, used: set[str]) -> CandidateSpec:
    if "=" in spec:
        label, path_text = spec.split("=", 1)
        path = resolve_project_path(path_text.strip()).resolve()
        return CandidateSpec(unique_label(label, used), path)
    path = resolve_project_path(spec.strip()).resolve()
    return CandidateSpec(unique_label(path.stem, used), path)


def strip_ass_text(value: str) -> str:
    text = ASS_OVERRIDE_RE.sub("", value)
    text = text.replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")
    return text.strip()


def load_ass_file(path: Path, source_name: str) -> list[Cue]:
    format_fields: list[str] = []
    cues: list[Cue] = []
    for raw_line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw_line.strip()
        if line.lower().startswith("format:"):
            format_fields = [field.strip().lower() for field in line.split(":", 1)[1].split(",")]
            continue
        if not line.lower().startswith("dialogue:") or not format_fields:
            continue
        payload = line.split(":", 1)[1].lstrip()
        parts = payload.split(",", max(0, len(format_fields) - 1))
        if len(parts) < len(format_fields):
            continue
        item = {format_fields[index]: parts[index].strip() for index in range(len(format_fields))}
        start = parse_timecode(item.get("start"))
        end = parse_timecode(item.get("end"))
        if start is None or end is None:
            continue
        text = strip_ass_text(item.get("text", ""))
        ja_text, zh_text = split_subtitle_lines(text.splitlines())
        cues.append(
            Cue(
                index=len(cues),
                mode=source_name,
                start=start,
                end=end,
                ja_text=ja_text,
                zh_text=zh_text,
                gender="",
                words=[],
            )
        )
    return cues


def load_subtitle(path: Path, label: str) -> list[Cue]:
    suffix = path.suffix.lower()
    if suffix == ".srt":
        cues = load_srt_file(path, label)
    elif suffix == ".json":
        cues = load_json_file(path, label)
    elif suffix == ".ass":
        cues = load_ass_file(path, label)
    else:
        raise SystemExit(f"unsupported subtitle file type: {path}")
    if not cues:
        raise SystemExit(f"no subtitle cues parsed from: {path}")
    return cues


def discover_reference(video_stem: str, reference_dir: Path) -> Path:
    patterns = (
        f"{video_stem}*.ja.srt",
        f"{video_stem}*.srt",
        f"{video_stem}*.ja.ass",
        f"{video_stem}*.ass",
        f"{video_stem}*.json",
    )
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(reference_dir.glob(pattern)))
        matches.extend(sorted(reference_dir.glob(f"**/{pattern}")))
    unique = []
    seen = set()
    for path in matches:
        if "source_audit" in path.parts:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    if not unique:
        available = ", ".join(path.name for path in sorted(reference_dir.glob("*"))) or "none"
        raise SystemExit(
            f"no reference subtitle found for {video_stem} under {reference_dir}; "
            f"available references: {available}"
        )
    return sorted(unique, key=lambda path: (len(path.relative_to(reference_dir).parts), path.as_posix()))[0]


def discover_candidates(video_stem: str) -> list[CandidateSpec]:
    used: set[str] = set()
    specs: list[CandidateSpec] = []
    for mode in DEFAULT_MODES:
        srt_path = video_artifact_path(video_stem, f"{video_stem}.{mode}.srt")
        json_path = video_artifact_path(video_stem, f"{video_stem}.{mode}.bilingual.json")
        if srt_path.exists():
            specs.append(CandidateSpec(unique_label(mode, used), srt_path))
        elif json_path.exists():
            specs.append(CandidateSpec(unique_label(mode, used), json_path))
    if specs:
        return specs
    for path in iter_video_artifacts(video_stem, f"{video_stem}.*.srt"):
        specs.append(CandidateSpec(unique_label(path.stem.removeprefix(f"{video_stem}."), used), path))
    for path in iter_video_artifacts(video_stem, f"{video_stem}.*.bilingual.json"):
        label = path.name.removeprefix(f"{video_stem}.").removesuffix(".bilingual.json")
        specs.append(CandidateSpec(unique_label(label, used), path))
    return specs


def japanese_char_count(cues: list[Cue]) -> int:
    return sum(len(JA_CHAR_RE.findall(cue.ja_text)) for cue in cues)


def cues_in_window(reference: Cue, candidates: list[Cue], time_pad: float) -> list[Cue]:
    start = reference.start - time_pad
    end = reference.end + time_pad
    return [cue for cue in candidates if cue.end >= start and cue.start <= end and cue.ja_text.strip()]


def joined_text(cues: list[Cue]) -> str:
    return "".join(cue.ja_text for cue in cues)


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.floor((len(ordered) - 1) * ratio)))
    return ordered[index]


def reference_weight(cue: Cue) -> int:
    return max(1, len(normalize_text(cue.ja_text)))


def evaluate_candidate(
    *,
    spec: CandidateSpec,
    reference_cues: list[Cue],
    candidate_cues: list[Cue],
    time_pad: float,
    low_threshold: float,
    good_threshold: float,
) -> tuple[CandidateMetrics, list[ReferenceCueResult]]:
    results: list[ReferenceCueResult] = []
    matched_candidate_indexes: set[int] = set()

    for reference in reference_cues:
        matched_cues = cues_in_window(reference, candidate_cues, time_pad)
        for cue in matched_cues:
            matched_candidate_indexes.add(cue.index)
        candidate_text = joined_text(matched_cues)
        similarity = text_similarity(reference.ja_text, candidate_text) if candidate_text else 0.0
        results.append(
            ReferenceCueResult(
                index=reference.index,
                start=reference.start,
                end=reference.end,
                reference_text=reference.ja_text,
                candidate_text=candidate_text,
                candidate_indexes=[cue.index for cue in matched_cues],
                similarity=similarity,
                matched=bool(candidate_text),
            )
        )

    similarities = [item.similarity for item in results]
    weights = [reference_weight(reference) for reference in reference_cues]
    weighted_similarity = (
        sum(score * weight for score, weight in zip(similarities, weights, strict=False)) / sum(weights)
        if weights
        else 0.0
    )
    matched_reference_count = sum(1 for item in results if item.matched)
    missing_reference_count = len(results) - matched_reference_count
    good_similarity_count = sum(1 for item in results if item.similarity >= good_threshold)
    low_similarity_count = sum(1 for item in results if item.similarity < low_threshold)
    extra_candidate_count = sum(
        1
        for cue in candidate_cues
        if cue.index not in matched_candidate_indexes and bool(normalize_text(cue.ja_text))
    )

    metrics = CandidateMetrics(
        label=spec.label,
        path=display_path(spec.path),
        cue_count=len(candidate_cues),
        reference_cue_count=len(reference_cues),
        matched_reference_count=matched_reference_count,
        missing_reference_count=missing_reference_count,
        low_similarity_count=low_similarity_count,
        good_similarity_count=good_similarity_count,
        extra_candidate_count=extra_candidate_count,
        covered_reference_ratio=matched_reference_count / max(1, len(reference_cues)),
        good_reference_ratio=good_similarity_count / max(1, len(reference_cues)),
        mean_similarity=sum(similarities) / max(1, len(similarities)),
        median_similarity=median(similarities) if similarities else 0.0,
        weighted_similarity=weighted_similarity,
        p10_similarity=percentile(similarities, 0.10),
    )
    return metrics, results


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(
    *,
    path: Path,
    video_path: Path,
    reference_path: Path,
    metrics: list[CandidateMetrics],
) -> None:
    lines = [
        f"# {video_path.stem} Reference Subtitle Evaluation",
        "",
        f"- Video: `{display_path(video_path)}`",
        f"- Reference: `{display_path(reference_path)}`",
        "",
        "| candidate | weighted | mean | median | p10 | covered | good | low | extra |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in sorted(metrics, key=lambda value: value.weighted_similarity, reverse=True):
        lines.append(
            "| "
            f"{item.label} | "
            f"{item.weighted_similarity:.3f} | "
            f"{item.mean_similarity:.3f} | "
            f"{item.median_similarity:.3f} | "
            f"{item.p10_similarity:.3f} | "
            f"{item.covered_reference_ratio:.1%} | "
            f"{item.good_reference_ratio:.1%} | "
            f"{item.low_similarity_count} | "
            f"{item.extra_candidate_count} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Japanese subtitles against an external reference subtitle.")
    parser.add_argument("--video", required=True, help="Path to source video, e.g. video/MKMP-577.mp4")
    parser.add_argument("--reference", help="Reference subtitle path. Defaults to video/reference/<video-stem>*")
    parser.add_argument("--reference-dir", default=str(REFERENCE_DIR), help="Directory for auto-discovered references")
    parser.add_argument(
        "--candidate",
        action="append",
        help="Candidate subtitle as label=path or path. Repeat for multiple candidates. Defaults to known VAD outputs.",
    )
    parser.add_argument("--time-pad", type=float, default=0.35, help="Seconds to expand each reference cue matching window")
    parser.add_argument("--low-threshold", type=float, default=0.55)
    parser.add_argument("--good-threshold", type=float, default=0.82)
    parser.add_argument("--review-limit", type=int, default=120, help="Worst per-cue rows to keep in CSV")
    parser.add_argument("--allow-non-japanese-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = resolve_project_path(args.video).resolve()
    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")

    reference_dir = resolve_project_path(args.reference_dir).resolve()
    reference_path = (
        resolve_project_path(args.reference).resolve()
        if args.reference
        else discover_reference(video_path.stem, reference_dir).resolve()
    )
    if not reference_path.exists():
        raise SystemExit(f"reference subtitle not found: {reference_path}")

    used: set[str] = set()
    candidate_specs = (
        [parse_input_spec(spec, used) for spec in args.candidate]
        if args.candidate
        else discover_candidates(video_path.stem)
    )
    if not candidate_specs:
        raise SystemExit(f"no candidate subtitles found for {video_path.stem}")

    reference_cues = load_subtitle(reference_path, "reference")
    if japanese_char_count(reference_cues) < 20 and not args.allow_non_japanese_reference:
        raise SystemExit(
            f"reference does not look like Japanese text: {reference_path}; "
            "pass --allow-non-japanese-reference to override"
        )

    output_dir = subtitle_qc_output_dir(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[CandidateMetrics] = []
    worst_rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "video": display_path(video_path),
        "reference": display_path(reference_path),
        "reference_cue_count": len(reference_cues),
        "time_pad": args.time_pad,
        "low_threshold": args.low_threshold,
        "good_threshold": args.good_threshold,
        "candidates": {},
    }

    for spec in candidate_specs:
        if not spec.path.exists():
            raise SystemExit(f"candidate subtitle not found: {spec.path}")
        candidate_cues = load_subtitle(spec.path, spec.label)
        metrics, cue_results = evaluate_candidate(
            spec=spec,
            reference_cues=reference_cues,
            candidate_cues=candidate_cues,
            time_pad=args.time_pad,
            low_threshold=args.low_threshold,
            good_threshold=args.good_threshold,
        )
        all_metrics.append(metrics)
        payload["candidates"][spec.label] = {
            "metrics": asdict(metrics),
            "worst_reference_cues": [
                asdict(item)
                for item in sorted(cue_results, key=lambda value: (value.similarity, value.start))[
                    : args.review_limit
                ]
            ],
        }
        for item in sorted(cue_results, key=lambda value: (value.similarity, value.start))[: args.review_limit]:
            worst_rows.append(
                {
                    "candidate": spec.label,
                    "reference_index": item.index + 1,
                    "start": fmt_time(item.start),
                    "end": fmt_time(item.end),
                    "similarity": f"{item.similarity:.4f}",
                    "matched": str(item.matched).lower(),
                    "candidate_indexes": " ".join(str(index + 1) for index in item.candidate_indexes),
                    "reference_text": item.reference_text,
                    "candidate_text": item.candidate_text,
                }
            )

    metrics_rows = [
        {
            "candidate": item.label,
            "path": item.path,
            "cue_count": item.cue_count,
            "reference_cue_count": item.reference_cue_count,
            "matched_reference_count": item.matched_reference_count,
            "missing_reference_count": item.missing_reference_count,
            "low_similarity_count": item.low_similarity_count,
            "good_similarity_count": item.good_similarity_count,
            "extra_candidate_count": item.extra_candidate_count,
            "covered_reference_ratio": f"{item.covered_reference_ratio:.6f}",
            "good_reference_ratio": f"{item.good_reference_ratio:.6f}",
            "mean_similarity": f"{item.mean_similarity:.6f}",
            "median_similarity": f"{item.median_similarity:.6f}",
            "weighted_similarity": f"{item.weighted_similarity:.6f}",
            "p10_similarity": f"{item.p10_similarity:.6f}",
        }
        for item in sorted(all_metrics, key=lambda value: value.weighted_similarity, reverse=True)
    ]

    json_path = output_dir / "reference_eval.json"
    metrics_csv_path = output_dir / "reference_eval_metrics.csv"
    worst_csv_path = output_dir / "reference_eval_worst_cues.csv"
    summary_path = output_dir / "reference_eval_summary.md"
    write_json(json_path, payload)
    write_csv(metrics_csv_path, metrics_rows, list(metrics_rows[0]))
    write_csv(
        worst_csv_path,
        worst_rows,
        [
            "candidate",
            "reference_index",
            "start",
            "end",
            "similarity",
            "matched",
            "candidate_indexes",
            "reference_text",
            "candidate_text",
        ],
    )
    write_markdown_summary(
        path=summary_path,
        video_path=video_path,
        reference_path=reference_path,
        metrics=all_metrics,
    )

    print(f"reference: {display_path(reference_path)}")
    print(f"outputs: {display_path(summary_path)}, {display_path(metrics_csv_path)}, {display_path(worst_csv_path)}")
    for item in sorted(all_metrics, key=lambda value: value.weighted_similarity, reverse=True):
        print(
            f"{item.label}: weighted={item.weighted_similarity:.3f} "
            f"mean={item.mean_similarity:.3f} covered={item.covered_reference_ratio:.1%} "
            f"good={item.good_reference_ratio:.1%} low={item.low_similarity_count} "
            f"extra={item.extra_candidate_count}"
        )


if __name__ == "__main__":
    main()
