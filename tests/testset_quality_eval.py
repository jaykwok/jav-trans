#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
REFERENCE_EXTS = {".ass", ".srt"}
DEFAULT_DATASET_ROOT = "video/test"
DEFAULT_INDEX = "video/test/index.json"
DEFAULT_OUTPUT_ROOT = "agents/temp/testset-quality-eval"

_ASS_TAG_RE = re.compile(r"\{[^{}]*\}")
_ASS_TIME_RE = re.compile(
    r"(?P<h>\d+):(?P<m>\d\d):(?P<s>\d\d)(?:[.](?P<cs>\d{1,2}))?"
)
_SRT_TIME_RE = re.compile(
    r"(?P<start>\d\d:\d\d:\d\d,\d{3})\s*-->\s*(?P<end>\d\d:\d\d:\d\d,\d{3})"
)
_HAN_RE = re.compile(r"[\u3400-\u9fff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_EVAL_KEEP_RE = re.compile(r"[^0-9A-Za-z\u3400-\u9fff]+")
_BRACKET_PREFIX_RE = re.compile(r"^\[[A-Z0-9_ -]{1,12}\]\s*")
_STYLE_EXCLUDE_RE = re.compile(
    r"(?:^|[_ -])(staff|title|logo|op|ed|ep|lyrics|karaoke)(?:$|[_ -])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SubtitleSegment:
    start: float
    end: float
    text: str
    source: str = ""
    style: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class DatasetCase:
    case_id: str
    video_path: str
    reference_path: str
    match_reason: str = "index"
    match_score: float = 100.0


def _project_path(path: Path | str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _project_relative(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _safe_case_id(video_path: Path | str) -> str:
    rel = _project_relative(video_path)
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    return f"testset_{digest}"


def _decode_subtitle_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030", "cp932"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "[Events]" in text or "Dialogue:" in text or "ScriptType" in text or "-->" in text:
            return text
    return data.decode("utf-8", errors="replace")


def read_subtitle_text(path: Path) -> str:
    return _decode_subtitle_bytes(path.read_bytes())


def _parse_ass_time(value: str) -> float:
    match = _ASS_TIME_RE.match(value.strip())
    if not match:
        raise ValueError(f"invalid ASS timestamp: {value!r}")
    centiseconds = match.group("cs") or "0"
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + int(centiseconds.ljust(2, "0")[:2]) / 100.0
    )


def _parse_srt_time(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


def clean_subtitle_text(text: str) -> str:
    text = _ASS_TAG_RE.sub("", text or "")
    text = text.replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")
    text = text.replace("\ufeff", "")
    text = re.sub(r"\\[A-Za-z]+(?:\([^)]*\))?", "", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def normalize_eval_text(text: str) -> str:
    text = clean_subtitle_text(text)
    text = _BRACKET_PREFIX_RE.sub("", text)
    return _EVAL_KEEP_RE.sub("", text)


def _style_is_reference_dialogue(style: str, text: str) -> bool:
    if _STYLE_EXCLUDE_RE.search(style or ""):
        return False
    return bool(normalize_eval_text(text))


def parse_ass_segments(path: Path) -> list[SubtitleSegment]:
    text = read_subtitle_text(path)
    in_events = False
    fields: list[str] = []
    segments: list[SubtitleSegment] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("["):
            in_events = line.strip().lower() == "[events]"
            continue
        if not in_events:
            continue
        if line.lower().startswith("format:"):
            fields = [part.strip() for part in line.split(":", 1)[1].split(",")]
            continue
        if not line.lower().startswith("dialogue:") or not fields:
            continue
        payload = line.split(":", 1)[1].lstrip()
        parts = payload.split(",", max(0, len(fields) - 1))
        if len(parts) < len(fields):
            continue
        values = {field.lower(): parts[index].strip() for index, field in enumerate(fields)}
        try:
            start = _parse_ass_time(values.get("start", ""))
            end = _parse_ass_time(values.get("end", ""))
        except ValueError:
            continue
        cleaned = clean_subtitle_text(values.get("text", ""))
        style = values.get("style", "")
        if end <= start or not _style_is_reference_dialogue(style, cleaned):
            continue
        segments.append(
            SubtitleSegment(
                start=start,
                end=end,
                text=cleaned,
                source=_project_relative(path),
                style=style,
            )
        )
    return sorted(segments, key=lambda segment: (segment.start, segment.end))


def extract_probable_chinese_text(lines: Iterable[str]) -> str:
    cleaned: list[str] = []
    for line in lines:
        line = _BRACKET_PREFIX_RE.sub("", clean_subtitle_text(line)).strip()
        if not line:
            continue
        han = len(_HAN_RE.findall(line))
        kana = len(_KANA_RE.findall(line))
        if han and han >= kana:
            cleaned.append(line)
    if cleaned:
        return "\n".join(cleaned)
    return "\n".join(
        _BRACKET_PREFIX_RE.sub("", clean_subtitle_text(line)).strip()
        for line in lines
        if line.strip()
    )


def parse_srt_segments(path: Path) -> list[SubtitleSegment]:
    text = read_subtitle_text(path)
    blocks = re.split(r"\n\s*\n", text.strip())
    segments: list[SubtitleSegment] = []
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_index = next((idx for idx, line in enumerate(lines) if _SRT_TIME_RE.search(line)), -1)
        if time_index < 0:
            continue
        match = _SRT_TIME_RE.search(lines[time_index])
        if match is None:
            continue
        start = _parse_srt_time(match.group("start"))
        end = _parse_srt_time(match.group("end"))
        content = extract_probable_chinese_text(lines[time_index + 1 :])
        if end > start and normalize_eval_text(content):
            segments.append(
                SubtitleSegment(
                    start=start,
                    end=end,
                    text=content,
                    source=_project_relative(path),
                )
            )
    return segments


def parse_reference_segments(path: Path) -> list[SubtitleSegment]:
    suffix = path.suffix.lower()
    if suffix == ".ass":
        return parse_ass_segments(path)
    if suffix == ".srt":
        return parse_srt_segments(path)
    raise ValueError(f"unsupported reference subtitle type: {path}")


def parse_subtitle_video_refs(path: Path) -> list[str]:
    if path.suffix.lower() != ".ass":
        return []
    refs: list[str] = []
    for line in read_subtitle_text(path).splitlines():
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key.strip().lower() in {"video file", "audio uri", "audio file"}:
            value = value.strip()
            if value and value != "?video":
                refs.append(Path(value).name)
    return list(dict.fromkeys(refs))


def _normalize_match_key(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"\[[^\]]*\]", " ", value)
    value = re.sub(r"\([^)]*\)", " ", value)
    return re.sub(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff]+", "", value)


def _score_reference_for_video(
    video: Path,
    reference: Path,
    reference_refs: list[str],
    reference_count: int,
) -> tuple[float, str]:
    if video.name in reference_refs:
        return 100.0, "reference_metadata_exact"
    if video.stem == reference.stem:
        return 96.0, "stem_exact"
    video_key = _normalize_match_key(video.stem)
    ref_key = _normalize_match_key(reference.stem)
    if video_key and video_key == ref_key:
        return 92.0, "normalized_stem_exact"
    if reference_count == 1:
        return 70.0, "only_reference_in_directory"

    score = 0.0
    for candidate in [reference.stem, *reference_refs]:
        candidate_key = _normalize_match_key(Path(candidate).stem)
        if not candidate_key or not video_key:
            continue
        ratio = SequenceMatcher(None, video_key, candidate_key).ratio()
        if video_key in candidate_key or candidate_key in video_key:
            ratio += 0.08
        score = max(score, min(1.0, ratio))
    return score * 65.0, "fuzzy_name"


def discover_dataset_cases(dataset_root: Path | str = DEFAULT_DATASET_ROOT) -> list[DatasetCase]:
    root = _project_path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    cases: list[DatasetCase] = []
    for directory in sorted({path.parent for path in root.rglob("*") if path.is_file()}):
        videos = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS
        )
        references = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in REFERENCE_EXTS
        )
        if not videos or not references:
            continue
        refs_by_reference = {reference: parse_subtitle_video_refs(reference) for reference in references}
        for video in videos:
            scored = [
                (
                    *_score_reference_for_video(
                        video,
                        reference,
                        refs_by_reference[reference],
                        len(references),
                    ),
                    reference,
                )
                for reference in references
            ]
            score, reason, reference = max(scored, key=lambda item: item[0])
            cases.append(
                DatasetCase(
                    case_id=_safe_case_id(video),
                    video_path=_project_relative(video),
                    reference_path=_project_relative(reference),
                    match_reason=reason,
                    match_score=round(float(score), 3),
                )
            )
    return sorted(cases, key=lambda case: case.video_path)


def _validate_case(case: DatasetCase) -> None:
    video = _project_path(case.video_path)
    reference = _project_path(case.reference_path)
    if not video.exists():
        raise FileNotFoundError(f"indexed video does not exist: {case.video_path}")
    if not reference.exists():
        raise FileNotFoundError(f"indexed reference subtitle does not exist: {case.reference_path}")
    if video.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"unsupported indexed video type: {case.video_path}")
    if reference.suffix.lower() not in REFERENCE_EXTS:
        raise ValueError(f"unsupported indexed reference type: {case.reference_path}")


def load_index(index_path: Path | str = DEFAULT_INDEX) -> list[DatasetCase]:
    path = _project_path(index_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(raw_cases, list):
        raise ValueError(f"index must contain a cases list: {path}")

    cases: list[DatasetCase] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"index case #{index} must be an object")
        video_path = str(raw_case.get("video_path") or "").strip()
        reference_path = str(raw_case.get("reference_path") or raw_case.get("ass_path") or "").strip()
        if not video_path or not reference_path:
            raise ValueError(f"index case #{index} missing video_path/reference_path")
        case = DatasetCase(
            case_id=str(raw_case.get("case_id") or _safe_case_id(video_path)),
            video_path=_project_relative(_project_path(video_path)),
            reference_path=_project_relative(_project_path(reference_path)),
            match_reason=str(raw_case.get("match_reason") or "index"),
            match_score=float(raw_case.get("match_score", 100.0)),
        )
        _validate_case(case)
        cases.append(case)
    return cases


def write_index(index_path: Path | str, cases: list[DatasetCase], dataset_root: Path | str) -> None:
    path = _project_path(index_path)
    payload = {
        "version": 1,
        "dataset_root": _project_relative(_project_path(dataset_root)),
        "case_count": len(cases),
        "cases": [asdict(case) for case in cases],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_or_discover_cases(
    *,
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    index_path: Path | str = DEFAULT_INDEX,
    use_index: bool = True,
) -> list[DatasetCase]:
    index = _project_path(index_path)
    if use_index and index.exists():
        return load_index(index)
    return discover_dataset_cases(dataset_root)


def load_bilingual_json_segments(path: Path) -> list[SubtitleSegment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    blocks = payload.get("blocks") if isinstance(payload, dict) else None
    if not isinstance(blocks, list):
        return []
    segments: list[SubtitleSegment] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        try:
            start = float(block.get("start"))
            end = float(block.get("end"))
        except (TypeError, ValueError):
            continue
        text = str(block.get("zh_text") or block.get("zh") or "").strip()
        if end > start and normalize_eval_text(text):
            segments.append(
                SubtitleSegment(
                    start=start,
                    end=end,
                    text=text,
                    source=_project_relative(path),
                )
            )
    return sorted(segments, key=lambda segment: (segment.start, segment.end))


def load_prediction_segments(paths: dict[str, str]) -> list[SubtitleSegment]:
    bilingual_json = paths.get("bilingual_json")
    if bilingual_json and Path(bilingual_json).exists():
        segments = load_bilingual_json_segments(Path(bilingual_json))
        if segments:
            return segments
    srt = paths.get("srt")
    if srt and Path(srt).exists():
        return parse_srt_segments(Path(srt))
    return []


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    normalized = sorted((float(start), float(end)) for start, end in intervals if end > start)
    merged: list[tuple[float, float]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def overlap_duration(
    left: Iterable[tuple[float, float]],
    right: Iterable[tuple[float, float]],
) -> float:
    a = merge_intervals(left)
    b = merge_intervals(right)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _segments_intervals(segments: list[SubtitleSegment]) -> list[tuple[float, float]]:
    return [(segment.start, segment.end) for segment in segments]


def levenshtein_counts(reference: str, hypothesis: str) -> dict[str, int]:
    ref = list(reference)
    hyp = list(hypothesis)
    previous = [(idx, idx, 0, 0) for idx in range(len(hyp) + 1)]
    for i, ref_char in enumerate(ref, 1):
        current = [(i, 0, i, 0)]
        for j, hyp_char in enumerate(hyp, 1):
            if ref_char == hyp_char:
                keep = previous[j - 1]
            else:
                cost, ins, dele, sub = previous[j - 1]
                keep = (cost + 1, ins, dele, sub + 1)
            cost, ins, dele, sub = current[j - 1]
            insert = (cost + 1, ins + 1, dele, sub)
            cost, ins, dele, sub = previous[j]
            delete = (cost + 1, ins, dele + 1, sub)
            current.append(min(keep, insert, delete, key=lambda item: (item[0], item[1] + item[2], item[3])))
        previous = current
    distance, insertions, deletions, substitutions = previous[-1]
    return {
        "distance": distance,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "reference_chars": len(ref),
        "hypothesis_chars": len(hyp),
    }


def lcs_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for j, right_char in enumerate(right, 1):
            if left_char == right_char:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def max_repeat_run(text: str) -> dict[str, object]:
    compact = normalize_eval_text(text)
    best: dict[str, object] = {"unit": "", "run": 0, "chars": 0, "ratio": 0.0}
    if not compact:
        return best
    max_unit_len = min(12, max(1, len(compact) // 2))
    for unit_len in range(1, max_unit_len + 1):
        index = 0
        while index <= len(compact) - unit_len:
            unit = compact[index : index + unit_len]
            run = 1
            cursor = index + unit_len
            while cursor + unit_len <= len(compact) and compact[cursor : cursor + unit_len] == unit:
                run += 1
                cursor += unit_len
            repeated_chars = run * unit_len
            ratio = repeated_chars / max(1, len(compact))
            if run > int(best["run"]) and run >= 2:
                best = {
                    "unit": unit,
                    "run": run,
                    "chars": repeated_chars,
                    "ratio": round(ratio, 4),
                }
            index += max(1, repeated_chars if run > 1 else 1)
    return best


def _overlapping_segments(
    anchor: SubtitleSegment,
    candidates: list[SubtitleSegment],
    min_overlap_s: float,
) -> list[SubtitleSegment]:
    matches: list[SubtitleSegment] = []
    for candidate in candidates:
        overlap = max(0.0, min(anchor.end, candidate.end) - max(anchor.start, candidate.start))
        if overlap >= min_overlap_s:
            matches.append(candidate)
    return sorted(matches, key=lambda segment: (segment.start, segment.end))


def evaluate_prediction_against_reference(
    reference_segments: list[SubtitleSegment],
    prediction_segments: list[SubtitleSegment],
    *,
    min_overlap_s: float = 0.10,
    max_examples: int = 24,
) -> dict:
    reference_intervals = _segments_intervals(reference_segments)
    prediction_intervals = _segments_intervals(prediction_segments)
    ref_duration = interval_duration(reference_intervals)
    pred_duration = interval_duration(prediction_intervals)
    overlap = overlap_duration(reference_intervals, prediction_intervals)

    totals = {
        "distance": 0,
        "insertions": 0,
        "deletions": 0,
        "substitutions": 0,
        "reference_chars": 0,
        "hypothesis_chars": 0,
        "lcs": 0,
    }
    matched_refs = 0
    examples: list[dict] = []
    worst: list[dict] = []
    for reference in reference_segments:
        preds = _overlapping_segments(reference, prediction_segments, min_overlap_s)
        ref_text = normalize_eval_text(reference.text)
        pred_text = normalize_eval_text("".join(pred.text for pred in preds))
        if preds:
            matched_refs += 1
        counts = levenshtein_counts(ref_text, pred_text)
        common = lcs_length(ref_text, pred_text)
        for key in ("distance", "insertions", "deletions", "substitutions", "reference_chars", "hypothesis_chars"):
            totals[key] += counts[key]
        totals["lcs"] += common
        item = {
            "start": reference.start,
            "end": reference.end,
            "reference_text": reference.text,
            "prediction_text": "".join(pred.text for pred in preds),
            "overlap_prediction_count": len(preds),
            "cer": round(counts["distance"] / max(1, counts["reference_chars"]), 4),
            "distance": counts["distance"],
        }
        if len(examples) < max_examples:
            examples.append(item)
        worst.append(item)

    unsupported_predictions = [
        prediction
        for prediction in prediction_segments
        if not _overlapping_segments(prediction, reference_segments, min_overlap_s)
    ]
    unsupported_duration = interval_duration(_segments_intervals(unsupported_predictions))
    repeated_predictions = [
        prediction
        for prediction in prediction_segments
        if int(max_repeat_run(prediction.text)["run"]) >= 4
    ]
    precision = totals["lcs"] / max(1, totals["hypothesis_chars"])
    recall = totals["lcs"] / max(1, totals["reference_chars"])
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    worst = sorted(worst, key=lambda item: (item["cer"], item["distance"]), reverse=True)[:max_examples]
    return {
        "reference_count": len(reference_segments),
        "prediction_count": len(prediction_segments),
        "matched_reference_count": matched_refs,
        "reference_match_rate": round(matched_refs / max(1, len(reference_segments)), 6),
        "reference_timeline_duration_s": round(ref_duration, 3),
        "prediction_timeline_duration_s": round(pred_duration, 3),
        "timeline_overlap_s": round(overlap, 3),
        "timeline_recall": round(overlap / max(1e-9, ref_duration), 6),
        "timeline_precision": round(overlap / max(1e-9, pred_duration), 6),
        "unsupported_prediction_count": len(unsupported_predictions),
        "unsupported_prediction_ratio": round(len(unsupported_predictions) / max(1, len(prediction_segments)), 6),
        "unsupported_prediction_duration_s": round(unsupported_duration, 3),
        "unsupported_prediction_duration_ratio": round(unsupported_duration / max(1e-9, pred_duration), 6),
        "repeated_prediction_count": len(repeated_predictions),
        "repeated_prediction_ratio": round(len(repeated_predictions) / max(1, len(prediction_segments)), 6),
        "zh_cer": round(totals["distance"] / max(1, totals["reference_chars"]), 6),
        "zh_insertion_rate": round(totals["insertions"] / max(1, totals["hypothesis_chars"]), 6),
        "zh_deletion_rate": round(totals["deletions"] / max(1, totals["reference_chars"]), 6),
        "zh_substitution_rate": round(totals["substitutions"] / max(1, totals["reference_chars"]), 6),
        "zh_char_precision": round(precision, 6),
        "zh_char_recall": round(recall, 6),
        "zh_char_f1": round(f1, 6),
        "examples": examples,
        "worst_examples": worst,
    }


def _load_json_if_exists(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _pick_newest(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def find_existing_prediction_paths(case: DatasetCase, predictions_root: Path | str) -> dict[str, str]:
    root = _project_path(predictions_root)
    stem = Path(case.video_path).stem
    candidates = {
        "bilingual_json": list(root.rglob(f"{stem}.bilingual.json")),
        "srt": list(root.rglob(f"{stem}.srt")) + list(root.rglob(f"{stem}.ja.srt")),
        "quality_report": list(root.rglob(f"{stem}.quality_report.json")),
        "quality_report_md": list(root.rglob(f"{stem}.quality_report.md")),
        "timings": list(root.rglob(f"{stem}.timings.json")),
    }
    return {
        key: str(path)
        for key, paths in candidates.items()
        for path in [_pick_newest(paths)]
        if path is not None
    }


def _asr_metrics_from_artifacts(paths: dict[str, str]) -> dict:
    timings = _load_json_if_exists(paths.get("timings"))
    quality = _load_json_if_exists(paths.get("quality_report"))
    asr_details = timings.get("asr_details") if isinstance(timings.get("asr_details"), dict) else {}
    counts = timings.get("counts") if isinstance(timings.get("counts"), dict) else {}
    return {
        "backend": timings.get("backend") or "",
        "segments": counts.get("segments"),
        "blocks": counts.get("blocks"),
        "asr_generation_error_count": quality.get("asr_generation_error_count", 0),
        "asr_generation_overflow_count": quality.get("asr_generation_overflow_count", 0),
        "asr_timeout_count": quality.get("asr_timeout_count", 0),
        "asr_quarantined_count": quality.get("asr_quarantined_count", 0),
        "quality_warnings": quality.get("warnings", []),
        "stage_timings": timings.get("stage_timings", {}),
    }


def evaluate_case(case: DatasetCase, prediction_paths: dict[str, str], *, min_overlap_s: float) -> dict:
    reference = parse_reference_segments(_project_path(case.reference_path))
    prediction = load_prediction_segments(prediction_paths)
    metrics = evaluate_prediction_against_reference(reference, prediction, min_overlap_s=min_overlap_s)
    return {
        "case": asdict(case),
        "prediction_paths": {
            key: _project_relative(value) for key, value in prediction_paths.items()
        },
        "translation_quality": metrics,
        "asr_and_hallucination_proxy": {
            **_asr_metrics_from_artifacts(prediction_paths),
            "timeline_recall": metrics["timeline_recall"],
            "timeline_precision": metrics["timeline_precision"],
            "no_reference_overlap_count": metrics["unsupported_prediction_count"],
            "no_reference_overlap_ratio": metrics["unsupported_prediction_ratio"],
            "no_reference_overlap_duration_ratio": metrics["unsupported_prediction_duration_ratio"],
            "repeated_prediction_ratio": metrics["repeated_prediction_ratio"],
            "zh_insertion_rate": metrics["zh_insertion_rate"],
        },
    }


def aggregate_results(case_results: list[dict]) -> dict:
    if not case_results:
        return {}
    keys = [
        "reference_count",
        "prediction_count",
        "matched_reference_count",
        "unsupported_prediction_count",
        "repeated_prediction_count",
    ]
    totals = {key: 0 for key in keys}
    weighted = {
        "timeline_recall": 0.0,
        "timeline_precision": 0.0,
        "zh_cer": 0.0,
        "zh_char_precision": 0.0,
        "zh_char_recall": 0.0,
        "zh_char_f1": 0.0,
        "unsupported_prediction_ratio": 0.0,
        "zh_insertion_rate": 0.0,
    }
    weight_total = 0
    asr_counts = {
        "asr_generation_error_count": 0,
        "asr_generation_overflow_count": 0,
        "asr_timeout_count": 0,
        "asr_quarantined_count": 0,
    }
    for result in case_results:
        quality = result["translation_quality"]
        proxy = result["asr_and_hallucination_proxy"]
        weight = max(1, int(quality.get("reference_count", 0)))
        weight_total += weight
        for key in keys:
            totals[key] += int(quality.get(key, 0) or 0)
        for key in weighted:
            weighted[key] += float(quality.get(key, 0.0) or 0.0) * weight
        for key in asr_counts:
            asr_counts[key] += int(proxy.get(key, 0) or 0)
    return {
        "case_count": len(case_results),
        **totals,
        **{key: round(value / max(1, weight_total), 6) for key, value in weighted.items()},
        **asr_counts,
    }


def run_pipeline_for_case(
    case: DatasetCase,
    *,
    backend: str,
    output_dir: Path,
    job_temp_root: Path,
    skip_translation: bool,
    translation_max_workers: int,
    run_log_enabled: bool,
) -> dict[str, str]:
    from core.job_context import JobContext
    from main import run_asr_alignment, run_translation_and_write

    video_path = _project_path(case.video_path)
    job_temp_dir = job_temp_root / case.case_id
    report_dir = output_dir / "quality_reports"
    advanced = {
        "QUALITY_REPORT_ENABLED": "1",
        "QUALITY_REPORT_DIR": str(report_dir),
        "QC_HARD_FAIL": "0",
        "RUN_LOG_ENABLED": "1" if run_log_enabled else "0",
        "RUN_LOG_DIR": str(output_dir / "run_logs"),
    }
    ctx = JobContext(
        asr_backend=backend,
        asr_context="",
        subtitle_mode="zh",
        skip_translation=skip_translation,
        target_lang="简体中文",
        translation_glossary="",
        translation_max_workers=translation_max_workers,
        translation_cache_path=str(job_temp_dir / "translation_cache.jsonl"),
        job_id=case.case_id,
        job_temp_dir=str(job_temp_dir),
        output_dir=str(output_dir / "generated"),
        keep_quality_report=True,
        keep_temp_files=True,
        run_log_enabled=run_log_enabled,
        run_log_dir=str(output_dir / "run_logs"),
        advanced=advanced,
    )
    artifacts = run_asr_alignment(str(video_path), ctx=ctx, job_id=case.case_id)
    run_translation_and_write(str(video_path), artifacts, ctx=ctx, job_id=case.case_id)
    return {
        key: value
        for key, value in {
            "srt": artifacts.srt_path,
            "bilingual_json": artifacts.bilingual_json_path,
            "quality_report": artifacts.quality_report_path,
            "timings": artifacts.timings_path,
        }.items()
        if value
    }


def write_html_report(path: Path, payload: dict) -> None:
    rows: list[str] = []
    for result in payload.get("cases", []):
        case = result["case"]
        quality = result["translation_quality"]
        proxy = result["asr_and_hallucination_proxy"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(case['case_id'])}</td>"
            f"<td>{html.escape(case['video_path'])}</td>"
            f"<td>{html.escape(case['reference_path'])}</td>"
            f"<td>{quality['reference_count']}</td>"
            f"<td>{quality['prediction_count']}</td>"
            f"<td>{quality['timeline_recall']:.3f}</td>"
            f"<td>{quality['timeline_precision']:.3f}</td>"
            f"<td>{quality['zh_cer']:.3f}</td>"
            f"<td>{quality['zh_char_f1']:.3f}</td>"
            f"<td>{proxy['no_reference_overlap_ratio']:.3f}</td>"
            f"<td>{proxy.get('asr_generation_error_count', 0)}</td>"
            "</tr>"
        )
    summary_items = "".join(
        f"<li><code>{html.escape(str(key))}</code>: {html.escape(str(value))}</li>"
        for key, value in payload.get("summary", {}).items()
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>JAVTrans Test Set Quality Eval</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f0f4f8; text-align: left; }}
    code {{ background: #f0f4f8; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>JAVTrans Test Set Quality Eval</h1>
  <h2>Summary</h2>
  <ul>{summary_items}</ul>
  <h2>Cases</h2>
  <table>
    <thead>
      <tr>
        <th>Case</th><th>Video</th><th>Reference</th><th>Ref</th><th>Pred</th>
        <th>Time Recall</th><th>Time Precision</th><th>ZH CER</th>
        <th>ZH F1</th><th>No-ref Pred</th><th>ASR Gen Errors</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _manifest_payload(cases: list[DatasetCase], dataset_root: str | Path, index_path: str | Path) -> dict:
    return {
        "dataset_root": _project_relative(_project_path(dataset_root)),
        "index_path": _project_relative(_project_path(index_path)),
        "case_count": len(cases),
        "cases": [asdict(case) for case in cases],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    from asr.backends.qwen import QWEN_ASR_06B_REPO_ID

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate arbitrary video + reference subtitle pairs from video/test. "
            "Use --write-index to refresh video/test/index.json after adding files."
        )
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--predictions-root", default="")
    parser.add_argument("--run-pipeline", action="store_true")
    parser.add_argument("--skip-translation", action="store_true")
    parser.add_argument("--backend", default=QWEN_ASR_06B_REPO_ID)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-contains", default="")
    parser.add_argument("--min-overlap-s", type=float, default=0.10)
    parser.add_argument("--translation-max-workers", type=int, default=4)
    parser.add_argument("--run-log", action="store_true")
    parser.add_argument("--list", action="store_true", help="Only write manifest and exit.")
    parser.add_argument("--write-index", action="store_true", help="Discover cases and write the index file.")
    parser.add_argument("--ignore-index", action="store_true", help="Discover cases instead of reading index.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_root = _project_path(args.output_root)
    if args.write_index:
        discovered = discover_dataset_cases(args.dataset_root)
        write_index(args.index, discovered, args.dataset_root)
        print(f"Wrote index: {_project_relative(args.index)} ({len(discovered)} cases)")
        return 0

    cases = load_or_discover_cases(
        dataset_root=args.dataset_root,
        index_path=args.index,
        use_index=not args.ignore_index,
    )
    if args.case_contains:
        needle = args.case_contains.casefold()
        cases = [
            case
            for case in cases
            if needle in case.video_path.casefold() or needle in case.reference_path.casefold()
        ]
    if args.limit > 0:
        cases = cases[: args.limit]

    manifest = _manifest_payload(cases, args.dataset_root, args.index)
    write_json(output_root / "manifest.json", manifest)
    if args.list:
        print(f"Wrote manifest: {_project_relative(output_root / 'manifest.json')}")
        return 0
    if not args.run_pipeline and not args.predictions_root:
        print(
            "No evaluation source selected. Use --run-pipeline to generate outputs, "
            "or --predictions-root <dir> to evaluate existing outputs."
        )
        print(f"Wrote manifest: {_project_relative(output_root / 'manifest.json')}")
        return 2

    results: list[dict] = []
    job_temp_root = PROJECT_ROOT / "tmp" / "testset-quality-eval" / "jobs"
    started = time.perf_counter()
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case.video_path}")
        if args.run_pipeline:
            prediction_paths = run_pipeline_for_case(
                case,
                backend=args.backend,
                output_dir=output_root,
                job_temp_root=job_temp_root,
                skip_translation=args.skip_translation,
                translation_max_workers=args.translation_max_workers,
                run_log_enabled=args.run_log,
            )
        else:
            prediction_paths = find_existing_prediction_paths(case, args.predictions_root)
        results.append(evaluate_case(case, prediction_paths, min_overlap_s=args.min_overlap_s))

    payload = {
        "dataset": manifest,
        "summary": aggregate_results(results),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "cases": results,
    }
    write_json(output_root / "summary.json", payload)
    write_html_report(output_root / "report.html", payload)
    print(f"Wrote summary: {_project_relative(output_root / 'summary.json')}")
    print(f"Wrote report:  {_project_relative(output_root / 'report.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
