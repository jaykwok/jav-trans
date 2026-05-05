from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.job_context import JobContext
import main as pipeline_main

BACKENDS = (
    "anime-whisper",
    "qwen3-asr-1.7b",
    "whisper-ja-1.5b",
    "whisper-ja-anime-v0.3",
)
LINE_PREVIEW_COUNT = 30
PREVIEW_CELL_WIDTH = 42


@dataclass
class BackendResult:
    backend: str
    status: str
    returncode: int | None
    wall_time_s: float
    asr_text_transcribe_s: float | None
    segment_count: int
    total_chars: int
    repeated_segments: int
    srt_path: Path | None
    aligned_segments_path: Path | None
    timings_path: Path | None
    log_path: Path | None
    run_log_path: Path | None
    preview_lines: list[str] = field(default_factory=list)
    error: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the supported ASR backends with skip-translation runs.",
    )
    parser.add_argument("video_path", help="Video file to benchmark")
    parser.add_argument(
        "--output-dir",
        default="temp/reports",
        help="Directory for per-backend outputs and the Markdown report",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Write compare stdout logs and child run logs under ./log",
    )
    parser.add_argument(
        "--log-dir",
        help="Write compare stdout logs and child run logs under this directory",
    )
    return parser.parse_args()


def _resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _project_relative(path: str | Path | None) -> str:
    if path is None:
        return ""
    raw = str(path)
    if not raw:
        return raw
    normalized = raw.replace("\\", "/")
    root_pattern = re.compile(
        re.escape(PROJECT_ROOT.resolve().as_posix()) + r"/?",
        re.IGNORECASE,
    )
    normalized = root_pattern.sub("", normalized)
    if normalized != raw.replace("\\", "/"):
        return normalized or "."
    candidate = Path(path)
    try:
        if candidate.is_absolute():
            return candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        return raw.replace("\\", "/")
    return raw.replace("\\", "/")


class _Spec:
    def __init__(
        self,
        *,
        backend: str,
        output_dir: Path,
        keep_temp_files: bool,
        skip_translation: bool,
    ) -> None:
        self.asr_backend = backend
        self.asr_context = ""
        self.subtitle_mode = "zh"
        self.show_gender = False
        self.multi_cue_split = False
        self.asr_recovery = False
        self.vad_threshold = 0.35
        self.skip_translation = skip_translation
        self.translation_batch_size = 100
        self.translation_max_workers = 1
        self.output_dir = str(output_dir)
        self.keep_quality_report = False
        self.keep_temp_files = keep_temp_files
        self.advanced = {}


def _delete_asr_checkpoints() -> int:
    checkpoint_root = PROJECT_ROOT / "temp"
    count = 0
    for path in checkpoint_root.glob("asr_checkpoint_*.json"):
        try:
            path.unlink()
            count += 1
        except OSError as exc:
            print(f"[WARN] failed to remove checkpoint {path}: {exc}")
    return count


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _extract_asr_text_seconds(
    aligned_payload: dict[str, Any],
    timings_payload: dict[str, Any],
) -> float | None:
    candidates = (
        aligned_payload.get("stage_timings", {}),
        aligned_payload.get("asr_details", {}).get("stage_timings", {}),
        timings_payload.get("asr_details", {}).get("stage_timings", {}),
        timings_payload.get("stage_timings", {}),
    )
    for stage_timings in candidates:
        if not isinstance(stage_timings, dict):
            continue
        value = stage_timings.get("asr_text_transcribe_s")
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_srt_texts(srt_path: Path | None) -> list[str]:
    text = _read_text(srt_path)
    if not text.strip():
        return []

    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
    parsed: list[str] = []
    for block in blocks:
        text_lines: list[str] = []
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if re.fullmatch(r"\d+", line):
                continue
            if "-->" in line:
                continue
            text_lines.append(line)
        if text_lines:
            parsed.append("".join(text_lines).strip())
    return parsed


def _summarize_srt(srt_path: Path | None) -> tuple[int, int, int, list[str]]:
    texts = _parse_srt_texts(srt_path)
    normalized = [re.sub(r"\s+", "", text) for text in texts]
    repeated = sum(
        1
        for prev, current in zip(normalized, normalized[1:])
        if current and current == prev
    )
    total_chars = sum(len(text) for text in normalized)
    preview = _read_text(srt_path).replace("\r\n", "\n").splitlines()[:LINE_PREVIEW_COUNT]
    return len(texts), total_chars, repeated, preview


def _run_backend(
    *,
    backend: str,
    video_path: Path,
    video_stem: str,
    run_root: Path,
    log_root: Path | None,
) -> BackendResult:
    backend_root = run_root / backend
    output_dir = backend_root / "output"
    job_temp_root = backend_root / "jobs"
    backend_log_root = (log_root / backend) if log_root is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)
    job_temp_root.mkdir(parents=True, exist_ok=True)
    if backend_log_root is not None:
        backend_log_root.mkdir(parents=True, exist_ok=True)
        log_path = backend_log_root / f"{backend}.stdout.log"
    else:
        log_path = None

    removed = _delete_asr_checkpoints()
    print(f"[{backend}] removed {removed} ASR checkpoint file(s)")
    print(f"[{backend}] running in-process skip-translation pipeline")

    ctx = JobContext.from_spec(
        _Spec(
            backend=backend,
            output_dir=output_dir,
            keep_temp_files=True,
            skip_translation=True,
        ),
        backend,
        str((job_temp_root / backend).resolve()),
        "",
    )

    started = time.perf_counter()
    returncode = 0
    previous_console = pipeline_main.console
    run_log_enabled = backend_log_root is not None
    if run_log_enabled:
        ctx.run_log_enabled = True
        ctx.run_log_dir = str(backend_log_root)
    try:
        pipeline_main.console = pipeline_main.Console(
            force_terminal=False,
            emoji=False,
        )
        if log_path is not None:
            with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
                with redirect_stdout(log_file), redirect_stderr(log_file):
                    artifacts = pipeline_main.run_asr_alignment_f0(
                        str(video_path),
                        ctx=ctx,
                        job_id=backend,
                    )
                    pipeline_main.run_translation_and_write(
                        str(video_path),
                        artifacts,
                        ctx=ctx,
                        job_id=artifacts.job_id,
                    )
        else:
            artifacts = pipeline_main.run_asr_alignment_f0(
                str(video_path),
                ctx=ctx,
                job_id=backend,
            )
            pipeline_main.run_translation_and_write(
                str(video_path),
                artifacts,
                ctx=ctx,
                job_id=artifacts.job_id,
            )
    except Exception as exc:
        returncode = 1
        if log_path is not None:
            with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"\n[ERROR] {type(exc).__name__}: {exc}\n")
        else:
            print(f"[{backend}] error: {type(exc).__name__}: {exc}")
    finally:
        pipeline_main.console = previous_console
    wall_time_s = time.perf_counter() - started

    aligned_path = _find_first(job_temp_root, f"{video_stem}.aligned_segments.json")
    timings_path = _find_first(job_temp_root, f"{video_stem}.timings.json")
    srt_path = output_dir / f"{video_stem}.ja.srt"
    if not srt_path.exists():
        srt_path = _find_first(output_dir, "*.ja.srt")

    aligned_payload = _read_json(aligned_path)
    timings_payload = _read_json(timings_path)
    run_log = timings_payload.get("outputs", {}).get("run_log")
    run_log_path = Path(run_log) if isinstance(run_log, str) and run_log else None
    if run_log_path is None and backend_log_root is not None:
        run_log_path = _find_first(backend_log_root, "*.run.log")
    asr_seconds = _extract_asr_text_seconds(aligned_payload, timings_payload)
    segment_count, total_chars, repeated_segments, preview_lines = _summarize_srt(srt_path)

    status = "ok" if returncode == 0 and srt_path is not None and srt_path.exists() else "failed"
    error = ""
    if status != "ok":
        error = f"returncode={returncode}"
        if log_path is not None:
            error += f"; see {_project_relative(log_path)}"

    print(
        f"[{backend}] done: status={status}, wall={wall_time_s:.1f}s, "
        f"log={_project_relative(log_path) if log_path is not None else 'disabled'}"
    )
    try:
        if job_temp_root.exists():
            shutil.rmtree(job_temp_root)
    except OSError as exc:
        print(f"[WARN] failed to remove temp jobs {_project_relative(job_temp_root)}: {exc}")
    return BackendResult(
        backend=backend,
        status=status,
        returncode=returncode,
        wall_time_s=wall_time_s,
        asr_text_transcribe_s=asr_seconds,
        segment_count=segment_count,
        total_chars=total_chars,
        repeated_segments=repeated_segments,
        srt_path=srt_path if srt_path and srt_path.exists() else None,
        aligned_segments_path=aligned_path,
        timings_path=timings_path,
        log_path=log_path,
        run_log_path=run_log_path,
        preview_lines=preview_lines,
        error=error,
    )


def _fmt_seconds(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def _fmt_path(path: Path | None) -> str:
    return _project_relative(path)


def _cell(text: str, width: int = PREVIEW_CELL_WIDTH) -> str:
    value = str(text).replace("|", "\\|").replace("\t", " ").strip()
    if len(value) <= width:
        return value
    return value[: max(0, width - 3)] + "..."


def _write_report(
    *,
    report_path: Path,
    video_path: Path,
    run_root: Path,
    log_root: Path | None,
    results: list[BackendResult],
) -> None:
    lines: list[str] = [
        f"# ASR backend comparison: {video_path.name}",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Video: `{_project_relative(video_path)}`",
        f"- Artifact root: `{_project_relative(run_root)}`",
        f"- Log root: `{_project_relative(log_root)}`" if log_root is not None else "- Log root: disabled",
        "",
        "## Timing comparison",
        "",
        "| Backend | Status | ASR text transcribe (s) | Wall time (s) | JA SRT | Stdout log | Run log |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.backend,
                    result.status,
                    _fmt_seconds(result.asr_text_transcribe_s),
                    f"{result.wall_time_s:.2f}",
                    f"`{_fmt_path(result.srt_path)}`",
                    f"`{_fmt_path(result.log_path)}`",
                    f"`{_fmt_path(result.run_log_path)}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Statistics comparison",
            "",
            "| Backend | Segment count | Total chars | Consecutive repeats |",
            "|---|---:|---:|---:|",
        ]
    )
    for result in results:
        lines.append(
            f"| {result.backend} | {result.segment_count} | "
            f"{result.total_chars} | {result.repeated_segments} |"
        )

    failures = [result for result in results if result.error]
    if failures:
        lines.extend(["", "## Failures", ""])
        for result in failures:
            lines.append(f"- `{result.backend}`: {result.error}")

    lines.extend(
        [
            "",
            f"## First {LINE_PREVIEW_COUNT} .ja.srt lines",
            "",
            "| Line | " + " | ".join(result.backend for result in results) + " |",
            "|---:|" + "|".join("---" for _ in results) + "|",
        ]
    )
    for index in range(LINE_PREVIEW_COUNT):
        row = [str(index + 1)]
        for result in results:
            row.append(_cell(result.preview_lines[index] if index < len(result.preview_lines) else ""))
        lines.append("| " + " | ".join(row) + " |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    video_path = _resolve_project_path(args.video_path)
    output_dir = _resolve_project_path(args.output_dir)
    log_dir = _resolve_project_path(args.log_dir or "log") if args.log or args.log_dir else None

    if not video_path.exists():
        print(f"BLOCKED. video not found: {_project_relative(video_path)}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = video_path.stem
    run_root = output_dir / f"asr_compare_work_{video_stem}_{timestamp}"
    report_path = output_dir / f"asr_compare_{video_stem}_{timestamp}.md"
    compare_log_root = (
        log_dir / f"asr_compare_{video_stem}_{timestamp}" if log_dir is not None else None
    )
    run_root.mkdir(parents=True, exist_ok=True)
    if compare_log_root is not None:
        compare_log_root.mkdir(parents=True, exist_ok=True)

    print(f"[compare] video: {_project_relative(video_path)}")
    print(f"[compare] output report: {_project_relative(report_path)}")
    print(f"[compare] run root: {_project_relative(run_root)}")
    print(f"[compare] log root: {_project_relative(compare_log_root) if compare_log_root is not None else 'disabled'}")

    results = [
        _run_backend(
            backend=backend,
            video_path=video_path,
            video_stem=video_stem,
            run_root=run_root,
            log_root=compare_log_root,
        )
        for backend in BACKENDS
    ]
    _write_report(
        report_path=report_path,
        video_path=video_path,
        run_root=run_root,
        log_root=compare_log_root,
        results=results,
    )
    print(f"[compare] report written: {_project_relative(report_path)}")
    return 0 if all(result.status == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

