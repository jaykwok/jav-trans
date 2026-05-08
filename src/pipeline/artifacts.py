from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

from utils.model_paths import PROJECT_ROOT


@dataclass
class AsrArtifacts:
    segments: list[dict]
    audio_path: str
    job_temp_dir: str
    asr_details: dict
    aligned_segments_path: str
    transcript_path: str
    asr_manifest_path: str
    pipeline_timings: dict
    logger: object
    run_log_path: object
    audio_cache_key: str
    video_stem: str
    output_dir: str
    srt_path: str
    bilingual_json_path: str
    quality_report_path: str
    bilingual: bool
    timings_path: str
    translation_cache_path: str
    asr_log: list[str]
    audio_cached: bool
    device: str
    backend_label: str
    video_duration_s: float | None
    pipeline_started: float
    f0_filtered_count: int
    f0_failed: bool
    job_id: str


TRANSLATION_ARTIFACTS_SNAPSHOT = "translation_artifacts.json"
ASR_ARTIFACT_PATH_FIELDS = {
    "audio_path",
    "job_temp_dir",
    "aligned_segments_path",
    "transcript_path",
    "asr_manifest_path",
    "run_log_path",
    "output_dir",
    "srt_path",
    "bilingual_json_path",
    "quality_report_path",
    "timings_path",
    "translation_cache_path",
}


def translation_artifacts_snapshot_path(job_temp_dir: str | Path) -> Path:
    return Path(job_temp_dir) / TRANSLATION_ARTIFACTS_SNAPSHOT


def serialize_asr_artifacts(artifacts: AsrArtifacts) -> dict:
    payload: dict[str, object] = {}
    for item in fields(AsrArtifacts):
        name = item.name
        if name == "logger":
            continue
        value = getattr(artifacts, name)
        if isinstance(value, Path):
            payload[name] = str(value)
        else:
            payload[name] = value
    return payload


def _project_relative(path: str | Path | None) -> str | None:
    if path is None:
        return None
    raw = str(path)
    if not raw:
        return raw
    project_root_text = PROJECT_ROOT.resolve().as_posix()
    normalized = raw.replace("\\", "/")
    root_pattern = re.compile(re.escape(project_root_text) + r"/?", re.IGNORECASE)
    normalized = root_pattern.sub("", normalized)
    if normalized != raw.replace("\\", "/"):
        return normalized or "."
    candidate = Path(raw)
    try:
        if candidate.is_absolute():
            return candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        return raw.replace("\\", "/")
    return raw.replace("\\", "/")


def _relativize_payload_paths(value):
    if isinstance(value, dict):
        return {key: _relativize_payload_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_payload_paths(item) for item in value]
    if isinstance(value, str):
        return _project_relative(value)
    return value


def _write_json_atomic(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(_relativize_payload_paths(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(target)


def write_translation_artifacts_snapshot(
    artifacts: AsrArtifacts,
    write_json_atomic: Callable[[Path, dict], None] | None = None,
) -> str:
    path = translation_artifacts_snapshot_path(artifacts.job_temp_dir)
    if write_json_atomic is None:
        write_json_atomic = _write_json_atomic
    write_json_atomic(path, serialize_asr_artifacts(artifacts))
    return str(path)


def _resolve_project_runtime_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def restore_snapshot_path(
    value,
    resolve_project_runtime_path: Callable[[str], Path] | None = None,
) -> str:
    if value is None:
        return ""
    raw = str(value)
    if not raw:
        return ""
    if resolve_project_runtime_path is None:
        resolve_project_runtime_path = _resolve_project_runtime_path
    return str(resolve_project_runtime_path(raw))


def load_translation_artifacts_snapshot(
    path: str | Path,
    resolve_project_runtime_path: Callable[[str], Path] | None = None,
) -> AsrArtifacts | None:
    if resolve_project_runtime_path is None:
        resolve_project_runtime_path = _resolve_project_runtime_path
    snapshot_path = Path(path)
    if snapshot_path.is_dir():
        snapshot_path = translation_artifacts_snapshot_path(snapshot_path)
    if not snapshot_path.exists():
        return None
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    values = {item.name: payload.get(item.name) for item in fields(AsrArtifacts)}
    for name in ASR_ARTIFACT_PATH_FIELDS:
        values[name] = restore_snapshot_path(
            values.get(name),
            resolve_project_runtime_path,
        )
    values["segments"] = list(values.get("segments") or [])
    values["asr_details"] = dict(values.get("asr_details") or {})
    values["pipeline_timings"] = dict(values.get("pipeline_timings") or {})
    values["asr_log"] = [str(item) for item in values.get("asr_log") or []]
    values["logger"] = None
    values["run_log_path"] = (
        Path(values["run_log_path"]) if values.get("run_log_path") else None
    )
    values["audio_cached"] = bool(values.get("audio_cached"))
    values["bilingual"] = bool(values.get("bilingual"))
    values["f0_filtered_count"] = int(values.get("f0_filtered_count") or 0)
    values["f0_failed"] = bool(values.get("f0_failed"))
    values["video_duration_s"] = (
        float(values["video_duration_s"])
        if values.get("video_duration_s") is not None
        else None
    )
    values["pipeline_started"] = time.perf_counter()
    values["audio_cache_key"] = str(values.get("audio_cache_key") or "")
    values["video_stem"] = str(values.get("video_stem") or snapshot_path.stem)
    values["device"] = str(values.get("device") or "cpu")
    values["backend_label"] = str(values.get("backend_label") or "")
    values["job_id"] = str(values.get("job_id") or "")
    try:
        return AsrArtifacts(**values)
    except Exception:
        return None
