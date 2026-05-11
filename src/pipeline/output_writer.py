from __future__ import annotations

import json
import os
import re
from pathlib import Path


def project_relative(path: str | Path | None, *, project_root: Path) -> str | None:
    if path is None:
        return None
    raw = str(path)
    if not raw:
        return raw
    project_root_text = project_root.resolve().as_posix()
    normalized = raw.replace("\\", "/")
    root_pattern = re.compile(re.escape(project_root_text) + r"/?", re.IGNORECASE)
    normalized = root_pattern.sub("", normalized)
    if normalized != raw.replace("\\", "/"):
        return normalized or "."
    candidate = Path(raw)
    try:
        if candidate.is_absolute():
            return candidate.resolve().relative_to(project_root).as_posix()
    except (OSError, ValueError):
        return raw.replace("\\", "/")
    return raw.replace("\\", "/")


def relativize_payload_paths(value, *, project_root: Path):
    if isinstance(value, dict):
        return {
            key: relativize_payload_paths(item, project_root=project_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            relativize_payload_paths(item, project_root=project_root)
            for item in value
        ]
    if isinstance(value, str):
        return project_relative(value, project_root=project_root)
    return value


def write_json(path: str, payload: dict, *, project_root: Path) -> None:
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(
            relativize_payload_paths(payload, project_root=project_root),
            writer,
            ensure_ascii=False,
            indent=2,
        )


def write_json_atomic(path: str | Path, payload: dict, *, project_root: Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(
            relativize_payload_paths(payload, project_root=project_root),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp_path.replace(target)


def timings_payload(
    *,
    video_path: str,
    audio_path: str,
    audio_cached: bool,
    job_id: str,
    job_temp_dir: str,
    device: str,
    backend: str,
    counts: dict,
    stage_timings: dict,
    asr_details: dict,
    translation_request_timings: list[dict],
    translation_api_retry_events: list[dict],
    outputs: dict,
    asr_log: list[str],
    asr_qc_blocked: bool | None = None,
) -> dict:
    payload = {
        "video_path": video_path,
        "audio_path": audio_path,
        "audio_cached": audio_cached,
        "job_id": job_id,
        "job_temp_dir": job_temp_dir,
        "device": device,
        "backend": backend,
        "counts": counts,
        "stage_timings": stage_timings,
        "asr_details": asr_details,
        "translation_request_timings": translation_request_timings,
        "translation_api_retry_events": translation_api_retry_events,
        "outputs": outputs,
        "asr_log": asr_log,
    }
    if asr_qc_blocked is not None:
        payload["asr_qc_blocked"] = asr_qc_blocked
    return payload
