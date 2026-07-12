import hashlib
import json
import os
import re
import uuid
import wave
from pathlib import Path
from typing import Any

from asr.backends.registry import (
    current_asr_backend,
    current_asr_worker_mode,
)
from asr.backends.qwen import active_qwen_asr_model_id
from asr.pre_asr_cueqc import runtime_signature as pre_asr_cueqc_runtime_signature


_LAST_BOUNDARY_SIGNATURE: dict = {}


def _checkpoint_enabled(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.getenv("ASR_CHECKPOINT_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _current_chunk_root(chunk_root: Path | str | None = None) -> Path:
    if chunk_root is not None:
        return Path(chunk_root).resolve()
    return Path(os.getenv("ASR_CHUNK_ROOT", Path("tmp") / "chunks")).resolve()


def _is_timed_out_result(result: dict) -> bool:
    return any("TIMEOUT:" in entry for entry in result.get("log", []))


def _is_quarantined_result(result: dict) -> bool:
    # Circuit-breaker quarantined chunks must NOT be persisted as completed
    # results: they carry empty text and must be re-transcribed on resume.
    # Tagged via asr_generation.policy == "quarantined_result" (and a
    # "QUARANTINED:" log line), distinct from timeout results.
    generation = result.get("asr_generation")
    if isinstance(generation, dict) and generation.get("policy") == "quarantined_result":
        return True
    return any(
        isinstance(entry, str) and entry.startswith("QUARANTINED:")
        for entry in result.get("log", [])
    )


def _checkpointable_text_results(
    text_results_by_index: dict[int, dict],
) -> dict[int, dict]:
    return {
        index: result
        for index, result in text_results_by_index.items()
        if not _is_timed_out_result(result) and not _is_quarantined_result(result)
    }


def _delete_path_for_cleanup(path: Path) -> None:
    if not path.exists():
        return

    try:
        if path.is_dir():
            import shutil

            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except Exception:
        pass


def _get_wav_duration(audio_path: str) -> float:
    with wave.open(audio_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    return frames / rate if rate else 0.0


def _env_text(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_lower(name: str, default: str = "") -> str:
    return _env_text(name, default).lower()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _jsonable(value.item())
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return _jsonable(value.item())
            array = np.ascontiguousarray(value)
            digest = hashlib.sha256(array.tobytes()).hexdigest()
            return {
                "array_type": "ndarray",
                "dtype": str(array.dtype),
                "shape": [int(item) for item in array.shape],
                "sha256": digest,
            }
    except Exception:
        pass
    return str(value)


def _signature_json(payload: dict) -> str:
    return json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _get_asr_runtime_signature(
    *,
    last_boundary_signature: dict | None = None,
) -> dict:
    boundary_signature = _LAST_BOUNDARY_SIGNATURE if last_boundary_signature is None else last_boundary_signature
    return {
        "version": 8,
        "backend": current_asr_backend(),
        "worker_mode": current_asr_worker_mode(),
        "timestamp": {
            "source": "boundary_chunk_timeline",
        },
        "model": {
            "asr_model_id": _env_text("ASR_MODEL_ID", ""),
            "resolved_asr_model_id": active_qwen_asr_model_id(),
            "asr_model_path": _env_text("ASR_MODEL_PATH", ""),
            "asr_dtype": _env_lower("ASR_DTYPE", "auto"),
            "asr_attention": _env_lower("ASR_ATTENTION", "auto"),
        },
        "language": {
            "asr_language": _env_text("ASR_LANGUAGE", "Japanese") or "Japanese",
            "asr_force_language": _env_lower("ASR_FORCE_LANGUAGE", "1"),
        },
        "generation": {
            "asr_max_new_tokens": _env_text("ASR_MAX_NEW_TOKENS", "128"),
            "asr_repetition_penalty": _env_text("ASR_REPETITION_PENALTY", "1.05"),
        },
        "pre_asr_cueqc": pre_asr_cueqc_runtime_signature(),
        "boundary": _jsonable(boundary_signature) if isinstance(boundary_signature, dict) else {},
    }


def _get_asr_checkpoint_path(
    audio_path: str,
    *,
    last_boundary_signature: dict | None = None,
    chunk_root: Path | str | None = None,
) -> Path:
    runtime_signature = _get_asr_runtime_signature(
        last_boundary_signature=last_boundary_signature,
    )
    key = hashlib.sha1(
        _signature_json(
            {
                "audio_path": audio_path,
                "runtime": runtime_signature,
            }
        ).encode()
    ).hexdigest()[:10]
    return _current_chunk_root(chunk_root).parent / f"asr_checkpoint_{key}.json"


def _get_asr_checkpoint_source(chunks: list[dict], text_stage_label: str) -> str:
    source = str(chunks[0].get("source_audio_path") or chunks[0].get("path", ""))
    return f"{source}|{text_stage_label}"


def _chunk_checkpoint_signature(
    chunks: list[dict],
    *,
    last_boundary_signature: dict | None = None,
) -> dict[str, dict[str, float | str]]:
    # last_boundary_signature is accepted for interface symmetry with the
    # checkpoint-path helpers but intentionally unused here: the previous
    # "boundary_method" entry read boundary_signature.get("backend"), which
    # never exists in the runtime boundary signature (it holds
    # result.parameters + boundary_pipeline), so the value was always the
    # default "unknown" -- pure noise. Chunk identity is fully captured by
    # start/end plus the runtime-signature hash baked into the checkpoint
    # filename, so the vestigial field and its stale-entry guard were removed.
    return {
        str(int(chunk["index"])): {
            "start": round(float(chunk.get("start", 0.0)), 3),
            "end": round(float(chunk.get("end", 0.0)), 3),
        }
        for chunk in chunks
    }


def _load_asr_checkpoint(
    checkpoint_path: Path,
    checkpoint_source: str,
    chunks: list[dict],
    run_id: str | None = None,
    *,
    last_boundary_signature: dict | None = None,
    checkpoint_enabled: bool | None = None,
) -> dict[int, dict]:
    if not _checkpoint_enabled(checkpoint_enabled) or not checkpoint_path.exists():
        return {}

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as reader:
            payload = json.load(reader)
    except Exception:
        return {}

    if payload.get("audio_path") != checkpoint_source:
        return {}

    raw_results = payload.get("results", {})
    if not isinstance(raw_results, dict):
        return {}

    chunk_by_index = {int(chunk["index"]): chunk for chunk in chunks}
    expected_signature = _chunk_checkpoint_signature(
        chunks,
        last_boundary_signature=last_boundary_signature,
    )
    saved_signature = payload.get("chunks", {})
    restored: dict[int, dict] = {}

    for key, value in raw_results.items():
        try:
            chunk_index = int(key)
        except (TypeError, ValueError):
            continue
        if chunk_index not in chunk_by_index or not isinstance(value, dict):
            continue
        if _is_timed_out_result(value) or _is_quarantined_result(value):
            continue
        saved_chunk_signature = (
            saved_signature.get(str(chunk_index))
            if isinstance(saved_signature, dict)
            else None
        )
        if (
            isinstance(saved_signature, dict)
            and saved_signature
            and saved_chunk_signature != expected_signature.get(str(chunk_index))
        ):
            continue

        chunk = chunk_by_index[chunk_index]
        result = dict(value)
        current_path = str(Path(chunk["path"]).resolve())
        result["normalized_path"] = current_path
        try:
            result["duration"] = _get_wav_duration(current_path)
        except Exception:
            result["duration"] = float(result.get("duration", 0.0))
        result.setdefault("language", "Japanese")
        result.setdefault("text", "")
        result.setdefault("raw_text", result.get("text", ""))
        result_log = list(result.get("log", []))
        result_log.append("ASR checkpoint resume: restored chunk text")
        saved_run_id = payload.get("run_id")
        if run_id and saved_run_id and saved_run_id != run_id:
            result_log.append(
                f"ASR checkpoint resume: saved_run_id={saved_run_id}, current_run_id={run_id}"
            )
        result["log"] = result_log
        restored[chunk_index] = result

    return restored


def _save_asr_checkpoint(
    checkpoint_path: Path,
    checkpoint_source: str,
    chunks: list[dict],
    results_by_index: dict[int, dict],
    run_id: str | None = None,
    *,
    last_boundary_signature: dict | None = None,
    checkpoint_enabled: bool | None = None,
) -> None:
    if not _checkpoint_enabled(checkpoint_enabled):
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio_path": checkpoint_source,
        "chunks": _chunk_checkpoint_signature(
            chunks,
            last_boundary_signature=last_boundary_signature,
        ),
        "results": _jsonable({str(key): value for key, value in sorted(results_by_index.items())}),
    }
    if run_id:
        payload["run_id"] = run_id
    tmp_path = checkpoint_path.with_name(
        f"{checkpoint_path.name}.{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        with open(tmp_path, "w", encoding="utf-8") as writer:
            json.dump(_jsonable(payload), writer, ensure_ascii=False)
        tmp_path.replace(checkpoint_path)
    finally:
        _delete_path_for_cleanup(tmp_path)


def aggregate_timeout_fragments(job_id: str) -> Path | None:
    normalized_job_id = re.sub(r"[^0-9A-Za-z._-]+", "_", (job_id or "").strip())
    normalized_job_id = normalized_job_id.strip("._-")
    if not normalized_job_id:
        return None

    out_dir = _current_chunk_root().parent / "asr_timeouts"
    if not out_dir.exists() or not out_dir.is_dir():
        return None

    fragments: list[Path] = []
    for path in out_dir.glob("timeouts_*.json"):
        if path.name.startswith("timeouts_summary_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates = {
            str(payload.get("job_id") or ""),
            str(payload.get("video_name") or ""),
            str(payload.get("video_stem") or ""),
            str(payload.get("source_job_id") or ""),
        }
        audio_path = str(payload.get("audio_path") or "")
        if normalized_job_id in candidates or f"/{normalized_job_id}/" in audio_path.replace("\\", "/"):
            fragments.append(path)

    if not fragments:
        return None

    records: list[dict] = []
    for path in sorted(fragments, key=lambda item: item.name):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            records.append({"source_file": path.name, "parse_error": repr(exc)})

    summary_path = out_dir / f"timeouts_summary_{normalized_job_id}.json"
    tmp_path = summary_path.with_name(f"{summary_path.name}.{uuid.uuid4().hex[:8]}.tmp")
    payload = {
        "job_id": normalized_job_id,
        "count": len(records),
        "fragments": [path.name for path in sorted(fragments, key=lambda item: item.name)],
        "records": records,
    }
    with open(tmp_path, "w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False, indent=2)
    tmp_path.replace(summary_path)

    for path in fragments:
        _delete_path_for_cleanup(path)

    return summary_path
