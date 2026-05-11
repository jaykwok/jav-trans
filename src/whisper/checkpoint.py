import hashlib
import json
import os
import re
import sys
import uuid
import wave
from pathlib import Path

from whisper.backends.registry import (
    _WHISPER_BACKENDS,
    current_asr_backend,
    current_asr_worker_mode,
)
from whisper.local_backend import ASR_DTYPE, ASR_MODEL_ID, ALIGNMENT_TIMESTAMP_MODE


_LAST_VAD_SIGNATURE: dict = {}
_ASR_CHUNK_ROOT = Path(
    os.getenv("ASR_CHUNK_ROOT", Path("temp") / "chunks")
).resolve()


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
    return Path(os.getenv("ASR_CHUNK_ROOT", _ASR_CHUNK_ROOT)).resolve()


def _is_timed_out_result(result: dict) -> bool:
    return any("TIMEOUT:" in entry for entry in result.get("log", []))


def _checkpointable_text_results(
    text_results_by_index: dict[int, dict],
) -> dict[int, dict]:
    return {
        index: result
        for index, result in text_results_by_index.items()
        if not _is_timed_out_result(result)
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


def _get_whisper_generation_checkpoint_signature(
    *,
    sliding_context_segs: int | None = None,
) -> str:
    backend_name = current_asr_backend()
    if backend_name not in _WHISPER_BACKENDS:
        return ""
    try:
        from whisper.model_backend import WHISPER_PRESETS

        preset = WHISPER_PRESETS.get(backend_name, {})
        beams = os.getenv("WHISPER_BEAMS", "").strip() or str(preset.get("beams", ""))
        no_repeat_ngram = (
            os.getenv("WHISPER_NO_REPEAT_NGRAM", "").strip()
            or str(preset.get("no_repeat_ngram", ""))
        )
        max_new_tokens = (
            os.getenv("WHISPER_MAX_NEW_TOKENS", "").strip()
            or str(preset.get("max_new_tokens", ""))
        )
        forced_fail_ratio = (
            os.getenv("WHISPER_FORCED_FAIL_RATIO", "").strip()
            or str(preset.get("forced_fail_ratio", ""))
        )
        model_path = os.getenv("WHISPER_MODEL_PATH", "").strip()
        if sliding_context_segs is None:
            sliding_context_segs = max(
                0,
                int(os.getenv("ASR_SLIDING_CONTEXT_SEGS", "2")),
            )
        return json.dumps(
            {
                "beams": beams,
                "no_repeat_ngram": no_repeat_ngram,
                "max_new_tokens": max_new_tokens,
                "forced_fail_ratio": forced_fail_ratio,
                "model_path": model_path,
                "sliding_context_segs": sliding_context_segs,
            },
            sort_keys=True,
        )
    except Exception:
        return ""


def _get_asr_checkpoint_path(
    audio_path: str,
    *,
    last_vad_signature: dict | None = None,
    chunk_root: Path | str | None = None,
    sliding_context_segs: int | None = None,
) -> Path:
    vad_signature = json.dumps(
        _LAST_VAD_SIGNATURE if last_vad_signature is None else last_vad_signature,
        sort_keys=True,
    )
    whisper_generation_signature = _get_whisper_generation_checkpoint_signature(
        sliding_context_segs=sliding_context_segs,
    )
    key = hashlib.sha1(
        (
            f"{audio_path}|{ASR_MODEL_ID}|{ASR_DTYPE}|"
            f"{current_asr_worker_mode()}|{ALIGNMENT_TIMESTAMP_MODE}|{vad_signature}|"
            f"{current_asr_backend()}|"
            f"{os.getenv('WHISPER_TIMESTAMP_MODE', 'forced').strip().lower()}|"
            f"{whisper_generation_signature}"
        ).encode()
    ).hexdigest()[:10]
    return _current_chunk_root(chunk_root).parent / f"asr_checkpoint_{key}.json"


def _get_asr_checkpoint_source(chunks: list[dict], text_stage_label: str) -> str:
    source = str(chunks[0].get("source_audio_path") or chunks[0].get("path", ""))
    return f"{source}|{text_stage_label}"


def _chunk_checkpoint_signature(
    chunks: list[dict],
    *,
    last_vad_signature: dict | None = None,
) -> dict[str, dict[str, float | str]]:
    vad_signature = _LAST_VAD_SIGNATURE if last_vad_signature is None else last_vad_signature
    return {
        str(int(chunk["index"])): {
            "start": round(float(chunk.get("start", 0.0)), 3),
            "end": round(float(chunk.get("end", 0.0)), 3),
            "vad_method": vad_signature.get("backend", "unknown"),
        }
        for chunk in chunks
    }


def _load_asr_checkpoint(
    checkpoint_path: Path,
    checkpoint_source: str,
    chunks: list[dict],
    run_id: str | None = None,
    *,
    last_vad_signature: dict | None = None,
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
        last_vad_signature=last_vad_signature,
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
        if _is_timed_out_result(value):
            continue
        saved_chunk_signature = (
            saved_signature.get(str(chunk_index))
            if isinstance(saved_signature, dict)
            else None
        )
        if isinstance(saved_chunk_signature, dict) and "vad_method" not in saved_chunk_signature:
            print(
                f"[WARN] ASR checkpoint resume: chunk {chunk_index} missing vad_method; skip stale checkpoint entry",
                file=sys.stderr,
            )
            continue
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
    last_vad_signature: dict | None = None,
    checkpoint_enabled: bool | None = None,
) -> None:
    if not _checkpoint_enabled(checkpoint_enabled):
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio_path": checkpoint_source,
        "chunks": _chunk_checkpoint_signature(
            chunks,
            last_vad_signature=last_vad_signature,
        ),
        "results": {str(key): value for key, value in sorted(results_by_index.items())},
    }
    if run_id:
        payload["run_id"] = run_id
    tmp_path = checkpoint_path.with_name(
        f"{checkpoint_path.name}.{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        with open(tmp_path, "w", encoding="utf-8") as writer:
            json.dump(payload, writer, ensure_ascii=False)
        tmp_path.replace(checkpoint_path)
    finally:
        _delete_path_for_cleanup(tmp_path)


def _build_quarantined_text_result(
    chunk: dict,
    *,
    kind: str,
    detail: str,
    respawn_count: int,
    run_id: str | None = None,
) -> dict:
    normalized_path = str(Path(chunk["path"]).resolve())
    try:
        duration = _get_wav_duration(normalized_path)
    except Exception:
        duration = max(
            0.0,
            float(chunk.get("end", 0.0)) - float(chunk.get("start", 0.0)),
        )

    return {
        "text": "",
        "raw_text": "",
        "duration": duration,
        "language": "Japanese",
        "normalized_path": normalized_path,
        "segments": [],
        "log": [
            (
                "QUARANTINED: "
                f"kind={kind}, respawn_count={respawn_count}, "
                f"run_id={run_id or ''}, detail={detail}"
            )
        ],
    }


def _quarantine_failed_chunks(
    checkpoint_source: str,
    chunks: list[dict],
    failure_records: list[dict],
    *,
    run_id: str | None = None,
    worker_mode: str | None = None,
) -> list[Path]:
    if not failure_records:
        return []

    chunk_by_index = {int(chunk["index"]): chunk for chunk in chunks}
    out_dir = _ASR_CHUNK_ROOT.parent / "asr_timeouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for record in failure_records:
        try:
            chunk_index = int(record["index"])
        except (KeyError, TypeError, ValueError):
            continue
        chunk = chunk_by_index.get(chunk_index)
        if chunk is None:
            continue

        record_run_id = str(record.get("run_id") or run_id or "unknown")
        chunk_key = hashlib.sha1(
            f"{checkpoint_source}|{chunk_index}".encode()
        ).hexdigest()[:10]
        target = out_dir / f"timeouts_{chunk_key}_{record_run_id}.json"
        tmp_path = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}.tmp")
        payload = {
            "run_id": record_run_id,
            "audio_path": checkpoint_source,
            "chunk_index": chunk_index,
            "start": float(chunk.get("start", 0.0)),
            "end": float(chunk.get("end", 0.0)),
            "model": ASR_MODEL_ID,
            "dtype": ASR_DTYPE,
            "timeout_s": float(os.getenv("TRANSCRIPTION_TIMEOUT_S", "180")),
            "respawn_count": int(record.get("respawn_count", 0)),
            "failure_kind": str(record.get("kind") or "crash"),
            "last_error": str(record.get("detail", "")),
            "worker_mode": str(record.get("worker_mode") or worker_mode or current_asr_worker_mode()),
        }
        try:
            with open(tmp_path, "w", encoding="utf-8") as writer:
                json.dump(payload, writer, ensure_ascii=False, indent=2)
            tmp_path.replace(target)
            written.append(target)
        except Exception:
            _delete_path_for_cleanup(tmp_path)

    return written


def aggregate_timeout_fragments(job_id: str) -> Path | None:
    normalized_job_id = re.sub(r"[^0-9A-Za-z._-]+", "_", (job_id or "").strip())
    normalized_job_id = normalized_job_id.strip("._-")
    if not normalized_job_id:
        return None

    out_dir = _ASR_CHUNK_ROOT.parent / "asr_timeouts"
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
