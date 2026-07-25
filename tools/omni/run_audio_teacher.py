#!/usr/bin/env python3
"""Run a named Omni/Gemini/Qwen audio teacher over files with resumable progress.

This tool is transport-only.  It never interprets a response as training truth;
task-specific validators and compilers must consume the JSON result separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.asr.cueqc.label_pre_asr_with_omni import (  # noqa: E402
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    GEMINI_THINKING_LEVELS,
)
from tools.omni.audio_teacher_transport import (  # noqa: E402
    KNOWN_AUDIO_TEACHER_PROFILES,
    create_audio_teacher_transport,
)
from tools.omni.audio_teacher_batch import (  # noqa: E402
    iter_completed_audio_teacher_items,
    resolve_worker_count,
)


RESULT_SCHEMA = "omni_audio_teacher_result_v1"
SUMMARY_SCHEMA = "omni_audio_teacher_summary_v1"
PROGRESS_SCHEMA = "omni_audio_teacher_progress_v1"
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


def _effective_max_tokens(profile: str, requested: int) -> int:
    if requested > 0:
        return int(requested)
    return 8192 if profile in {"openrouter", "gemini"} else 2048


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _resolve_env(args: argparse.Namespace) -> Path:
    profile = str(args.env_file)
    return (Path.home() / ".config" / "omni" / profile).resolve()


def _audio_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths: list[Path] = []
    if args.folder:
        root = Path(args.folder).expanduser().resolve()
        iterator = root.rglob("*") if args.recursive else root.glob("*")
        paths.extend(path for path in iterator if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    paths.extend(Path(value).expanduser().resolve() for value in args.file)
    if args.manifest:
        manifest = Path(args.manifest).expanduser().resolve()
        for row in _rows(manifest):
            raw = Path(str(row.get(args.audio_field) or ""))
            candidate = raw if raw.is_absolute() else (manifest.parent / raw)
            paths.append(candidate.resolve())
    unique = sorted({path for path in paths if path.is_file()}, key=lambda path: str(path).lower())
    if not unique:
        raise ValueError("no audio files found; use --folder, --file, or --manifest")
    return [{"item_id": path.stem, "audio": str(path), "audio_sha256": _sha256(path)} for path in unique]


def run(args: argparse.Namespace) -> dict[str, Any]:
    env_file = _resolve_env(args)
    profile = env_file.name.lower()
    model_env = tuple(value for value in args.model_env.split(",") if value)
    key_env = tuple(value for value in args.api_key_env.split(",") if value)
    base_env = tuple(value for value in args.base_url_env.split(",") if value)
    transport = create_audio_teacher_transport(
        profile=profile,
        env_file=env_file,
        model_override=str(args.model or ""),
        timeout_s=float(args.timeout_s),
        log=lambda message: print(message, flush=True),
        model_env=model_env,
        api_key_env=key_env,
        base_url_env=base_env,
    )
    model = transport.model
    rows = _audio_rows(args)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.jsonl"
    raw_path = output / "raw_responses.jsonl"
    progress_path = output / "progress.json"
    reasoning_effort = (
        str(args.reasoning_effort).lower() if args.enable_thinking else "none"
    ) if profile in {"openrouter", "gemini"} else ""
    effective_thinking_budget = (
        int(args.thinking_budget) if profile == "qwen" else 0
    )
    effective_max_tokens = _effective_max_tokens(profile, int(args.max_tokens))
    existing = {
        str(row.get("item_id")): row
        for row in _rows(result_path)
        if row.get("schema") == RESULT_SCHEMA
        and row.get("model") == model
        and row.get("profile") == profile
        and row.get("reasoning_effort") == reasoning_effort
        and bool(row.get("enable_thinking")) == bool(args.enable_thinking)
        and int(row.get("thinking_budget") or 0) == effective_thinking_budget
        and int(row.get("max_tokens") or 0) == effective_max_tokens
    }
    pending = [row for row in rows if row["item_id"] not in existing]
    if args.limit > 0:
        pending = pending[: args.limit]
    mode = transport.audio_content_mode
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8-sig")
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8-sig")
    if not prompt:
        raise ValueError("--prompt or --prompt-file is required")
    worker_count = (
        resolve_worker_count(
            requested=int(args.workers),
            provider_limit=int(transport.max_concurrency),
            item_count=len(pending),
        )
        if pending
        else 0
    )
    started = time.perf_counter()
    _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running", "profile": profile, "model": model, "audio_content_mode": mode, "worker_count": worker_count, "completed": len(existing), "total": len(rows), "pending": len(pending)})

    def execute_row(row: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, args.max_attempts + 1):
            request_started = time.perf_counter()
            print(f"omni_request provider={profile} item={row['item_id']} attempt={attempt}/{args.max_attempts} workers={worker_count}", flush=True)
            try:
                response = transport.call_json(
                    audio_path=Path(row["audio"]),
                    system_prompt=system_prompt,
                    prompt=prompt,
                    response_schema=None,
                    enable_thinking=bool(args.enable_thinking),
                    thinking_level=(
                        reasoning_effort if reasoning_effort != "none" else "minimal"
                    ),
                    thinking_budget=int(args.thinking_budget),
                    max_tokens=effective_max_tokens,
                    store_stream_chunks=bool(args.store_stream_chunks),
                )
                parsed, raw = response.parsed, response.raw
                result = {"schema": RESULT_SCHEMA, "item_id": row["item_id"], "audio": row["audio"], "audio_sha256": row["audio_sha256"], "model": model, "profile": profile, "transport": transport.transport_name, "audio_content_mode": mode, "reasoning_effort": reasoning_effort, "enable_thinking": bool(args.enable_thinking), "thinking_budget": effective_thinking_budget, "max_tokens": effective_max_tokens, "response": parsed, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                attempts.append(
                    {"item_id": row["item_id"], "attempt": attempt, "response": raw}
                )
                return {
                    "ok": True,
                    "result": result,
                    "attempts": attempts,
                    "request_s": time.perf_counter() - request_started,
                }
            except Exception as error:  # noqa: BLE001
                last_error = error
                print(f"omni_error provider={profile} item={row['item_id']} attempt={attempt}/{args.max_attempts} error={type(error).__name__}: {error}", flush=True)
                attempts.append(
                    {"item_id": row["item_id"], "attempt": attempt, "error": repr(error)}
                )
                if attempt < args.max_attempts:
                    time.sleep(min(8.0, float(attempt)))
        return {
            "ok": False,
            "attempts": attempts,
            "last_error": last_error,
        }

    completed_items = iter_completed_audio_teacher_items(
        items=pending,
        worker=execute_row,
        max_workers=max(1, worker_count),
        sequential_interval_s=(
            float(args.request_interval_s) if worker_count == 1 else 0.0
        ),
    )
    failures: list[tuple[str, Exception]] = []
    for completed in completed_items:
        row = completed.item
        outcome = completed.result
        with raw_path.open("a", encoding="utf-8") as handle:
            for attempt_record in outcome["attempts"]:
                handle.write(
                    json.dumps(attempt_record, ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
        if not outcome["ok"]:
            last_error = outcome["last_error"]
            failures.append((str(row["item_id"]), last_error))
            _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running_with_failures", "profile": profile, "model": model, "worker_count": worker_count, "completed": len(existing), "failed": len(failures), "total": len(rows), "pending": len(rows)-len(existing)-len(failures), "last_item_id": row["item_id"], "last_error": repr(last_error)})
            continue
        result = outcome["result"]
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        existing[row["item_id"]] = result
        total_elapsed = time.perf_counter() - started
        rate = len(existing) / max(total_elapsed, 1e-9)
        eta = (len(rows) - len(existing)) / max(rate, 1e-9)
        print(f"omni_result={len(existing)}/{len(rows)} provider={profile} item={row['item_id']} request_s={outcome['request_s']:.1f} eta_s={eta:.0f}", flush=True)
        _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running_with_failures" if failures else "running", "profile": profile, "model": model, "audio_content_mode": mode, "worker_count": worker_count, "completed": len(existing), "failed": len(failures), "total": len(rows), "pending": len(rows)-len(existing)-len(failures), "last_item_id": row["item_id"], "last_request_s": round(outcome["request_s"], 3), "elapsed_s": round(total_elapsed, 3), "eta_s": round(eta, 3)})
    if failures:
        item_id, last_error = failures[0]
        _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "failed", "profile": profile, "model": model, "worker_count": worker_count, "completed": len(existing), "failed": len(failures), "total": len(rows), "pending": len(rows)-len(existing), "last_item_id": item_id, "last_error": repr(last_error)})
        raise RuntimeError(
            f"Teacher failed for {len(failures)} item(s); "
            f"first={item_id}: {last_error}"
        ) from last_error
    summary = {"schema": SUMMARY_SCHEMA, "profile": profile, "model": model, "transport": transport.transport_name, "api_key_count": transport.api_key_count, "worker_count": worker_count, "audio_content_mode": mode, "reasoning_effort": reasoning_effort, "enable_thinking": bool(args.enable_thinking), "thinking_budget": effective_thinking_budget, "max_tokens": effective_max_tokens, "omitted_sampling_parameters": ["temperature", "top_p", "top_k"], "source_count": len(rows), "result_count": len(existing), "results": str(result_path), "raw_responses": str(raw_path), "training_manifest_allowed": False}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "completed", "profile": profile, "model": model, "audio_content_mode": mode, "completed": len(existing), "total": len(rows), "pending": 0, "elapsed_s": round(time.perf_counter()-started, 3), "summary": str(output / "summary.json")})
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default="openrouter",
        choices=KNOWN_AUDIO_TEACHER_PROFILES,
        help="Named profile: qwen/openrouter use compatible APIs; gemini uses Google AI Studio.",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--model-env", default="OMNI_MODEL,QWEN_OMNI_MODEL")
    parser.add_argument("--api-key-env", default=",".join(DEFAULT_API_KEY_ENV_CANDIDATES))
    parser.add_argument("--base-url-env", default=",".join(DEFAULT_BASE_URL_ENV_CANDIDATES))
    parser.add_argument("--folder", default="")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--manifest", default="")
    parser.add_argument("--audio-field", default="audio")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--system-prompt-file", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="0 selects OpenRouter/Gemini=8192 or Qwen=2048.",
    )
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument(
        "--reasoning-effort",
        choices=GEMINI_THINKING_LEVELS,
        default="medium",
        help="OpenRouter/native Gemini thinking level; Qwen uses --thinking-budget.",
    )
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store-stream-chunks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-interval-s", type=float, default=0.5)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 uses one worker per native Gemini key; compatible providers remain single-worker.",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers < 0 or args.max_attempts <= 0 or args.request_interval_s < 0:
        parser.error("worker/attempt/interval values are invalid")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
