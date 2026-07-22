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
    call_omni,
    first_env_value,
    load_env_file,
    normalize_openai_compat_base_url,
)


RESULT_SCHEMA = "omni_audio_teacher_result_v1"
SUMMARY_SCHEMA = "omni_audio_teacher_summary_v1"
PROGRESS_SCHEMA = "omni_audio_teacher_progress_v1"
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


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
    load_env_file(env_file)
    model_env = tuple(value for value in args.model_env.split(",") if value)
    key_env = tuple(value for value in args.api_key_env.split(",") if value)
    base_env = tuple(value for value in args.base_url_env.split(",") if value)
    _model_name, configured_model = first_env_value(model_env)
    model = args.model or configured_model
    _key_name, api_key = first_env_value(key_env)
    _base_name, raw_base_url = first_env_value(base_env)
    base_url = normalize_openai_compat_base_url(raw_base_url)
    if not model or not api_key:
        raise RuntimeError("model and API key are required")
    rows = _audio_rows(args)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.jsonl"
    raw_path = output / "raw_responses.jsonl"
    progress_path = output / "progress.json"
    existing = {str(row.get("item_id")): row for row in _rows(result_path) if row.get("schema") == RESULT_SCHEMA and row.get("model") == model}
    pending = [row for row in rows if row["item_id"] not in existing]
    if args.limit > 0:
        pending = pending[: args.limit]
    profile = env_file.name
    mode = {"qwen": "input_audio", "gemini": "input_audio_raw"}[profile.lower()]
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8-sig")
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8-sig")
    if not prompt:
        raise ValueError("--prompt or --prompt-file is required")
    started = time.perf_counter()
    _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running", "profile": profile, "model": model, "audio_content_mode": mode, "completed": len(existing), "total": len(rows), "pending": len(pending)})
    for position, row in enumerate(pending, start=1):
        last_error: Exception | None = None
        for attempt in range(1, args.max_attempts + 1):
            request_started = time.perf_counter()
            print(f"omni_request={len(existing)+1}/{len(rows)} provider={profile} item={row['item_id']} attempt={attempt}/{args.max_attempts}", flush=True)
            try:
                parsed, raw = call_omni(audio_path=Path(row["audio"]), fmt=Path(row["audio"]).suffix.lstrip(".") or "wav", audio_content_mode=mode, model=model, api_key=api_key, base_url=base_url, timeout_s=args.timeout_s, store_stream_chunks=args.store_stream_chunks, prompt=prompt, system_prompt=system_prompt, max_tokens=args.max_tokens, enable_thinking=args.enable_thinking, thinking_budget=args.thinking_budget)
                result = {"schema": RESULT_SCHEMA, "item_id": row["item_id"], "audio": row["audio"], "audio_sha256": row["audio_sha256"], "model": model, "profile": profile, "audio_content_mode": mode, "response": parsed, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                with result_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                with raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"item_id": row["item_id"], "attempt": attempt, "response": raw}, ensure_ascii=False, sort_keys=True) + "\n")
                existing[row["item_id"]] = result
                elapsed = time.perf_counter() - request_started
                total_elapsed = time.perf_counter() - started
                rate = len(existing) / max(total_elapsed, 1e-9)
                eta = (len(rows) - len(existing)) / max(rate, 1e-9)
                print(f"omni_result={len(existing)}/{len(rows)} provider={profile} item={row['item_id']} request_s={elapsed:.1f} eta_s={eta:.0f}", flush=True)
                _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running", "profile": profile, "model": model, "audio_content_mode": mode, "completed": len(existing), "total": len(rows), "pending": len(rows)-len(existing), "last_item_id": row["item_id"], "last_request_s": round(elapsed, 3), "elapsed_s": round(total_elapsed, 3), "eta_s": round(eta, 3)})
                last_error = None
                break
            except Exception as error:  # noqa: BLE001
                last_error = error
                print(f"omni_error provider={profile} item={row['item_id']} attempt={attempt}/{args.max_attempts} error={type(error).__name__}: {error}", flush=True)
                with raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"item_id": row["item_id"], "attempt": attempt, "error": repr(error)}, ensure_ascii=False, sort_keys=True) + "\n")
                if attempt < args.max_attempts:
                    time.sleep(min(8.0, float(attempt)))
        if last_error is not None:
            _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "failed", "profile": profile, "model": model, "completed": len(existing), "total": len(rows), "pending": len(rows)-len(existing), "last_item_id": row["item_id"], "last_error": repr(last_error)})
            raise RuntimeError(f"teacher failed for {row['item_id']}: {last_error}") from last_error
        if position < len(pending) and args.request_interval_s > 0:
            time.sleep(args.request_interval_s)
    summary = {"schema": SUMMARY_SCHEMA, "profile": profile, "model": model, "audio_content_mode": mode, "source_count": len(rows), "result_count": len(existing), "results": str(result_path), "raw_responses": str(raw_path), "training_manifest_allowed": False}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "completed", "profile": profile, "model": model, "audio_content_mode": mode, "completed": len(existing), "total": len(rows), "pending": 0, "elapsed_s": round(time.perf_counter()-started, 3), "summary": str(output / "summary.json")})
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default="gemini",
        choices=("qwen", "gemini"),
        help="Named ~/.config/omni profile. Gemini is the default; use qwen explicitly.",
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
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store-stream-chunks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-interval-s", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
