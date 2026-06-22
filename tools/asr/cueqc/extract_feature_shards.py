#!/usr/bin/env python3
"""Run CueQC v4 binary feature extraction in resumable shards."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _timestamp_prefix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_output_dir() -> Path:
    return PROJECT_ROOT / "agents" / "temp" / f"{_timestamp_prefix()}_cueqc-feature-shards"


def _project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def _project_rel(value: str | Path) -> str:
    path = Path(value)
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(payload)
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _range_count(total_rows: int, *, start_index: int, max_samples: int | None) -> int:
    if start_index >= total_rows:
        return 0
    available = total_rows - start_index
    return available if max_samples is None else min(available, max_samples)


def _extract_command(
    args: argparse.Namespace,
    *,
    input_path: Path,
    output_path: Path,
    start_index: int,
    max_samples: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tools.asr.cueqc.extract_features_v4_binary",
        "--audio-root",
        str(_project_path(args.audio_root)),
        "--output",
        str(output_path),
        "--device",
        args.device,
        "--start-index",
        str(start_index),
        "--max-samples",
        str(max_samples),
        "--audio-cache-size",
        str(args.audio_cache_size),
    ]
    command.extend(["--train" if args.train else "--input", str(input_path)])
    if args.model_spec:
        command.extend(["--model-spec", args.model_spec])
    return command


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_shard(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[str]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=_child_env(),
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )


def _merge_shards(shards: list[Path], output: Path) -> int:
    command = [sys.executable, "-m", "tools.asr.cueqc.merge_features_v4_binary"]
    for shard in shards:
        command.extend(["--input", str(shard)])
    command.extend(["--output", str(output)])
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=_child_env(), text=True, check=False)
    return int(result.returncode)


def run(args: argparse.Namespace) -> int:
    input_path = _project_path(args.train or args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_dir = _project_path(args.output_dir) if args.output_dir else _default_output_dir()
    shards_dir = output_dir / "shards"
    logs_dir = output_dir / "logs"
    status_path = output_dir / "status.jsonl"
    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    total_rows = _count_jsonl_rows(input_path)
    planned_rows = _range_count(total_rows, start_index=args.start_index, max_samples=args.max_samples)
    _append_status(
        status_path,
        {
            "event": "start",
            "input": _project_rel(input_path),
            "output_dir": _project_rel(output_dir),
            "total_rows": total_rows,
            "start_index": args.start_index,
            "planned_rows": planned_rows,
            "shard_size": args.shard_size,
        },
    )

    shards: list[Path] = []
    failures: list[dict[str, Any]] = []
    shard_index = 0
    for start in range(args.start_index, args.start_index + planned_rows, args.shard_size):
        count = min(args.shard_size, args.start_index + planned_rows - start)
        end = start + count - 1
        shard_name = f"shard_{shard_index:05d}_{start:06d}_{end:06d}.pt"
        shard_path = shards_dir / shard_name
        stdout_path = logs_dir / f"{shard_name}.stdout.log"
        stderr_path = logs_dir / f"{shard_name}.stderr.log"
        shards.append(shard_path)
        if shard_path.exists() and shard_path.stat().st_size > 0 and not args.force:
            _append_status(status_path, {"event": "skip_existing", "shard": shard_name, "start_index": start, "max_samples": count})
            shard_index += 1
            continue

        command = _extract_command(args, input_path=input_path, output_path=shard_path, start_index=start, max_samples=count)
        _append_status(
            status_path,
            {
                "event": "shard_start",
                "shard": shard_name,
                "start_index": start,
                "max_samples": count,
                "command": command,
            },
        )
        result = _run_shard(command, stdout_path=stdout_path, stderr_path=stderr_path)
        if result.returncode != 0:
            failure = {
                "event": "shard_failed",
                "shard": shard_name,
                "start_index": start,
                "max_samples": count,
                "exit_code": result.returncode,
                "stdout": _project_rel(stdout_path),
                "stderr": _project_rel(stderr_path),
            }
            failures.append(failure)
            _append_status(status_path, failure)
            break
        _append_status(
            status_path,
            {
                "event": "shard_done",
                "shard": shard_name,
                "start_index": start,
                "max_samples": count,
                "bytes": shard_path.stat().st_size if shard_path.exists() else 0,
                "stdout": _project_rel(stdout_path),
                "stderr": _project_rel(stderr_path),
            },
        )
        shard_index += 1

    merged_output = ""
    merge_exit_code: int | None = None
    if not failures and args.merged_output:
        merged_path = _project_path(args.merged_output)
        _append_status(status_path, {"event": "merge_start", "output": _project_rel(merged_path), "shards": len(shards)})
        merge_exit_code = _merge_shards(shards, merged_path)
        if merge_exit_code == 0:
            merged_output = _project_rel(merged_path)
            _append_status(status_path, {"event": "merge_done", "output": merged_output})
        else:
            failures.append({"event": "merge_failed", "exit_code": merge_exit_code, "output": _project_rel(merged_path)})
            _append_status(status_path, failures[-1])

    summary = {
        "schema": "cueqc_feature_shards_run_v1",
        "input": _project_rel(input_path),
        "output_dir": _project_rel(output_dir),
        "status_path": _project_rel(status_path),
        "total_rows": total_rows,
        "start_index": args.start_index,
        "planned_rows": planned_rows,
        "shard_size": args.shard_size,
        "shards": [_project_rel(path) for path in shards],
        "failures": failures,
        "merged_output": merged_output,
        "merge_exit_code": merge_exit_code,
    }
    _write_json(summary_path, summary)
    _append_status(status_path, {"event": "done" if not failures else "failed", "summary": _project_rel(summary_path)})
    print(f"summary={_project_rel(summary_path)}")
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CueQC v4 binary features in resumable shards.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train", default="", help="Labeled cueqc_train.jsonl.")
    source.add_argument("--input", default="", help="Unlabeled cueqc_candidate_v4 JSONL.")
    parser.add_argument("--audio-root", required=True, help="Root containing per-video baseline wav artifacts.")
    parser.add_argument("--output-dir", default="", help="Defaults to agents/temp/YYYYMMDD_HHMMSS_cueqc-feature-shards.")
    parser.add_argument("--merged-output", default="", help="Optional merged .pt output after all shards succeed.")
    parser.add_argument("--model-spec", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--audio-cache-size", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Recreate existing non-empty shard files.")
    args = parser.parse_args(argv)
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.shard_size <= 0:
        parser.error("--shard-size must be positive")
    if args.audio_cache_size <= 0:
        parser.error("--audio-cache-size must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
