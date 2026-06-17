from __future__ import annotations

import argparse
import time
from pathlib import Path

from tools.web.smoke.common import TERMINAL_STATUSES, compact_progress, default_run_dir, http_json, write_json


def _resolve_job_id(args: argparse.Namespace) -> str:
    if args.job_id:
        return args.job_id.strip()
    if args.job_id_file:
        return Path(args.job_id_file).read_text(encoding="utf-8").strip()
    raise SystemExit("provide --job-id or --job-id-file")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Poll a Web API job and persist latest state.")
    parser.add_argument("--base-url", default="http://127.0.0.1:17321")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--job-id-file", default="")
    parser.add_argument("--run-dir", default="", help="Defaults to agents/temp/YYYYMMDD_HHMMSS_web-smoke-poll")
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0, help="0 means no timeout.")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    job_id = _resolve_job_id(args)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir("web-smoke-poll")
    run_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout_seconds if args.timeout_seconds > 0 else None

    while True:
        state = http_json("GET", args.base_url.rstrip("/") + f"/api/jobs/{job_id}")
        write_json(run_dir / "latest_job_state.json", state)
        print(compact_progress(state), flush=True)
        status = str(state.get("status") or "")
        if args.once or status in TERMINAL_STATUSES:
            return 0 if status != "failed" else 2
        if deadline is not None and time.monotonic() >= deadline:
            return 124
        sleep_s = args.interval_seconds
        if deadline is not None:
            sleep_s = min(sleep_s, max(0.0, deadline - time.monotonic()))
        time.sleep(sleep_s)


if __name__ == "__main__":
    raise SystemExit(main())
