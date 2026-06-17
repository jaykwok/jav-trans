from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.web.smoke.common import default_run_dir, is_port_open, wait_for_port, write_json


def _parse_env(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--env must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--env key is empty: {item!r}")
        parsed[key] = value
    return parsed


def _pythonpath_with_src(env: dict[str, str]) -> str:
    src = str(Path("src").resolve())
    current = env.get("PYTHONPATH", "")
    if not current:
        return src
    parts = current.split(os.pathsep)
    if src in parts:
        return current
    return src + os.pathsep + current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the JAVTrans Web server for smoke tests.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17321)
    parser.add_argument("--events-port", default="17322")
    parser.add_argument("--run-dir", default="", help="Defaults to agents/temp/YYYYMMDD_HHMMSS_web-smoke-server")
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--allow-existing", action="store_true", help="Return success if the port is already open.")
    parser.add_argument("--env", action="append", default=[], help="Extra environment override, KEY=VALUE.")
    args = parser.parse_args(argv)

    if is_port_open(args.host, args.port):
        if args.allow_existing:
            print(f"server already listening on {args.host}:{args.port}")
            return 0
        raise SystemExit(f"port already in use: {args.host}:{args.port}")

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir("web-smoke-server")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "uvicorn.out.log"
    err_path = run_dir / "uvicorn.err.log"

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = _pythonpath_with_src(env)
    env["JAVTRANS_EVENTS_PORT"] = str(args.events_port)
    env.update(_parse_env(args.env))

    command = [
        "uv",
        "run",
        "python",
        "-m",
        "uvicorn",
        "web.app:create_app",
        "--factory",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    with out_path.open("w", encoding="utf-8") as stdout, err_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )

    (run_dir / "uvicorn.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
    write_json(
        run_dir / "server.json",
        {
            "pid": process.pid,
            "host": args.host,
            "port": args.port,
            "events_port": args.events_port,
            "command": command,
            "run_dir": str(run_dir),
            "stdout": str(out_path),
            "stderr": str(err_path),
        },
    )
    if not wait_for_port(args.host, args.port, timeout_s=args.startup_timeout):
        raise SystemExit(
            "server did not start before timeout; see "
            + json.dumps({"stdout": str(out_path), "stderr": str(err_path)}, ensure_ascii=False)
        )
    print(f"started pid={process.pid} url=http://{args.host}:{args.port} run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
