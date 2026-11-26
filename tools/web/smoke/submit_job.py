from __future__ import annotations

import argparse
from pathlib import Path

from tools.web.smoke.common import default_run_dir, http_json, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit a Web API smoke job.")
    parser.add_argument("--base-url", default="http://127.0.0.1:2233")
    parser.add_argument("--video-path", action="append", required=True, help="Video path; repeat for batch jobs.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-dir", default="", help="Defaults to dev/temp/YYYYMMDD_HHMMSS_web-smoke-submit")
    parser.add_argument("--target-lang", default="")
    parser.add_argument("--subtitle-mode", choices=["zh", "bilingual"], default="zh")
    parser.add_argument("--translate", action="store_true", help="Run translation. Default is no-translation smoke.")
    args = parser.parse_args(argv)

    payload = {
        "video_paths": args.video_path,
        "subtitle_mode": args.subtitle_mode,
        "skip_translation": not args.translate,
    }
    if args.output_dir:
        payload["output_dir"] = args.output_dir
    if args.target_lang:
        payload["target_lang"] = args.target_lang

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir("web-smoke-submit")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "submit_payload.json", payload)

    response = http_json("POST", args.base_url.rstrip("/") + "/api/jobs", payload)
    write_json(run_dir / "submit_response.json", response)
    ids = response.get("ids") if isinstance(response, dict) else []
    if ids:
        (run_dir / "job_id.txt").write_text(str(ids[0]) + "\n", encoding="utf-8")
    print(f"submitted ids={ids} run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
