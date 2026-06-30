from __future__ import annotations

import argparse
from pathlib import Path

from tools.web.smoke.common import default_run_dir, http_json, write_json


DEFAULT_ASR_BACKEND = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"


def _parse_advanced(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--advanced must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--advanced key is empty: {item!r}")
        parsed[key] = value
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit a Web API smoke job.")
    parser.add_argument("--base-url", default="http://127.0.0.1:17321")
    parser.add_argument("--video-path", action="append", required=True, help="Video path; repeat for batch jobs.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-dir", default="", help="Defaults to agents/temp/YYYYMMDD_HHMMSS_web-smoke-submit")
    parser.add_argument("--asr-backend", default=DEFAULT_ASR_BACKEND)
    parser.add_argument("--target-lang", default="")
    parser.add_argument("--subtitle-mode", choices=["zh", "bilingual"], default="zh")
    parser.add_argument("--translate", action="store_true", help="Run translation. Default is no-translation smoke.")
    parser.add_argument("--keep-quality-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-temp-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--speech-boundary-scorer-checkpoint-by-repo",
        default="",
        help="Optional repo-id scorer map. Empty uses the registered repo-id scorer when available.",
    )
    parser.add_argument("--advanced", action="append", default=[], help="Extra advanced env override, KEY=VALUE.")
    args = parser.parse_args(argv)

    advanced = _parse_advanced(args.advanced)
    if args.speech_boundary_scorer_checkpoint_by_repo:
        advanced.setdefault(
            "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO",
            args.speech_boundary_scorer_checkpoint_by_repo,
        )

    payload = {
        "video_paths": args.video_path,
        "asr_backend": args.asr_backend,
        "subtitle_mode": args.subtitle_mode,
        "skip_translation": not args.translate,
        "keep_quality_report": bool(args.keep_quality_report),
        "keep_temp_files": bool(args.keep_temp_files),
        "advanced": advanced,
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
