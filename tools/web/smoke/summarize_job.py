from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.web.smoke.common import read_json, write_json


def _find_job_dir(job_id: str, explicit: str) -> Path:
    if explicit:
        return Path(explicit)
    return Path("tmp") / "web" / "jobs" / job_id


def _load_timings(job_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    matches = sorted(job_dir.glob("*.timings.json"))
    if not matches:
        return None, {}
    path = matches[0]
    return path, read_json(path)


def _postgate_summary(asr_details: dict[str, Any]) -> dict[str, Any]:
    """What the post-gate marked, which is the only gate left in the run.

    This used to read `asr_details.pre_asr_cueqc`, a key nothing has written
    since the pre-ASR chain was retired on 2026-07-31 - so the smoke summary
    reported blanks and looked like a job that simply had no drops.
    """
    report = (
        asr_details.get("postgate")
        if isinstance(asr_details.get("postgate"), dict)
        else {}
    )
    chunks = (
        asr_details.get("transcript_chunks")
        if isinstance(asr_details.get("transcript_chunks"), list)
        else []
    )
    return {
        "source": "asr_details.postgate",
        "schema": report.get("schema"),
        "reviewed": report.get("reviewed"),
        "flagged": report.get("flagged"),
        "flags": report.get("flags") or {},
        "transcript_chunks": len(chunks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Web smoke artifacts and post-gate findings."
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-dir", default="")
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args(argv)

    job_dir = _find_job_dir(args.job_id, args.job_dir)
    run_dir = Path(args.run_dir) if args.run_dir else job_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    timings_path, timings = _load_timings(job_dir)
    asr_details = timings.get("asr_details") if isinstance(timings.get("asr_details"), dict) else {}
    postgate_summary = _postgate_summary(asr_details)
    summary = {
        "job_id": args.job_id,
        "job_dir": str(job_dir),
        "timings_path": str(timings_path) if timings_path else "",
        "status": "artifacts_found" if timings_path else "timings_missing",
        "counts": timings.get("counts", {}),
        "stage_timings": timings.get("stage_timings", {}),
        "postgate": postgate_summary,
    }
    write_json(run_dir / "job_summary.json", summary)
    markdown = [
        f"# Web smoke summary: {args.job_id}",
        "",
        f"- status: {summary['status']}",
        f"- job_dir: `{job_dir}`",
        f"- timings: `{timings_path}`" if timings_path else "- timings: missing",
        f"- counts: `{summary['counts']}`",
        f"- post-gate reviewed/flagged: "
        f"`{postgate_summary.get('reviewed', '')}/"
        f"{postgate_summary.get('flagged', '')}` (flags are marks, not drops)",
        f"- post-gate flags: `{postgate_summary.get('flags', {})}`",
        f"- transcript_chunks: `{postgate_summary['transcript_chunks']}`",
        "",
    ]
    (run_dir / "job_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))
    return 0 if timings_path else 2


if __name__ == "__main__":
    raise SystemExit(main())
