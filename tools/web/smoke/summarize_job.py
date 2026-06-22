from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from tools.web.smoke.common import read_json, write_json


def _counter_add(counter: Counter[str], value: Any) -> None:
    key = str(value or "").strip() or "unknown"
    counter[key] += 1


def _cueqc_from_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    modes: Counter[str] = Counter()
    display: Counter[str] = Counter()
    fallback_stage: Counter[str] = Counter()
    fallback_reason: Counter[str] = Counter()
    fallback_detail: Counter[str] = Counter()
    drop_count = 0
    model_count = 0
    for chunk in chunks:
        decision = chunk.get("cueqc_shadow") if isinstance(chunk.get("cueqc_shadow"), dict) else {}
        mode = str(decision.get("mode") or "")
        _counter_add(modes, mode)
        _counter_add(display, decision.get("display_hint"))
        if mode == "cueqc_mamba_v4_binary":
            model_count += 1
        if mode == "cueqc_mamba_v4_binary" and decision.get("display_hint") == "drop":
            drop_count += 1
        if mode != "cueqc_mamba_v4_binary":
            _counter_add(fallback_stage, decision.get("fallback_stage"))
            reasons = decision.get("reasons") if isinstance(decision.get("reasons"), list) else []
            _counter_add(fallback_reason, reasons[0] if reasons else "")
            if decision.get("fallback_detail"):
                _counter_add(fallback_detail, decision.get("fallback_detail"))
    return {
        "source": "transcript_chunks",
        "candidate_count": len(chunks),
        "transcript_chunks": len(chunks),
        "model_count": model_count,
        "drop_count": drop_count,
        "fallback_count": sum(fallback_stage.values()),
        "mode": dict(modes),
        "display_hint": dict(display),
        "fallback_stage": dict(fallback_stage),
        "fallback_reason": dict(fallback_reason),
        "fallback_detail_top": dict(fallback_detail.most_common(5)),
    }


def _cueqc_from_decisions(
    decisions: list[dict[str, Any]],
    *,
    report: dict[str, Any],
    transcript_chunks: list[dict[str, Any]],
    stage_timings: dict[str, Any],
) -> dict[str, Any]:
    modes: Counter[str] = Counter()
    display: Counter[str] = Counter()
    fallback_stage: Counter[str] = Counter()
    fallback_reason: Counter[str] = Counter()
    fallback_detail: Counter[str] = Counter()
    drop_count = 0
    model_count = 0
    fallback_count = 0
    for decision in decisions:
        mode = str(decision.get("mode") or "")
        _counter_add(modes, mode)
        _counter_add(display, decision.get("display_hint"))
        if mode == "cueqc_mamba_v4_binary":
            model_count += 1
        else:
            fallback_count += 1
            _counter_add(fallback_stage, decision.get("fallback_stage"))
            reasons = decision.get("reasons") if isinstance(decision.get("reasons"), list) else []
            _counter_add(fallback_reason, reasons[0] if reasons else "")
            if decision.get("fallback_detail"):
                _counter_add(fallback_detail, decision.get("fallback_detail"))
        if decision.get("display_hint") == "drop":
            drop_count += 1

    report_counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    fallback_summary = report.get("fallback_summary") if isinstance(report.get("fallback_summary"), dict) else {}
    return {
        "source": "cueqc_shadow.decisions",
        "candidate_count": int(report.get("candidate_count") or len(decisions)),
        "transcript_chunks": len(transcript_chunks),
        "model_count": model_count,
        "drop_count": drop_count,
        "fallback_count": fallback_count,
        "stage_cueqc_drop_count": stage_timings.get("cueqc_drop_count"),
        "mode": dict(modes),
        "display_hint": dict(display),
        "fallback_stage": dict(fallback_stage),
        "fallback_reason": dict(fallback_reason),
        "fallback_detail_top": dict(fallback_detail.most_common(5)),
        "report_counts": report_counts,
        "report_fallback_summary": fallback_summary,
    }


def _cueqc_summary(asr_details: dict[str, Any]) -> dict[str, Any]:
    chunks = asr_details.get("transcript_chunks") if isinstance(asr_details.get("transcript_chunks"), list) else []
    stage_timings = asr_details.get("stage_timings") if isinstance(asr_details.get("stage_timings"), dict) else {}
    report = asr_details.get("cueqc_shadow") if isinstance(asr_details.get("cueqc_shadow"), dict) else {}
    decisions = report.get("decisions") if isinstance(report.get("decisions"), list) else []
    if decisions:
        return _cueqc_from_decisions(
            decisions,
            report=report,
            transcript_chunks=chunks,
            stage_timings=stage_timings,
        )
    return _cueqc_from_chunks(chunks)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Web smoke artifacts and CueQC runtime decisions.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-dir", default="")
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args(argv)

    job_dir = _find_job_dir(args.job_id, args.job_dir)
    run_dir = Path(args.run_dir) if args.run_dir else job_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    timings_path, timings = _load_timings(job_dir)
    asr_details = timings.get("asr_details") if isinstance(timings.get("asr_details"), dict) else {}
    cueqc_summary = _cueqc_summary(asr_details)
    summary = {
        "job_id": args.job_id,
        "job_dir": str(job_dir),
        "timings_path": str(timings_path) if timings_path else "",
        "status": "artifacts_found" if timings_path else "timings_missing",
        "counts": timings.get("counts", {}),
        "stage_timings": timings.get("stage_timings", {}),
        "cueqc": cueqc_summary,
    }
    write_json(run_dir / "job_summary.json", summary)
    markdown = [
        f"# Web smoke summary: {args.job_id}",
        "",
        f"- status: {summary['status']}",
        f"- job_dir: `{job_dir}`",
        f"- timings: `{timings_path}`" if timings_path else "- timings: missing",
        f"- counts: `{summary['counts']}`",
        f"- CueQC source: `{cueqc_summary['source']}`",
        f"- CueQC candidates/transcript_chunks: `{cueqc_summary['candidate_count']}/{cueqc_summary['transcript_chunks']}`",
        f"- CueQC mode: `{cueqc_summary['mode']}`",
        f"- CueQC display: `{cueqc_summary['display_hint']}`",
        f"- CueQC model/drop/fallback: `{cueqc_summary['model_count']}/{cueqc_summary['drop_count']}/{cueqc_summary['fallback_count']}`",
        f"- CueQC stage_drop_count: `{cueqc_summary.get('stage_cueqc_drop_count', '')}`",
        f"- CueQC fallback_stage: `{cueqc_summary['fallback_stage']}`",
        f"- CueQC fallback_reason: `{cueqc_summary['fallback_reason']}`",
        f"- CueQC fallback_detail_top: `{cueqc_summary['fallback_detail_top']}`",
        "",
    ]
    (run_dir / "job_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))
    return 0 if timings_path else 2


if __name__ == "__main__":
    raise SystemExit(main())
