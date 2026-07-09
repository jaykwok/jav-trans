#!/usr/bin/env python3
"""Generate an audio audit page for Split operating-point candidates."""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SRC_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from tools.datasets.analyze_split_adaptive_operating_point import (  # noqa: E402
    _selected_adaptive_cuts,
)
from tools.datasets.analyze_split_threshold_sensitivity import _selected_cuts  # noqa: E402

SUMMARY_SCHEMA = "split_operating_point_audit_summary_v1"
MANIFEST_SCHEMA = "split_operating_point_audit_item_v1"


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


def _source_windows(reexport_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("window_id") or row.get("audio_id") or "").strip(): row
        for row in read_jsonl(reexport_dir / "source_windows.jsonl")
    }


def _feature_dirs(reexport_dir: Path) -> list[Path]:
    return sorted(path for path in (reexport_dir / "features").iterdir() if path.is_dir())


def _cuts_inside_chunk(
    cuts: list[Mapping[str, Any]],
    *,
    start: float,
    end: float,
    min_chunk_after_split_s: float,
) -> list[dict[str, Any]]:
    return [
        dict(cut)
        for cut in cuts
        if start + min_chunk_after_split_s <= _safe_float(cut.get("time_s")) <= end - min_chunk_after_split_s
    ]


def _policy_from_summary(path: Path, policy_index: int) -> dict[str, float]:
    summary = read_json(path)
    rows = summary.get("rows") if isinstance(summary, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"adaptive summary has no rows: {path}")
    index = max(0, min(policy_index, len(rows) - 1))
    policy = rows[index].get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError(f"adaptive summary row has no policy: {path}")
    return {
        "abs_floor": float(policy["abs_floor"]),
        "percentile_floor": float(policy["percentile_floor"]),
        "z_floor": float(policy["z_floor"]),
    }


def collect_items(
    *,
    reexport_dir: Path,
    fixed_threshold: float,
    adaptive_policy: Mapping[str, float],
    long_chunk_min_s: float,
    min_chunk_after_split_s: float,
    limit: int,
) -> list[dict[str, Any]]:
    windows = _source_windows(reexport_dir)
    items: list[dict[str, Any]] = []
    for feature_dir in _feature_dirs(reexport_dir):
        candidate_path = feature_dir / "pre_asr_candidates.jsonl"
        split_path = feature_dir / "semantic_split_features.jsonl"
        if not candidate_path.exists() or not split_path.exists():
            continue
        candidates = read_jsonl(candidate_path)
        split_rows = read_jsonl(split_path)
        fixed_cuts = [
            cut
            for cut in _selected_cuts(
                split_rows,
                threshold=fixed_threshold,
                min_chunk_after_split_s=min_chunk_after_split_s,
            )
            if not bool(cut.get("accepted_current"))
        ]
        adaptive_cuts = [
            cut
            for cut in _selected_adaptive_cuts(
                split_rows,
                abs_floor=float(adaptive_policy["abs_floor"]),
                percentile_floor=float(adaptive_policy["percentile_floor"]),
                z_floor=float(adaptive_policy["z_floor"]),
                min_chunk_after_split_s=min_chunk_after_split_s,
            )
            if not bool(cut.get("accepted_current"))
        ]
        for candidate in candidates:
            start = _safe_float(candidate.get("start"))
            end = _safe_float(candidate.get("end"), start)
            duration = _safe_float(candidate.get("duration_s"), end - start)
            if duration < long_chunk_min_s:
                continue
            fixed_inside = _cuts_inside_chunk(
                fixed_cuts,
                start=start,
                end=end,
                min_chunk_after_split_s=min_chunk_after_split_s,
            )
            adaptive_inside = _cuts_inside_chunk(
                adaptive_cuts,
                start=start,
                end=end,
                min_chunk_after_split_s=min_chunk_after_split_s,
            )
            if not fixed_inside and not adaptive_inside:
                continue
            candidate_id = str(candidate.get("candidate_id") or candidate.get("sample_id") or "")
            window_id = str(candidate.get("audio_id") or candidate.get("window_id") or candidate.get("video_id") or "")
            proposed = sorted(
                {round(_safe_float(cut.get("time_s")), 6): cut for cut in [*fixed_inside, *adaptive_inside]}.values(),
                key=lambda cut: (-_safe_float(cut.get("p_cut")), _safe_float(cut.get("time_s"))),
            )
            best_cut = proposed[0]
            source = windows.get(window_id, {})
            fixed_times = {round(_safe_float(cut.get("time_s")), 6) for cut in fixed_inside}
            adaptive_times = {round(_safe_float(cut.get("time_s")), 6) for cut in adaptive_inside}
            items.append(
                {
                    "schema": MANIFEST_SCHEMA,
                    "candidate_id": candidate_id,
                    "window_id": window_id,
                    "chunk_index": int(_safe_float(candidate.get("chunk_index", candidate.get("index")), 0.0)),
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "duration_s": round(duration, 6),
                    "best_cut_time_s": round(_safe_float(best_cut.get("time_s")), 6),
                    "best_cut_p_cut": round(_safe_float(best_cut.get("p_cut")), 6),
                    "best_cut_fixed": round(_safe_float(best_cut.get("time_s")), 6) in fixed_times,
                    "best_cut_adaptive": round(_safe_float(best_cut.get("time_s")), 6) in adaptive_times,
                    "fixed_cut_times_s": sorted(fixed_times),
                    "adaptive_cut_times_s": sorted(adaptive_times),
                    "fixed_internal_cut_count": len(fixed_inside),
                    "adaptive_internal_cut_count": len(adaptive_inside),
                    "source_video": str(source.get("source_video") or ""),
                    "source_start_s": _safe_float(source.get("source_start_s")),
                    "source_video_time_start_s": round(_safe_float(source.get("source_start_s")) + start, 6),
                    "source_video_time_end_s": round(_safe_float(source.get("source_start_s")) + end, 6),
                    "audio_wav": str(source.get("audio_wav") or ""),
                }
            )
    items.sort(
        key=lambda row: (
            -int(bool(row["best_cut_fixed"]) and bool(row["best_cut_adaptive"])),
            -_safe_float(row["duration_s"]),
            str(row["candidate_id"]),
        )
    )
    return items[: max(0, limit)]


def _slice_audio(
    *,
    source_audio: Path,
    output_path: Path,
    start_s: float,
    end_s: float,
    cut_audio: bool,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cut_audio:
        return str(output_path)
    duration = max(0.05, end_s - start_s)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, start_s):.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(source_audio),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return str(output_path)


def _attach_audio(items: list[dict[str, Any]], *, output_dir: Path, cut_audio: bool) -> None:
    audio_dir = output_dir / "audio"
    for index, row in enumerate(items, start=1):
        source_audio = Path(str(row.get("audio_wav") or ""))
        if not source_audio.is_absolute():
            source_audio = (PROJECT_ROOT / source_audio).resolve()
        start = _safe_float(row.get("start"))
        end = _safe_float(row.get("end"), start)
        cut = _safe_float(row.get("best_cut_time_s"))
        prefix = f"{index:03d}_{_safe_name(str(row['candidate_id']))}"
        clips = {
            "full": (start, end),
            "before_cut": (max(start, cut - 6.0), cut),
            "after_cut": (cut, min(end, cut + 6.0)),
            "cut_context": (max(start, cut - 4.0), min(end, cut + 4.0)),
        }
        row["clips"] = {}
        for name, (clip_start, clip_end) in clips.items():
            path = audio_dir / f"{prefix}_{name}.wav"
            _slice_audio(
                source_audio=source_audio,
                output_path=path,
                start_s=clip_start,
                end_s=clip_end,
                cut_audio=cut_audio,
            )
            row["clips"][name] = path.relative_to(output_dir).as_posix()


def _audio_tag(path: str) -> str:
    return f'<audio controls preload="none" src="{html.escape(Path(path).name if "/" not in path else path)}"></audio>'


def _html_page(*, rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> str:
    cards = []
    for row in rows:
        clips = row.get("clips") if isinstance(row.get("clips"), Mapping) else {}
        rel_clips = {key: str(value).replace("\\", "/") for key, value in clips.items()}
        badges = []
        if row.get("best_cut_fixed"):
            badges.append("fixed 0.70")
        if row.get("best_cut_adaptive"):
            badges.append("adaptive")
        cards.append(
            f"""
      <article class="card">
        <h2>{html.escape(str(row["candidate_id"]))}</h2>
        <div class="meta">
          <span>{html.escape(str(row["window_id"]))}</span>
          <span>chunk {int(row["chunk_index"])}</span>
          <span>{float(row["duration_s"]):.2f}s</span>
          <span>cut {float(row["best_cut_time_s"]):.2f}s / p={float(row["best_cut_p_cut"]):.3f}</span>
          <span>{html.escape(" + ".join(badges))}</span>
        </div>
        <div class="grid">
          <section><h3>full</h3>{_audio_tag(rel_clips.get("full", ""))}</section>
          <section><h3>cut context</h3>{_audio_tag(rel_clips.get("cut_context", ""))}</section>
          <section><h3>before</h3>{_audio_tag(rel_clips.get("before_cut", ""))}</section>
          <section><h3>after</h3>{_audio_tag(rel_clips.get("after_cut", ""))}</section>
        </div>
        <pre>{html.escape(json.dumps({k: row[k] for k in ("fixed_cut_times_s", "adaptive_cut_times_s", "source_video_time_start_s", "source_video_time_end_s")}, ensure_ascii=False, indent=2))}</pre>
      </article>"""
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Split Operating Point Audit</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; background: #f7f7f4; color: #1f2328; }}
    header {{ max-width: 1120px; margin: 0 auto 20px; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 8px; font-size: 13px; }}
    .summary span, .meta span {{ background: #fff; border: 1px solid #d8d8d0; border-radius: 6px; padding: 4px 8px; }}
    .card {{ max-width: 1120px; margin: 14px auto; background: #fff; border: 1px solid #d8d8d0; border-radius: 8px; padding: 14px; }}
    h2 {{ font-size: 16px; margin: 0 0 8px; }}
    h3 {{ font-size: 13px; margin: 0 0 6px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    audio {{ width: 100%; }}
    pre {{ white-space: pre-wrap; background: #f4f4ef; padding: 8px; border-radius: 6px; font-size: 12px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  </style>
</head>
<body>
  <header>
    <h1>Split Operating Point Audit</h1>
    <div class="summary">
      <span>items {int(summary["item_count"])}</span>
      <span>fixed threshold {float(summary["fixed_threshold"]):.2f}</span>
      <span>adaptive {html.escape(json.dumps(summary["adaptive_policy"], ensure_ascii=False, sort_keys=True))}</span>
      <span>long chunk >= {float(summary["long_chunk_min_s"]):.1f}s</span>
    </div>
  </header>
  {''.join(cards)}
</body>
</html>
"""


def build_audit(
    *,
    reexport_dir: Path,
    adaptive_summary: Path,
    output_dir: Path,
    fixed_threshold: float,
    adaptive_policy_index: int,
    long_chunk_min_s: float,
    min_chunk_after_split_s: float,
    limit: int,
    cut_audio: bool,
) -> dict[str, Any]:
    policy = _policy_from_summary(adaptive_summary, adaptive_policy_index)
    rows = collect_items(
        reexport_dir=reexport_dir,
        fixed_threshold=fixed_threshold,
        adaptive_policy=policy,
        long_chunk_min_s=long_chunk_min_s,
        min_chunk_after_split_s=min_chunk_after_split_s,
        limit=limit,
    )
    _attach_audio(rows, output_dir=output_dir, cut_audio=cut_audio)
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.html"
    write_jsonl(manifest_path, rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "output_dir": str(output_dir.resolve()),
        "reexport_dir": repo_rel(reexport_dir),
        "adaptive_summary": repo_rel(adaptive_summary),
        "fixed_threshold": float(fixed_threshold),
        "adaptive_policy": policy,
        "adaptive_policy_index": int(adaptive_policy_index),
        "long_chunk_min_s": float(long_chunk_min_s),
        "min_chunk_after_split_s": float(min_chunk_after_split_s),
        "item_count": len(rows),
        "manifest": repo_rel(manifest_path),
        "index_html": repo_rel(index_path),
    }
    write_json(summary_path, summary)
    index_path.write_text(_html_page(rows=rows, summary=summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reexport-dir", required=True)
    parser.add_argument("--adaptive-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fixed-threshold", type=float, default=0.70)
    parser.add_argument("--adaptive-policy-index", type=int, default=4)
    parser.add_argument("--long-chunk-min-s", type=float, default=15.0)
    parser.add_argument("--min-chunk-after-split-s", type=float, default=1.2)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--no-cut-audio", action="store_true")
    args = parser.parse_args(argv)
    summary = build_audit(
        reexport_dir=project_path(args.reexport_dir),
        adaptive_summary=project_path(args.adaptive_summary),
        output_dir=project_path(args.output_dir),
        fixed_threshold=float(args.fixed_threshold),
        adaptive_policy_index=int(args.adaptive_policy_index),
        long_chunk_min_s=float(args.long_chunk_min_s),
        min_chunk_after_split_s=float(args.min_chunk_after_split_s),
        limit=int(args.limit),
        cut_audio=not bool(args.no_cut_audio),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
