#!/usr/bin/env python3
"""Compare two Scorer v11 teacher preaudits on the same frozen sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


SUMMARY_SCHEMA = "candidate_island_preaudit_comparison_summary_v1"
DETAIL_SCHEMA = "candidate_island_preaudit_comparison_item_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _index(path: Path, *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _rows(path):
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in result:
            raise ValueError(f"{name} requires unique source_id")
        result[source_id] = row
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_bounds(span: dict[str, Any]) -> tuple[int, int]:
    start = (
        int(span["start_frame"])
        if "start_frame" in span
        else round(float(span["start_s"]) / 0.02)
    )
    end = (
        int(span["end_frame"])
        if "end_frame" in span
        else round(float(span["end_s"]) / 0.02)
    )
    return start, end


def _labels(row: dict[str, Any], *, frame_count: int) -> list[str]:
    result = ["outside_candidate"] * frame_count
    for label, field in (
        ("inside_candidate", "islands"),
        ("unsure", "unsure_spans"),
    ):
        for span in row.get(field) or ():
            raw_start, raw_end = _frame_bounds(span)
            if not -1 <= raw_start < raw_end <= frame_count + 1:
                raise ValueError(
                    f"invalid {label} span for {row.get('source_id')}: "
                    f"{raw_start}..{raw_end}"
                )
            start = max(0, min(frame_count, raw_start))
            end = max(start, min(frame_count, raw_end))
            if end <= start:
                raise ValueError(f"empty clamped {label} span for {row.get('source_id')}")
            if any(value != "outside_candidate" for value in result[start:end]):
                raise ValueError(f"overlapping teacher spans for {row.get('source_id')}")
            result[start:end] = [label] * (end - start)
    return result


def _spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label, field in (
        ("inside_candidate", "islands"),
        ("unsure", "unsure_spans"),
    ):
        for span in row.get(field) or ():
            start, end = _frame_bounds(span)
            result.append(
                {
                    "label": label,
                    "start_s": float(span["start_s"]) if "start_s" in span else start * 0.02,
                    "end_s": float(span["end_s"]) if "end_s" in span else end * 0.02,
                }
            )
    return sorted(result, key=lambda span: (span["start_s"], span["end_s"], span["label"]))


def _audio_url(value: str, *, manifest: Path) -> str:
    raw = Path(value)
    audio = raw if raw.is_absolute() else manifest.parent / raw
    audio = audio.resolve()
    if not audio.is_file():
        raise FileNotFoundError(audio)
    return "/" + audio.relative_to(PROJECT_ROOT.resolve()).as_posix()


def _page(payload: list[dict[str, Any]], *, base_name: str, candidate_name: str) -> str:
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scorer v11 teacher prompt A/B</title><style>
body{{margin:0;background:#f4f7fa;color:#18212b;font-family:Segoe UI,Microsoft YaHei,sans-serif}}header{{position:sticky;top:0;z-index:3;display:flex;gap:10px;align-items:center;background:#122233;color:#fff;padding:12px 18px}}header #status{{margin-left:auto}}main{{max-width:1400px;margin:auto;padding:16px}}article{{background:#fff;border:1px solid #ccd6df;border-radius:10px;padding:14px;margin-bottom:14px}}audio{{width:100%}}.lane{{display:grid;grid-template-columns:130px 1fr;gap:8px;align-items:center;margin:8px 0}}.track{{position:relative;height:38px;background:#e6eaee;border-radius:5px;overflow:hidden}}.span{{position:absolute;top:0;height:100%;border:0;min-width:2px;cursor:pointer;color:#fff;font-size:10px;overflow:hidden}}.base-inside{{background:#3078bf}}.base-unsure{{background:#a56a00}}.candidate-inside{{background:#258b57}}.candidate-unsure{{background:#d87800}}.changed{{background:#c53a3a}}button.playing{{outline:3px solid #111;outline-offset:-3px}}.metrics{{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#40566c}}.positive{{color:#087443}}.negative{{color:#b3261e}}small{{color:#607080}}</style></head><body><header><b>Scorer v11 · Prompt A/B（同一冻结 {len(payload)} source）</b><button id="stop" type="button">停止播放</button><span id="status"></span></header><main><section><p>蓝色/棕色={base_name} inside/unsure；绿色/橙色={candidate_name} inside/unsure；红色=两版逐帧标签不同。灰色差集为 outside_candidate。点击色块只播放该区间，完整播放器用于听上下文。</p></section><div id="list"></div></main><script>
const rows={encoded};let activeAudio=null,activeButton=null,stopFn=null;function esc(v){{return String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function stop(){{if(activeAudio&&stopFn)activeAudio.removeEventListener('timeupdate',stopFn);if(activeAudio)activeAudio.pause();if(activeButton)activeButton.classList.remove('playing');activeAudio=activeButton=stopFn=null;}}function play(audio,button,start,end){{stop();activeAudio=audio;activeButton=button;button.classList.add('playing');const begin=()=>{{audio.currentTime=start;stopFn=()=>{{if(audio.currentTime>=end)stop();}};audio.addEventListener('timeupdate',stopFn);audio.play();}};if(audio.readyState<1){{audio.addEventListener('loadedmetadata',begin,{{once:true}});audio.load();}}else begin();}}function lane(card,audio,row,name,spans,kind){{const line=document.createElement('div');line.className='lane';line.innerHTML=`<b>${{esc(name)}}</b><div class="track"></div>`;const track=line.querySelector('.track');for(const span of spans){{const button=document.createElement('button'),start=Number(span.start_s),end=Number(span.end_s);button.className=`span ${{kind}}-${{span.label==='unsure'?'unsure':'inside'}}`;if(kind==='diff')button.className='span changed';button.style.left=`${{100*start/row.duration_s}}%`;button.style.width=`${{Math.max(.12,100*(end-start)/row.duration_s)}}%`;button.title=`${{name}} ${{span.label}} ${{start.toFixed(2)}}–${{end.toFixed(2)}}s`;button.onclick=()=>play(audio,button,start,end);track.appendChild(button);}}card.appendChild(line);}}const root=document.getElementById('list');for(const row of rows){{const card=document.createElement('article'),failure=row.base_failed_closed||row.candidate_failed_closed;card.innerHTML=`<h2>${{esc(row.source_id)}}</h2><small>${{Number(row.duration_s).toFixed(2)}}s${{failure?' / ⚠ teacher validation failed-closed，不作为语义差异':''}}</small><audio controls preload="none" src="${{esc(row.audio)}}"></audio>`;const audio=card.querySelector('audio');lane(card,audio,row,'{base_name}',row.base_spans,'base');lane(card,audio,row,'{candidate_name}',row.candidate_spans,'candidate');lane(card,audio,row,'changed',row.changed_spans,'diff');const m=document.createElement('div');m.className='metrics';const delta=100*row.inside_delta_ratio;m.innerHTML=`<span>inside ${{(100*row.base_inside_ratio).toFixed(1)}}% → ${{(100*row.candidate_inside_ratio).toFixed(1)}}%</span><span class="${{delta>=0?'positive':'negative'}}">Δ=${{delta>=0?'+':''}}${{delta.toFixed(1)}}pp</span><span>unsure ${{(100*row.base_unsure_ratio).toFixed(1)}}% → ${{(100*row.candidate_unsure_ratio).toFixed(1)}}%</span><span>changed=${{(100*row.changed_ratio).toFixed(1)}}%</span>`;card.appendChild(m);if(failure){{const note=document.createElement('p');note.className='negative';note.textContent=`fail-closed: ${{row.base_failed_closed?'base ':''}}${{row.candidate_failed_closed?'candidate ':''}}${{row.candidate_reason||row.base_reason||''}}`;card.appendChild(note);}}root.appendChild(card);}}document.getElementById('stop').onclick=stop;document.getElementById('status').textContent=`${{rows.length}} sources / changed ${{rows.filter(r=>r.changed_frames>0).length}} / failed-closed ${{rows.filter(r=>r.base_failed_closed||r.candidate_failed_closed).length}}`;</script></body></html>"""


def compare(
    *,
    manifest: Path,
    base: Path,
    candidate: Path,
    output_dir: Path,
    base_name: str = "v5",
    candidate_name: str = "v6",
) -> dict[str, Any]:
    manifest = manifest.resolve()
    base = base.resolve()
    candidate = candidate.resolve()
    sources = _index(manifest, name="source manifest")
    base_rows = _index(base, name="base preaudit")
    candidate_rows = _index(candidate, name="candidate preaudit")
    missing_base = set(sources) - set(base_rows)
    missing_candidate = set(sources) - set(candidate_rows)
    if missing_base or missing_candidate:
        raise ValueError(
            "every comparison source must exist in both preaudits: "
            f"missing_base={sorted(missing_base)[:3]}, "
            f"missing_candidate={sorted(missing_candidate)[:3]}"
        )
    details: list[dict[str, Any]] = []
    totals = {
        "frame_count": 0,
        "base_inside_frames": 0,
        "candidate_inside_frames": 0,
        "base_unsure_frames": 0,
        "candidate_unsure_frames": 0,
        "changed_frames": 0,
    }
    for source_id, source in sources.items():
        frame_count = int(source["frame_count"])
        for teacher_name, teacher in ((base_name, base_rows[source_id]), (candidate_name, candidate_rows[source_id])):
            if int(teacher.get("frame_count") or 0) != frame_count:
                raise ValueError(f"{teacher_name} frame geometry mismatch: {source_id}")
            if str(teacher.get("audio_sha256") or "") != str(source.get("audio_sha256") or ""):
                raise ValueError(f"{teacher_name} audio identity mismatch: {source_id}")
        left = _labels(base_rows[source_id], frame_count=frame_count)
        right = _labels(candidate_rows[source_id], frame_count=frame_count)
        changed_values = [a != b for a, b in zip(left, right)]
        changed_spans: list[dict[str, Any]] = []
        start: int | None = None
        for index, changed in enumerate(changed_values + [False]):
            if changed and start is None:
                start = index
            elif not changed and start is not None:
                changed_spans.append({"label": "changed", "start_s": start * 0.02, "end_s": index * 0.02})
                start = None
        counts = {
            "base_inside_frames": left.count("inside_candidate"),
            "candidate_inside_frames": right.count("inside_candidate"),
            "base_unsure_frames": left.count("unsure"),
            "candidate_unsure_frames": right.count("unsure"),
            "changed_frames": sum(changed_values),
        }
        for key, value in counts.items():
            totals[key] += value
        totals["frame_count"] += frame_count
        details.append(
            {
                "schema": DETAIL_SCHEMA,
                "source_id": source_id,
                "duration_s": float(source["duration_s"]),
                "frame_count": frame_count,
                "audio": _audio_url(str(source["audio"]), manifest=manifest),
                "base_prompt_version": str(base_rows[source_id].get("prompt_version") or ""),
                "candidate_prompt_version": str(candidate_rows[source_id].get("prompt_version") or ""),
                "base_failed_closed": bool(base_rows[source_id].get("teacher_failed_closed")),
                "candidate_failed_closed": bool(candidate_rows[source_id].get("teacher_failed_closed")),
                "base_reason": str(base_rows[source_id].get("overall_reason") or ""),
                "candidate_reason": str(candidate_rows[source_id].get("overall_reason") or ""),
                "base_spans": _spans(base_rows[source_id]),
                "candidate_spans": _spans(candidate_rows[source_id]),
                "changed_spans": changed_spans,
                **counts,
                "base_inside_ratio": counts["base_inside_frames"] / max(frame_count, 1),
                "candidate_inside_ratio": counts["candidate_inside_frames"] / max(frame_count, 1),
                "inside_delta_ratio": (counts["candidate_inside_frames"] - counts["base_inside_frames"]) / max(frame_count, 1),
                "base_unsure_ratio": counts["base_unsure_frames"] / max(frame_count, 1),
                "candidate_unsure_ratio": counts["candidate_unsure_frames"] / max(frame_count, 1),
                "changed_ratio": counts["changed_frames"] / max(frame_count, 1),
            }
        )
    details.sort(key=lambda row: (-row["changed_ratio"], row["source_id"]))
    valid_details = [
        row
        for row in details
        if not row["base_failed_closed"] and not row["candidate_failed_closed"]
    ]
    valid_frames = sum(row["frame_count"] for row in valid_details)
    valid_base_inside = sum(row["base_inside_frames"] for row in valid_details)
    valid_candidate_inside = sum(row["candidate_inside_frames"] for row in valid_details)
    valid_base_unsure = sum(row["base_unsure_frames"] for row in valid_details)
    valid_candidate_unsure = sum(row["candidate_unsure_frames"] for row in valid_details)
    valid_changed = sum(row["changed_frames"] for row in valid_details)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "per_source.jsonl"
    detail_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in details),
        encoding="utf-8",
    )
    total_frames = max(totals["frame_count"], 1)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "base": str(base),
        "base_sha256": _sha256(base),
        "base_name": base_name,
        "candidate": str(candidate),
        "candidate_sha256": _sha256(candidate),
        "candidate_name": candidate_name,
        "source_count": len(details),
        "base_extra_source_count": len(set(base_rows) - set(sources)),
        "candidate_extra_source_count": len(set(candidate_rows) - set(sources)),
        "sources_changed": sum(row["changed_frames"] > 0 for row in details),
        "inside_increase_source_count": sum(row["inside_delta_ratio"] > 0 for row in details),
        "inside_decrease_source_count": sum(row["inside_delta_ratio"] < 0 for row in details),
        "inside_unchanged_source_count": sum(row["inside_delta_ratio"] == 0 for row in details),
        "base_failed_closed_count": sum(row["base_failed_closed"] for row in details),
        "candidate_failed_closed_count": sum(row["candidate_failed_closed"] for row in details),
        "base_empty_source_count": sum(row["base_inside_frames"] == 0 for row in details),
        "candidate_empty_source_count": sum(row["candidate_inside_frames"] == 0 for row in details),
        "base_full_source_count": sum(row["base_inside_frames"] == row["frame_count"] for row in details),
        "candidate_full_source_count": sum(row["candidate_inside_frames"] == row["frame_count"] for row in details),
        "valid_source_count": len(valid_details),
        "valid_frame_count": valid_frames,
        "valid_base_inside_ratio": valid_base_inside / max(valid_frames, 1),
        "valid_candidate_inside_ratio": valid_candidate_inside / max(valid_frames, 1),
        "valid_inside_delta_ratio": (valid_candidate_inside - valid_base_inside) / max(valid_frames, 1),
        "valid_base_unsure_ratio": valid_base_unsure / max(valid_frames, 1),
        "valid_candidate_unsure_ratio": valid_candidate_unsure / max(valid_frames, 1),
        "valid_changed_ratio": valid_changed / max(valid_frames, 1),
        **totals,
        "base_inside_ratio": totals["base_inside_frames"] / total_frames,
        "candidate_inside_ratio": totals["candidate_inside_frames"] / total_frames,
        "inside_delta_ratio": (totals["candidate_inside_frames"] - totals["base_inside_frames"]) / total_frames,
        "base_unsure_ratio": totals["base_unsure_frames"] / total_frames,
        "candidate_unsure_ratio": totals["candidate_unsure_frames"] / total_frames,
        "changed_ratio": totals["changed_frames"] / total_frames,
        "per_source": str(detail_path),
        "training_manifest_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    index = output_dir / "index.html"
    index.write_text(_page(details, base_name=base_name, candidate_name=candidate_name), encoding="utf-8")
    update_audit_entrypoints(latest_html=index, title="Scorer v11 teacher prompt A/B")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--base-name", default="v5")
    parser.add_argument("--candidate-name", default="v6")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            compare(
                manifest=Path(args.manifest),
                base=Path(args.base),
                candidate=Path(args.candidate),
                base_name=args.base_name,
                candidate_name=args.candidate_name,
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
