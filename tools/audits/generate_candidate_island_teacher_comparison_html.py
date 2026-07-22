#!/usr/bin/env python3
"""Render human/Qwen/Gemini candidate-island timelines on the same sources."""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _audio_path(value: str, *, manifest: Path) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [manifest.parent / raw]
    summary_path = manifest.parent / "summary.json"
    if not raw.is_absolute() and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        source_audit_dir = str(summary.get("source_audit_dir") or "")
        if source_audit_dir:
            candidates.append(PROJECT_ROOT / source_audit_dir / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"comparison audio is missing for {value!r}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _audio_url(value: str, *, manifest: Path) -> str:
    path = _audio_path(value, manifest=manifest)
    return "/" + path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def build(*, manifest: Path, human: Path, teacher_specs: list[str], metrics: Path, output_dir: Path) -> Path:
    source_rows = {str(row["source_id"]): row for row in _rows(manifest)}
    human_rows = {str(row["source_id"]): row for row in _rows(human)}
    teachers: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in teacher_specs:
        name, separator, path_text = spec.partition("=")
        if not separator:
            raise ValueError("teacher must use name=path")
        teachers[name] = {str(row["source_id"]): row for row in _rows(Path(path_text))}
    metric_rows = {str(row["source_id"]): row for row in _rows(Path(json.loads(metrics.read_text(encoding="utf-8"))["per_source"]))}
    payload = []
    for source_id, source in source_rows.items():
        if source_id not in human_rows or any(source_id not in rows for rows in teachers.values()):
            raise ValueError(f"comparison source missing: {source_id}")
        human_spans = [span for span in human_rows[source_id].get("spans") or () if span.get("label") == "inside_candidate"]
        payload.append({"source_id": source_id, "partition": source.get("partition"), "duration_s": source.get("duration_s"), "audio": _audio_url(str(source["audio"]), manifest=manifest), "lanes": {"human": human_spans, **{name: rows[source_id].get("islands") or [] for name, rows in teachers.items()}}, "metrics": metric_rows[source_id].get("teachers") or {}})
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Scorer v11 teacher vs human</title><style>
body{{margin:0;background:#f4f7fa;color:#18212b;font-family:Segoe UI,Microsoft YaHei,sans-serif}}header{{position:sticky;top:0;z-index:2;background:#122233;color:white;padding:12px 18px}}main{{max-width:1400px;margin:auto;padding:16px}}article{{background:white;border:1px solid #ccd6df;border-radius:10px;padding:14px;margin-bottom:14px}}audio{{width:100%}}.lane{{display:grid;grid-template-columns:90px 1fr;gap:8px;align-items:center;margin:8px 0}}.track{{height:34px;background:#e9eef3;position:relative;border-radius:5px;overflow:hidden}}.span{{position:absolute;height:100%;border:0;border-radius:4px;cursor:pointer;min-width:2px}}.human{{background:#2ca25f}}.qwen{{background:#3182bd}}.gemini{{background:#8e63ce}}.metrics{{font-size:12px;color:#40566c;display:flex;gap:14px;flex-wrap:wrap}}button.playing{{outline:3px solid #ef9b20}}small{{color:#607080}}</style></head><body><header><b>Scorer v11 · 人工 / Qwen / Gemini 连续对话岛对比</b>　<span id=\"status\"></span></header><main><section><p>每条均为同一完整 source。绿色=人工，蓝色=Qwen，紫色=Gemini。点击色块只播放该区间；完整播放器保留全上下文。本页只比较 Scorer candidate membership，不评价句子切分。</p></section><div id=\"list\"></div></main><script>
const rows={encoded};let activeAudio=null,activeButton=null,stopFn=null;function esc(s){{return String(s??'').replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));}}function stop(){{if(activeAudio&&stopFn)activeAudio.removeEventListener('timeupdate',stopFn);if(activeAudio)activeAudio.pause();if(activeButton)activeButton.classList.remove('playing');activeAudio=activeButton=stopFn=null;}}function play(audio,button,start,end){{stop();activeAudio=audio;activeButton=button;button.classList.add('playing');const go=()=>{{audio.currentTime=start;stopFn=()=>{{if(audio.currentTime>=end)stop();}};audio.addEventListener('timeupdate',stopFn);audio.play();}};if(audio.readyState<1){{audio.addEventListener('loadedmetadata',go,{{once:true}});audio.load();}}else go();}}const root=document.getElementById('list');for(const row of rows){{const card=document.createElement('article');card.innerHTML=`<h2>${{esc(row.source_id)}}</h2><small>${{esc(row.partition)}} / ${{Number(row.duration_s).toFixed(2)}}s</small><audio controls preload=\"metadata\" src=\"${{esc(row.audio)}}\"></audio><div class=\"lanes\"></div><div class=\"metrics\"></div>`;const audio=card.querySelector('audio'),lanes=card.querySelector('.lanes');for(const [name,spans] of Object.entries(row.lanes)){{const lane=document.createElement('div');lane.className='lane';lane.innerHTML=`<b>${{esc(name)}}</b><div class=\"track\"></div>`;const track=lane.querySelector('.track');for(const span of spans){{const start=Number(span.start_s??span.start_frame*.02),end=Number(span.end_s??span.end_frame*.02),button=document.createElement('button');button.className=`span ${{name}}`;button.style.left=`${{100*start/row.duration_s}}%`;button.style.width=`${{Math.max(.15,100*(end-start)/row.duration_s)}}%`;button.title=`${{name}} ${{start.toFixed(2)}}–${{end.toFixed(2)}}s`;button.onclick=()=>play(audio,button,start,end);track.appendChild(button);}}lanes.appendChild(lane);}}const metrics=card.querySelector('.metrics');for(const [name,m] of Object.entries(row.metrics))metrics.innerHTML+=`<span><b>${{esc(name)}}</b> inside=${{(100*m.inside_candidate_recall).toFixed(1)}}% outside=${{(100*m.outside_candidate_recall).toFixed(1)}}% extra=${{(100*m.extra_inside_rate_on_human_outside).toFixed(1)}}%</span>`;root.appendChild(card);}}document.getElementById('status').textContent=`${{rows.length}} sources`;</script></body></html>"""
    output_dir.mkdir(parents=True, exist_ok=True)
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    summary = {"schema": "candidate_island_teacher_human_comparison_page_v1", "source_count": len(payload), "manifest": str(manifest), "human": str(human), "teacher_specs": teacher_specs, "metrics": str(metrics), "training_manifest_allowed": False}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_audit_entrypoints(latest_html=index, title="Scorer v11 teacher vs human")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--human", required=True)
    parser.add_argument("--teacher", action="append", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(build(manifest=Path(args.manifest), human=Path(args.human), teacher_specs=args.teacher, metrics=Path(args.metrics), output_dir=Path(args.output_dir)))
