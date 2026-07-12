#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _candidate_id(row: dict[str, Any]) -> str:
    return f"{row['window_id']}#f{int(row['feature_index']):05d}"


def _priority(row: dict[str, Any]) -> tuple[int, float, str]:
    if row.get("expected_gate_label"):
        rank = 0
    elif row.get("current_label") != row.get("label"):
        rank = 1
    elif row.get("legacy_label") and row.get("legacy_label") != row.get("label"):
        rank = 2
    elif row.get("label") == "cut":
        rank = 3
    elif float(row.get("confidence") or 0.0) < 0.9:
        rank = 4
    else:
        rank = 5
    return rank, float(row.get("confidence") or 0.0), _candidate_id(row)


def _slice_context(
    *,
    source: Path,
    output: Path,
    center_s: float,
    duration_s: float,
    context_s: float,
) -> tuple[float, float]:
    start = max(0.0, center_s - context_s)
    end = min(duration_s, center_s + context_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-to",
            f"{end:.6f}",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output),
        ],
        check=True,
    )
    return start, center_s - start


def build_audit(
    *,
    selected_windows: Path,
    labels: Path,
    output_dir: Path,
    limit: int,
    context_s: float,
    update_nav: bool,
) -> dict[str, Any]:
    windows = {str(row["window_id"]): row for row in _read_jsonl(selected_windows)}
    rows = sorted(_read_jsonl(labels), key=_priority)
    if limit > 0:
        rows = rows[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(rows, start=1):
        window = windows[str(row["window_id"])]
        source = Path(window["audio_path"])
        center = float(row["time_s"])
        clip = output_dir / "audio" / f"{index:04d}_{row['window_id']}_f{int(row['feature_index']):05d}.wav"
        clip_start, candidate_offset = _slice_context(
            source=source,
            output=clip,
            center_s=center,
            duration_s=float(window["duration_s"]),
            context_s=context_s,
        )
        manifest.append(
            {
                **row,
                "candidate_id": _candidate_id(row),
                "audio": clip.relative_to(output_dir).as_posix(),
                "clip_start_s": clip_start,
                "candidate_offset_s": candidate_offset,
            }
        )
    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )
    payload = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Split v3 Plus Hard-case Audit</title>
<style>
body{{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f5f6f8;color:#20242a}}header{{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #d9dde3;padding:12px 20px}}h1{{font-size:20px;margin:0 0 8px}}.toolbar{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}button,select,input{{font:inherit}}button{{border:1px solid #b9c0c9;background:#fff;padding:7px 11px;border-radius:4px;cursor:pointer}}button.active{{background:#1769aa;color:#fff;border-color:#1769aa}}button.cut.active{{background:#b42318}}button.continue.active{{background:#197642}}button.unsure.active{{background:#8a5a00}}main{{max-width:1080px;margin:18px auto;padding:0 16px}}article{{background:#fff;border:1px solid #d9dde3;border-radius:6px;padding:14px;margin-bottom:12px}}article.done{{border-left:5px solid #197642}}.meta{{display:flex;gap:10px;flex-wrap:wrap;font-size:13px;color:#59616c}}.labels{{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}}.pill{{background:#eef1f4;padding:3px 7px;border-radius:3px}}audio{{width:100%;margin:8px 0}}.verdict{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}.note{{flex:1;min-width:240px;padding:7px;border:1px solid #b9c0c9;border-radius:4px}}pre{{white-space:pre-wrap;background:#f6f7f9;padding:8px;border-radius:4px;font-size:12px}}#status{{font-size:13px;color:#59616c}}
</style></head><body><header><h1>Split v3 Plus Hard-case Audit</h1><div class="toolbar"><button id="save">保存裁决</button><select id="filter"><option value="all">全部</option><option value="todo">未审</option><option value="disagree">模型改判</option></select><span id="status"></span></div></header><main id="list"></main>
<script>
const rows={payload}; const key='split-v3-hardcase-audit-v1'; const ann=JSON.parse(localStorage.getItem(key)||'{{}}');
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function render(){{const filter=document.getElementById('filter').value; const list=document.getElementById('list'); list.innerHTML=''; let shown=0;
for(const row of rows){{const a=ann[row.candidate_id]||{{verdict:'',note:''}}; const disagree=row.current_label!==row.label; if(filter==='todo'&&a.verdict)continue;if(filter==='disagree'&&!disagree)continue;shown++;
const article=document.createElement('article');article.dataset.id=row.candidate_id;if(a.verdict)article.classList.add('done');article.innerHTML=`<strong>${{esc(row.candidate_id)}}</strong><div class="meta"><span>${{esc(row.partition)}}</span><span>t=${{Number(row.time_s).toFixed(3)}}s</span><span>confidence=${{Number(row.confidence).toFixed(2)}}</span><span>${{esc((row.hard_case_categories||[]).join(', '))}}</span></div><div class="labels"><span class="pill">current: ${{esc(row.current_label)}}</span><span class="pill">legacy: ${{esc(row.legacy_label||'-')}}</span><span class="pill">Plus: ${{esc(row.label)}}</span><span class="pill">raw: ${{esc(row.omni_label)}}</span></div><audio controls preload="none" src="${{esc(row.audio)}}"></audio><button class="seek">跳到候选点 ${{Number(row.candidate_offset_s).toFixed(2)}}s</button><div class="verdict"><button class="cut" data-v="cut">cut</button><button class="continue" data-v="continue">continue</button><button class="unsure" data-v="unsure">unsure</button><input class="note" placeholder="备注" value="${{esc(a.note)}}"></div><pre>${{esc(JSON.stringify({{left_complete:row.left_complete,right_complete:row.right_complete,merged_better:row.merged_better,flags:row.flags,expected_gate_label:row.expected_gate_label}},null,2))}}</pre>`;
article.querySelector('.seek').onclick=()=>{{const audio=article.querySelector('audio');audio.currentTime=Math.max(0,row.candidate_offset_s-0.5);audio.play();}};
for(const b of article.querySelectorAll('[data-v]')){{if(b.dataset.v===a.verdict)b.classList.add('active');b.onclick=()=>{{ann[row.candidate_id]={{verdict:b.dataset.v,note:article.querySelector('.note').value,updated_at:new Date().toISOString()}};localStorage.setItem(key,JSON.stringify(ann));render();}};}}
article.querySelector('.note').onchange=e=>{{const old=ann[row.candidate_id]||{{verdict:'',note:''}};ann[row.candidate_id]={{...old,note:e.target.value,updated_at:new Date().toISOString()}};localStorage.setItem(key,JSON.stringify(ann));}};list.appendChild(article);}}
document.getElementById('status').textContent=`显示 ${{shown}} / ${{rows.length}}，已审 ${{Object.values(ann).filter(x=>x.verdict).length}}`;}}
document.getElementById('filter').onchange=render;document.getElementById('save').onclick=async()=>{{const content=rows.filter(r=>ann[r.candidate_id]?.verdict).map(r=>JSON.stringify({{schema:'semantic_split_v3_manual_verdict_v1',candidate_id:r.candidate_id,window_id:r.window_id,feature_index:r.feature_index,verdict:ann[r.candidate_id].verdict,note:ann[r.candidate_id].note||'',model_label:r.label,current_label:r.current_label,updated_at:ann[r.candidate_id].updated_at}})).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();
</script></body></html>"""
    index_path = output_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")
    summary = {
        "schema": "semantic_split_v3_hard_case_audit_summary_v1",
        "title": "Split v3 Plus Hard-case Audit",
        "item_count": len(manifest),
        "label_counts": dict(Counter(str(row["label"]) for row in manifest)),
        "current_disagreement_count": sum(row["current_label"] != row["label"] for row in manifest),
        "manual_verdicts": str(output_dir / "manual_verdicts.jsonl"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if update_nav:
        update_audit_entrypoints(latest_html=index_path, title=summary["title"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-windows", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--context-s", type=float, default=5.0)
    parser.add_argument("--no-update-nav", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_audit(
                selected_windows=Path(args.selected_windows),
                labels=Path(args.labels),
                output_dir=Path(args.output_dir),
                limit=args.limit,
                context_s=args.context_s,
                update_nav=not args.no_update_nav,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
