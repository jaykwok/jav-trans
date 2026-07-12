#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_audit(*, selected: Path, labels: Path, output_dir: Path) -> dict[str, Any]:
    selection = {str(row["island_id"]): row for row in _read_jsonl(selected)}
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in _read_jsonl(labels):
        island = selection[str(label["island_id"])]
        source = Path(label["island_id"].replace("#", "__") + ".wav")
        request_audio = labels.parent / "request_audio" / source
        target_audio = output_dir / "audio" / source
        target_audio.parent.mkdir(parents=True, exist_ok=True)
        if not target_audio.exists():
            target_audio.write_bytes(request_audio.read_bytes())
        accepted = sorted(
            {
                round(float(candidate["relative_time_s"]), 3)
                for candidate in island["candidates"]
                if candidate.get("accepted")
            }
        )
        rows.append(
            {
                **label,
                "audio": target_audio.relative_to(output_dir).as_posix(),
                "current_accepted_cuts": accepted,
            }
        )
    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Split v3 Speech Island Smoke</title>
<style>body{{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f5f6f8;color:#20242a}}header{{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #d9dde3;padding:12px 20px}}main{{max-width:1080px;margin:18px auto;padding:0 16px}}article{{background:#fff;border:1px solid #d9dde3;border-radius:6px;padding:14px;margin-bottom:14px}}article.done{{border-left:5px solid #197642}}audio{{width:100%;margin:10px 0}}button,input{{font:inherit}}button{{border:1px solid #b9c0c9;background:#fff;padding:7px 10px;border-radius:4px;cursor:pointer}}button.active{{background:#1769aa;color:#fff}}.meta,.legend,.controls,.verdict{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}.meta{{font-size:13px;color:#59616c}}.timeline{{height:32px;position:relative;background:#eef1f4;border:1px solid #b9c0c9;border-radius:3px;margin:8px 0}}.mark{{position:absolute;top:0;bottom:0;width:3px;cursor:pointer}}.omni{{background:#b42318}}.current{{background:#1769aa;top:16px}}.legend span{{font-size:13px}}.swatch{{display:inline-block;width:12px;height:3px;margin-right:4px;vertical-align:middle}}.note{{flex:1;min-width:250px;padding:7px;border:1px solid #b9c0c9;border-radius:4px}}pre{{white-space:pre-wrap;background:#f6f7f9;padding:8px;border-radius:4px;font-size:12px}}</style></head>
<body><header><strong>Split v3 Speech Island Smoke · 5 samples</strong> <button id="save">保存裁决</button> <span id="status"></span></header><main id="list"></main>
<script>const rows={payload};const key='split-v3-island-smoke-v1';const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let active=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function playAt(a,t){{if(active&&active!==a)active.pause();active=a;a.currentTime=Math.max(0,t-1);const end=Math.min(a.duration||1e9,t+1);const stop=()=>{{if(a.currentTime>=end){{a.pause();a.removeEventListener('timeupdate',stop)}}}};a.addEventListener('timeupdate',stop);a.play();}}
function render(){{const list=document.getElementById('list');list.innerHTML='';for(const row of rows){{const a=ann[row.island_id]||{{verdict:'',note:''}};const el=document.createElement('article');if(a.verdict)el.classList.add('done');const dur=Number(row.duration_s);const omni=(row.cuts||[]).map(x=>Number(x.time_s));const current=(row.current_accepted_cuts||[]).map(Number);const marks=[...omni.map(t=>`<button class="mark omni" data-t="${{t}}" title="Omni ${{t.toFixed(3)}}s" style="left:${{t/dur*100}}%"></button>`),...current.map(t=>`<button class="mark current" data-t="${{t}}" title="现役 ${{t.toFixed(3)}}s" style="left:${{t/dur*100}}%"></button>`)].join('');el.innerHTML=`<strong>${{esc(row.island_id)}}</strong><div class="meta"><span>时长 ${{dur.toFixed(2)}}s</span><span>Omni cuts=${{omni.length}}</span><span>现役 cuts=${{current.length}}</span></div><audio controls preload="metadata" src="${{esc(row.audio)}}"></audio><div class="timeline">${{marks}}</div><div class="legend"><span><i class="swatch" style="background:#b42318"></i>Omni</span><span><i class="swatch" style="background:#1769aa"></i>现役 accepted</span><span>点击竖线跨点试听 ±1s</span></div><div class="verdict"><button data-v="correct">整体正确</button><button data-v="missed_cut">有漏切</button><button data-v="false_cut">有误切</button><button data-v="timing_error">时间不准</button><button data-v="unsure">不确定</button><input class="note" placeholder="指出具体秒数" value="${{esc(a.note)}}"></div><pre>${{esc(JSON.stringify({{omni_cuts:row.cuts,current_accepted_cuts:current,reason:row.reason}},null,2))}}</pre>`;const audio=el.querySelector('audio');audio.onplay=()=>{{if(active&&active!==audio)active.pause();active=audio}};el.querySelectorAll('.mark').forEach(b=>b.onclick=()=>playAt(audio,Number(b.dataset.t)));el.querySelectorAll('[data-v]').forEach(b=>{{if(b.dataset.v===a.verdict)b.classList.add('active');b.onclick=()=>{{ann[row.island_id]={{verdict:b.dataset.v,note:el.querySelector('.note').value,updated_at:new Date().toISOString()}};localStorage.setItem(key,JSON.stringify(ann));render()}}}});el.querySelector('.note').onchange=e=>{{ann[row.island_id]={{...(ann[row.island_id]||{{verdict:'',note:''}}),note:e.target.value,updated_at:new Date().toISOString()}};localStorage.setItem(key,JSON.stringify(ann))}};list.appendChild(el)}}document.getElementById('status').textContent=`已审 ${{Object.values(ann).filter(x=>x.verdict).length}} / ${{rows.length}}`;}}
document.getElementById('save').onclick=async()=>{{const content=rows.filter(r=>ann[r.island_id]?.verdict).map(r=>JSON.stringify({{schema:'semantic_split_v3_island_manual_verdict_v1',island_id:r.island_id,verdict:ann[r.island_id].verdict,note:ann[r.island_id].note||'',updated_at:ann[r.island_id].updated_at}})).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error}};render();</script></body></html>"""
    index_path = output_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")
    summary = {"schema": "semantic_split_v3_island_audit_summary_v1", "item_count": len(rows), "total_omni_cuts": sum(len(row["cuts"]) for row in rows), "manual_verdicts": str(output_dir / "manual_verdicts.jsonl")}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_audit_entrypoints(latest_html=index_path, title="Split v3 Speech Island Smoke")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_audit(selected=Path(args.selected), labels=Path(args.labels), output_dir=Path(args.output_dir)), ensure_ascii=False))


if __name__ == "__main__":
    main()
