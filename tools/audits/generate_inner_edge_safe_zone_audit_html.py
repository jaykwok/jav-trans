#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
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
    selection = {str(row["boundary_id"]): row for row in _read_jsonl(selected)}
    rows = []
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for label in _read_jsonl(labels):
        boundary = selection[str(label["boundary_id"])]
        original_src = Path(boundary["original_audio"])
        original_dst = audio_dir / f"{boundary['boundary_id'].replace('#', '__')}__original.wav"
        shutil.copy2(original_src, original_dst)
        plans = []
        by_id = {str(row["candidate_id"]): row for row in label["candidates"]}
        for candidate in boundary["candidates"]:
            src = Path(candidate["audio"])
            dst = audio_dir / f"{boundary['boundary_id'].replace('#', '__')}__{candidate['candidate_id']}.wav"
            shutil.copy2(src, dst)
            plans.append(
                {
                    **candidate,
                    **by_id[candidate["candidate_id"]],
                    "audio": dst.relative_to(output_dir).as_posix(),
                }
            )
        rows.append({"boundary_id": boundary["boundary_id"], "original_audio": original_dst.relative_to(output_dir).as_posix(), "plans": plans})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Inner Edge Safe-zone Smoke</title>
<style>
body{{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f4f6f8;color:#20242a}}header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccd2d9;padding:12px 18px;z-index:2}}main{{max-width:1050px;margin:18px auto;padding:0 14px}}.help,article{{background:#fff;border:1px solid #ccd2d9;border-radius:7px;padding:16px;margin-bottom:16px}}.help{{border-left:5px solid #1769aa}}article.done{{border-left:5px solid #197642}}.legend{{display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:8px;margin:12px 0}}.legend div,.plan{{border:1px solid #dfe3e8;border-radius:6px;padding:11px}}.plan{{margin:10px 0}}.plan.reviewed{{background:#f2f8f4;border-color:#78a989}}.plan-head,.choices{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}.candidate-audio{{display:none}}audio.original{{width:100%}}button,input{{font:inherit}}button{{padding:7px 10px;border:1px solid #aeb6c0;border-radius:4px;background:#fff;cursor:pointer}}button.active{{background:#1769aa;color:#fff;border-color:#1769aa}}button.play{{background:#eef5fb}}input.note{{width:min(96%,720px);padding:8px;margin-top:8px}}code{{font-size:12px}}.teacher{{font-weight:700}}.left_clipped{{color:#b42318}}.safe{{color:#197642}}.right_clipped{{color:#9b4d00}}.unsure{{color:#666}}small{{display:block;color:#59616c;margin:6px 0}}.boundary-status{{font-weight:600;color:#59616c}}
</style></head><body><header><strong>Inner Edge Safe-zone 人工审计</strong>　<button id="save">保存全部裁决</button> <span id="status"></span></header><main>
<section class="help"><h2>你要做什么</h2><p>这里只核对当前候选方案；页面不显示旧切点、旧 Refiner 或 Timeline。</p><ol><li>每个 boundary 先听一次“原始 island”，了解前后两句。</li><li>下面每个候选都在一个不同位置插入了<strong>恰好 1 秒静音</strong>。灰色长条只是浏览器音频播放器，不是评分进度。</li><li>点击“播放插入点附近”，只听该候选静音前后约 2 秒。</li><li>按听感为每个候选选择：太早、安全吗、太晚，或无法判断。页面上的“模型建议”只是待核对答案。</li></ol><div class="legend"><div><b class="left_clipped">太早 / left_clipped</b><br>左边那句话还没完整说完就插入静音。</div><div><b class="safe">安全 / safe</b><br>静音正好落在左句结束、右句开始之间。</div><div><b class="right_clipped">太晚 / right_clipped</b><br>右边那句话已经开始，句首被留到静音前。</div><div><b class="unsure">无法判断 / unsure</b><br>连续或重叠语音，听不出明确安全区。</div></div></section>
<div id="list"></div></main><script>
const rows={payload};const key='inner-edge-safe-zone-smoke-v2';const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let active=null;
function esc(s){{return String(s??'').replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));}}
function ensure(row){{ann[row.boundary_id]??={{candidates:{{}},note:''}};ann[row.boundary_id].candidates??={{}};return ann[row.boundary_id];}}
function labelName(v){{return ({{left_clipped:'太早',safe:'安全',right_clipped:'太晚',unsure:'无法判断'}})[v]||v;}}
function status(){{let done=0,total=0,boundaries=0;for(const row of rows){{const a=ensure(row);let rowDone=0;for(const p of row.plans){{total++;if(a.candidates[p.candidate_id]){{done++;rowDone++;}}}}if(rowDone===row.plans.length)boundaries++;}}document.getElementById('status').textContent=`候选已审 ${{done}} / ${{total}}；完整边界 ${{boundaries}} / ${{rows.length}}`;}}
function playNear(audio,cut){{if(active&&active!==audio)active.pause();active=audio;audio.currentTime=Math.max(0,Number(cut)-2);const stopAt=Number(cut)+3.2;const stop=()=>{{if(audio.currentTime>=stopAt){{audio.pause();audio.removeEventListener('timeupdate',stop);}}}};audio.addEventListener('timeupdate',stop);audio.play();}}
function render(){{const list=document.getElementById('list');list.innerHTML='';for(const row of rows){{const a=ensure(row);const el=document.createElement('article');const reviewed=Object.keys(a.candidates).length;if(reviewed===row.plans.length)el.classList.add('done');el.innerHTML=`<h3>${{esc(row.boundary_id)}}</h3><p class="boundary-status">本边界已审 ${{reviewed}} / ${{row.plans.length}} 个候选</p><p><strong>① 原始 island</strong>（只用于理解前后语音）</p><audio class="original" controls preload="metadata" src="${{esc(row.original_audio)}}"></audio><p><strong>② 逐个核对候选</strong></p>${{row.plans.map(p=>{{const manual=a.candidates[p.candidate_id]||'';return `<section class="plan ${{manual?'reviewed':''}}" data-cid="${{esc(p.candidate_id)}}"><div class="plan-head"><code>${{esc(p.candidate_id)}}</code><button class="play">▶ 播放插入点附近</button><span>模型建议：<b class="teacher ${{esc(p.label)}}">${{esc(labelName(p.label))}}</b></span></div><audio class="candidate-audio" preload="metadata" src="${{esc(p.audio)}}"></audio><small>模型理由：${{esc(p.reason||'无')}}</small><div class="choices"><span><b>你的判断：</b></span>${{[['left_clipped','太早'],['safe','安全'],['right_clipped','太晚'],['unsure','无法判断']].map(([v,t])=>`<button data-label="${{v}}" class="${{manual===v?'active':''}}">${{t}}</button>`).join('')}}</div></section>`;}}).join('')}}<input class="note" placeholder="边界备注（可选）" value="${{esc(a.note)}}">`;el.querySelectorAll('audio').forEach(x=>x.onplay=()=>{{if(active&&active!==x)active.pause();active=x;}});el.querySelectorAll('.plan').forEach(plan=>{{const cid=plan.dataset.cid;const p=row.plans.find(x=>x.candidate_id===cid);const audio=plan.querySelector('.candidate-audio');plan.querySelector('.play').onclick=()=>playNear(audio,p.relative_time_s);plan.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>{{a.candidates[cid]=b.dataset.label;a.updated_at=new Date().toISOString();localStorage.setItem(key,JSON.stringify(ann));render();}});}});el.querySelector('.note').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();localStorage.setItem(key,JSON.stringify(ann));}};list.appendChild(el);}}status();}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(row=>{{const a=ensure(row);return JSON.stringify({{schema:'inner_edge_safe_zone_manual_verdict_v2',boundary_id:row.boundary_id,candidates:row.plans.map(p=>({{candidate_id:p.candidate_id,teacher_label:p.label,manual_label:a.candidates[p.candidate_id]||'unreviewed',matches_teacher:a.candidates[p.candidate_id]===p.label}})),note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();
</script></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    summary = {"schema": "inner_edge_safe_zone_audit_summary_v1", "item_count": len(rows), "manual_verdicts": str(output_dir / "manual_verdicts.jsonl")}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_audit_entrypoints(latest_html=index, title="Inner Edge Safe-zone Smoke")
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
