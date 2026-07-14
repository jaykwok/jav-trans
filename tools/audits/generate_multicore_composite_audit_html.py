#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_audit(*, manifest: Path, output_dir: Path) -> Path:
    rows = _rows(manifest)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload_rows = []
    for row in rows:
        source = Path(row["audio"])
        destination = audio_dir / f"{row['sample_id']}{source.suffix}"
        shutil.copy2(source, destination)
        payload_rows.append(
            {
                **row,
                "audio": destination.relative_to(output_dir).as_posix(),
                "inner_review_required": bool(row.get("semantic_events")),
            }
        )

    payload = json.dumps(payload_rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic Split / Inner Multi-core Composite Smoke</title>
<style>
body{{margin:0;background:#f2f4f7;color:#20242a;font-family:Segoe UI,Arial,sans-serif}}header{{position:sticky;top:0;z-index:5;padding:12px 18px;background:#fff;border-bottom:1px solid #cbd2da}}main{{max-width:1180px;margin:18px auto;padding:0 14px}}article,.help{{background:#fff;border:1px solid #cbd2da;border-radius:8px;padding:16px;margin-bottom:16px}}article.done{{border-left:6px solid #1b7a3a}}.help{{border-left:6px solid #1769aa}}.panels{{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:10px}}.panel{{border:1px solid #d6dce3;border-radius:7px;padding:12px}}audio{{width:100%}}.track{{position:relative;height:66px;background:#e7e9ec;border-radius:6px;overflow:hidden;margin:12px 0}}.bar{{position:absolute;box-sizing:border-box;border:1px solid rgba(0,0,0,.3);overflow:hidden;font-size:12px;padding:3px}}.core{{top:8px;height:24px;background:#64bd79;z-index:2}}.safe{{top:38px;height:20px;background:#73a9d8}}.overlap{{top:38px;height:20px;background:#dc7d70}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d6dce3;padding:7px;text-align:left;vertical-align:top}}button,textarea{{font:inherit}}button{{padding:6px 9px;margin:2px}}button.active{{background:#1769aa;color:#fff}}button.reject.active{{background:#a92c2c}}textarea{{box-sizing:border-box;width:100%;min-height:58px}}.verdict{{background:#f7f9fb;border-radius:6px;padding:10px;margin-top:10px}}small,.muted{{color:#5a626d}}code{{background:#eef1f4;padding:2px 4px;border-radius:3px}}@media(max-width:850px){{.panels{{grid-template-columns:1fr}}table{{font-size:13px}}}}
</style></head><body>
<header><strong>1.7B Split v3 + Inner · multi-core composite smoke5</strong>　<button id="save">保存全部裁决</button> <span id="status"></span></header>
<main><section class="help"><h2>这 5 条只验证合成数据契约，不训练</h2><div class="panels">
<div class="panel"><h3>1 · composite/core</h3><p>确认绿色 semantic cores 内容完整，拼接、BGM 或负面声音没有误伤清楚语义，也没有明显不可接受的合成伪影。</p></div>
<div class="panel"><h3>2 · Semantic Split</h3><p>独立 cores 之间应是 <code>cut</code>；单一 maximal core 即使 BGM 改变也应全部 <code>continue</code>。这里只审语义事件，不把区间当最终切点。</p></div>
<div class="panel"><h3>3 · Inner edge</h3><p>蓝色为可移除 semantic-safe gap；红色为 overlap，必须 abstain。没有 semantic event 的单-core control 自动记为 not_applicable，不重复要求第三次裁决。</p></div>
</div><p>gap 时长来自真实 residual 分位数；music、breathing/moaning 来自保留的 hardmix definite-drop 原始资产。人工 5/5 前禁止扩 100 条、跑 proposer 或训练。</p></section><div id="list"></div></main>
<script>
const rows={payload};
const key='semantic-split-multicore-composite-audit-v1:'+location.pathname;
const ann=JSON.parse(localStorage.getItem(key)||'{{}}');
let activeAudio=null;
let activeStop=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function ensure(row){{ann[row.sample_id]??={{composition_verdict:'',split_verdict:'',inner_verdict:'',note:''}};return ann[row.sample_id];}}
function approved(row,a){{return a.composition_verdict==='approve'&&a.split_verdict==='approve'&&(!row.inner_review_required||a.inner_verdict==='approve');}}
function persist(){{localStorage.setItem(key,JSON.stringify(ann));status();}}
function status(){{const done=rows.filter(row=>approved(row,ensure(row))).length;document.getElementById('status').textContent=`全部必要项通过 ${{done}} / ${{rows.length}}`;}}
function pct(value,duration){{return Math.max(0,Math.min(100,Number(value)*100/Math.max(Number(duration),.001)));}}
function span(start,end){{return `${{Number(start).toFixed(3)}}–${{Number(end).toFixed(3)}}s`;}}
function clearStop(){{if(activeAudio&&activeStop)activeAudio.removeEventListener('timeupdate',activeStop);activeStop=null;}}
function playSpan(id,start,end){{const audio=document.getElementById(id);if(activeAudio){{clearStop();if(activeAudio!==audio)activeAudio.pause();}}activeAudio=audio;const begin=()=>{{audio.currentTime=Number(start);activeStop=()=>{{if(audio.currentTime>=Number(end)){{clearStop();audio.pause();}}}};audio.addEventListener('timeupdate',activeStop);audio.play();}};if(audio.readyState<1){{audio.addEventListener('loadedmetadata',begin,{{once:true}});audio.load();}}else begin();}}
function verdict(label,field,value,a){{return `<button data-field="${{field}}" data-value="${{value}}" class="${{value==='reject'?'reject ':''}}${{a[field]===value?'active':''}}">${{label}}</button>`;}}
function timeline(row){{let bars='';for(const core of row.core_spans)bars+=`<div class="bar core" style="left:${{pct(core.start_s,row.duration_s)}}%;width:${{pct(core.end_s-core.start_s,row.duration_s)}}%" title="${{esc(core.core_id)}} ${{span(core.start_s,core.end_s)}}">${{esc(core.core_id)}}</div>`;for(const gap of row.gap_spans||[]){{const cls=gap.kind==='overlap'?'overlap':'safe';bars+=`<div class="bar ${{cls}}" style="left:${{pct(gap.start_s,row.duration_s)}}%;width:${{pct(gap.end_s-gap.start_s,row.duration_s)}}%" title="${{esc(gap.kind)}} ${{span(gap.start_s,gap.end_s)}}">${{esc(gap.kind)}}</div>`;}}return `<div class="track">${{bars}}</div><small>0s　　　　　　　　　　　　　　　　　　　　　　　　　${{Number(row.duration_s).toFixed(3)}}s　（绿=semantic core，蓝=safe gap，红=overlap abstain）</small>`;}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const row of rows){{const a=ensure(row);const card=document.createElement('article');if(approved(row,a))card.classList.add('done');const audioId='audio-'+row.sample_id;const cores=row.core_spans.map(core=>`<tr><td>${{esc(core.core_id)}}</td><td>${{esc(core.text)}}</td><td>${{span(core.start_s,core.end_s)}}</td><td><button data-play-start="${{core.start_s}}" data-play-end="${{core.end_s}}">播放 core</button></td></tr>`).join('');const events=row.semantic_events.length?row.semantic_events.map(event=>{{const inner=event.inner_target;const playStart=Math.max(0,event.event_interval_start_s-.8),playEnd=Math.min(row.duration_s,event.event_interval_end_s+.8);return `<tr><td>${{esc(event.event_id)}}</td><td>${{esc(event.semantic_decision)}}</td><td>${{span(event.event_interval_start_s,event.event_interval_end_s)}}</td><td>${{esc(inner.status)}} / ${{esc(inner.gap_kind)}}</td><td>${{inner.status==='safe'?span(inner.left_speech_end_s,inner.right_speech_start_s):'保持单 chunk'}}</td><td><button data-play-start="${{playStart}}" data-play-end="${{playEnd}}">播放事件上下文</button></td></tr>`;}}).join(''):'<tr><td colspan="6"><b>无 semantic event：</b>单一 maximal semantic core；BGM 变化不是 cut。</td></tr>';const innerPanel=row.inner_review_required?`<div class="panel"><h3>3 · Inner safe/abstain</h3><p>逐 event 核对蓝色 safe gap 或红色 overlap abstain。</p><div class="verdict">${{verdict('Inner 通过','inner_verdict','approve',a)}}${{verdict('Inner 不通过','inner_verdict','reject',a)}}</div></div>`:`<div class="panel"><h3>3 · Inner</h3><p><b>自动去重：</b>本条没有 semantic event，Inner 为 not_applicable，不需要重复裁决。</p></div>`;card.innerHTML=`<h2>${{esc(row.sample_id)}}</h2><p><b>审计重点：</b>${{esc(row.audit_focus)}}</p><small>axes=${{esc(JSON.stringify(row.sampling_axes))}}；时长=${{Number(row.duration_s).toFixed(3)}}s</small><p><b>完整 composite</b></p><audio id="${{audioId}}" controls preload="metadata" src="${{esc(row.audio)}}"></audio>${{timeline(row)}}<table><thead><tr><th>core</th><th>可信文本</th><th>精确 span</th><th>试听</th></tr></thead><tbody>${{cores}}</tbody></table><h3>事件真值</h3><table><thead><tr><th>event</th><th>Split</th><th>event interval</th><th>Inner</th><th>paired edges</th><th>试听</th></tr></thead><tbody>${{events}}</tbody></table><div class="panels"><div class="panel"><h3>1 · composite/core</h3><div class="verdict">${{verdict('合成/core 通过','composition_verdict','approve',a)}}${{verdict('合成/core 不通过','composition_verdict','reject',a)}}</div></div><div class="panel"><h3>2 · Semantic Split</h3><div class="verdict">${{verdict('Split 通过','split_verdict','approve',a)}}${{verdict('Split 不通过','split_verdict','reject',a)}}</div></div>${{innerPanel}}</div><label><b>备注</b><textarea placeholder="指出 core、event、safe gap 或合成伪影问题">${{esc(a.note)}}</textarea></label>`;card.querySelectorAll('[data-field]').forEach(button=>button.onclick=()=>{{a[button.dataset.field]=button.dataset.value;a.updated_at=new Date().toISOString();persist();render();}});card.querySelectorAll('[data-play-start]').forEach(button=>button.onclick=()=>playSpan(audioId,button.dataset.playStart,button.dataset.playEnd));card.querySelector('textarea').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};const audio=card.querySelector('audio');audio.onplay=e=>{{if(activeAudio&&activeAudio!==e.target){{clearStop();activeAudio.pause();}}activeAudio=e.target;}};audio.onpause=()=>clearStop();root.appendChild(card);}}status();}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(row=>{{const a=ensure(row);const inner=row.inner_review_required?(a.inner_verdict||'unreviewed'):'not_applicable';const rejected=a.composition_verdict==='reject'||a.split_verdict==='reject'||(row.inner_review_required&&a.inner_verdict==='reject');return JSON.stringify({{schema:'semantic_split_multicore_composite_manual_verdict_v1',sample_id:row.sample_id,composition_verdict:a.composition_verdict||'unreviewed',split_verdict:a.split_verdict||'unreviewed',inner_verdict:inner,verdict:approved(row,a)?'approve':(rejected?'reject':'unreviewed'),note:a.note||'',semantic_events:row.semantic_events,updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};
render();
</script></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the fixed multi-core Split/Inner smoke audit.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(build_audit(manifest=Path(args.manifest), output_dir=Path(args.output_dir)))
