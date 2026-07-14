#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_audit(*, labels: Path, output_dir: Path) -> Path:
    rows = _rows(labels)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload_rows = []
    for row in rows:
        source = Path(row["audio"])
        sample_dir = audio_dir / str(row["sample_id"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        destination = sample_dir / f"full{source.suffix}"
        shutil.copy2(source, destination)
        payload_rows.append(
            {
                **row,
                "audio": destination.relative_to(output_dir).as_posix(),
            }
        )

    payload = json.dumps(payload_rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic Source Text Alignment Smoke</title>
<style>
body{{margin:0;background:#f2f4f7;color:#20242a;font-family:Segoe UI,Arial,sans-serif}}header{{position:sticky;top:0;z-index:5;padding:12px 18px;background:#fff;border-bottom:1px solid #cbd2da}}main{{max-width:1120px;margin:18px auto;padding:0 14px}}article,.help{{background:#fff;border:1px solid #cbd2da;border-radius:8px;padding:16px;margin-bottom:16px}}article.done{{border-left:6px solid #1b7a3a}}.help{{border-left:6px solid #1769aa}}.steps{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:10px}}.step,.panel{{border:1px solid #d6dce3;border-radius:7px;padding:12px}}.step h3,.panel h4{{margin-top:0}}.units{{display:flex;flex-wrap:wrap;gap:5px;padding:10px;background:#f7f9fb;border-radius:6px}}.unit{{padding:5px 8px;border-radius:5px;border:1px solid #cbd2da}}.semantic{{background:#d9f2df}}.nonsemantic{{background:#eceff2}}.unsure{{background:#fff0c8}}audio{{width:100%}}.track{{position:relative;height:52px;background:#e7e9ec;border-radius:6px;overflow:hidden;margin:10px 0}}.bar{{position:absolute;top:0;height:100%;box-sizing:border-box;border:1px solid rgba(0,0,0,.25);overflow:hidden;font-size:12px;padding:3px}}.keep{{background:#bcd8f2;top:0;height:52px}}.meaning{{background:#55b56d;top:15px;height:30px;z-index:2}}.uncertain{{background:#f0bd54;top:8px;height:36px;z-index:3}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d6dce3;padding:7px;text-align:left;vertical-align:top}}button,textarea{{font:inherit}}button{{padding:6px 9px;margin:2px}}button.active{{background:#1769aa;color:#fff}}button.reject.active{{background:#a92c2c}}textarea{{box-sizing:border-box;width:100%;min-height:58px}}.verdict{{background:#f7f9fb;border-radius:6px;padding:10px;margin-top:10px}}.contexts{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}small,.muted{{color:#5a626d}}code{{background:#eef1f4;padding:2px 4px;border-radius:3px}}@media(max-width:800px){{.steps,.contexts{{grid-template-columns:1fr}}table{{font-size:13px}}}}
</style></head><body>
<header><strong>1.7B Semantic Source · 单请求两阶段 teacher smoke</strong>　<button id="save">保存全部裁决</button> <span id="status"></span></header>
<main><section class="help"><h2>这页分开审三件事</h2><div class="steps">
<div class="step"><h3>1 · 文本语义拆分</h3><p>只看参考文本是否正确分成 <code>semantic</code>、<code>nonsemantic</code>、<code>unsure</code>。连续正常句子不按单词切碎；这里也不决定 Split chunk。</p></div>
<div class="step"><h3>2 · 语义音频对齐</h3><p>只检查绿色 semantic 文本在完整音频中的起止时间。每行“播放该语义 span”只播放表中明确写出的区间；相邻 semantic span 必须不重叠。</p></div>
<div class="step"><h3>3 · keep span 归属</h3><p>蓝色 keep span 是 <b>source membership teacher</b>：应包含全部绿色语义，并可包含同一前景 utterance 的附属喘息/呻吟/亲吻声；纯 BGM、环境噪声、远处背景人声应留在蓝色之外。它不是最终 Outer edge，也不是 Split 切点。</p></div>
</div><p><b>请求契约：</b>每个样本只上传一次完整音频和一次完整参考文本；Omni 在同一个响应里先拆文本、再对齐音频。此页不显示旧切点或 learned candidates。</p></section><div id="list"></div></main>
<script>
const rows={payload};
const key='semantic-source-text-alignment-audit-v1:'+location.pathname;
const ann=JSON.parse(localStorage.getItem(key)||'{{}}');
let activeAudio=null;
let activeStop=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function ensure(row){{ann[row.sample_id]??={{text_verdict:'',timeline_verdict:'',keep_verdict:'',note:''}};return ann[row.sample_id];}}
function approved(a){{return a.text_verdict==='approve'&&a.timeline_verdict==='approve'&&a.keep_verdict==='approve';}}
function persist(){{localStorage.setItem(key,JSON.stringify(ann));status();}}
function status(){{const done=rows.filter(row=>approved(ensure(row))).length;document.getElementById('status').textContent=`三项全通过 ${{done}} / ${{rows.length}}`;}}
function pct(value,duration){{return Math.max(0,Math.min(100,Number(value)*100/Math.max(Number(duration),.001)));}}
function spanLabel(start,end){{return `${{Number(start).toFixed(3)}}–${{Number(end).toFixed(3)}}s`;}}
function clearStop(){{if(activeAudio&&activeStop)activeAudio.removeEventListener('timeupdate',activeStop);activeStop=null;}}
function playSpan(id,start,end){{const audio=document.getElementById(id);if(activeAudio){{clearStop();if(activeAudio!==audio)activeAudio.pause();}}activeAudio=audio;const begin=()=>{{audio.currentTime=Number(start);activeStop=()=>{{if(audio.currentTime>=Number(end)){{clearStop();audio.pause();}}}};audio.addEventListener('timeupdate',activeStop);audio.play();}};if(audio.readyState<1){{audio.addEventListener('loadedmetadata',begin,{{once:true}});audio.load();}}else begin();}}
function verdict(label,field,value,a){{return `<button data-field="${{field}}" data-value="${{value}}" class="${{value==='reject'?'reject ':''}}${{a[field]===value?'active':''}}">${{label}}</button>`;}}
function timeline(row){{const keep=row.keep_span;let bars='';if(keep.status==='matched')bars+=`<div class="bar keep" style="left:${{pct(keep.start_s,row.duration_s)}}%;width:${{pct(keep.end_s-keep.start_s,row.duration_s)}}%" title="keep ${{spanLabel(keep.start_s,keep.end_s)}}">keep</div>`;for(const item of row.semantic_alignments){{if(item.status==='matched')bars+=`<div class="bar meaning" style="left:${{pct(item.start_s,row.duration_s)}}%;width:${{pct(item.end_s-item.start_s,row.duration_s)}}%" title="${{esc(item.unit_id)}} ${{spanLabel(item.start_s,item.end_s)}}">${{esc(item.unit_id)}}</div>`;}}for(const item of row.unsure_audio_spans||[])bars+=`<div class="bar uncertain" style="left:${{pct(item.start_s,row.duration_s)}}%;width:${{pct(item.end_s-item.start_s,row.duration_s)}}%" title="unsure ${{spanLabel(item.start_s,item.end_s)}}">?</div>`;return `<div class="track">${{bars}}</div><small>0s　　　　　　　　　　　　　　　　　　　　　　　　　${{Number(row.duration_s).toFixed(3)}}s　（蓝=keep，绿=semantic，黄=unsure）</small>`;}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const row of rows){{const a=ensure(row);const card=document.createElement('article');if(approved(a))card.classList.add('done');const audioId='audio-'+row.sample_id;const units=row.text_units.map(u=>`<span class="unit ${{esc(u.kind)}}" title="${{esc(u.reason)}}"><b>${{esc(u.unit_id)}}</b> ${{esc(u.text)}} <small>${{esc(u.kind)}} ${{Number(u.confidence).toFixed(2)}}</small></span>`).join('');const alignments=row.semantic_alignments.length?row.semantic_alignments.map(x=>`<tr><td>${{esc(x.unit_id)}}</td><td>${{esc(x.status)}}</td><td>${{x.status==='matched'?spanLabel(x.start_s,x.end_s):'—'}}</td><td>${{x.status==='matched'?`<button data-play-start="${{x.start_s}}" data-play-end="${{x.end_s}}">播放该语义 span</button>`:''}}</td><td>${{Number(x.confidence).toFixed(2)}} · ${{esc(x.reason)}}</td></tr>`).join(''):'<tr><td colspan="5">没有 semantic 文本单元；应重点核对这是否确实是纯非语义样本。</td></tr>';const keep=row.keep_span;card.innerHTML=`<h2>${{esc(row.sample_id)}}</h2><p><b>参考文本：</b>${{esc(row.reference_text)}}</p><p><b>审计重点：</b>${{esc(row.audit_focus)}}</p><small>来源：${{esc(row.source)}}；时长 ${{Number(row.duration_s).toFixed(3)}}s；一次完整音频 + 完整文本请求</small><p><b>完整原音频（独立播放）</b></p><audio id="${{esc(audioId)}}" controls preload="metadata" src="${{esc(row.audio)}}"></audio>${{timeline(row)}}<section class="panel"><h4>第 1 项 · 文本语义拆分</h4><div class="units">${{units}}</div><div class="verdict">${{verdict('文本拆分通过','text_verdict','approve',a)}}${{verdict('文本拆分不通过','text_verdict','reject',a)}}</div></section><section class="panel"><h4>第 2 项 · semantic timeline</h4><table><thead><tr><th>unit</th><th>状态</th><th>精确区间</th><th>试听</th><th>证据</th></tr></thead><tbody>${{alignments}}</tbody></table><div class="verdict">${{verdict('时间轴通过','timeline_verdict','approve',a)}}${{verdict('时间轴不通过','timeline_verdict','reject',a)}}</div></section><section class="panel"><h4>第 3 项 · membership keep span</h4><p><b>状态：</b>${{esc(keep.status)}}；<b>区间：</b>${{keep.status==='matched'?spanLabel(keep.start_s,keep.end_s):'—'}} ${{keep.status==='matched'?`<button data-play-start="${{keep.start_s}}" data-play-end="${{keep.end_s}}">播放完整 keep span</button>`:''}}</p><div class="contexts"><div><b>前侧归属：</b>${{esc(keep.leading_context)}}</div><div><b>后侧归属：</b>${{esc(keep.trailing_context)}}</div></div><p>${{esc(keep.reason)}}（confidence=${{Number(keep.confidence).toFixed(2)}}）</p><div class="verdict">${{verdict('keep span 通过','keep_verdict','approve',a)}}${{verdict('keep span 不通过','keep_verdict','reject',a)}}</div></section><label><b>备注</b><textarea placeholder="指出具体文本 unit、时间或上下文归属问题">${{esc(a.note)}}</textarea></label>`;card.querySelectorAll('[data-field]').forEach(button=>button.onclick=()=>{{a[button.dataset.field]=button.dataset.value;a.updated_at=new Date().toISOString();persist();render();}});card.querySelectorAll('[data-play-start]').forEach(button=>button.onclick=()=>playSpan(audioId,button.dataset.playStart,button.dataset.playEnd));card.querySelector('textarea').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};const audio=card.querySelector('audio');audio.onplay=e=>{{if(activeAudio&&activeAudio!==e.target){{clearStop();activeAudio.pause();}}activeAudio=e.target;}};audio.onpause=()=>clearStop();root.appendChild(card);}}status();}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(row=>{{const a=ensure(row);return JSON.stringify({{schema:'semantic_source_text_alignment_manual_verdict_v1',sample_id:row.sample_id,reference_text:row.reference_text,text_units:row.text_units,semantic_alignments:row.semantic_alignments,keep_span:row.keep_span,unsure_audio_spans:row.unsure_audio_spans,text_verdict:a.text_verdict||'unreviewed',timeline_verdict:a.timeline_verdict||'unreviewed',keep_verdict:a.keep_verdict||'unreviewed',verdict:approved(a)?'approve':(a.text_verdict==='reject'||a.timeline_verdict==='reject'||a.keep_verdict==='reject'?'reject':'unreviewed'),note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};
render();
</script></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the three-part semantic source text alignment audit."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(build_audit(labels=Path(args.labels), output_dir=Path(args.output_dir)))
