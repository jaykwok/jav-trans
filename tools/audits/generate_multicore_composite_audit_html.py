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
        copied = dict(row)
        for field, suffix in (
            ("clean_audio", "clean"),
            ("overlay_audio", "overlay"),
            ("mixed_audio", "mixed"),
        ):
            source = Path(row[field])
            destination = audio_dir / f"{row['sample_id']}.{suffix}{source.suffix}"
            shutil.copy2(source, destination)
            copied[field] = destination.relative_to(output_dir).as_posix()
        copied["inner_review_required"] = bool(row.get("semantic_events"))
        payload_rows.append(copied)

    payload = json.dumps(payload_rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic Split / Inner Additive-overlay Smoke</title>
<style>
body{{margin:0;background:#f2f4f7;color:#20242a;font-family:Segoe UI,Arial,sans-serif}}header{{position:sticky;top:0;z-index:5;padding:12px 18px;background:#fff;border-bottom:1px solid #cbd2da}}main{{max-width:1260px;margin:18px auto;padding:0 14px}}article,.help{{background:#fff;border:1px solid #cbd2da;border-radius:8px;padding:16px;margin-bottom:16px}}article.done{{border-left:6px solid #1b7a3a}}.help{{border-left:6px solid #1769aa}}.panels{{display:grid;grid-template-columns:repeat(4,minmax(210px,1fr));gap:10px}}.panel{{border:1px solid #d6dce3;border-radius:7px;padding:12px}}audio{{width:100%}}.audio-grid{{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:10px}}.track{{position:relative;height:92px;background:#e7e9ec;border-radius:6px;overflow:hidden;margin:12px 0}}.bar{{position:absolute;box-sizing:border-box;border:1px solid rgba(0,0,0,.3);overflow:hidden;font-size:12px;padding:3px}}.core{{top:8px;height:24px;background:#64bd79;z-index:2}}.safe{{top:38px;height:20px;background:#73a9d8}}.overlap{{top:38px;height:20px;background:#dc7d70}}.overlay{{top:64px;height:20px;background:#efad5b}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d6dce3;padding:7px;text-align:left;vertical-align:top}}button,textarea{{font:inherit}}button{{padding:6px 9px;margin:2px}}button.active{{background:#1769aa;color:#fff}}button.reject.active{{background:#a92c2c}}textarea{{box-sizing:border-box;width:100%;min-height:58px}}.verdict{{background:#f7f9fb;border-radius:6px;padding:10px;margin-top:10px}}small,.muted{{color:#5a626d}}code{{background:#eef1f4;padding:2px 4px;border-radius:3px}}@media(max-width:980px){{.panels,.audio-grid{{grid-template-columns:1fr}}table{{font-size:13px}}}}
</style></head><body>
<header><strong>1.7B Split v3 + Inner · additive-overlay smoke5</strong>　<button id="save">保存全部裁决</button> <span id="status"></span></header>
<main><section class="help"><h2>审计目标：semantic core 与背景/噪声必须同时可听</h2><div class="panels">
<div class="panel"><h3>1 · 语义完整可懂</h3><p>先听 clean，再听 mixed。mixed 中所有绿色 semantic cores 的字句必须完整、清楚，不能被混音或拼接遮坏。</p></div>
<div class="panel"><h3>2 · overlay 合格</h3><p>mixed 中必须确实可听到橙色 overlay、强度不过弱也不过强，并且 overlay 本身不能含清楚可字幕词语。noise-only 只用于辨认来源。</p></div>
<div class="panel"><h3>3 · Semantic Split</h3><p>独立 cores 之间是 <code>cut</code>；单一 maximal core 内的 BGM switch 是 <code>continue</code>。overlay 的开始、循环或切换不产生语义 event。</p></div>
<div class="panel"><h3>4 · Inner edge</h3><p>蓝色是无 semantic speech 的可移除区，哪怕其中仍有 BGM/非语义声音；红色 overlap 必须 abstain。无 event 时自动 not_applicable。</p></div>
</div><p>每条 overlay 的 SNR 来自既有 hardmix 背景混音经验分布的分位数，不使用单一固定阈值。相邻 event 的试听上下文按 event 代表点中点自适应分区，互不重叠。人工 5/5 前禁止扩 100 条、跑 proposer 或训练。</p></section><div id="list"></div></main>
<script>
const rows={payload};
const key='semantic-split-multicore-additive-overlay-audit-v2:'+location.pathname;
const ann=JSON.parse(localStorage.getItem(key)||'{{}}');
let activeAudio=null;
let activeStop=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function ensure(row){{ann[row.sample_id]??={{intelligibility_verdict:'',overlay_audibility_verdict:'',split_verdict:'',inner_verdict:'',note:''}};return ann[row.sample_id];}}
function approved(row,a){{return a.intelligibility_verdict==='approve'&&a.overlay_audibility_verdict==='approve'&&a.split_verdict==='approve'&&(!row.inner_review_required||a.inner_verdict==='approve');}}
function persist(){{localStorage.setItem(key,JSON.stringify(ann));status();}}
function status(){{const done=rows.filter(row=>approved(row,ensure(row))).length;document.getElementById('status').textContent=`全部必要项通过 ${{done}} / ${{rows.length}}`;}}
function pct(value,duration){{return Math.max(0,Math.min(100,Number(value)*100/Math.max(Number(duration),.001)));}}
function span(start,end){{return `${{Number(start).toFixed(3)}}–${{Number(end).toFixed(3)}}s`;}}
function clearStop(){{if(activeAudio&&activeStop)activeAudio.removeEventListener('timeupdate',activeStop);activeStop=null;}}
function playSpan(id,start,end){{const audio=document.getElementById(id);if(activeAudio){{clearStop();if(activeAudio!==audio)activeAudio.pause();}}activeAudio=audio;const begin=()=>{{audio.currentTime=Number(start);activeStop=()=>{{if(audio.currentTime>=Number(end)){{clearStop();audio.pause();}}}};audio.addEventListener('timeupdate',activeStop);audio.play();}};if(audio.readyState<1){{audio.addEventListener('loadedmetadata',begin,{{once:true}});audio.load();}}else begin();}}
function verdict(label,field,value,a){{return `<button data-field="${{field}}" data-value="${{value}}" class="${{value==='reject'?'reject ':''}}${{a[field]===value?'active':''}}">${{label}}</button>`;}}
function timeline(row){{let bars='';for(const core of row.core_spans)bars+=`<div class="bar core" style="left:${{pct(core.start_s,row.duration_s)}}%;width:${{pct(core.end_s-core.start_s,row.duration_s)}}%" title="${{esc(core.core_id)}} ${{span(core.start_s,core.end_s)}}">${{esc(core.core_id)}}</div>`;for(const gap of row.gap_spans||[]){{const cls=gap.kind==='overlap'?'overlap':'safe';bars+=`<div class="bar ${{cls}}" style="left:${{pct(gap.start_s,row.duration_s)}}%;width:${{pct(gap.end_s-gap.start_s,row.duration_s)}}%" title="${{esc(gap.kind)}} ${{span(gap.start_s,gap.end_s)}}">${{esc(gap.kind)}}</div>`;}}bars+=`<div class="bar overlay" style="left:0;width:100%" title="additive overlay ${{span(row.overlay.start_s,row.overlay.end_s)}}">additive overlay（全程与 core 同时存在）</div>`;return `<div class="track">${{bars}}</div><small>绿=semantic core；蓝=safe；红=overlap abstain；橙=additive overlay。时长 ${{Number(row.duration_s).toFixed(3)}}s</small>`;}}
function eventContexts(row){{
  const events=[...row.semantic_events].sort(function(a,b){{return a.representative_s-b.representative_s;}});
  return events.map(function(event,index){{
    return {{
      event:event,
      start:index===0?0:(events[index-1].representative_s+event.representative_s)/2,
      end:index===events.length-1?row.duration_s:(event.representative_s+events[index+1].representative_s)/2
    }};
  }});
}}
function coreRows(row,ids){{
  return row.core_spans.map(function(core){{
    return `<tr><td>${{esc(core.core_id)}}</td><td>${{esc(core.text)}}</td><td>${{span(core.start_s,core.end_s)}}</td><td><button data-audio="${{ids.clean}}" data-play-start="${{core.start_s}}" data-play-end="${{core.end_s}}">clean core</button><button data-audio="${{ids.mixed}}" data-play-start="${{core.start_s}}" data-play-end="${{core.end_s}}">mixed core</button></td></tr>`;
  }}).join('');
}}
function eventRows(row,ids){{
  const contexts=eventContexts(row);
  if(!contexts.length)return '<tr><td colspan="6"><b>无 semantic event：</b>单一 maximal semantic core；BGM switch 全部 continue。</td></tr>';
  return contexts.map(function(item){{
    const event=item.event;
    const inner=event.inner_target;
    const paired=inner.status==='safe'?span(inner.left_speech_end_s,inner.right_speech_start_s):'保持单 chunk';
    return `<tr><td>${{esc(event.event_id)}}</td><td>${{esc(event.semantic_decision)}}</td><td>${{span(event.event_interval_start_s,event.event_interval_end_s)}}</td><td>${{esc(inner.status)}} / ${{esc(inner.gap_kind)}}</td><td>${{paired}}</td><td>${{span(item.start,item.end)}}<br><button data-audio="${{ids.clean}}" data-play-start="${{item.start}}" data-play-end="${{item.end}}">clean 上下文</button><button data-audio="${{ids.mixed}}" data-play-start="${{item.start}}" data-play-end="${{item.end}}">mixed 上下文</button></td></tr>`;
  }}).join('');
}}
function innerPanel(row,a){{
  if(!row.inner_review_required)return '<div class="panel"><h3>4 · Inner</h3><p><b>自动去重：</b>无 semantic event，Inner=not_applicable。</p></div>';
  return `<div class="panel"><h3>4 · Inner safe/abstain</h3><div class="verdict">${{verdict('Inner 通过','inner_verdict','approve',a)}}${{verdict('Inner 不通过','inner_verdict','reject',a)}}</div></div>`;
}}
function bindCard(card,a){{
  for(const button of card.querySelectorAll('[data-field]'))button.onclick=function(){{a[button.dataset.field]=button.dataset.value;a.updated_at=new Date().toISOString();persist();render();}};
  for(const button of card.querySelectorAll('[data-play-start]'))button.onclick=function(){{playSpan(button.dataset.audio,button.dataset.playStart,button.dataset.playEnd);}};
  card.querySelector('textarea').onchange=function(event){{a.note=event.target.value;a.updated_at=new Date().toISOString();persist();}};
  for(const audio of card.querySelectorAll('audio')){{
    audio.onplay=function(event){{if(activeAudio&&activeAudio!==event.target){{clearStop();activeAudio.pause();}}activeAudio=event.target;}};
    audio.onpause=function(){{clearStop();}};
  }}
}}
function render(){{
  const root=document.getElementById('list');
  root.innerHTML='';
  for(const row of rows){{
    const a=ensure(row);
    const card=document.createElement('article');
    if(approved(row,a))card.classList.add('done');
    const ids={{clean:'clean-'+row.sample_id,overlay:'overlay-'+row.sample_id,mixed:'mixed-'+row.sample_id}};
    const overlay=row.overlay;
    const sourceText=overlay.sources.map(function(source){{return `${{source.audio_id}} / ${{source.background_type}} / ${{source.source_partition}} / ${{Number(source.source_duration_s).toFixed(2)}}s${{source.tiled?'（循环）':''}}`;}}).join('；');
    card.innerHTML=`<h2>${{esc(row.sample_id)}}</h2><p><b>审计重点：</b>${{esc(row.audit_focus)}}</p><small>axes=${{esc(JSON.stringify(row.sampling_axes))}}；overlay=${{esc(sourceText)}}；target/achieved SNR=${{Number(overlay.mix.target_snr_db).toFixed(2)}}/${{Number(overlay.mix.achieved_snr_db).toFixed(2)}} dB</small><div class="audio-grid"><div><b>Clean composite</b><audio id="${{ids.clean}}" controls preload="metadata" src="${{esc(row.clean_audio)}}"></audio></div><div><b>Mixed（训练输入）</b><audio id="${{ids.mixed}}" controls preload="metadata" src="${{esc(row.mixed_audio)}}"></audio></div><div><b>Overlay-only（检查可听度与语义泄漏）</b><audio id="${{ids.overlay}}" controls preload="metadata" src="${{esc(row.overlay_audio)}}"></audio></div></div>${{timeline(row)}}<table><thead><tr><th>core</th><th>可信文本</th><th>精确 span</th><th>clean / mixed 对照</th></tr></thead><tbody>${{coreRows(row,ids)}}</tbody></table><h3>事件真值</h3><table><thead><tr><th>event</th><th>Split</th><th>event interval</th><th>Inner</th><th>paired edges</th><th>不重叠自适应上下文</th></tr></thead><tbody>${{eventRows(row,ids)}}</tbody></table><div class="panels"><div class="panel"><h3>1 · 语义完整可懂</h3><div class="verdict">${{verdict('语义通过','intelligibility_verdict','approve',a)}}${{verdict('语义不通过','intelligibility_verdict','reject',a)}}</div></div><div class="panel"><h3>2 · Overlay 合格</h3><div class="verdict">${{verdict('可听且无字幕语义','overlay_audibility_verdict','approve',a)}}${{verdict('不可听/过强/含清楚词语','overlay_audibility_verdict','reject',a)}}</div></div><div class="panel"><h3>3 · Semantic Split</h3><div class="verdict">${{verdict('Split 通过','split_verdict','approve',a)}}${{verdict('Split 不通过','split_verdict','reject',a)}}</div></div>${{innerPanel(row,a)}}</div><label><b>备注</b><textarea placeholder="指出语义遮蔽、overlay 强弱/语义泄漏、event、safe gap 或合成伪影">${{esc(a.note)}}</textarea></label>`;
    bindCard(card,a);
    root.appendChild(card);
  }}
  status();
}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(row=>{{const a=ensure(row);const inner=row.inner_review_required?(a.inner_verdict||'unreviewed'):'not_applicable';const rejected=a.intelligibility_verdict==='reject'||a.overlay_audibility_verdict==='reject'||a.split_verdict==='reject'||(row.inner_review_required&&a.inner_verdict==='reject');return JSON.stringify({{schema:'semantic_split_multicore_additive_overlay_manual_verdict_v2',sample_id:row.sample_id,intelligibility_verdict:a.intelligibility_verdict||'unreviewed',overlay_audibility_verdict:a.overlay_audibility_verdict||'unreviewed',split_verdict:a.split_verdict||'unreviewed',inner_verdict:inner,verdict:approved(row,a)?'approve':(rejected?'reject':'unreviewed'),note:a.note||'',semantic_events:row.semantic_events,overlay:row.overlay,updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};
render();
</script></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the fixed additive-overlay Split/Inner smoke audit."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(build_audit(manifest=Path(args.manifest), output_dir=Path(args.output_dir)))
