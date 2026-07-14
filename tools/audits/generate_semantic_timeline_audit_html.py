#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_audit(*, labels: Path, output_dir: Path, update_latest: bool = True) -> Path:
    rows = _rows(labels)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload_rows = []
    for row in rows:
        source = Path(row["audio"])
        destination_dir = audio_dir / str(row["sample_id"])
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"full{source.suffix}"
        shutil.copy2(source, destination)
        payload_rows.append(
            {**row, "audio": destination.relative_to(output_dir).as_posix()}
        )

    payload = json.dumps(payload_rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic Timeline Training-label Smoke</title>
<style>
body{{margin:0;background:#f2f4f7;color:#20242a;font-family:Segoe UI,Arial,sans-serif}}header{{position:sticky;top:0;z-index:5;padding:12px 18px;background:#fff;border-bottom:1px solid #cbd2da}}main{{max-width:1180px;margin:18px auto;padding:0 14px}}article,.help{{background:#fff;border:1px solid #cbd2da;border-radius:8px;padding:16px;margin-bottom:16px}}article.done{{border-left:6px solid #1b7a3a}}.help{{border-left:6px solid #1769aa}}.views{{display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:10px}}.panel{{border:1px solid #d6dce3;border-radius:7px;padding:12px;margin-top:10px}}.units{{display:flex;flex-wrap:wrap;gap:6px;padding:10px;background:#f7f9fb;border-radius:6px}}.unit{{padding:5px 8px;border-radius:5px;border:1px solid #cbd2da}}.semantic{{background:#d9f2df}}.nonsemantic{{background:#eceff2}}.unsure{{background:#fff0c8}}audio{{width:100%}}.track{{position:relative;height:56px;background:#e7e9ec;border-radius:6px;overflow:hidden;margin:10px 0}}.bar{{position:absolute;box-sizing:border-box;border:1px solid rgba(0,0,0,.25);overflow:hidden;font-size:12px;padding:3px}}.meaning{{background:#55b56d;top:8px;height:32px;z-index:2}}.uncertain{{background:#f0bd54;top:4px;height:44px;z-index:3}}.event{{position:absolute;top:0;height:56px;width:2px;background:#b3261e;z-index:4}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d6dce3;padding:7px;text-align:left;vertical-align:top}}button,textarea{{font:inherit}}button{{padding:6px 9px;margin:2px}}button.active{{background:#1769aa;color:#fff}}button.reject.active{{background:#a92c2c}}textarea{{box-sizing:border-box;width:100%;min-height:58px}}small,.muted{{color:#5a626d}}code{{background:#eef1f4;padding:2px 4px;border-radius:3px}}@media(max-width:800px){{.views{{grid-template-columns:1fr}}table{{font-size:13px}}}}
</style></head><body>
<header><strong>1.7B Semantic Timeline · 训练标签 smoke</strong>　<button id="save">保存全部裁决</button> <span id="status"></span></header>
<main><section class="help"><h2>这不是 atomic-core 审计</h2><p>每条保留完整可信 Galgame source。人工只确认两项：文本是否拆成正确的最小完整语义单元，以及每个单元的 semantic timeline 是否正确。下面的 Scorer / Outer / Split / Inner 卡片是同一标签的确定性训练路由，不要求你重复裁决。</p><div class="views"><div><b>Scorer</b><br>绿色 semantic span 为正标签；可靠差集只作 frame-level nonsemantic 负标签；membership envelope 只负责高召回粗 island。</div><div><b>Outer</b><br>最早 semantic start 与最晚 semantic end 是成对 outer target，前后 BGM、喘息、呻吟、亲吻声、背景人声应在外。</div><div><b>Split</b><br>相邻 semantic units 形成有序 event，再投影到真实 proposer candidates；这里不把 Omni 秒数当最终 cut。</div><div><b>Inner</b><br>本页不给 safe 标签。每个 semantic event 仍需单独的 candidate safe-zone teacher；重叠或无安全区时 abstain。</div></div><p><b>CueQC 不在本页：</b>必须等新边界链实际导出 chunk 后再逐 chunk 独立标注。</p></section><div id="list"></div></main>
<script>
const rows={payload};
const key='semantic-timeline-training-label-audit-v1:'+location.pathname;
const ann=JSON.parse(localStorage.getItem(key)||'{{}}');
let activeAudio=null;let activeStop=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function ensure(row){{ann[row.sample_id]??={{units_verdict:'',timeline_verdict:'',note:''}};return ann[row.sample_id];}}
function approved(a){{return a.units_verdict==='approve'&&a.timeline_verdict==='approve';}}
function persist(){{localStorage.setItem(key,JSON.stringify(ann));status();}}
function status(){{document.getElementById('status').textContent=`训练标签通过 ${{rows.filter(r=>approved(ensure(r))).length}} / ${{rows.length}}`;}}
function pct(value,duration){{return Math.max(0,Math.min(100,Number(value)*100/Math.max(Number(duration),.001)));}}
function spanLabel(start,end){{return `${{Number(start).toFixed(3)}}–${{Number(end).toFixed(3)}}s`;}}
function clearStop(){{if(activeAudio&&activeStop)activeAudio.removeEventListener('timeupdate',activeStop);activeStop=null;}}
function playSpan(id,start,end){{const audio=document.getElementById(id);if(activeAudio){{clearStop();if(activeAudio!==audio)activeAudio.pause();}}activeAudio=audio;const begin=()=>{{audio.currentTime=Number(start);activeStop=()=>{{if(audio.currentTime>=Number(end)){{clearStop();audio.pause();}}}};audio.addEventListener('timeupdate',activeStop);audio.play();}};if(audio.readyState<1){{audio.addEventListener('loadedmetadata',begin,{{once:true}});audio.load();}}else begin();}}
function verdict(label,field,value,a){{return `<button data-field="${{field}}" data-value="${{value}}" class="${{value==='reject'?'reject ':''}}${{a[field]===value?'active':''}}">${{label}}</button>`;}}
function timeline(row){{let bars='';for(const item of row.semantic_timeline||[])bars+=`<div class="bar meaning" style="left:${{pct(item.start_s,row.duration_s)}}%;width:${{pct(item.end_s-item.start_s,row.duration_s)}}%" title="${{esc(item.unit_id)}} ${{spanLabel(item.start_s,item.end_s)}}">${{esc(item.unit_id)}}</div>`;for(const item of row.unsure_audio_spans||[])bars+=`<div class="bar uncertain" style="left:${{pct(item.start_s,row.duration_s)}}%;width:${{pct(item.end_s-item.start_s,row.duration_s)}}%" title="unsure ${{spanLabel(item.start_s,item.end_s)}}">?</div>`;for(const event of row.semantic_events||[]){{if(event.status==='matched'){{const x=(Number(event.interval_start_s)+Number(event.interval_end_s))/2;bars+=`<div class="event" style="left:${{pct(x,row.duration_s)}}%" title="${{esc(event.event_id)}} semantic event"></div>`;}}}}return `<div class="track">${{bars}}</div><small>绿=semantic timeline；红线=语义 event 的区间中点示意，不是最终 cut；黄=unsure</small>`;}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const row of rows){{const a=ensure(row);const card=document.createElement('article');if(approved(a))card.classList.add('done');const audioId='audio-'+row.sample_id;const units=row.text_units.map(u=>`<span class="unit ${{esc(u.kind)}}" title="${{esc(u.reason)}}"><b>${{esc(u.unit_id)}}</b> ${{esc(u.text)}} <small>${{esc(u.kind)}} ${{Number(u.confidence).toFixed(2)}}</small></span>`).join('');const alignments=row.semantic_alignments.length?row.semantic_alignments.map(x=>`<tr><td>${{esc(x.unit_id)}}</td><td>${{esc(x.status)}}</td><td>${{x.status==='matched'?spanLabel(x.start_s,x.end_s):'—'}}</td><td>${{x.status==='matched'?`<button data-play-start="${{x.start_s}}" data-play-end="${{x.end_s}}">播放此 semantic unit</button>`:''}}</td><td>${{Number(x.confidence).toFixed(2)}} · ${{esc(x.reason)}}</td></tr>`).join(''):'<tr><td colspan="5">没有 semantic unit</td></tr>';const events=(row.semantic_events||[]).map(e=>`<tr><td>${{esc(e.event_id)}}</td><td>${{esc(e.left_unit_id)}} → ${{esc(e.right_unit_id)}}</td><td>${{esc(e.status)}}</td><td>${{e.status==='matched'?spanLabel(e.interval_start_s,e.interval_end_s):'—'}}</td><td>${{e.overlap===true?'overlap → Inner abstain':e.overlap===false?'待 safe-zone teacher':'unsure'}}</td></tr>`).join('')||'<tr><td colspan="5">没有内部 semantic event；仍可用于 Scorer / Outer，不作为 Split 正例。</td></tr>';const membership=row.scorer_view.source_membership;card.innerHTML=`<h2>${{esc(row.sample_id)}}</h2><p><b>参考文本：</b>${{esc(row.reference_text)}}</p><small>来源：${{esc(row.source)}}；模型：${{esc(row.model)}}；时长 ${{Number(row.duration_s).toFixed(3)}}s</small><p><b>完整原音频</b></p><audio id="${{esc(audioId)}}" controls preload="metadata" src="${{esc(row.audio)}}"></audio>${{timeline(row)}}<section class="panel"><h3>1 · 最小完整语义单元</h3><div class="units">${{units}}</div><p>${{verdict('语义单元正确','units_verdict','approve',a)}}${{verdict('语义单元错误','units_verdict','reject',a)}}</p></section><section class="panel"><h3>2 · Semantic timeline</h3><table><thead><tr><th>unit</th><th>状态</th><th>区间</th><th>试听</th><th>理由</th></tr></thead><tbody>${{alignments}}</tbody></table><p>${{verdict('时间轴正确','timeline_verdict','approve',a)}}${{verdict('时间轴错误','timeline_verdict','reject',a)}}</p></section><section class="panel"><h3>派生训练路由（只读）</h3><p><b>Scorer membership：</b>${{membership.status==='matched'?spanLabel(membership.start_s,membership.end_s):esc(membership.status)}}；<b>frame negatives：</b>${{row.scorer_view.nonsemantic_complement_spans.length}} 段。</p><p><b>Outer target：</b>${{row.outer_refiner_view.status==='matched'?spanLabel(row.outer_refiner_view.left_speech_start_s,row.outer_refiner_view.right_speech_end_s):esc(row.outer_refiner_view.status)}}</p><table><thead><tr><th>event</th><th>语义关系</th><th>状态</th><th>event interval</th><th>Inner 路由</th></tr></thead><tbody>${{events}}</tbody></table></section><label><b>备注</b><textarea placeholder="指出具体 unit 或时间轴问题">${{esc(a.note)}}</textarea></label>`;card.querySelectorAll('[data-field]').forEach(button=>button.onclick=()=>{{a[button.dataset.field]=button.dataset.value;a.updated_at=new Date().toISOString();persist();render();}});card.querySelectorAll('[data-play-start]').forEach(button=>button.onclick=()=>playSpan(audioId,button.dataset.playStart,button.dataset.playEnd));card.querySelector('textarea').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};const audio=card.querySelector('audio');audio.onplay=e=>{{if(activeAudio&&activeAudio!==e.target){{clearStop();activeAudio.pause();}}activeAudio=e.target;}};audio.onpause=()=>clearStop();root.appendChild(card);}}status();}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(row=>{{const a=ensure(row);const rejected=a.units_verdict==='reject'||a.timeline_verdict==='reject';return JSON.stringify({{schema:'semantic_timeline_manual_verdict_v1',sample_id:row.sample_id,units_verdict:a.units_verdict||'unreviewed',timeline_verdict:a.timeline_verdict||'unreviewed',verdict:approved(a)?'approve':(rejected?'reject':'unreviewed'),note:a.note||'',text_units:row.text_units,semantic_alignments:row.semantic_alignments,semantic_events:row.semantic_events,updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};
render();
</script></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    if update_latest:
        update_audit_entrypoints(
            latest_html=index, title="Semantic Timeline Training-label Smoke"
        )
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a semantic-timeline audit focused on trainable labels."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        build_audit(
            labels=Path(args.labels),
            output_dir=Path(args.output_dir),
            update_latest=not args.no_update_latest,
        )
    )
