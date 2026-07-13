#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_audit(*, labels: Path, output_dir: Path) -> Path:
    rows = _rows(labels)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload_rows = []
    for row in rows:
        source = Path(row["audio"])
        destination = audio_dir / source.name
        shutil.copy2(source, destination)
        payload_rows.append(
            {**row, "audio": destination.relative_to(output_dir).as_posix()}
        )
    payload = json.dumps(payload_rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Semantic Speech / Outer Smoke</title>
<style>body{{margin:0;background:#f3f5f7;color:#20242a;font-family:Segoe UI,Arial,sans-serif}}header{{position:sticky;top:0;z-index:3;background:#fff;border-bottom:1px solid #ccd2d9;padding:12px 18px}}main{{max-width:1100px;margin:18px auto;padding:0 14px}}.help,article{{background:#fff;border:1px solid #ccd2d9;border-radius:7px;padding:16px;margin-bottom:16px}}.help{{border-left:5px solid #1769aa}}audio{{width:100%}}.legend{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:8px}}.legend div,.segment{{border:1px solid #d8dde3;border-radius:6px;padding:10px}}.timeline{{height:46px;position:relative;background:#e7eaee;border-radius:5px;margin:12px 0;overflow:hidden}}.bar{{position:absolute;top:0;height:100%;border-right:1px solid #fff;display:flex;align-items:center;justify-content:center;font-size:11px;overflow:hidden}}.semantic_target{{background:#8bd49d}}.discardable{{background:#c7ccd2}}.unsure{{background:#f4c66b}}.segments{{display:grid;gap:8px}}.segment{{display:grid;grid-template-columns:95px 95px 180px 1fr auto;gap:8px;align-items:center}}input,select,button{{font:inherit;padding:6px}}input.reason{{width:95%}}.verdict{{margin-top:12px;padding:10px;background:#f6f8fa}}button.active{{background:#1769aa;color:#fff}}article.done{{border-left:5px solid #197642}}small{{color:#59616c}}</style></head><body><header><strong>1.7B Semantic Speech / Full-island Outer · 5 样本人工审计</strong>　<button id="save">保存全部裁决</button> <span id="status"></span></header><main>
<section class="help"><h2>你要核对什么</h2><p><strong>彩色长条是时间轴，不是进度条；灰色播放器才是音频播放器。</strong></p><ol><li>完整听一遍音频。</li><li>核对每个区间的起止时间与三类标签；可直接修改数字和下拉框。</li><li>绿色只允许清楚、可辨、值得字幕化的前景语言。喘息、呻吟、亲吻声、笑声、短促拟声、BGM、噪声、远处不可辨背景人声都应是灰色。</li><li>疑似有词但听不清，或和语义语音重叠无法分离，必须橙色 unsure。</li><li>修改后选择“通过”或“仍不通过”。5/5 通过前不会扩大数据或训练。</li></ol><div class="legend"><div class="semantic_target"><b>semantic_target</b><br>清楚可辨、有字幕价值的前景语言</div><div class="discardable"><b>discardable</b><br>BGM/噪声/非语言 vocalization/不可辨背景人声</div><div class="unsure"><b>unsure</b><br>疑似词语、重叠或无法可靠判断</div></div></section><div id="list"></div></main>
<script>const rows={payload};const key='semantic-speech-outer-smoke5-v1';const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let active=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function ensure(row){{ann[row.sample_id]??={{segments:row.segments.map(x=>({{...x}})),verdict:'',note:''}};return ann[row.sample_id];}}
function persist(){{localStorage.setItem(key,JSON.stringify(ann));status();}}
function status(){{const done=rows.filter(r=>ensure(r).verdict==='approve').length;document.getElementById('status').textContent=`人工通过 ${{done}} / ${{rows.length}}`;}}
function play(audio,start,end){{if(active&&active!==audio)active.pause();active=audio;audio.currentTime=Math.max(0,Number(start));const stopAt=Number(end);const fn=()=>{{if(audio.currentTime>=stopAt){{audio.pause();audio.removeEventListener('timeupdate',fn);}}}};audio.addEventListener('timeupdate',fn);audio.play();}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const row of rows){{const a=ensure(row);const card=document.createElement('article');if(a.verdict==='approve')card.classList.add('done');card.innerHTML=`<h3>${{esc(row.sample_id)}}</h3><p><b>审计重点：</b>${{esc(row.audit_focus)}}</p><small>来源：${{esc(row.source)}}；时长 ${{Number(row.duration_s).toFixed(3)}}s</small><audio controls preload="metadata" src="${{esc(row.audio)}}"></audio><div class="timeline">${{a.segments.map((s,i)=>`<div class="bar ${{esc(s.label)}}" style="left:${{100*Number(s.start_s)/row.duration_s}}%;width:${{100*(Number(s.end_s)-Number(s.start_s))/row.duration_s}}%">${{i+1}} ${{esc(s.label)}}</div>`).join('')}}</div><div class="segments">${{a.segments.map((s,i)=>`<div class="segment" data-index="${{i}}"><input data-field="start_s" type="number" step="0.01" value="${{s.start_s}}"><input data-field="end_s" type="number" step="0.01" value="${{s.end_s}}"><select data-field="label">${{['semantic_target','discardable','unsure'].map(v=>`<option ${{s.label===v?'selected':''}}>${{v}}</option>`).join('')}}</select><input class="reason" data-field="reason" value="${{esc(s.reason)}}"><button class="play">▶ 仅此区间</button></div>`).join('')}}</div><div class="verdict"><b>本样本：</b> <button data-verdict="approve" class="${{a.verdict==='approve'?'active':''}}">通过（含我的修正）</button> <button data-verdict="reject" class="${{a.verdict==='reject'?'active':''}}">仍不通过</button><br><input class="reason note" placeholder="备注" value="${{esc(a.note)}}"></div>`;const audio=card.querySelector('audio');card.querySelectorAll('.segment').forEach(el=>{{const i=Number(el.dataset.index);el.querySelectorAll('[data-field]').forEach(input=>input.onchange=()=>{{const f=input.dataset.field;a.segments[i][f]=f==='start_s'||f==='end_s'?Number(input.value):input.value;persist();render();}});el.querySelector('.play').onclick=()=>play(audio,a.segments[i].start_s,a.segments[i].end_s);}});card.querySelectorAll('[data-verdict]').forEach(b=>b.onclick=()=>{{a.verdict=b.dataset.verdict;a.updated_at=new Date().toISOString();persist();render();}});card.querySelector('.note').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};root.appendChild(card);}}status();}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(row=>{{const a=ensure(row);return JSON.stringify({{schema:'semantic_speech_outer_manual_verdict_v1',sample_id:row.sample_id,audio:row.audio,duration_s:row.duration_s,segments:a.segments,verdict:a.verdict||'unreviewed',note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();</script></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate semantic speech / Outer audit.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(build_audit(labels=Path(args.labels), output_dir=Path(args.output_dir)))
