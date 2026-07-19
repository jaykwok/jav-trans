#!/usr/bin/env python3
"""Generate an exact-span listening page for Scorer v10 residuals."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_audit(*, selection: Path, output_dir: Path) -> Path:
    rows = _rows(selection)
    if not rows:
        raise ValueError("Scorer v10 prediction audit selection is empty")
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source = Path(str(row["audio"]))
        if not source.is_file():
            raise ValueError(f"Scorer v10 audit audio is missing: {source}")
        destination = audio_dir / f"item-{index:03d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        payload.append(
            {
                **row,
                "audit_id": f"{row['category']}:{row['source_id']}",
                "audio": destination.relative_to(output_dir).as_posix(),
            }
        )
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scorer v10 prediction audit</title>
<style>body{{margin:0;background:#f3f5f7;color:#20242a;font-family:Segoe UI,Arial,sans-serif}}header{{position:sticky;top:0;z-index:4;background:#fff;border-bottom:1px solid #c9d0d8;padding:12px 18px}}main{{max-width:1180px;margin:18px auto;padding:0 14px}}section,article{{background:#fff;border:1px solid #c9d0d8;border-radius:8px;padding:14px;margin-bottom:14px}}article.done{{border-left:6px solid #267443}}audio{{width:100%}}.track{{display:flex;min-height:40px;margin:8px 0;background:#e1e5e9}}.span{{min-width:3px;min-height:40px;margin:0;padding:7px 4px;border:0;border-right:1px solid rgba(0,0,0,.2);box-sizing:border-box;overflow:hidden;white-space:nowrap;font-size:11px;text-align:left;cursor:pointer}}.span.playing{{outline:3px solid #111;outline-offset:-3px;color:#fff}}.truth_speech{{background:#58aa70}}.truth_background{{background:#d0a14b}}.model_speech{{background:#417fc2;color:#fff}}button,textarea{{font:inherit}}button{{padding:6px 9px;margin:3px}}button.active{{background:#1769aa;color:#fff}}button.risk.active{{background:#a52f2f}}textarea{{width:100%;box-sizing:border-box;min-height:56px}}small{{color:#5c6570}}h3{{margin:8px 0 2px}}@media(max-width:700px){{.span{{font-size:9px}}}}</style></head><body>
<header><strong>1.7B Scorer v10 · prediction residual audit</strong> <button id="save">保存裁决</button> <span id="status"></span></header><main><section><b>精确播放：</b>绿色/黄色是真值 speech/background，蓝色是模型 argmax=speech。每个条只从自己的 start 播到 end 并立即停止；不添加任何上下文。此页包含全部 row-level true-speech deletion、val/test false-keep/edge hard case，以及所有预测 speech run &gt;8s residual。人工保存前 gate 保持 pending。</section><div id="list"></div></main>
<script>const rows={encoded};const key='scorer-v10-prediction-audit-v1:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let activeAudio=null,activeButton=null,activeStop=null;function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function ensure(r){{ann[r.audit_id]??={{verdict:'',note:''}};return ann[r.audit_id];}}function stopPlayback(){{if(activeAudio&&activeStop)activeAudio.removeEventListener('timeupdate',activeStop);if(activeAudio)activeAudio.pause();if(activeButton)activeButton.classList.remove('playing');activeAudio=null;activeButton=null;activeStop=null;}}function playExact(audio,button,start,end){{if(activeAudio===audio&&activeButton===button&&!audio.paused){{stopPlayback();return;}}stopPlayback();activeAudio=audio;activeButton=button;button.classList.add('playing');const begin=()=>{{audio.currentTime=start;activeStop=()=>{{if(audio.currentTime>=end)stopPlayback();}};audio.addEventListener('timeupdate',activeStop);audio.play();}};if(audio.readyState<1){{audio.addEventListener('loadedmetadata',begin,{{once:true}});audio.load();}}else begin();}}function choices(r){{if(r.category==='background_false_keep')return [['model_false_keep','应全段 drop'],['contains_target_speech','真值可能错：含目标对白'],['unsure','不确定']];if(r.category==='speech_deletion')return [['true_speech_deleted','真语音被整段删'],['teacher_span_wrong','真值 span 错'],['acceptable_nonsemantic','实际可 drop'],['unsure','不确定']];if(r.category==='long_residual')return [['acceptable_long_residual','长段可接受'],['missed_background_or_gap','含应分离背景/停顿'],['edge_clipped','有真语音截断'],['unsure','不确定']];return [['edge_correct','边缘可接受'],['edge_clipped','真语音被截'],['teacher_span_too_wide','真值过宽'],['unsure','不确定']];}}function spans(r,list){{return list.map(s=>`<button type="button" class="span ${{s.label}}" style="width:${{Math.max(.3,100*(s.end_s-s.start_s)/r.duration_s)}}%" data-start="${{s.start_s}}" data-end="${{s.end_s}}">${{esc(s.label)}} ${{Number(s.start_s).toFixed(2)}}–${{Number(s.end_s).toFixed(2)}}s</button>`).join('');}}function persist(){{localStorage.setItem(key,JSON.stringify(ann));render();}}function render(){{stopPlayback();const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r),card=document.createElement('article');if(a.verdict)card.classList.add('done');card.innerHTML=`<h2>${{esc(r.source_id)}}</h2><small>${{esc(r.partition)}} / ${{esc(r.row_role)}} / ${{esc(r.category)}} / duration=${{Number(r.duration_s).toFixed(2)}}s / FN=${{r.false_negative_frames}} / FP=${{r.false_positive_frames}} / max model run=${{Number(r.max_predicted_speech_run_s).toFixed(2)}}s</small><audio controls preload="metadata" src="${{esc(r.audio)}}"></audio><h3>真值（绿色 speech / 黄色 background）</h3><div class="track">${{spans(r,r.truth_spans)}}</div><h3>模型 argmax=speech（蓝色）</h3><div class="track">${{spans(r,r.prediction_spans)}}</div><div>${{choices(r).map(v=>`<button data-v="${{v[0]}}" class="${{v[0].includes('clipped')||v[0].includes('false')?'risk ':''}}${{a.verdict===v[0]?'active':''}}">${{v[1]}}</button>`).join('')}}</div><textarea placeholder="记录精确时间和原因">${{esc(a.note)}}</textarea>`;const audio=card.querySelector('audio');card.querySelectorAll('[data-start]').forEach(b=>b.onclick=()=>playExact(audio,b,Number(b.dataset.start),Number(b.dataset.end)));audio.addEventListener('ended',stopPlayback);card.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{{a.verdict=b.dataset.v;a.updated_at=new Date().toISOString();persist();}});card.querySelector('textarea').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};root.appendChild(card);}}document.getElementById('status').textContent=`已裁决 ${{rows.filter(r=>ensure(r).verdict).length}}/${{rows.length}}`;}}document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r);return JSON.stringify({{schema:'speech_scorer_v10_prediction_manual_verdict_v1',audit_id:r.audit_id,source_id:r.source_id,partition:r.partition,row_role:r.row_role,category:r.category,verdict:a.verdict||'unreviewed',note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();</script></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    manifest = output_dir / "audit_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload),
        encoding="utf-8",
    )
    summary = {
        "schema": "speech_scorer_v10_prediction_audit_summary_v1",
        "title": "Scorer v10 prediction residual audit",
        "review_item_count": len(payload),
        "category_counts": dict(Counter(str(row["category"]) for row in payload)),
        "selection": str(selection),
        "audit_manifest": str(manifest),
        "manual_gate_status": "pending",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_audit_entrypoints(latest_html=index, title=summary["title"])
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(build_audit(selection=Path(args.selection), output_dir=Path(args.output_dir)))
