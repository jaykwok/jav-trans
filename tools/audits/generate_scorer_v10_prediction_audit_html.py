#!/usr/bin/env python3
"""Generate an exact-span listening page for Scorer v10 residuals."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


FRAME_HOP_S = 0.02
SUMMARY_SCHEMA = "speech_scorer_v10_prediction_audit_summary_v2"
VERDICT_SCHEMA = "speech_scorer_v10_prediction_manual_verdict_v2"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def truth_drop_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return exact canonical-speech intervals where binary argmax is background."""

    def frame(span: dict[str, Any], side: str) -> int:
        key = f"{side}_frame"
        if key in span:
            return int(span[key])
        return round(float(span[f"{side}_s"]) / FRAME_HOP_S)

    predictions = sorted(
        (
            (frame(span, "start"), frame(span, "end"))
            for span in row.get("prediction_spans", [])
            if str(span.get("label") or "") == "model_speech"
        ),
        key=lambda item: item,
    )
    dropped: list[dict[str, Any]] = []
    for truth in row.get("truth_spans", []):
        if str(truth.get("label") or "") != "truth_speech":
            continue
        truth_start = frame(truth, "start")
        truth_end = frame(truth, "end")
        cursor = truth_start
        for predicted_start, predicted_end in predictions:
            if predicted_end <= cursor or predicted_start >= truth_end:
                continue
            if predicted_start > cursor:
                end = min(predicted_start, truth_end)
                dropped.append(
                    {
                        "label": "truth_speech_model_background",
                        "start_frame": cursor,
                        "end_frame": end,
                        "start_s": cursor * FRAME_HOP_S,
                        "end_s": end * FRAME_HOP_S,
                    }
                )
            cursor = max(cursor, min(predicted_end, truth_end))
            if cursor >= truth_end:
                break
        if cursor < truth_end:
            dropped.append(
                {
                    "label": "truth_speech_model_background",
                    "start_frame": cursor,
                    "end_frame": truth_end,
                    "start_s": cursor * FRAME_HOP_S,
                    "end_s": truth_end * FRAME_HOP_S,
                }
            )
    return dropped


def _render_page(payload: list[dict[str, Any]]) -> str:
    encoded = (
        json.dumps(payload, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    schema = json.dumps(VERDICT_SCHEMA)
    return (
        """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scorer v10 prediction residual audit</title>
<style>
:root{color-scheme:light;--border:#c9d0d8;--text:#20242a;--muted:#5c6570;--ok:#267443;--risk:#a52f2f;--speech:#417fc2;--drop:#b53a3a}
*{box-sizing:border-box}
body{margin:0;background:#f3f5f7;color:var(--text);font-family:Segoe UI,Arial,sans-serif}
header{position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:8px;flex-wrap:wrap;background:#fff;border-bottom:1px solid var(--border);padding:10px 18px}
header strong{margin-right:auto}
main{max-width:1180px;margin:18px auto;padding:0 14px}
section,article{background:#fff;border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:14px}
article.done{border-left:6px solid var(--ok)}
audio{width:100%;margin:6px 0}
.guide{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:10px}
.guide div{border-left:4px solid var(--border);padding:7px 9px;background:#f7f8fa}
.track{position:relative;min-height:42px;margin:8px 0;background:#e1e5e9;overflow:hidden}
.span{position:absolute;top:0;bottom:0;min-width:3px;min-height:42px;margin:0;padding:7px 4px;border:0;border-right:1px solid rgba(0,0,0,.2);box-sizing:border-box;overflow:hidden;white-space:nowrap;font-size:11px;text-align:left;cursor:pointer}
.span.playing{outline:3px solid #111;outline-offset:-3px;color:#fff}
.truth_speech{background:#58aa70}.truth_background{background:#d0a14b}.model_speech{background:var(--speech);color:#fff}.truth_speech_model_background{background:var(--drop);color:#fff}
button,select{font:inherit}button,select{padding:6px 9px}button{margin:3px;border:1px solid #69737e;border-radius:5px;background:#fff;cursor:pointer}button.active{background:#1769aa;color:#fff}button.risk.active{background:var(--risk);color:#fff}
small{color:var(--muted)}h2{font-size:18px;margin:0 0 4px;overflow-wrap:anywhere}h3{margin:10px 0 2px}.empty{color:var(--muted);padding:10px}
@media(max-width:760px){.guide{grid-template-columns:1fr}header strong{width:100%}.span{font-size:9px}}
</style>
</head>
<body>
<header>
  <strong>1.7B Scorer v10 · prediction residual audit</strong>
  <label>类别 <select id="filter"></select></label>
  <button id="next" type="button">下一个未裁决</button>
  <button id="stop" type="button">停止播放</button>
  <button id="save" type="button">保存裁决</button>
  <span id="status"></span>
</header>
<main>
  <section>
    <div><b>实际工作流：</b>蓝色是 Scorer 二分类 argmax=speech，会成为独立 downstream island；不做 threshold、gap merge 或时长规则。红色是 canonical truth_speech 中被模型判为 background、实际不会送出的精确区间。绿色/黄色是完整 canonical speech/background，整条 source 播放器只用于判断上下文。</div>
    <div><b>播放合同：</b>每个色条只播放自身 start–end 后立即停止，不添加上下文。点击原生播放器可听整条 source。只需选择按钮，不要求备注。</div>
    <div class="guide">
      <div><b>speech_deletion</b>：优先听红条；判断是真语音被整段删除，还是 canonical 应改为 background。</div>
      <div><b>speech_edge_or_partial</b>：优先听红条；判断是否截掉真语音/尾音，还是标签边缘过宽。</div>
      <div><b>long_residual</b>：听蓝条并用整条 source 判断；确认是否含应独立 drop 的背景/非语义段。</div>
      <div><b>background_false_keep</b>：听蓝条；判断模型是否误留背景，或 all-background canonical 漏了目标语音。</div>
    </div>
  </section>
  <div id="list"></div>
</main>
<script>
const rows=__ROWS__;
const verdictSchema=__VERDICT_SCHEMA__;
const key='scorer-v10-prediction-audit-v2:'+location.pathname;
let ann={};
try{ann=JSON.parse(localStorage.getItem(key)||'{}');}catch(_error){ann={};}
let activeAudio=null,activeButton=null,activeCheck=null,activeLoadHandler=null,activeTimer=null,activeFrame=null;
function esc(value){return String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));}
function ensure(row){ann[row.audit_id]??={verdict:''};return ann[row.audit_id];}
function stopPlayback(){
  if(activeAudio&&activeCheck){activeAudio.removeEventListener('timeupdate',activeCheck);activeAudio.removeEventListener('ended',activeCheck);}
  if(activeAudio&&activeLoadHandler) activeAudio.removeEventListener('loadedmetadata',activeLoadHandler);
  if(activeTimer!==null) clearTimeout(activeTimer);
  if(activeFrame!==null) cancelAnimationFrame(activeFrame);
  if(activeAudio) activeAudio.pause();
  if(activeButton) activeButton.classList.remove('playing');
  activeAudio=null;activeButton=null;activeCheck=null;activeLoadHandler=null;activeTimer=null;activeFrame=null;
}
function playExact(audio,button,start,end){
  if(activeAudio===audio&&activeButton===button&&!audio.paused){stopPlayback();return;}
  stopPlayback();activeAudio=audio;activeButton=button;button.classList.add('playing');
  const begin=async()=>{
    activeLoadHandler=null;
    if(activeAudio!==audio||activeButton!==button)return;
    audio.currentTime=start;
    activeCheck=()=>{if(audio.currentTime>=end||audio.ended)stopPlayback();};
    audio.addEventListener('timeupdate',activeCheck);audio.addEventListener('ended',activeCheck);
    const watch=()=>{if(activeAudio!==audio)return;if(audio.currentTime>=end){stopPlayback();return;}activeFrame=requestAnimationFrame(watch);};
    try{
      await audio.play();
      if(activeAudio!==audio){audio.pause();return;}
      activeFrame=requestAnimationFrame(watch);
      activeTimer=setTimeout(stopPlayback,Math.max(100,(end-start)*1000+120));
    }catch(error){stopPlayback();document.getElementById('status').textContent='播放失败: '+error.message;}
  };
  if(audio.readyState<1){activeLoadHandler=begin;audio.addEventListener('loadedmetadata',begin,{once:true});audio.load();}else begin();
}
function choices(row){
  if(row.category==='background_false_keep')return [['model_false_keep','确实全是背景/非语义，模型误留'],['canonical_contains_target_speech','含目标语音，canonical all-background 错'],['unsure','不确定']];
  if(row.category==='speech_deletion')return [['true_speech_deleted','红段是真语音，模型整段删除'],['canonical_should_be_background','红段可 drop，canonical 应为 background'],['unsure','不确定']];
  if(row.category==='long_residual')return [['acceptable_long_residual','蓝色长段整体可保留'],['missed_background_or_gap','蓝段内含应独立 drop 的背景/非语义'],['true_speech_edge_clipped','实际仍有真语音边缘被截'],['unsure','不确定']];
  return [['true_speech_clipped','红段含真语音/尾音，模型截断'],['canonical_should_be_background','红段可 drop，canonical 边缘过宽'],['unsure','不确定']];
}
function spans(row,list){return [...(list||[])].sort((a,b)=>a.start_s-b.start_s||a.end_s-b.end_s).map(span=>`<button type="button" class="span ${esc(span.label)}" style="left:${Math.max(0,100*span.start_s/row.duration_s)}%;width:${Math.max(.3,100*(span.end_s-span.start_s)/row.duration_s)}%" data-start="${span.start_s}" data-end="${span.end_s}">${esc(span.label)} ${Number(span.start_s).toFixed(2)}–${Number(span.end_s).toFixed(2)}s</button>`).join('');}
function saveLocal(){localStorage.setItem(key,JSON.stringify(ann));}
function updateStatus(){document.getElementById('status').textContent=`已裁决 ${rows.filter(row=>ensure(row).verdict).length}/${rows.length}`;}
function riskVerdict(value){return ['model_false_keep','true_speech_deleted','true_speech_clipped','true_speech_edge_clipped','missed_background_or_gap'].includes(value);}
function setVerdict(card,row,value){
  const state=ensure(row);state.verdict=value;state.updated_at=new Date().toISOString();saveLocal();
  card.classList.add('done');
  card.querySelectorAll('[data-v]').forEach(button=>button.classList.toggle('active',button.dataset.v===value));
  updateStatus();
}
function render(){
  stopPlayback();const root=document.getElementById('list');root.innerHTML='';const filter=document.getElementById('filter').value;
  for(const row of rows){
    if(filter!=='all'&&row.category!==filter)continue;
    const state=ensure(row),card=document.createElement('article');card.dataset.auditId=row.audit_id;if(state.verdict)card.classList.add('done');
    const dropped=(row.truth_drop_spans||[]).length?`<h3>实际未送出（红色：truth_speech ∩ model_background）</h3><div class="track">${spans(row,row.truth_drop_spans)}</div>`:'';
    card.innerHTML=`<h2>${esc(row.source_id)}</h2><small>${esc(row.partition)} / ${esc(row.row_role)} / ${esc(row.category)} / duration=${Number(row.duration_s).toFixed(2)}s / FN=${row.false_negative_frames} / FP=${row.false_positive_frames} / max model run=${Number(row.max_predicted_speech_run_s).toFixed(2)}s</small><h3>整条 source（仅供完整上下文）</h3><audio controls preload="none" src="${esc(row.audio)}"></audio><h3>canonical（绿色 speech / 黄色 background）</h3><div class="track">${spans(row,row.truth_spans)}</div>${dropped}<h3>实际候选工作流输出（蓝色 argmax=speech）</h3><div class="track">${spans(row,row.prediction_spans)}</div><div class="choices">${choices(row).map(choice=>`<button type="button" data-v="${choice[0]}" class="${riskVerdict(choice[0])?'risk ':''}${state.verdict===choice[0]?'active':''}">${choice[1]}</button>`).join('')}</div>`;
    const audio=card.querySelector('audio');
    audio.addEventListener('play',()=>{if(activeAudio&&activeAudio!==audio)stopPlayback();activeAudio=audio;});
    audio.addEventListener('ended',stopPlayback);
    card.querySelectorAll('[data-start]').forEach(button=>button.onclick=()=>playExact(audio,button,Number(button.dataset.start),Number(button.dataset.end)));
    card.querySelectorAll('[data-v]').forEach(button=>button.onclick=()=>setVerdict(card,row,button.dataset.v));
    root.appendChild(card);
  }
  if(!root.children.length)root.innerHTML='<div class="empty">当前筛选没有项目。</div>';
  updateStatus();
}
const categories=[['all','全部类别'],['speech_deletion','整段删除'],['speech_edge_or_partial','边缘/部分漏掉'],['long_residual','>8s residual'],['background_false_keep','背景误留']];
document.getElementById('filter').innerHTML=categories.map(item=>`<option value="${item[0]}">${item[1]} (${item[0]==='all'?rows.length:rows.filter(row=>row.category===item[0]).length})</option>`).join('');
document.getElementById('filter').onchange=render;
document.getElementById('stop').onclick=stopPlayback;
document.getElementById('next').onclick=()=>{const next=[...document.querySelectorAll('article')].find(card=>!card.classList.contains('done'));if(next)next.scrollIntoView({behavior:'smooth',block:'start'});};
document.getElementById('save').onclick=async()=>{
  const content=rows.map(row=>{const state=ensure(row);return JSON.stringify({schema:verdictSchema,audit_id:row.audit_id,source_id:row.source_id,partition:row.partition,row_role:row.row_role,category:row.category,verdict:state.verdict||'unreviewed',updated_at:state.updated_at||new Date().toISOString()});}).join('\\n')+'\\n';
  const response=await fetch('/__audit_api__/save-labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({href:location.pathname,filename:'manual_verdicts.jsonl',content})});
  const result=await response.json();document.getElementById('status').textContent=result.ok?'已保存到 '+result.path:'保存失败: '+result.error;
};
render();
</script>
</body>
</html>"""
        .replace("__ROWS__", encoded)
        .replace("__VERDICT_SCHEMA__", schema)
    )


def build_audit(*, selection: Path, output_dir: Path) -> Path:
    rows = _rows(selection)
    if not rows:
        raise ValueError("Scorer v10 prediction audit selection is empty")
    category_priority = {
        "speech_deletion": 0,
        "speech_edge_or_partial": 1,
        "long_residual": 2,
        "background_false_keep": 3,
        "normal": 4,
    }
    partition_priority = {"val": 0, "test": 1, "train": 2}

    def audit_category_priority(row: dict[str, Any]) -> int:
        category = str(row.get("category") or "")
        if (
            category == "normal"
            and float(row.get("max_predicted_speech_run_s") or 0.0) > 8.0
        ):
            category = "long_residual"
        return category_priority.get(category, 5)

    rows.sort(
        key=lambda row: (
            audit_category_priority(row),
            partition_priority.get(str(row.get("partition") or ""), 3),
            -int(row.get("false_negative_frames") or 0),
            -int(row.get("false_positive_frames") or 0),
            str(row.get("source_id") or ""),
        )
    )
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source = Path(str(row["audio"]))
        if not source.is_file():
            raise ValueError(f"Scorer v10 audit audio is missing: {source}")
        destination = audio_dir / f"item-{index:03d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        category = str(row["category"])
        if (
            category == "normal"
            and float(row.get("max_predicted_speech_run_s") or 0.0) > 8.0
        ):
            category = "long_residual"
        payload.append(
            {
                **row,
                "category": category,
                "audit_id": f"{category}:{row['source_id']}",
                "audio": destination.relative_to(output_dir).as_posix(),
                "truth_drop_spans": truth_drop_spans(row),
            }
        )
    page = _render_page(payload)
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    manifest = output_dir / "audit_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload),
        encoding="utf-8",
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "title": "Scorer v10 prediction residual audit",
        "review_item_count": len(payload),
        "category_counts": dict(Counter(str(row["category"]) for row in payload)),
        "selection": str(selection),
        "audit_manifest": str(manifest),
        "selection_contract": (
            "all_truth_keep_model_drop_rows_plus_all_heldout_hard_cases_"
            "plus_all_over_8s_residuals"
        ),
        "exact_truth_drop_playback": True,
        "manual_verdict_schema": VERDICT_SCHEMA,
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
