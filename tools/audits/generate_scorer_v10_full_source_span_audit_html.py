#!/usr/bin/env python3
"""Generate a model-independent full-source span audit for Scorer v10."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


FRAME_HOP_S = 0.02
SUMMARY_SCHEMA = "speech_scorer_v10_full_source_span_audit_summary_v1"
ITEM_SCHEMA = "speech_scorer_v10_full_source_span_audit_item_v1"
MANUAL_VERDICT_SCHEMA = "speech_scorer_v10_full_source_span_manual_verdict_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_page(rows: list[dict[str, Any]]) -> str:
    encoded = (
        json.dumps(rows, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    schema = json.dumps(MANUAL_VERDICT_SCHEMA)
    return (
        """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scorer v10 full-source truth repair</title>
<style>
:root{color-scheme:light;--border:#c9d0d8;--text:#20242a;--muted:#5c6570;--speech:#4b9b68;--background:#d0a14b;--unsure:#8065b3;--risk:#a52f2f;--ok:#267443}
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:var(--text);font-family:Segoe UI,Arial,sans-serif}
header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:8px;flex-wrap:wrap;background:#fff;border-bottom:1px solid var(--border);padding:10px 18px}header strong{margin-right:auto}
main{max-width:1180px;margin:18px auto;padding:0 14px}section,article{background:#fff;border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:14px}article.done{border-left:6px solid var(--ok)}
audio{width:100%;margin:7px 0}.track{position:relative;height:48px;margin:9px 0;background:#e1e5e9;overflow:hidden}.span{position:absolute;top:0;bottom:0;min-width:3px;border-right:1px solid rgba(0,0,0,.28);overflow:hidden;white-space:nowrap;padding:6px 4px;font-size:11px}.speech{background:var(--speech);color:#fff}.background{background:var(--background)}.unsure{background:var(--unsure);color:#fff}
.editor{display:grid;grid-template-columns:repeat(2,minmax(150px,1fr));gap:8px;align-items:end;margin:10px 0}.editor label{display:grid;gap:4px}.editor input{width:100%;padding:7px}.controls,.source-state{display:flex;gap:5px;flex-wrap:wrap;margin:7px 0}.range-row{display:grid;grid-template-columns:minmax(230px,1fr) auto;gap:8px;align-items:center;border-top:1px solid #d7dde3;padding:7px 0}
button,input{font:inherit}button{padding:7px 10px;border:1px solid #69737e;border-radius:5px;background:#fff;cursor:pointer}button:disabled{cursor:not-allowed;opacity:.48}button.active{background:#1769aa;color:#fff}button.risk{border-color:var(--risk)}button.playing{outline:3px solid #111;outline-offset:-3px}.muted,small{color:var(--muted)}h2{font-size:18px;overflow-wrap:anywhere}.error{color:var(--risk);font-weight:600}.complete-note{padding:8px;background:#eef5ef;border-left:4px solid var(--ok)}
@media(max-width:760px){.editor{grid-template-columns:1fr}.range-row{grid-template-columns:1fr}.span{font-size:9px}header strong{width:100%}}
</style>
</head>
<body>
<header><strong>1.7B Scorer v10 · 完整 source 真值修复</strong><button id="next" type="button">下一个未完成 source</button><button id="stop" type="button">停止播放</button><button id="save" type="button">保存裁决</button><span id="status"></span></header>
<main>
<section>
  <div><b>本页不显示模型输出：</b>没有蓝条、Proposal、ASR 文本或旧 all-background 分段；它们均不得成为人工真值候选。</div>
  <div><b>标注方式：</b>从头到尾听完整条 source，添加所有目标语音区间；听不清或标签不确定的区间添加为 unsure。每条添加完后必须点击“已从头听到尾，以上区间完整”，顶部达到 2/2 后才能保存。时间自动对齐到 Scorer 的 20ms frame。</div>
  <div><b>background 合同：</b>未标出的差集只有在勾选“已从头听到尾”后才成为 background。unsure 会保留在 canonical，并在 normalization、split、loss、metrics 和 gate 中映射为 ignore=-100。</div>
  <div><b>禁止补救：</b>不按时长合并、不扩大边界、不使用 runtime threshold，也不因为 CueQC 可能删除短片段而忽略 Scorer 漏召回。</div>
</section>
<div id="list"></div>
</main>
<script>
const rows=__ROWS__,verdictSchema=__VERDICT_SCHEMA__,frameHop=0.02,key='scorer-v10-full-source-span-audit-v1:'+location.pathname;
let ann={};try{ann=JSON.parse(localStorage.getItem(key)||'{}');}catch(_error){ann={};}
let activeAudio=null,activeButton=null,activeCheck=null,activeLoad=null,activeTimer=null,activeFrame=null;
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function ensure(row){ann[row.source_id]??={ranges:[],reviewed_full_source:false,updated_at:''};ann[row.source_id].ranges??=[];return ann[row.source_id];}
function persist(){localStorage.setItem(key,JSON.stringify(ann));}
function stopPlayback(){if(activeAudio&&activeCheck){activeAudio.removeEventListener('timeupdate',activeCheck);activeAudio.removeEventListener('ended',activeCheck);}if(activeAudio&&activeLoad)activeAudio.removeEventListener('loadedmetadata',activeLoad);if(activeTimer!==null)clearTimeout(activeTimer);if(activeFrame!==null)cancelAnimationFrame(activeFrame);if(activeAudio)activeAudio.pause();if(activeButton)activeButton.classList.remove('playing');activeAudio=null;activeButton=null;activeCheck=null;activeLoad=null;activeTimer=null;activeFrame=null;}
function playExact(audio,button,start,end){if(!(end>start)){setMessage('播放区间必须满足 end > start',true);return;}if(activeAudio===audio&&activeButton===button&&!audio.paused){stopPlayback();return;}stopPlayback();activeAudio=audio;activeButton=button;button.classList.add('playing');const begin=async()=>{activeLoad=null;if(activeAudio!==audio)return;audio.currentTime=start;activeCheck=()=>{if(audio.currentTime>=end||audio.ended)stopPlayback();};audio.addEventListener('timeupdate',activeCheck);audio.addEventListener('ended',activeCheck);const watch=()=>{if(activeAudio!==audio)return;if(audio.currentTime>=end){stopPlayback();return;}activeFrame=requestAnimationFrame(watch);};try{await audio.play();if(activeAudio!==audio){audio.pause();return;}activeFrame=requestAnimationFrame(watch);activeTimer=setTimeout(stopPlayback,Math.max(100,(end-start)*1000+120));}catch(error){stopPlayback();setMessage('播放失败: '+error.message,true);}};if(audio.readyState<1){activeLoad=begin;audio.addEventListener('loadedmetadata',begin,{once:true});audio.load();}else begin();}
function clampFrame(value,row){return Math.max(0,Math.min(row.frame_count,Math.round(Number(value)/frameHop)));}
function seconds(frame){return Number(frame*frameHop).toFixed(2);}
function normalizeRanges(row,state){return [...state.ranges].map((range,index)=>({id:String(range.id||('range-'+index)),label:String(range.label),start_frame:Number(range.start_frame),end_frame:Number(range.end_frame)})).sort((a,b)=>a.start_frame-b.start_frame||a.end_frame-b.end_frame||a.label.localeCompare(b.label));}
function validateExplicit(row,state){const ranges=normalizeRanges(row,state);let previous=0;for(const range of ranges){if(!['speech','unsure'].includes(range.label))return '只允许显式 speech/unsure 区间';if(!Number.isInteger(range.start_frame)||!Number.isInteger(range.end_frame)||range.start_frame<0||range.end_frame>row.frame_count||range.end_frame<=range.start_frame)return '存在无效区间';if(range.start_frame<previous)return '区间不能重叠';previous=range.end_frame;}return '';}
function materialize(row,state){const error=validateExplicit(row,state);if(error)throw new Error(error);const result=[];const pushSpan=(label,start_frame,end_frame)=>{if(end_frame<=start_frame)return;const previous=result[result.length-1];if(previous&&previous.label===label&&previous.end_frame===start_frame)previous.end_frame=end_frame;else result.push({label,start_frame,end_frame});};let cursor=0;for(const range of normalizeRanges(row,state)){if(cursor<range.start_frame)pushSpan('background',cursor,range.start_frame);pushSpan(range.label,range.start_frame,range.end_frame);cursor=range.end_frame;}if(cursor<row.frame_count)pushSpan('background',cursor,row.frame_count);return result.map(span=>({...span,start_s:Number(seconds(span.start_frame)),end_s:Number(seconds(span.end_frame))}));}
function verdict(row,state){if(!state.reviewed_full_source)return 'unreviewed';const labels=new Set(materialize(row,state).map(span=>span.label));if(labels.has('speech'))return 'complete_with_target_speech';if(labels.has('unsure'))return 'complete_with_unsure_only';return 'complete_all_background';}
function setMessage(text,isError=false){const node=document.getElementById('status');node.textContent=text;node.classList.toggle('error',isError);}
function timeline(row,state){const spans=state.reviewed_full_source?materialize(row,state):normalizeRanges(row,state);return spans.map(span=>`<div class="span ${esc(span.label)}" style="left:${100*span.start_frame/row.frame_count}%;width:${Math.max(.3,100*(span.end_frame-span.start_frame)/row.frame_count)}%" title="${esc(span.label)} ${seconds(span.start_frame)}–${seconds(span.end_frame)}s">${esc(span.label)} ${seconds(span.start_frame)}–${seconds(span.end_frame)}s</div>`).join('');}
function updateStatus(){const done=rows.filter(row=>ensure(row).reviewed_full_source).length,pending=rows.length-done,save=document.getElementById('save');save.disabled=pending!==0;save.textContent=pending===0?'保存完整裁决':`保存裁决（还差 ${pending} 条）`;setMessage(`完整 source 已确认 ${done}/${rows.length}${pending?'；未完成时不会写文件':''}`,pending!==0);}
function addRange(row,state,label,startInput,endInput){const start=clampFrame(startInput.value,row),end=clampFrame(endInput.value,row);if(end<=start){setMessage('新增区间必须满足 end > start',true);return;}const candidate={id:`r-${Date.now()}-${Math.random().toString(16).slice(2)}`,label,start_frame:start,end_frame:end};state.ranges=[...state.ranges,candidate];const error=validateExplicit(row,state);if(error){state.ranges=state.ranges.filter(item=>item.id!==candidate.id);setMessage(error,true);return;}state.reviewed_full_source=false;state.updated_at=new Date().toISOString();persist();render();}
function removeRange(row,state,id){state.ranges=state.ranges.filter(range=>String(range.id)!==id);state.reviewed_full_source=false;state.updated_at=new Date().toISOString();persist();render();}
function render(){stopPlayback();const root=document.getElementById('list');root.innerHTML='';for(const row of rows){const state=ensure(row),card=document.createElement('article');if(state.reviewed_full_source)card.classList.add('done');card.dataset.sourceId=row.source_id;const currentVerdict=verdict(row,state);card.innerHTML=`<h2>${esc(row.source_id)}</h2><small>${esc(row.partition)} / ${row.frame_count} frames / ${Number(row.duration_s).toFixed(2)}s / 当前=${esc(currentVerdict)}</small><audio controls preload="none" src="${esc(row.audio)}"></audio><div class="track">${timeline(row,state)}</div><div class="editor"><label>区间开始（秒）<input class="start" type="number" min="0" max="${row.duration_s}" step="0.02" value="0.00"></label><label>区间结束（秒）<input class="end" type="number" min="0" max="${row.duration_s}" step="0.02" value="${Number(row.duration_s).toFixed(2)}"></label></div><div class="controls"><button class="capture-start" type="button">开始 = 当前播放位置</button><button class="capture-end" type="button">结束 = 当前播放位置</button><button class="preview" type="button">试听当前区间</button><button class="add-speech" type="button">添加 speech</button><button class="add-unsure risk" type="button">添加 unsure</button></div><div class="ranges"></div><div class="source-state"><button class="confirm ${state.reviewed_full_source?'active':''}" type="button">已从头听到尾，以上区间完整</button><button class="reopen" type="button">撤销完整确认</button></div><div class="complete-note">确认后，未标出的完整差集才会写为 background；没有显式区间即表示人工复核为全段 background。</div>`;const audio=card.querySelector('audio'),startInput=card.querySelector('.start'),endInput=card.querySelector('.end');audio.addEventListener('play',()=>{if(activeAudio&&activeAudio!==audio)stopPlayback();activeAudio=audio;});card.querySelector('.capture-start').onclick=()=>{startInput.value=seconds(clampFrame(audio.currentTime,row));};card.querySelector('.capture-end').onclick=()=>{endInput.value=seconds(clampFrame(audio.currentTime,row));};const preview=card.querySelector('.preview');preview.onclick=()=>playExact(audio,preview,clampFrame(startInput.value,row)*frameHop,clampFrame(endInput.value,row)*frameHop);card.querySelector('.add-speech').onclick=()=>addRange(row,state,'speech',startInput,endInput);card.querySelector('.add-unsure').onclick=()=>addRange(row,state,'unsure',startInput,endInput);const rangeRoot=card.querySelector('.ranges');for(const range of normalizeRanges(row,state)){const line=document.createElement('div');line.className='range-row';line.innerHTML=`<button type="button" class="${esc(range.label)}">播放 ${esc(range.label)} ${seconds(range.start_frame)}–${seconds(range.end_frame)}s</button><button type="button" class="remove risk">删除此区间</button>`;const play=line.querySelector(`.${range.label}`);play.onclick=()=>playExact(audio,play,range.start_frame*frameHop,range.end_frame*frameHop);line.querySelector('.remove').onclick=()=>removeRange(row,state,range.id);rangeRoot.appendChild(line);}card.querySelector('.confirm').onclick=()=>{const error=validateExplicit(row,state);if(error){setMessage(error,true);return;}state.reviewed_full_source=true;state.updated_at=new Date().toISOString();persist();render();};card.querySelector('.reopen').onclick=()=>{state.reviewed_full_source=false;state.updated_at=new Date().toISOString();persist();render();};root.appendChild(card);}updateStatus();}
document.getElementById('stop').onclick=stopPlayback;document.getElementById('next').onclick=()=>{const card=[...document.querySelectorAll('article')].find(item=>!item.classList.contains('done'));if(card)card.scrollIntoView({behavior:'smooth',block:'start'});};
document.getElementById('save').onclick=async()=>{const pending=rows.filter(row=>!ensure(row).reviewed_full_source);if(pending.length){setMessage(`拒绝保存：还有 ${pending.length} 条完整 source 未确认`,true);const card=document.querySelector(`article[data-source-id="${CSS.escape(pending[0].source_id)}"]`);if(card)card.scrollIntoView({behavior:'smooth',block:'start'});return;}let content='';try{content=rows.map(row=>{const state=ensure(row);return JSON.stringify({schema:verdictSchema,boundary_serialization_contract_id:row.boundary_serialization_contract_id,source_id:row.source_id,partition:row.partition,frame_count:row.frame_count,frame_hop_s:frameHop,reviewed_full_source:true,verdict:verdict(row,state),spans:materialize(row,state),updated_at:state.updated_at||new Date().toISOString()});}).join('\\n')+'\\n';}catch(error){setMessage('保存前校验失败: '+error.message,true);return;}const response=await fetch('/__audit_api__/save-labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({href:location.pathname,filename:'manual_verdicts.jsonl',content})});const result=await response.json();setMessage(result.ok?'已保存到 '+result.path:'保存失败: '+result.error,!result.ok);};render();
</script>
</body>
</html>"""
        .replace("__ROWS__", encoded)
        .replace("__VERDICT_SCHEMA__", schema)
    )


def build_audit(
    *,
    prediction_audit_manifest: Path,
    output_dir: Path,
    source_ids: set[str],
) -> Path:
    if not source_ids:
        raise ValueError("full-source span audit requires explicit source ids")
    selected: list[dict[str, Any]] = []
    available: set[str] = set()
    seen: set[str] = set()
    for row in _rows(prediction_audit_manifest):
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in seen:
            raise ValueError("prediction audit manifest requires unique source ids")
        seen.add(source_id)
        available.add(source_id)
        if source_id not in source_ids:
            continue
        if (
            str(row.get("row_role") or "") != "all_background"
            or str(row.get("category") or "") != "background_false_keep"
        ):
            raise ValueError("full-source repair requires an all-background discovery row")
        frame_count = int(row.get("frame_count") or 0)
        duration_s = float(row.get("duration_s") or 0.0)
        if frame_count <= 0 or duration_s <= 0.0:
            raise ValueError(f"invalid full-source frame extent: {source_id}")
        expected_duration = frame_count * FRAME_HOP_S
        if abs(duration_s - expected_duration) > FRAME_HOP_S + 1e-9:
            raise ValueError(f"full-source duration/frame mismatch: {source_id}")
        selected.append(row)
    missing = sorted(source_ids - available)
    if missing:
        raise ValueError(f"full-source audit sources are missing: {missing}")
    if len(selected) != len(source_ids):
        raise ValueError("full-source audit did not select every requested source")

    selected.sort(
        key=lambda row: (
            {"val": 0, "test": 1, "train": 2}.get(str(row.get("partition") or ""), 3),
            str(row["source_id"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        source = Path(str(row.get("audio") or ""))
        if not source.is_absolute():
            source = prediction_audit_manifest.parent / source
        if not source.is_file():
            raise ValueError(f"full-source audit audio is missing: {source}")
        destination = audio_dir / f"source-{index:03d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        payload.append(
            {
                "schema": ITEM_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "source_id": str(row["source_id"]),
                "partition": str(row.get("partition") or ""),
                "frame_count": int(row["frame_count"]),
                "frame_hop_s": FRAME_HOP_S,
                "duration_s": float(row["duration_s"]),
                "audio": destination.relative_to(output_dir).as_posix(),
            }
        )

    manifest = output_dir / "audit_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload),
        encoding="utf-8",
    )
    index = output_dir / "index.html"
    index.write_text(_render_page(payload), encoding="utf-8")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "title": "Scorer v10 model-independent full-source truth repair",
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "prediction_audit_manifest": str(prediction_audit_manifest),
        "prediction_audit_manifest_sha256": _sha256(prediction_audit_manifest),
        "source_count": len(payload),
        "partition_counts": dict(Counter(row["partition"] for row in payload)),
        "source_ids": [row["source_id"] for row in payload],
        "frame_hop_s": FRAME_HOP_S,
        "audit_manifest": str(manifest),
        "audit_manifest_sha256": _sha256(manifest),
        "manual_verdict_schema": MANUAL_VERDICT_SCHEMA,
        "manual_gate_status": "pending",
        "model_output_used_as_annotation_seed": False,
        "asr_output_used_as_annotation_seed": False,
        "unmarked_complement_becomes_background_only_after_full_source_confirmation": True,
        "unsure_training_label": -100,
        "training_manifest_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_audit_entrypoints(latest_html=index, title=summary["title"])
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-audit-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--only-source-id", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        build_audit(
            prediction_audit_manifest=Path(args.prediction_audit_manifest),
            output_dir=Path(args.output_dir),
            source_ids=set(args.only_source_id),
        )
    )
