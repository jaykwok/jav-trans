#!/usr/bin/env python3
"""Generate exact-island repair audit for contaminated Scorer all-background rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.audits.generate_scorer_v10_prediction_audit_html import (  # noqa: E402
    FRAME_HOP_S,
    VERDICT_SCHEMA as PREDICTION_VERDICT_SCHEMA,
)


SUMMARY_SCHEMA = "speech_scorer_v10_background_speech_repair_audit_summary_v1"
ISLAND_SCHEMA = "speech_scorer_v10_background_speech_repair_island_v1"
LINK_SCHEMA = "speech_scorer_v10_background_speech_repair_link_v1"
MANUAL_VERDICT_SCHEMA = (
    "speech_scorer_v10_background_speech_repair_manual_verdict_v1"
)
TARGET_SOURCE_VERDICT = "canonical_contains_target_speech"
SAMPLES_PER_FRAME_16K = round(16000 * FRAME_HOP_S)


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_model_spans(
    row: dict[str, Any], *, sample_count: int
) -> list[dict[str, Any]]:
    frame_count = int(row.get("frame_count") or 0)
    expected_frames = math.ceil(sample_count / SAMPLES_PER_FRAME_16K)
    if frame_count != expected_frames:
        raise ValueError(
            f"prediction/canonical frame_count mismatch: {row.get('source_id')}"
        )
    spans: list[dict[str, Any]] = []
    previous_end = -1
    for raw in sorted(
        (
            span
            for span in row.get("prediction_spans") or ()
            if str(span.get("label") or "") == "model_speech"
        ),
        key=lambda span: (int(span["start_frame"]), int(span["end_frame"])),
    ):
        start_frame = int(raw["start_frame"])
        end_frame = int(raw["end_frame"])
        if (
            start_frame < 0
            or end_frame <= start_frame
            or end_frame > frame_count
            or start_frame < previous_end
        ):
            raise ValueError("prediction speech islands are invalid or overlapping")
        start_sample = min(sample_count, start_frame * SAMPLES_PER_FRAME_16K)
        end_sample = min(sample_count, end_frame * SAMPLES_PER_FRAME_16K)
        if end_sample <= start_sample:
            raise ValueError("prediction speech island is empty in sample space")
        spans.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_s": start_sample / 16000.0,
                "end_s": end_sample / 16000.0,
            }
        )
        previous_end = end_frame
    if not spans:
        raise ValueError("target all-background repair row has no model-speech islands")
    return spans


def _render_page(payload: list[dict[str, Any]]) -> str:
    encoded = (
        json.dumps(payload, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    verdict_schema = json.dumps(MANUAL_VERDICT_SCHEMA)
    return (
        """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scorer v10 all-background speech repair</title>
<style>
:root{color-scheme:light;--border:#c9d0d8;--text:#20242a;--muted:#5c6570;--speech:#417fc2;--gap:#d0a14b;--ok:#267443;--risk:#a52f2f}
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:var(--text);font-family:Segoe UI,Arial,sans-serif}
header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:8px;flex-wrap:wrap;background:#fff;border-bottom:1px solid var(--border);padding:10px 18px}header strong{margin-right:auto}
main{max-width:1180px;margin:18px auto;padding:0 14px}section,article{background:#fff;border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:14px}article.done{border-left:6px solid var(--ok)}
audio{width:100%;margin:6px 0}.track{position:relative;min-height:42px;margin:8px 0;background:#e1e5e9;overflow:hidden}.span{position:absolute;top:0;bottom:0;min-width:3px;min-height:42px;margin:0;padding:7px 4px;border:0;border-right:1px solid rgba(0,0,0,.2);overflow:hidden;white-space:nowrap;font-size:11px;text-align:left;cursor:pointer}.model_speech{background:var(--speech);color:#fff}.model_gap{background:var(--gap)}.span.playing,.play.playing{outline:3px solid #111;outline-offset:-3px;color:#fff}
.item{display:grid;grid-template-columns:minmax(250px,1fr) minmax(420px,auto);gap:10px;align-items:center;padding:9px 0;border-top:1px solid #d7dde3}.play{width:100%;border:0;padding:9px;text-align:left;cursor:pointer}.island-play{background:var(--speech);color:#fff}.gap-play{background:var(--gap)}button{font:inherit;padding:6px 9px;margin:2px;border:1px solid #69737e;border-radius:5px;background:#fff;cursor:pointer}button.active{background:#1769aa;color:#fff}button.risk.active{background:var(--risk);color:#fff}small{color:var(--muted)}h2{overflow-wrap:anywhere}.muted{color:var(--muted)}
@media(max-width:820px){.item{grid-template-columns:1fr}header strong{width:100%}.span{font-size:9px}}
</style>
</head>
<body>
<header><strong>1.7B Scorer v10 · all-background 精确 speech 修标</strong><button id="next" type="button">下一个未完成 source</button><button id="stop" type="button">停止播放</button><button id="save" type="button">保存裁决</button><span id="status"></span></header>
<main>
<section>
  <div><b>职责：</b>这些 source 曾被标为 all-background，但人工已确认包含目标语音。不能把整条粗改 speech；本页逐个审计 Scorer 实际蓝岛。</div>
  <div><b>蓝岛：</b>选择“目标语音且边界可用”、 “含目标语音但蓝段截字/边界不完整”、 “背景/无语义人声”或“不确定”。边界不完整会阻止自动修标并进入后续精确边界页。</div>
  <div><b>黄色间隙：</b>只有左右蓝岛都被选为边界可用的目标语音时才需要判断；同一 ASR 单元表示修标时连同间隙保持连续，独立事件表示间隙保留 background。</div>
  <div><b>播放：</b>蓝岛和黄色间隙都只播放自身区间并立即停止，不加上下文；整条 source 播放器只用于判断话语关系。页面不做 runtime merge 或时长规则。</div>
</section>
<div id="list"></div>
</main>
<script>
const rows=__ROWS__,verdictSchema=__VERDICT_SCHEMA__,key='scorer-v10-background-speech-repair-v1:'+location.pathname;
let ann={};try{ann=JSON.parse(localStorage.getItem(key)||'{}');}catch(_error){ann={};}
let activeAudio=null,activeButton=null,activeCheck=null,activeTimer=null,activeFrame=null,activeLoad=null;
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function ensure(item){ann[item.item_id]??={verdict:''};return ann[item.item_id];}
function stopPlayback(){if(activeAudio&&activeCheck){activeAudio.removeEventListener('timeupdate',activeCheck);activeAudio.removeEventListener('ended',activeCheck);}if(activeAudio&&activeLoad)activeAudio.removeEventListener('loadedmetadata',activeLoad);if(activeTimer!==null)clearTimeout(activeTimer);if(activeFrame!==null)cancelAnimationFrame(activeFrame);if(activeAudio)activeAudio.pause();if(activeButton)activeButton.classList.remove('playing');activeAudio=null;activeButton=null;activeCheck=null;activeTimer=null;activeFrame=null;activeLoad=null;}
function playExact(audio,button,start,end){if(activeAudio===audio&&activeButton===button&&!audio.paused){stopPlayback();return;}stopPlayback();activeAudio=audio;activeButton=button;button.classList.add('playing');const begin=async()=>{activeLoad=null;if(activeAudio!==audio)return;audio.currentTime=start;activeCheck=()=>{if(audio.currentTime>=end||audio.ended)stopPlayback();};audio.addEventListener('timeupdate',activeCheck);audio.addEventListener('ended',activeCheck);const watch=()=>{if(activeAudio!==audio)return;if(audio.currentTime>=end){stopPlayback();return;}activeFrame=requestAnimationFrame(watch);};try{await audio.play();if(activeAudio!==audio){audio.pause();return;}activeFrame=requestAnimationFrame(watch);activeTimer=setTimeout(stopPlayback,Math.max(100,(end-start)*1000+120));}catch(error){stopPlayback();document.getElementById('status').textContent='播放失败: '+error.message;}};if(audio.readyState<1){activeLoad=begin;audio.addEventListener('loadedmetadata',begin,{once:true});audio.load();}else begin();}
function islandChoices(item){return [['target_speech_span_ok','目标语音，蓝段边界可用于修标'],['target_speech_boundary_incomplete','含目标语音，但蓝段截字/边界不完整'],['background_or_nonsemantic','背景/无语义人声'],['unsure','不确定']];}
function linkChoices(item){return [['same_asr_unit','同一 ASR 单元，间隙也应保持连续'],['separate_target_events','独立目标事件，间隙保留 background'],['unsure','不确定']];}
function risk(v){return ['target_speech_boundary_incomplete','same_asr_unit','unsure'].includes(v);}
function setVerdict(card,item,value){const state=ensure(item);state.verdict=value;state.updated_at=new Date().toISOString();localStorage.setItem(key,JSON.stringify(ann));card.querySelectorAll(`[data-item="${CSS.escape(item.item_id)}"] [data-v]`).forEach(b=>b.classList.toggle('active',b.dataset.v===value));updateCard(card);updateStatus();}
function requiredItems(row){const result=[...row.islands];for(const link of row.links){const left=ensure(row.islands[link.left_island_index]).verdict,right=ensure(row.islands[link.right_island_index]).verdict;if(left==='target_speech_span_ok'&&right==='target_speech_span_ok')result.push(link);}return result;}
function updateCard(card){const row=rows.find(r=>r.source_id===card.dataset.sourceId),required=requiredItems(row);card.classList.toggle('done',required.every(item=>ensure(item).verdict&&ensure(item).verdict!=='unreviewed'));card.querySelectorAll('.link-item').forEach(line=>{const link=row.links[Number(line.dataset.linkIndex)],left=ensure(row.islands[link.left_island_index]).verdict,right=ensure(row.islands[link.right_island_index]).verdict;line.style.opacity=(left==='target_speech_span_ok'&&right==='target_speech_span_ok')?'1':'.45';});}
function updateStatus(){const required=rows.flatMap(requiredItems),done=required.filter(item=>ensure(item).verdict&&ensure(item).verdict!=='unreviewed').length;document.getElementById('status').textContent=`必审已裁决 ${done}/${required.length}`;}
function trackSpans(row){return [...row.islands.map(x=>({...x,label:'model_speech'})),...row.links.map(x=>({...x,label:'model_gap'}))].sort((a,b)=>a.start_s-b.start_s).map(x=>`<button type="button" class="span ${x.label}" style="left:${Math.max(0,100*x.start_s/row.duration_s)}%;width:${Math.max(.3,100*(x.end_s-x.start_s)/row.duration_s)}%" data-start="${x.start_s}" data-end="${x.end_s}">${x.label} ${Number(x.start_s).toFixed(2)}–${Number(x.end_s).toFixed(2)}s</button>`).join('');}
function choiceHtml(item,choices){const state=ensure(item);return choices(item).map(c=>`<button type="button" data-v="${c[0]}" class="${risk(c[0])?'risk ':''}${state.verdict===c[0]?'active':''}">${c[1]}</button>`).join('');}
function render(){stopPlayback();const root=document.getElementById('list');root.innerHTML='';for(const row of rows){const card=document.createElement('article');card.dataset.sourceId=row.source_id;card.innerHTML=`<h2>${esc(row.source_id)}</h2><small>${esc(row.partition)} / ${row.islands.length} blue islands / ${row.links.length} gaps / ${Number(row.duration_s).toFixed(2)}s</small><h3>整条 source</h3><audio controls preload="none" src="${esc(row.audio)}"></audio><h3>实际蓝岛与相邻间隙</h3><div class="track">${trackSpans(row)}</div><h3>逐蓝岛</h3><div class="islands"></div><h3>相邻蓝岛关系</h3><div class="links"></div>`;const audio=card.querySelector('audio');audio.addEventListener('play',()=>{if(activeAudio&&activeAudio!==audio)stopPlayback();activeAudio=audio;});card.querySelectorAll('[data-start]').forEach(b=>b.onclick=()=>playExact(audio,b,Number(b.dataset.start),Number(b.dataset.end)));const islandRoot=card.querySelector('.islands');for(const item of row.islands){const line=document.createElement('div');line.className='item';line.dataset.item=item.item_id;line.innerHTML=`<button type="button" class="play island-play">蓝岛 ${item.island_index+1} · ${Number(item.start_s).toFixed(2)}–${Number(item.end_s).toFixed(2)}s</button><div>${choiceHtml(item,islandChoices)}</div>`;line.querySelector('.play').onclick=()=>playExact(audio,line.querySelector('.play'),item.start_s,item.end_s);line.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>setVerdict(card,item,b.dataset.v));islandRoot.appendChild(line);}const linkRoot=card.querySelector('.links');if(!row.links.length)linkRoot.innerHTML='<div class="muted">本 source 只有一个蓝岛，无相邻关系需要裁决。</div>';for(const item of row.links){const line=document.createElement('div');line.className='item link-item';line.dataset.item=item.item_id;line.dataset.linkIndex=item.link_index;line.innerHTML=`<button type="button" class="play gap-play">间隙 ${item.link_index+1} · ${Number(item.start_s).toFixed(2)}–${Number(item.end_s).toFixed(2)}s</button><div>${choiceHtml(item,linkChoices)}</div>`;line.querySelector('.play').onclick=()=>playExact(audio,line.querySelector('.play'),item.start_s,item.end_s);line.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>setVerdict(card,item,b.dataset.v));linkRoot.appendChild(line);}root.appendChild(card);updateCard(card);}updateStatus();}
document.getElementById('stop').onclick=stopPlayback;document.getElementById('next').onclick=()=>{const card=[...document.querySelectorAll('article')].find(x=>!x.classList.contains('done'));if(card)card.scrollIntoView({behavior:'smooth',block:'start'});};document.getElementById('save').onclick=async()=>{const items=rows.flatMap(r=>[...r.islands,...r.links]);const content=items.map(item=>{const state=ensure(item);return JSON.stringify({schema:verdictSchema,item_id:item.item_id,item_type:item.item_type,source_id:item.source_id,verdict:state.verdict||'unreviewed',updated_at:state.updated_at||new Date().toISOString()});}).join('\\n')+'\\n';const response=await fetch('/__audit_api__/save-labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({href:location.pathname,filename:'manual_verdicts.jsonl',content})});const result=await response.json();document.getElementById('status').textContent=result.ok?'已保存到 '+result.path:'保存失败: '+result.error;};render();
</script>
</body>
</html>"""
        .replace("__ROWS__", encoded)
        .replace("__VERDICT_SCHEMA__", verdict_schema)
    )


def build_audit(
    *,
    canonical_sources: Path,
    prediction_audit_manifest: Path,
    prediction_manual_verdicts: Path,
    output_dir: Path,
) -> Path:
    canonical = {str(row["source_id"]): row for row in _rows(canonical_sources)}
    predictions: dict[str, dict[str, Any]] = {}
    for row in _rows(prediction_audit_manifest):
        audit_id = str(row.get("audit_id") or "")
        if not audit_id or audit_id in predictions:
            raise ValueError("prediction audit manifest requires unique audit_id values")
        predictions[audit_id] = row

    selected: list[dict[str, Any]] = []
    for verdict in _rows(prediction_manual_verdicts):
        if verdict.get("schema") != PREDICTION_VERDICT_SCHEMA:
            raise ValueError("invalid Scorer prediction verdict schema")
        if str(verdict.get("verdict") or "") != TARGET_SOURCE_VERDICT:
            continue
        audit_id = str(verdict.get("audit_id") or "")
        prediction = predictions.get(audit_id)
        if prediction is None:
            raise ValueError(f"prediction verdict target is missing: {audit_id}")
        for field in ("source_id", "partition", "row_role", "category"):
            if str(verdict.get(field) or "") != str(prediction.get(field) or ""):
                raise ValueError(f"prediction verdict {field} mismatch: {audit_id}")
        source_id = str(prediction["source_id"])
        source = canonical.get(source_id)
        if source is None:
            raise ValueError(f"canonical all-background source is missing: {source_id}")
        if (
            str(source.get("row_role") or "") != "all_background"
            or str(prediction.get("row_role") or "") != "all_background"
            or str(prediction.get("category") or "") != "background_false_keep"
            or str(source.get("partition") or "")
            != str(prediction.get("partition") or "")
        ):
            raise ValueError("background speech repair requires matching all-background rows")
        if (
            source.get("boundary_serialization_contract_id")
            != ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError("background speech repair has the wrong central contract")
        if int(source.get("sample_rate") or 0) != 16000:
            raise ValueError("background speech repair requires 16 kHz canonical audio")
        if any(
            str(span.get("label") or "") != "background"
            for span in source.get("canonical_spans") or ()
        ):
            raise ValueError("background speech repair source is not canonically all-background")
        selected.append({"source": source, "prediction": prediction})
    if not selected:
        raise ValueError("no canonical_contains_target_speech rows require repair")

    selected.sort(
        key=lambda item: (
            {"val": 0, "test": 1, "train": 2}.get(
                str(item["source"].get("partition") or ""), 3
            ),
            str(item["source"]["source_id"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for source_index, item in enumerate(selected):
        source = item["source"]
        prediction = item["prediction"]
        source_id = str(source["source_id"])
        sample_count = int(source["sample_count"])
        model_spans = _normalized_model_spans(prediction, sample_count=sample_count)
        audio_source = Path(str(source["audio"]))
        if not audio_source.is_file():
            raise ValueError(f"background speech repair audio is missing: {audio_source}")
        audio_target = audio_dir / f"source-{source_index:02d}{audio_source.suffix.lower()}"
        shutil.copy2(audio_source, audio_target)

        islands: list[dict[str, Any]] = []
        for island_index, span in enumerate(model_spans):
            island = {
                "schema": ISLAND_SCHEMA,
                "item_id": f"{source_id}::island{island_index:02d}",
                "item_type": "island",
                "source_id": source_id,
                "partition": str(source["partition"]),
                "background_id": str(source["background_id"]),
                "island_index": island_index,
                **span,
            }
            islands.append(island)
            manifest_rows.append(island)

        links: list[dict[str, Any]] = []
        for link_index, (left, right) in enumerate(zip(islands, islands[1:])):
            if int(right["start_sample"]) <= int(left["end_sample"]):
                raise ValueError("background speech repair islands do not have a positive gap")
            link = {
                "schema": LINK_SCHEMA,
                "item_id": f"{source_id}::link{link_index:02d}",
                "item_type": "link",
                "source_id": source_id,
                "partition": str(source["partition"]),
                "background_id": str(source["background_id"]),
                "link_index": link_index,
                "left_island_id": str(left["item_id"]),
                "right_island_id": str(right["item_id"]),
                "left_island_index": int(left["island_index"]),
                "right_island_index": int(right["island_index"]),
                "start_frame": int(left["end_frame"]),
                "end_frame": int(right["start_frame"]),
                "start_sample": int(left["end_sample"]),
                "end_sample": int(right["start_sample"]),
                "start_s": float(left["end_s"]),
                "end_s": float(right["start_s"]),
            }
            links.append(link)
            manifest_rows.append(link)
        payload.append(
            {
                "source_id": source_id,
                "partition": str(source["partition"]),
                "background_id": str(source["background_id"]),
                "duration_s": sample_count / 16000.0,
                "audio": audio_target.relative_to(output_dir).as_posix(),
                "islands": islands,
                "links": links,
            }
        )

    audit_manifest = output_dir / "audit_manifest.jsonl"
    audit_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    index = output_dir / "index.html"
    index.write_text(_render_page(payload), encoding="utf-8")
    island_count = sum(len(row["islands"]) for row in payload)
    link_count = sum(len(row["links"]) for row in payload)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "title": "Scorer v10 all-background exact speech repair",
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_sources": str(canonical_sources),
        "canonical_sources_sha256": _sha256(canonical_sources),
        "prediction_audit_manifest": str(prediction_audit_manifest),
        "prediction_audit_manifest_sha256": _sha256(prediction_audit_manifest),
        "prediction_manual_verdicts": str(prediction_manual_verdicts),
        "prediction_manual_verdicts_sha256": _sha256(prediction_manual_verdicts),
        "source_count": len(payload),
        "island_count": island_count,
        "link_count": link_count,
        "review_item_count": island_count + link_count,
        "source_ids": [str(row["source_id"]) for row in payload],
        "audit_manifest": str(audit_manifest),
        "audit_manifest_sha256": _sha256(audit_manifest),
        "manual_verdict_schema": MANUAL_VERDICT_SCHEMA,
        "manual_gate_status": "pending",
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
    parser.add_argument("--canonical-sources", required=True)
    parser.add_argument("--prediction-audit-manifest", required=True)
    parser.add_argument("--prediction-manual-verdicts", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        build_audit(
            canonical_sources=Path(args.canonical_sources),
            prediction_audit_manifest=Path(args.prediction_audit_manifest),
            prediction_manual_verdicts=Path(args.prediction_manual_verdicts),
            output_dir=Path(args.output_dir),
        )
    )
