#!/usr/bin/env python3
"""Generate an exact-span audit for Scorer v10 internal truth-speech gaps."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


FRAME_HOP_S = 0.02
VERDICT_SCHEMA = "speech_scorer_v10_fragmentation_gap_manual_verdict_v3"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_internal_truth_gaps(
    predictions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only argmax holes bounded by speech inside canonical speech spans."""

    selected: list[dict[str, Any]] = []
    partition_priority = {"val": 0, "test": 1, "train": 2}
    for row in predictions:
        if str(row.get("row_role") or "") != "speech":
            continue
        truth_spans = sorted(
            (
                span
                for span in row.get("truth_spans", [])
                if span.get("label") == "truth_speech"
            ),
            key=lambda span: (int(span["start_frame"]), int(span["end_frame"])),
        )
        prediction_spans = sorted(
            row.get("prediction_spans", []),
            key=lambda span: (int(span["start_frame"]), int(span["end_frame"])),
        )
        for truth_index, truth in enumerate(truth_spans):
            truth_start = int(truth["start_frame"])
            truth_end = int(truth["end_frame"])
            overlaps = [
                span
                for span in prediction_spans
                if int(span["end_frame"]) > truth_start
                and int(span["start_frame"]) < truth_end
            ]
            for gap_index, (left, right) in enumerate(zip(overlaps, overlaps[1:])):
                gap_start = max(truth_start, int(left["end_frame"]))
                gap_end = min(truth_end, int(right["start_frame"]))
                if gap_end <= gap_start:
                    continue
                left_start = max(truth_start, int(left["start_frame"]))
                left_end = min(truth_end, int(left["end_frame"]))
                right_start = max(truth_start, int(right["start_frame"]))
                right_end = min(truth_end, int(right["end_frame"]))
                source_id = str(row["source_id"])
                selected.append(
                    {
                        "audit_id": (
                            f"{source_id}:truth{truth_index}:gap{gap_index}:"
                            f"{gap_start}-{gap_end}"
                        ),
                        "source_id": source_id,
                        "audio": str(row["audio"]),
                        "partition": str(row["partition"]),
                        "row_role": "speech",
                        "truth_run_index": truth_index,
                        "gap_index": gap_index,
                        "gap_frames": gap_end - gap_start,
                        "gap_ms": (gap_end - gap_start) * 20,
                        "left_span": {
                            "label": "model_speech_left",
                            "start_frame": left_start,
                            "end_frame": left_end,
                            "start_s": left_start * FRAME_HOP_S,
                            "end_s": left_end * FRAME_HOP_S,
                        },
                        "gap_span": {
                            "label": "truth_speech_model_background",
                            "start_frame": gap_start,
                            "end_frame": gap_end,
                            "start_s": gap_start * FRAME_HOP_S,
                            "end_s": gap_end * FRAME_HOP_S,
                        },
                        "right_span": {
                            "label": "model_speech_right",
                            "start_frame": right_start,
                            "end_frame": right_end,
                            "start_s": right_start * FRAME_HOP_S,
                            "end_s": right_end * FRAME_HOP_S,
                        },
                    }
                )
    selected.sort(
        key=lambda row: (
            partition_priority.get(str(row["partition"]), 3),
            -int(row["gap_frames"]),
            str(row["source_id"]),
            int(row["gap_span"]["start_frame"]),
        )
    )
    cluster_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in selected:
        cluster_groups.setdefault(
            (str(row["source_id"]), int(row["truth_run_index"])), []
        ).append(row)
    for (source_id, truth_run_index), cluster_rows in cluster_groups.items():
        cluster_id = f"{source_id}:truth{truth_run_index}"
        unique_runs = {
            (
                int(part["start_frame"]),
                int(part["end_frame"]),
            )
            for row in cluster_rows
            for part in (row["left_span"], row["right_span"])
        }
        cluster_start_s = min(
            float(row["left_span"]["start_s"]) for row in cluster_rows
        )
        cluster_end_s = max(
            float(row["right_span"]["end_s"]) for row in cluster_rows
        )
        for row in cluster_rows:
            row.update(
                {
                    "cluster_id": cluster_id,
                    "cluster_gap_count": len(cluster_rows),
                    "cluster_model_run_count": len(unique_runs),
                    "cluster_start_s": cluster_start_s,
                    "cluster_end_s": cluster_end_s,
                }
            )
    return selected


def _render_page(payload: list[dict[str, Any]]) -> str:
    encoded = (
        json.dumps(payload, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return (
        """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scorer v10 target-speech gap audit</title>
<style>
:root{color-scheme:light;--border:#c9d0d8;--text:#20242a;--muted:#5c6570;--speech:#417fc2;--gap:#efc75e;--ok:#267443;--risk:#a52f2f}
*{box-sizing:border-box}
body{margin:0;background:#f3f5f7;color:var(--text);font-family:Segoe UI,Arial,sans-serif}
header{position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:#fff;border-bottom:1px solid var(--border);padding:10px 18px}
header strong{margin-right:auto}
main{max-width:1120px;margin:18px auto;padding:0 14px}
section,article{background:#fff;border:1px solid var(--border);border-radius:6px;padding:14px;margin-bottom:14px}
article.done{border-left:6px solid var(--ok)}
h2{font-size:18px;margin:0 0 4px;overflow-wrap:anywhere}
small{color:var(--muted)}
audio{display:none}
button{font:inherit;border:1px solid #69737e;border-radius:5px;background:#fff;padding:7px 10px;cursor:pointer}
button:hover{border-color:#20242a}
button.active{background:#1769aa;color:#fff}
button.risk.active{background:var(--risk);color:#fff}
button.cluster-action{margin-top:9px}
.contract{display:grid;gap:7px}
.legend{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:4px}
.legend div{border-left:4px solid var(--border);padding:6px 9px;background:#f7f8fa}
.workflow-status{margin-top:12px;padding:9px;border-left:4px solid var(--speech);background:#eef4fb;color:#374151}
.sequence-label{margin-top:12px;color:#374151;font-weight:600}
.sequence{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px;margin:8px 0 14px}
.play{width:100%;min-height:62px;margin:0;padding:8px;overflow-wrap:anywhere}
.speech{background:var(--speech);color:#fff}
.gap{background:var(--gap);color:#1d2329}
.context-label{margin-top:2px;color:#374151;font-weight:600}
.context-playback{margin:8px 0 14px}
.context-playback .play{background:#eef4fb;color:#20242a;border-color:var(--speech);min-height:58px}
.play.playing{outline:3px solid #111;outline-offset:-3px}
.gap-reviews{border-top:1px solid var(--border)}
.gap-review{padding:13px 0;border-bottom:1px solid #e1e5e9}
.gap-review:last-child{border-bottom:0;padding-bottom:0}
.gap-review.done>strong::after{content:' · 已裁决';color:var(--ok)}
.segments{display:grid;grid-template-columns:minmax(0,1fr) minmax(160px,.55fr) minmax(0,1fr);gap:8px;margin:9px 0}
.choices{display:flex;flex-wrap:wrap;gap:6px}
.choices button{margin:0}
@media(max-width:760px){.legend,.segments,.sequence{grid-template-columns:1fr}header strong{width:100%}}
</style>
</head>
<body>
<header>
  <strong>1.7B Scorer v10 · 目标语音内部断点</strong>
  <span id="status"></span>
  <button id="stop" type="button">停止播放</button>
  <button id="save" type="button">保存裁决</button>
</header>
<main>
  <section class="contract">
    <div>按同一 truth run 组织审计，但不合并模型输出。蓝色是独立 Scorer island；黄色是 canonical truth_speech 内被模型判为 background、实际不会发送的 gap。每个 island/gap 条只播放自身精确区间；整段 truth-run 条仅用于听感判断，不代表合并或送 ASR。</div>
    <div class="legend">
      <div><b>同一 ASR 单元</b>：前后属于同一句或同一发声；中间即使是短非语义声，也不能让一句话断开。</div>
      <div><b>非语义一侧</b>：至少一侧不是 speech core，应与目标语音分离并交给 CueQC 独立 drop；适用于语音+杂音、杂音+语音或多段杂音后接语音。</div>
      <div><b>两侧独立语音</b>：两侧都是完整但不同的目标语音事件，均保留并分别送 ASR。</div>
      <div><b>整串均非 speech core</b>：当前 truth run 显示的所有片段都应退出 canonical speech 标注。</div>
    </div>
  </section>
  <div id="list"></div>
</main>
<script>
const rows=__ROWS__;
const key='scorer-v10-fragmentation-gap-audit-v3:'+location.pathname;
let ann={};
try{ann=JSON.parse(localStorage.getItem(key)||'{}');}catch(_error){ann={};}
const clusterMap=new Map();
for(const row of rows){
  const id=row.cluster_id||`${row.source_id}:truth${row.truth_run_index}`;
  if(!clusterMap.has(id)) clusterMap.set(id,{cluster_id:id,source_id:row.source_id,partition:row.partition,audio:row.audio,gaps:[]});
  clusterMap.get(id).gaps.push(row);
}
for(const cluster of clusterMap.values()) cluster.gaps.sort((a,b)=>a.gap_span.start_frame-b.gap_span.start_frame);
const clusters=[...clusterMap.values()].sort((a,b)=>{
  const order={val:0,test:1,train:2};
  const aMax=Math.max(...a.gaps.map(row=>row.gap_frames));
  const bMax=Math.max(...b.gaps.map(row=>row.gap_frames));
  return (order[a.partition]??3)-(order[b.partition]??3)||bMax-aMax||a.source_id.localeCompare(b.source_id)||a.gaps[0].truth_run_index-b.gaps[0].truth_run_index;
});
let activeAudio=null;
let activeButton=null;
let activeCheck=null;
let activeLoadHandler=null;
let activeTimer=null;
let activeFrame=null;
function esc(value){return String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));}
function ensureGap(row){ann[row.audit_id]??={verdict:'',note:''};return ann[row.audit_id];}
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
  stopPlayback();
  activeAudio=audio;activeButton=button;button.classList.add('playing');
  const begin=async()=>{
    activeLoadHandler=null;
    if(activeAudio!==audio||activeButton!==button) return;
    audio.currentTime=start;
    activeCheck=()=>{if(audio.ended||audio.currentTime>=end) stopPlayback();};
    audio.addEventListener('timeupdate',activeCheck);
    audio.addEventListener('ended',activeCheck);
    const watch=()=>{if(activeAudio!==audio)return;if(audio.currentTime>=end){stopPlayback();return;}activeFrame=requestAnimationFrame(watch);};
    try{
      await audio.play();
      if(activeAudio!==audio){audio.pause();return;}
      activeFrame=requestAnimationFrame(watch);
      activeTimer=setTimeout(stopPlayback,Math.max(100,(end-start)*1000+120));
    }catch(error){stopPlayback();document.getElementById('status').textContent=`播放失败：${error.message}`;}
  };
  if(audio.readyState<1){activeLoadHandler=begin;audio.addEventListener('loadedmetadata',begin,{once:true});audio.load();}else begin();
}
function sequenceParts(cluster){
  const parts=[];const seen=new Set();
  for(const row of cluster.gaps){for(const [span,kind] of [[row.left_span,'speech'],[row.gap_span,'gap'],[row.right_span,'speech']]){const id=`${kind}:${span.start_frame}-${span.end_frame}`;if(seen.has(id))continue;seen.add(id);parts.push({span,kind,title:''});}}
  parts.sort((a,b)=>a.span.start_frame-b.span.start_frame||a.span.end_frame-b.span.end_frame);
  let islandIndex=0;let gapIndex=0;
  for(const item of parts) item.title=item.kind==='speech'?`独立 Scorer island #${++islandIndex}`:`未发送 gap #${++gapIndex}`;
  return parts;
}
function playButton(span,kind,title){
  const durationMs=Math.round((Number(span.end_s)-Number(span.start_s))*1000);
  return `<button type="button" class="play ${kind}" data-start="${span.start_s}" data-end="${span.end_s}">${esc(title)} · ${durationMs}ms<br>${Number(span.start_s).toFixed(2)}–${Number(span.end_s).toFixed(2)}s</button>`;
}
function choices(row){
  const verdict=ensureGap(row).verdict;
  const options=[
    ['same_asr_unit_keep_continuous','同一 ASR 单元：应连续','risk'],
    ['separate_drop_nonsemantic','非语义一侧：可独立丢弃',''],
    ['separate_keep_both_speech','两侧独立语音：都保留',''],
    ['cluster_not_speech_core','整串均非 speech core','risk'],
    ['unsure','不确定','']
  ];
  return options.map(([value,label,kind])=>`<button type="button" data-v="${value}" class="${kind} ${verdict===value?'active':''}">${label}</button>`).join('');
}
function syncReview(review,row){
  const verdict=ensureGap(row).verdict;
  review.classList.toggle('done',Boolean(verdict));
  review.querySelectorAll('[data-v]').forEach(button=>button.classList.toggle('active',button.dataset.v===verdict));
}
function syncCard(card,cluster){card.classList.toggle('done',cluster.gaps.every(row=>Boolean(ensureGap(row).verdict)));}
function updateStatus(){document.getElementById('status').textContent=`已裁决 ${rows.filter(row=>ensureGap(row).verdict).length}/${rows.length} gaps · ${clusters.length} truth runs`;}
function persist(){localStorage.setItem(key,JSON.stringify(ann));updateStatus();}
function render(){
  stopPlayback();
  const root=document.getElementById('list');root.innerHTML='';
  for(const cluster of clusters){
    const card=document.createElement('article');
    card.innerHTML=`<h2>${esc(cluster.source_id)}</h2><small>${esc(cluster.partition)} / truth run ${cluster.gaps[0].truth_run_index} / ${cluster.gaps[0].cluster_model_run_count} independent islands / ${cluster.gaps.length} gaps / ${Number(cluster.gaps[0].cluster_start_s).toFixed(2)}–${Number(cluster.gaps[0].cluster_end_s).toFixed(2)}s</small><div class="workflow-status">每个蓝色 island 单独进入后续链；若 CueQC keep，也会作为独立 ASR chunk。此页不合并、不补 gap、不连续播放。</div><div class="sequence-label">实际候选工作流输出顺序</div><div class="sequence"></div><button type="button" class="cluster-action risk" data-cluster-v="cluster_not_speech_core">整串均非 speech core（应用到全部断点）</button><audio preload="none" src="${esc(cluster.audio)}"></audio><div class="gap-reviews"></div>`;
    card.querySelector('.workflow-status').textContent='每个蓝色 island 单独进入后续链；若 CueQC keep，也会作为独立 ASR chunk。下游不合并、不补 gap；完整审计条只用于听感判断。';
    const audio=card.querySelector('audio');
    const sequence=card.querySelector('.sequence');
    sequence.innerHTML=sequenceParts(cluster).map(item=>playButton(item.span,item.kind,item.title)).join('');
    sequence.querySelectorAll('[data-start]').forEach(button=>button.onclick=()=>playExact(audio,button,Number(button.dataset.start),Number(button.dataset.end)));
    const contextLabel=document.createElement('div');
    contextLabel.className='context-label';
    contextLabel.textContent='完整 island 串审计播放（首个 island 至末个 island，含 gap；仅判断上下文，不送 ASR）';
    const context=document.createElement('div');
    context.className='context-playback';
    const contextSpan={start_s:cluster.gaps[0].cluster_start_s,end_s:cluster.gaps[0].cluster_end_s};
    context.innerHTML=playButton(contextSpan,'context','完整 island 串 · 审计上下文');
    context.querySelectorAll('[data-start]').forEach(button=>button.onclick=()=>playExact(audio,button,Number(button.dataset.start),Number(button.dataset.end)));
    sequence.insertAdjacentElement('afterend',context);
    sequence.insertAdjacentElement('afterend',contextLabel);
    const reviews=card.querySelector('.gap-reviews');
    reviews.innerHTML=cluster.gaps.map((row,index)=>`<div class="gap-review"><strong>断点 ${index+1} · ${Number(row.gap_span.start_s).toFixed(2)}–${Number(row.gap_span.end_s).toFixed(2)}s / ${row.gap_ms}ms</strong><div class="segments">${playButton(row.left_span,'speech','左侧 model_speech')}${playButton(row.gap_span,'gap','truth_speech / model_background')}${playButton(row.right_span,'speech','右侧 model_speech')}</div><div class="choices">${choices(row)}</div></div>`).join('');
    reviews.querySelectorAll('.gap-review').forEach((review,index)=>{
      const row=cluster.gaps[index];
      review.querySelectorAll('[data-start]').forEach(button=>button.onclick=()=>playExact(audio,button,Number(button.dataset.start),Number(button.dataset.end)));
      review.querySelectorAll('[data-v]').forEach(button=>button.onclick=()=>{const verdict=ensureGap(row);verdict.verdict=button.dataset.v;verdict.updated_at=new Date().toISOString();syncReview(review,row);syncCard(card,cluster);persist();});
      syncReview(review,row);
    });
    card.querySelector('[data-cluster-v]').onclick=()=>{for(const row of cluster.gaps){const verdict=ensureGap(row);verdict.verdict='cluster_not_speech_core';verdict.updated_at=new Date().toISOString();}reviews.querySelectorAll('.gap-review').forEach((review,index)=>syncReview(review,cluster.gaps[index]));syncCard(card,cluster);persist();};
    syncCard(card,cluster);root.appendChild(card);
  }
  updateStatus();
}
document.getElementById('stop').onclick=stopPlayback;
document.getElementById('save').onclick=async()=>{
  const content=rows.map(row=>{const verdict=ensureGap(row);return JSON.stringify({schema:'__VERDICT_SCHEMA__',audit_id:row.audit_id,source_id:row.source_id,partition:row.partition,gap_start_s:row.gap_span.start_s,gap_end_s:row.gap_span.end_s,gap_ms:row.gap_ms,verdict:verdict.verdict||'unreviewed',note:verdict.note||'',updated_at:verdict.updated_at||new Date().toISOString()});}).join('\\n')+'\\n';
  try{
    const response=await fetch('/__audit_api__/save-labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({href:location.pathname,filename:'manual_verdicts.jsonl',content})});
    const output=await response.json();
    document.getElementById('status').textContent=response.ok&&output.ok?'已保存到 '+output.path:'保存失败: '+(output.error||response.status);
  }catch(error){document.getElementById('status').textContent='保存失败: '+error.message;}
};
render();
</script>
</body>
</html>
"""
        .replace("__ROWS__", encoded)
        .replace("__VERDICT_SCHEMA__", VERDICT_SCHEMA)
    )


def build_audit(*, predictions: Path, output_dir: Path) -> Path:
    gaps = select_internal_truth_gaps(_rows(predictions))
    if not gaps:
        raise ValueError("Scorer v10 fragmentation audit has no internal truth-speech gaps")
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[Path, str] = {}
    payload: list[dict[str, Any]] = []
    for row in gaps:
        source = Path(str(row["audio"])).resolve()
        if not source.is_file():
            raise ValueError(f"Scorer v10 fragmentation audio is missing: {source}")
        if source not in copied:
            destination = audio_dir / f"item-{len(copied):03d}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            copied[source] = destination.relative_to(output_dir).as_posix()
        payload.append({**row, "audio": copied[source]})

    page = _render_page(payload)
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    manifest = output_dir / "audit_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload),
        encoding="utf-8",
    )
    partition_counts = Counter(str(row["partition"]) for row in payload)
    truth_run_count = len(
        {
            (str(row["source_id"]), int(row["truth_run_index"]))
            for row in payload
        }
    )
    summary = {
        "schema": "speech_scorer_v10_fragmentation_gap_audit_summary_v3",
        "title": "Scorer v10 target-speech internal gap audit",
        "review_item_count": len(payload),
        "source_count": len({str(row["source_id"]) for row in payload}),
        "truth_run_count": truth_run_count,
        "partition_counts": dict(partition_counts),
        "gap_frame_count": sum(int(row["gap_frames"]) for row in payload),
        "max_gap_frames": max(int(row["gap_frames"]) for row in payload),
        "selection_contract": "all_internal_argmax_gaps_inside_canonical_truth_speech",
        "all_background_gaps_excluded": True,
        "playback_context_s": 0.0,
        "full_island_cluster_audit_playback": True,
        "full_island_cluster_playback_runtime_effect": "none_audit_only",
        "full_island_cluster_playback_span_contract": (
            "first_model_island_start_to_last_model_island_end_including_internal_gaps"
        ),
        "merged_playback": False,
        "runtime_gap_merge": False,
        "audit_grouping_runtime_effect": "none_truth_run_organization_only",
        "workflow_view_contract": (
            "each_argmax_speech_run_is_an_independent_downstream_island"
        ),
        "predictions": str(predictions),
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
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(build_audit(predictions=Path(args.predictions), output_dir=Path(args.output_dir)))
