#!/usr/bin/env python3
"""Generate a listenable Scorer v11 held-out and residual audit page."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


SUMMARY_SCHEMA = "candidate_island_scorer_v11_prediction_audit_summary_v1"
VERDICT_SCHEMA = "candidate_island_scorer_v11_prediction_manual_verdict_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def build_items(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in predictions:
        source_id = str(row["source_id"])
        common = {
            "source_id": source_id,
            "partition": str(row["partition"]),
            "duration_s": float(row["duration_s"]),
            "frame_count": int(row["frame_count"]),
            "truth_spans": list(row.get("truth_spans") or ()),
            "prediction_spans": list(row.get("prediction_spans") or ()),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
        }
        items.append(
            {
                **common,
                "audit_id": f"{source_id}::heldout_full_source",
                "category": "heldout_full_source",
                "focus_span": None,
            }
        )
        for index, span in enumerate(row.get("prediction_drop_truth_keep_spans") or ()):
            items.append(
                {
                    **common,
                    "audit_id": f"{source_id}::drop_truth_keep::{index:03d}",
                    "category": "prediction_drop_truth_keep",
                    "focus_span": dict(span),
                }
            )
        for index, span in enumerate(row.get("long_residual_spans") or ()):
            items.append(
                {
                    **common,
                    "audit_id": f"{source_id}::long_residual::{index:03d}",
                    "category": "long_residual_over_8s",
                    "focus_span": dict(span),
                }
            )
    return items


def _page(items: list[dict[str, Any]]) -> str:
    encoded = (
        json.dumps(items, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scorer v11 prediction audit</title>
<style>
:root{--border:#c9d0d8;--text:#20242a;--muted:#5c6570;--inside:#54a96d;--outside:#d3a64e;--model:#397fc4;--risk:#bf3d3d}
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:var(--text);font-family:Segoe UI,Arial,sans-serif}header{position:sticky;top:0;z-index:5;display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 16px;background:#fff;border-bottom:1px solid var(--border)}header strong{margin-right:auto}main{max-width:1180px;margin:16px auto;padding:0 12px}section,article{background:#fff;border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:12px}article.done{border-left:6px solid #2b7c48}audio{width:100%}.track{position:relative;height:42px;background:#e3e6e9;margin:7px 0;overflow:hidden}.span{position:absolute;top:0;bottom:0;min-width:3px;border:0;border-right:1px solid rgba(0,0,0,.2);padding:6px 3px;overflow:hidden;white-space:nowrap;font-size:10px;text-align:left;cursor:pointer}.truth_inside_candidate{background:var(--inside)}.truth_outside_candidate{background:var(--outside)}.truth_unsure{background:#888;color:#fff}.model_inside_candidate{background:var(--model);color:#fff}.focus{background:var(--risk);color:#fff}.span.playing{outline:3px solid #111;outline-offset:-3px}button,select{font:inherit;padding:6px 9px}button{margin:3px;border:1px solid #6a747e;border-radius:5px;background:#fff;cursor:pointer}button.active{background:#1769aa;color:#fff}.muted{color:var(--muted)}h2{font-size:17px;margin:0 0 4px;overflow-wrap:anywhere}h3{font-size:14px;margin:8px 0 2px}
</style></head><body>
<header><strong>1.7B Scorer v11 · held-out / residual 人工审计</strong><label>类别 <select id="filter"></select></label><button id="next">下一个未裁决</button><button id="stop">停止播放</button><button id="save">保存裁决</button><span id="status"></span></header>
<main><section>
<div><b>实际工作流：</b>蓝色是 two-logit argmax=`inside_candidate` 的真实输出，不做 threshold、gap merge或时长规则。绿色/黄色是完整 canonical。红色只标当前需要判断的精确 residual。</div>
<div><b>播放：</b>每个色条只播放自身区间并立即停止，不附加上下文；完整 source 播放器用于判断是否同一句话及整体连续性。</div>
<div><b>标签原则：</b>真语音删除/尾音截断零容忍。同一句内部短停顿即使无语义，若切开会伤害ASR，也判连续性有害。孤立呻吟/喘息允许Scorer先保留给CueQC。</div>
</section><div id="list"></div></main>
<script>
const rows=__ROWS__, verdictSchema=__SCHEMA__, key='scorer-v11-prediction-audit:'+location.pathname;const listNode=document.getElementById('list'),filterNode=document.getElementById('filter'),statusNode=document.getElementById('status'),stopNode=document.getElementById('stop'),nextNode=document.getElementById('next'),saveNode=document.getElementById('save');let ann={};try{ann=JSON.parse(localStorage.getItem(key)||'{}')}catch(_e){}
let activeAudio=null,activeButton=null,check=null,timer=null,raf=null;
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function stopPlayback(){if(activeAudio&&check){activeAudio.removeEventListener('timeupdate',check);activeAudio.removeEventListener('ended',check)}if(timer)clearTimeout(timer);if(raf)cancelAnimationFrame(raf);if(activeAudio)activeAudio.pause();if(activeButton)activeButton.classList.remove('playing');activeAudio=activeButton=check=timer=raf=null}
async function play(audio,button,start,end){if(activeAudio===audio&&activeButton===button&&!audio.paused){stopPlayback();return}stopPlayback();activeAudio=audio;activeButton=button;button.classList.add('playing');const begin=async()=>{if(activeAudio!==audio)return;audio.currentTime=start;check=()=>{if(audio.currentTime>=end||audio.ended)stopPlayback()};audio.addEventListener('timeupdate',check);audio.addEventListener('ended',check);try{await audio.play();const watch=()=>{if(activeAudio!==audio)return;if(audio.currentTime>=end){stopPlayback();return}raf=requestAnimationFrame(watch)};raf=requestAnimationFrame(watch);timer=setTimeout(stopPlayback,Math.max(100,(end-start)*1000+120))}catch(e){stopPlayback();statusNode.textContent='播放失败: '+e.message}};if(audio.readyState<1){audio.addEventListener('loadedmetadata',begin,{once:true});audio.load()}else await begin()}
function spans(row,list,focus=false){return (list||[]).map(s=>`<button class="span ${focus?'focus':esc(s.label)}" style="left:${100*s.start_s/row.duration_s}%;width:${Math.max(.25,100*(s.end_s-s.start_s)/row.duration_s)}%" data-start="${s.start_s}" data-end="${s.end_s}">${esc(s.label)} ${Number(s.start_s).toFixed(2)}–${Number(s.end_s).toFixed(2)}s</button>`).join('')}
function choices(row){if(row.category==='heldout_full_source')return [['pass_no_true_speech_loss','整条输出无真语音损失'],['true_speech_loss','存在真语音删除或边缘截断'],['continuity_harm','存在会伤害ASR的碎片化'],['canonical_error','canonical需修正'],['unsure','不确定']];if(row.category==='long_residual_over_8s')return [['acceptable_candidate','长蓝段整体可送下游'],['contains_independent_outside','含应独立切出的背景/非语义'],['canonical_error','canonical需修正'],['unsure','不确定']];return [['true_speech_deleted_or_clipped','红段含真语音/尾音，模型误删'],['same_asr_unit_continuity_harm','红段虽无语义但切开同一ASR单元'],['canonical_should_be_outside','红段可安全提前删除，canonical应改outside'],['unsure','不确定']]}
function update(){statusNode.textContent=`已裁决 ${rows.filter(r=>ann[r.audit_id]?.verdict).length}/${rows.length}`}
function render(){stopPlayback();listNode.innerHTML='';for(const row of rows){if(filterNode.value!=='all'&&row.category!==filterNode.value)continue;const card=document.createElement('article'),state=ann[row.audit_id]||{};card.dataset.auditId=row.audit_id;if(state.verdict)card.classList.add('done');card.innerHTML=`<h2>${esc(row.source_id)}</h2><div class="muted">${esc(row.partition)} / ${esc(row.category)} / ${Number(row.duration_s).toFixed(2)}s</div><audio controls preload="none" src="${esc(row.audio)}"></audio><h3>canonical</h3><div class="track">${spans(row,row.truth_spans)}</div><h3>model inside_candidate</h3><div class="track">${spans(row,row.prediction_spans)}</div>${row.focus_span?`<h3>当前精确审计区间</h3><div class="track">${spans(row,[row.focus_span],true)}</div>`:''}<div>${choices(row).map(([v,t])=>`<button data-v="${v}" class="${state.verdict===v?'active':''}">${t}</button>`).join('')}</div>`;const audio=card.querySelector('audio');card.querySelectorAll('.span').forEach(b=>b.onclick=()=>play(audio,b,Number(b.dataset.start),Number(b.dataset.end)));card.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{ann[row.audit_id]={schema:verdictSchema,audit_id:row.audit_id,source_id:row.source_id,partition:row.partition,category:row.category,focus_span:row.focus_span,checkpoint_sha256:row.checkpoint_sha256,verdict:b.dataset.v,updated_at:new Date().toISOString()};localStorage.setItem(key,JSON.stringify(ann));render()});listNode.appendChild(card)}update()}
const categories=['all',...new Set(rows.map(r=>r.category))];filterNode.innerHTML=categories.map(v=>`<option value="${v}">${v}</option>`).join('');filterNode.onchange=render;stopNode.onclick=stopPlayback;nextNode.onclick=()=>{const row=rows.find(r=>!ann[r.audit_id]?.verdict);if(!row)return;filterNode.value='all';render();document.querySelector(`[data-audit-id="${CSS.escape(row.audit_id)}"]`)?.scrollIntoView({behavior:'smooth'})};
saveNode.onclick=async()=>{const missing=rows.filter(r=>!ann[r.audit_id]?.verdict);if(missing.length){statusNode.textContent=`仍有 ${missing.length} 条未裁决`;return}const content=rows.map(r=>JSON.stringify(ann[r.audit_id])).join('\\n')+'\\n';try{const response=await fetch('/__audit_api__/save-labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({href:location.pathname,filename:'manual_verdicts.jsonl',content})});const result=await response.json();statusNode.textContent=result.ok?'已保存到 '+result.path:'保存失败: '+result.error}catch(e){statusNode.textContent='保存失败: '+e.message}};render();
</script></body></html>""".replace("__ROWS__", encoded).replace("__SCHEMA__", json.dumps(VERDICT_SCHEMA))


def build(*, predictions_path: Path, output_dir: Path) -> Path:
    predictions = _rows(predictions_path)
    items = build_items(predictions)
    if not items:
        raise ValueError("Scorer v11 prediction audit has no items")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    by_source = {str(row["source_id"]): row for row in predictions}
    for index, source_id in enumerate(sorted(by_source)):
        source = _resolve(by_source[source_id]["audio"])
        if not source.is_file():
            raise FileNotFoundError(f"Scorer v11 audit audio is missing: {source}")
        target = audio_dir / f"source-{index:03d}.wav"
        shutil.copy2(source, target)
        copied[source_id] = target.relative_to(output_dir).as_posix()
    for item in items:
        item["audio"] = copied[item["source_id"]]
    manifest = output_dir / "audit_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items),
        encoding="utf-8",
    )
    index = output_dir / "index.html"
    index.write_text(_page(items), encoding="utf-8")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "title": "Scorer v11 held-out and residual audit",
        "source_predictions": str(predictions_path),
        "source_count": len(predictions),
        "item_count": len(items),
        "category_counts": {
            category: sum(item["category"] == category for item in items)
            for category in sorted({item["category"] for item in items})
        },
        "manual_verdict_schema": VERDICT_SCHEMA,
        "manual_gate_status": "pending",
        "training_manifest_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_audit_entrypoints(latest_html=index, title=summary["title"])
    return index


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(build(predictions_path=Path(args.source_predictions), output_dir=Path(args.output_dir)))
