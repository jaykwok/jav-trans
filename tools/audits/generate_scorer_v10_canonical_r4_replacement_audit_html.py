#!/usr/bin/env python3
"""Generate an exact rendered-placement audit for Scorer v10 canonical r4."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.boundary.ja.apply_speech_island_scorer_v10_canonical_r4_repairs import (  # noqa: E402
    PLACEMENT_SCHEMA,
    SUMMARY_SCHEMA as CANONICAL_R4_SUMMARY_SCHEMA,
)


SUMMARY_SCHEMA = "speech_scorer_v10_canonical_r4_replacement_audit_summary_v1"
ITEM_SCHEMA = "speech_scorer_v10_canonical_r4_replacement_audit_item_v1"
MANUAL_VERDICT_SCHEMA = (
    "speech_scorer_v10_canonical_r4_replacement_manual_verdict_v1"
)


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_page(payload: list[dict[str, Any]]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    verdict_schema = json.dumps(MANUAL_VERDICT_SCHEMA)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scorer v10 canonical r4 replacement audit</title>
<style>
:root{{--border:#c9d0d8;--text:#20242a;--muted:#5c6570;--ok:#267443;--risk:#a52f2f;--speech:#58aa70;--background:#d0a14b;--mapped:#397db8}}
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f7;color:var(--text);font-family:Segoe UI,Arial,sans-serif}}
header{{position:sticky;top:0;z-index:4;display:flex;gap:8px;align-items:center;flex-wrap:wrap;background:#fff;border-bottom:1px solid var(--border);padding:10px 18px}}header strong{{margin-right:auto}}
main{{max-width:1120px;margin:18px auto;padding:0 14px}}section,article{{background:#fff;border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:14px}}article.done{{border-left:6px solid var(--ok)}}
audio{{width:100%;margin:5px 0}}button{{font:inherit;padding:6px 9px;margin:3px;border:1px solid #69737e;border-radius:5px;background:#fff;cursor:pointer}}button.active{{background:#1769aa;color:#fff}}button.risk.active{{background:var(--risk);color:#fff}}button.playing{{outline:3px solid #111;outline-offset:-3px}}
.track{{position:relative;height:42px;background:#e1e5e9;overflow:hidden;margin:8px 0}}.span{{position:absolute;top:0;bottom:0;min-width:3px;margin:0;padding:7px 4px;border:0;border-right:1px solid rgba(0,0,0,.2);overflow:hidden;white-space:nowrap;font-size:11px;text-align:left}}.speech{{background:var(--speech)}}.background{{background:var(--background)}}
.exact{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:8px 0}}.exact div{{border:1px solid var(--border);border-radius:6px;padding:8px}}.mapped{{background:#e6f0fa}}.changed{{background:#fff4d7}}small{{color:var(--muted)}}h2{{font-size:17px;overflow-wrap:anywhere}}@media(max-width:760px){{.exact{{grid-template-columns:1fr}}header strong{{width:100%}}}}
</style></head><body>
<header><strong>1.7B Scorer v10 · canonical r4 replacement audit</strong><button id="next">下一个未裁决</button><button id="stop">停止播放</button><button id="save">保存裁决</button><span id="status"></span></header>
<main><section>
<div><b>审计对象：</b>5 个原 all-background control 与其 6 个 active composite 依赖，共 19 个精确 crop/tile occurrence。每条都来自已人工确认的 source repair event；这里复核它经过裁剪、循环平铺或 additive mix 后，实际听感是否仍是目标语音且边界完整。</div>
<div><b>播放合同：</b>“源事件”“实际映射”“实际改标签区间”和绿色/黄色 canonical 条都只播放自己的 start–end，到点立即停止，不添加上下文。最下面的完整 island 播放器只用于判断整句/整段关系。</div>
<div><b>标签：</b>“映射语音正确”表示 r4 处理可保留；“渲染后不是目标语音”表示该 occurrence 不能据此新增 speech；“边界不完整”表示 crop/tile 只留下截断残片，需精确返修。黄色改绿为 0 的 occurrence 本来就在已有 speech 内，不会重复创建 core，但仍需确认依赖映射。</div>
</section><div id="list"></div></main>
<script>
const rows={encoded};const verdictSchema={verdict_schema};const key='scorer-v10-canonical-r4-replacement-v1:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let activeAudio=null,activeButton=null,activeStop=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function ensure(r){{ann[r.item_id]??={{verdict:'',updated_at:''}};return ann[r.item_id];}}
function stopPlayback(){{if(activeAudio&&activeStop)activeAudio.removeEventListener('timeupdate',activeStop);if(activeAudio)activeAudio.pause();if(activeButton)activeButton.classList.remove('playing');activeAudio=null;activeButton=null;activeStop=null;}}
function pauseOtherAudio(except){{document.querySelectorAll('audio').forEach(a=>{{if(a!==except&&!a.paused)a.pause();}});}}
function playExact(audio,button,start,end){{if(activeAudio===audio&&activeButton===button&&!audio.paused){{stopPlayback();return;}}stopPlayback();pauseOtherAudio(audio);activeAudio=audio;activeButton=button;button.classList.add('playing');const begin=()=>{{audio.currentTime=start;activeStop=()=>{{if(audio.currentTime>=end)stopPlayback();}};audio.addEventListener('timeupdate',activeStop);audio.play();}};if(audio.readyState<1){{audio.addEventListener('loadedmetadata',begin,{{once:true}});audio.load();}}else begin();}}
function pct(v,d){{return Math.max(0,Math.min(100,100*v/d));}}function span(a,b){{return `${{Number(a).toFixed(3)}}–${{Number(b).toFixed(3)}}s`;}}
function verdictButtons(a){{const values=[['repair_speech_correct','映射语音正确',false],['not_target_after_render','渲染后不是目标语音',true],['boundary_incomplete','边界不完整',true],['unsure','不确定',true]];return values.map(v=>`<button data-v="${{v[0]}}" class="${{v[2]?'risk ':''}}${{a.verdict===v[0]?'active':''}}">${{v[1]}}</button>`).join('');}}
function render(){{stopPlayback();const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r),card=document.createElement('article');card.id='item-'+r.item_index;if(a.verdict==='repair_speech_correct')card.classList.add('done');const bars=r.canonical_spans.map(s=>`<button type="button" class="span ${{s.label}}" style="left:${{pct(s.start_s,r.target_duration_s)}}%;width:${{Math.max(.2,pct(s.end_s-s.start_s,r.target_duration_s))}}%" data-target-start="${{s.start_s}}" data-target-end="${{s.end_s}}" title="${{esc(s.label_source)}} ${{span(s.start_s,s.end_s)}}">${{esc(s.label)}}</button>`).join('');const changes=r.background_label_change_ranges.map((s,i)=>`<button type="button" data-target-start="${{s.start_s}}" data-target-end="${{s.end_s}}">改标签 ${{i+1}} · ${{span(s.start_s,s.end_s)}}</button>`).join('')||'<small>该 occurrence 全部落在已有 speech 内，没有新增 core。</small>';card.innerHTML=`<h2>${{esc(r.item_id)}}</h2><small>${{esc(r.partition)}} / ${{esc(r.role)}} / mapped=${{span(r.mapped_start_s,r.mapped_end_s)}} / changed=${{r.background_label_change_sample_count}} samples / already-speech=${{r.already_speech_sample_count}} samples</small><div class="exact"><div><b>原 source repair event</b><audio class="source-audio" preload="none" src="${{esc(r.source_audio)}}"></audio><button type="button" data-source-start="${{r.source_event_start_s}}" data-source-end="${{r.source_event_end_s}}">直接播放/停止 · ${{span(r.source_event_start_s,r.source_event_end_s)}}</button></div><div class="mapped"><b>实际 crop/tile/mix 后 occurrence</b><audio class="target-exact-audio" preload="none" src="${{esc(r.target_audio)}}"></audio><button type="button" data-target-start="${{r.mapped_start_s}}" data-target-end="${{r.mapped_end_s}}">直接播放/停止 · ${{span(r.mapped_start_s,r.mapped_end_s)}}</button></div><div class="changed"><b>实际 background→speech 子区间</b>${{changes}}</div></div><div class="track">${{bars}}</div><div>${{verdictButtons(a)}}</div><h3>实际候选工作流完整 island</h3><audio class="target-full-audio" controls preload="none" src="${{esc(r.target_audio)}}"></audio>`;const sourceAudio=card.querySelector('.source-audio'),targetAudio=card.querySelector('.target-exact-audio');card.querySelectorAll('[data-source-start]').forEach(b=>b.onclick=()=>playExact(sourceAudio,b,Number(b.dataset.sourceStart),Number(b.dataset.sourceEnd)));card.querySelectorAll('[data-target-start]').forEach(b=>b.onclick=()=>playExact(targetAudio,b,Number(b.dataset.targetStart),Number(b.dataset.targetEnd)));card.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{{a.verdict=b.dataset.v;a.updated_at=new Date().toISOString();localStorage.setItem(key,JSON.stringify(ann));render();}});card.querySelector('.target-full-audio').addEventListener('play',e=>{{stopPlayback();pauseOtherAudio(e.currentTarget);}});root.appendChild(card);}}document.getElementById('status').textContent=`已裁决 ${{rows.filter(r=>ensure(r).verdict).length}}/${{rows.length}}`;}}
document.getElementById('stop').onclick=stopPlayback;document.getElementById('next').onclick=()=>{{const row=rows.find(r=>!ensure(r).verdict);if(row)document.getElementById('item-'+row.item_index)?.scrollIntoView({{behavior:'smooth'}});}};
document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r);return JSON.stringify({{schema:verdictSchema,item_id:r.item_id,placement_id:r.placement_id,event_id:r.event_id,source_id:r.source_id,target_source_id:r.target_source_id,partition:r.partition,role:r.role,core_registered:r.core_registered,verdict:a.verdict||'unreviewed',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();
</script></body></html>"""


def build_audit(*, canonical_summary: Path, output_dir: Path) -> Path:
    summary = json.loads(canonical_summary.read_text(encoding="utf-8-sig"))
    if summary.get("schema") != CANONICAL_R4_SUMMARY_SCHEMA:
        raise ValueError("invalid Scorer canonical r4 summary schema")
    canonical_sources = Path(str(summary.get("canonical_sources") or ""))
    placements_path = Path(str(summary.get("repair_placements") or ""))
    for path, sha_key in (
        (canonical_sources, "canonical_sources_sha256"),
        (placements_path, "repair_placements_sha256"),
    ):
        if not path.is_file() or _sha256(path) != str(summary.get(sha_key) or ""):
            raise ValueError(f"Scorer canonical r4 evidence changed: {path}")
    gate_path = Path(str(summary.get("background_speech_repair_gate") or ""))
    if (
        not gate_path.is_file()
        or _sha256(gate_path)
        != str(summary.get("background_speech_repair_gate_sha256") or "")
    ):
        raise ValueError("Scorer canonical r4 speech repair gate changed")
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    events_path = Path(str(gate.get("repair_events") or ""))
    if not events_path.is_file() or _sha256(events_path) != str(
        gate.get("repair_events_sha256") or ""
    ):
        raise ValueError("Scorer canonical r4 speech repair events changed")

    canonical_by_id = {str(row["source_id"]): row for row in _rows(canonical_sources)}
    events = {str(row["event_id"]): row for row in _rows(events_path)}
    placements = _rows(placements_path)
    if len(placements) != int(summary.get("repair_placement_count") or -1):
        raise ValueError("Scorer canonical r4 placement count mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[tuple[str, str], str] = {}

    def copy_audio(kind: str, source_id: str, source: Path) -> str:
        key = (kind, source_id)
        if key in copied:
            return copied[key]
        if not source.is_file():
            raise ValueError(f"Scorer canonical r4 audit audio is missing: {source}")
        destination = audio_dir / f"{kind}-{len(copied):03d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        copied[key] = destination.relative_to(output_dir).as_posix()
        return copied[key]

    payload: list[dict[str, Any]] = []
    for index, placement in enumerate(placements):
        if placement.get("schema") != PLACEMENT_SCHEMA:
            raise ValueError("invalid Scorer canonical r4 placement schema")
        target_id = str(placement["target_source_id"])
        source_id = str(placement["source_id"])
        target = canonical_by_id[target_id]
        source = canonical_by_id[source_id]
        event = events[str(placement["event_id"])]
        payload.append(
            {
                **placement,
                "schema": ITEM_SCHEMA,
                "item_index": index,
                "item_id": str(placement["placement_id"]),
                "target_audio": copy_audio("target", target_id, Path(str(target["audio"]))),
                "source_audio": copy_audio("source", source_id, Path(str(source["audio"]))),
                "target_duration_s": float(target["duration_s"]),
                "source_event_start_s": int(event["start_sample"]) / int(source["sample_rate"]),
                "source_event_end_s": int(event["end_sample"]) / int(source["sample_rate"]),
                "canonical_spans": [
                    {
                        **span,
                        "start_s": int(span["start_sample"]) / int(target["sample_rate"]),
                        "end_s": int(span["end_sample"]) / int(target["sample_rate"]),
                    }
                    for span in target["canonical_spans"]
                ],
                "background_label_change_ranges": [
                    {
                        **span,
                        "start_s": int(span["start_sample"]) / int(target["sample_rate"]),
                        "end_s": int(span["end_sample"]) / int(target["sample_rate"]),
                    }
                    for span in placement["background_label_change_ranges"]
                ],
            }
        )

    manifest = output_dir / "audit_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload),
        encoding="utf-8",
    )
    index = output_dir / "index.html"
    index.write_text(_render_page(payload), encoding="utf-8")
    result = {
        "schema": SUMMARY_SCHEMA,
        "title": "Scorer v10 canonical r4 replacement audit",
        "canonical_summary": str(canonical_summary),
        "canonical_summary_sha256": _sha256(canonical_summary),
        "canonical_sources": str(canonical_sources),
        "canonical_sources_sha256": _sha256(canonical_sources),
        "repair_placements": str(placements_path),
        "repair_placements_sha256": _sha256(placements_path),
        "repair_events": str(events_path),
        "repair_events_sha256": _sha256(events_path),
        "audit_manifest": str(manifest),
        "audit_manifest_sha256": _sha256(manifest),
        "review_item_count": len(payload),
        "registered_core_item_count": sum(bool(row["core_registered"]) for row in payload),
        "no_label_change_item_count": sum(not bool(row["core_registered"]) for row in payload),
        "manual_verdict_schema": MANUAL_VERDICT_SCHEMA,
        "manual_gate_status": "pending",
        "training_manifest_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_audit_entrypoints(latest_html=index, title=result["title"])
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        build_audit(
            canonical_summary=Path(args.canonical_summary),
            output_dir=Path(args.output_dir),
        )
    )
