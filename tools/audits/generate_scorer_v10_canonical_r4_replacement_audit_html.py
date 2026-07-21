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

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.boundary.ja.apply_speech_island_scorer_v10_canonical_r4_repairs import (  # noqa: E402
    DEPENDENCY_MAPPING_SCHEMA,
    PLACEMENT_SCHEMA,
    SUMMARY_SCHEMA as CANONICAL_R4_SUMMARY_SCHEMA,
)


SUMMARY_SCHEMA = "speech_scorer_v10_canonical_r4_replacement_audit_summary_v3"
ITEM_SCHEMA = "speech_scorer_v10_canonical_r4_replacement_audit_item_v3"
MANUAL_VERDICT_SCHEMA = (
    "speech_scorer_v10_canonical_r4_replacement_manual_verdict_v2"
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
.exact{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:8px 0}}.exact>div{{border:1px solid var(--border);border-radius:6px;padding:8px}}.mapped{{background:#e6f0fa}}.changed{{background:#fff4d7}}.mapping-pair{{display:grid;grid-template-columns:1fr;gap:5px;margin-top:5px}}.mapping-arrow{{font-weight:600;color:#24547c}}small{{display:block;color:var(--muted)}}h2{{font-size:17px;overflow-wrap:anywhere}}@media(max-width:760px){{.exact{{grid-template-columns:1fr}}header strong{{width:100%}}}}
</style></head><body>
<header><strong>1.7B Scorer v10 · canonical r4 replacement audit</strong><button id="next">下一个未裁决</button><button id="stop">停止播放</button><button id="save">保存裁决</button><span id="status"></span></header>
<main><section>
<div><b>审计对象：</b>5 个原 all-background control 与其 6 个 active composite 依赖，共 19 个精确 crop/tile occurrence。先判断原 source event 本身是否为目标语音，再判断它经过裁剪、循环平铺或 additive mix 后是否仍完整有效。</div>
<div><b>三栏含义：</b>第一栏是原 source event 全段，用于判断上游证据本身是否成立；第二栏是同一次 occurrence 在 source 文件中实际取用的子段，以及它在 target 合成文件中 crop/tile/mix 后的结果；第三栏只列本次会把 canonical 从 background 改成 speech 的 target 子区间，不是第三种音频来源。若第三栏为空，表示 occurrence 已落在原有 speech 标签内。</div>
<div><b>时间轴：</b>source 与 target 是两个不同音频文件，绝对秒数不要求重合。页面明确显示 `source 子段 → target occurrence`；所有播放器都是物理裁好的独立 WAV，从 0 播到文件结尾，不依赖浏览器 seek，也不添加上下文。最下面的完整 island 只用于判断整句/整段关系。</div>
<div><b>标签：</b>如果源 event 本身就不是目标语音，选“源事件非目标语音（整组撤销）”，同一 event 的 control、crop、tile 和 overlay occurrence 会一次性全部标记；不要逐条选“渲染后不是目标语音”。后者只用于源 event 正确、但某个具体渲染 occurrence 因裁剪/混音已不再构成目标语音的情况。</div>
</section><div id="list"></div></main>
<script>
const rows={encoded};const verdictSchema={verdict_schema};const key='scorer-v10-canonical-r4-replacement-v3:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let activeClipButton=null,activeClipAudio=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function ensure(r){{ann[r.item_id]??={{verdict:'',updated_at:''}};return ann[r.item_id];}}
function clearClipState(){{if(activeClipButton)activeClipButton.classList.remove('playing');activeClipButton=null;activeClipAudio=null;}}
function stopPlayback(){{document.querySelectorAll('audio').forEach(a=>a.pause());clearClipState();}}
function pauseOtherAudio(except){{document.querySelectorAll('audio').forEach(a=>{{if(a!==except&&!a.paused)a.pause();}});if(activeClipAudio&&activeClipAudio!==except)clearClipState();}}
function playClip(audio,button,src){{if(activeClipAudio===audio&&activeClipButton===button&&!audio.paused){{audio.pause();clearClipState();return;}}pauseOtherAudio(audio);audio.src=src;audio.currentTime=0;button.classList.add('playing');activeClipButton=button;activeClipAudio=audio;audio.play();}}
function pct(v,d){{return Math.max(0,Math.min(100,100*v/d));}}function span(a,b){{return `${{Number(a).toFixed(3)}}–${{Number(b).toFixed(3)}}s`;}}
function verdictButtons(a){{const values=[['repair_speech_correct','映射语音正确',false],['source_event_not_target','源事件非目标语音（整组撤销）',true],['not_target_after_render','仅此渲染后不是目标语音',true],['boundary_incomplete','边界不完整',true],['unsure','不确定',true]];return values.map(v=>`<button data-v="${{v[0]}}" class="${{v[2]?'risk ':''}}${{a.verdict===v[0]?'active':''}}">${{v[1]}}</button>`).join('');}}
function setVerdict(r,value){{const now=new Date().toISOString();if(value==='source_event_not_target'){{rows.filter(x=>x.event_id===r.event_id).forEach(x=>{{const a=ensure(x);a.verdict=value;a.updated_at=now;}});}}else{{const group=rows.filter(x=>x.event_id===r.event_id);if(group.some(x=>ensure(x).verdict==='source_event_not_target'))group.forEach(x=>{{const a=ensure(x);a.verdict='';a.updated_at=now;}});const a=ensure(r);a.verdict=value;a.updated_at=now;}}localStorage.setItem(key,JSON.stringify(ann));render();}}
function render(){{stopPlayback();const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r),card=document.createElement('article');card.id='item-'+r.item_index;if(['repair_speech_correct','source_event_not_target'].includes(a.verdict))card.classList.add('done');const bars=r.canonical_spans.map(s=>`<button type="button" class="span ${{s.label}}" style="left:${{pct(s.start_s,r.target_duration_s)}}%;width:${{Math.max(.2,pct(s.end_s-s.start_s,r.target_duration_s))}}%" data-clip="${{esc(s.clip_audio)}}" title="${{esc(s.label_source)}} ${{span(s.start_s,s.end_s)}}">${{esc(s.label)}}</button>`).join('');const changes=r.background_label_change_ranges.map((s,i)=>`<div><small>target 标签变化 ${{i+1}} · ${{span(s.start_s,s.end_s)}}</small><audio controls preload="none" src="${{esc(s.clip_audio)}}"></audio></div>`).join('')||'<small>无标签变化：该 occurrence 已全部位于原有 speech 内，不新增 core。</small>';const omitted=[];if(r.source_occurrence_start_s>r.source_event_start_s+1e-9)omitted.push('event 前部未进入');if(r.source_occurrence_end_s<r.source_event_end_s-1e-9)omitted.push('event 后部未进入');const truncation=omitted.length?' / '+omitted.join('、'):'';card.innerHTML=`<h2>${{esc(r.item_id)}}</h2><small>${{esc(r.partition)}} / ${{esc(r.role)}} / tile=${{r.tile_index}} / occurrence=${{r.occurrence_index}} / changed=${{r.background_label_change_sample_count}} samples / already-speech=${{r.already_speech_sample_count}} samples</small><div class="exact"><div><b>1. 原 source event 全段</b><audio controls preload="none" src="${{esc(r.source_clip_audio)}}"></audio><small>source 文件坐标 ${{span(r.source_event_start_s,r.source_event_end_s)}} / ${{Number(r.source_clip_duration_s).toFixed(3)}}s</small></div><div class="mapped"><b>2. 本次 occurrence 的精确映射</b><div class="mapping-pair"><small>source 实际取用 ${{span(r.source_occurrence_start_s,r.source_occurrence_end_s)}} / ${{Number(r.source_occurrence_duration_s).toFixed(3)}}s${{truncation}}</small><audio controls preload="none" src="${{esc(r.source_occurrence_clip_audio)}}"></audio><div class="mapping-arrow">↓ crop / tile / mix</div><small>target 合成坐标 ${{span(r.mapped_start_s,r.mapped_end_s)}} / ${{Number(r.rendered_clip_duration_s).toFixed(3)}}s</small><audio controls preload="none" src="${{esc(r.rendered_clip_audio)}}"></audio></div></div><div class="changed"><b>3. 仅 canonical 标签变化区间</b>${{changes}}</div></div><div class="track">${{bars}}</div><audio class="canonical-clip-audio" controls preload="none"></audio><div>${{verdictButtons(a)}}</div><h3>实际候选工作流完整 island</h3><audio class="target-full-audio" controls preload="none" src="${{esc(r.target_audio)}}"></audio>`;const clipAudio=card.querySelector('.canonical-clip-audio');card.querySelectorAll('[data-clip]').forEach(b=>b.onclick=()=>playClip(clipAudio,b,b.dataset.clip));card.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>setVerdict(r,b.dataset.v));card.querySelectorAll('audio').forEach(audio=>{{audio.addEventListener('play',()=>pauseOtherAudio(audio));audio.addEventListener('ended',()=>{{if(activeClipAudio===audio)clearClipState();}});}});root.appendChild(card);}}document.getElementById('status').textContent=`已裁决 ${{rows.filter(r=>ensure(r).verdict).length}}/${{rows.length}}`;}}
document.getElementById('stop').onclick=stopPlayback;document.getElementById('next').onclick=()=>{{const row=rows.find(r=>!ensure(r).verdict);if(row)document.getElementById('item-'+row.item_index)?.scrollIntoView({{behavior:'smooth'}});}};
document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r);return JSON.stringify({{schema:verdictSchema,item_id:r.item_id,placement_id:r.placement_id,event_id:r.event_id,source_id:r.source_id,target_source_id:r.target_source_id,partition:r.partition,role:r.role,core_registered:r.core_registered,verdict:a.verdict||'unreviewed',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();
</script></body></html>"""


def build_audit(*, canonical_summary: Path, output_dir: Path) -> Path:
    summary = json.loads(canonical_summary.read_text(encoding="utf-8-sig"))
    if summary.get("schema") != CANONICAL_R4_SUMMARY_SCHEMA:
        raise ValueError("invalid Scorer canonical r4 summary schema")
    canonical_sources = Path(str(summary.get("canonical_sources") or ""))
    placements_path = Path(str(summary.get("repair_placements") or ""))
    dependency_mappings_path = Path(str(summary.get("dependency_mappings") or ""))
    for path, sha_key in (
        (canonical_sources, "canonical_sources_sha256"),
        (placements_path, "repair_placements_sha256"),
        (dependency_mappings_path, "dependency_mappings_sha256"),
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
    dependency_mappings = _rows(dependency_mappings_path)
    dependency_mappings_by_id: dict[str, dict[str, Any]] = {}
    for mapping in dependency_mappings:
        if mapping.get("schema") != DEPENDENCY_MAPPING_SCHEMA:
            raise ValueError("invalid Scorer canonical r4 dependency mapping schema")
        mapping_id = str(mapping.get("mapping_id") or "")
        if not mapping_id or mapping_id in dependency_mappings_by_id:
            raise ValueError("invalid or duplicate Scorer dependency mapping id")
        dependency_mappings_by_id[mapping_id] = mapping
    if len(placements) != int(summary.get("repair_placement_count") or -1):
        raise ValueError("Scorer canonical r4 placement count mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[tuple[str, str], str] = {}
    clips: dict[tuple[str, int, int], str] = {}

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

    def clip_audio(source: Path, start_sample: int, end_sample: int) -> str:
        key = (str(source.resolve()), int(start_sample), int(end_sample))
        if key in clips:
            return clips[key]
        if end_sample <= start_sample:
            raise ValueError("Scorer canonical r4 audit clip is empty")
        destination = audio_dir / f"clip-{len(clips):03d}.wav"
        with sf.SoundFile(source) as handle:
            if start_sample < 0 or end_sample > len(handle):
                raise ValueError("Scorer canonical r4 audit clip is outside its audio")
            handle.seek(start_sample)
            samples = handle.read(
                frames=end_sample - start_sample,
                dtype="int16",
                always_2d=True,
            )
            sample_rate = int(handle.samplerate)
        if len(samples) != end_sample - start_sample:
            raise ValueError("Scorer canonical r4 audit clip length mismatch")
        sf.write(destination, samples, sample_rate, subtype="PCM_16")
        with sf.SoundFile(destination) as rendered:
            if len(rendered) != end_sample - start_sample:
                raise ValueError("Scorer canonical r4 rendered clip length mismatch")
        clips[key] = destination.relative_to(output_dir).as_posix()
        return clips[key]

    payload: list[dict[str, Any]] = []
    for index, placement in enumerate(placements):
        if placement.get("schema") != PLACEMENT_SCHEMA:
            raise ValueError("invalid Scorer canonical r4 placement schema")
        target_id = str(placement["target_source_id"])
        source_id = str(placement["source_id"])
        target = canonical_by_id[target_id]
        source = canonical_by_id[source_id]
        event = events[str(placement["event_id"])]
        target_path = Path(str(target["audio"]))
        source_path = Path(str(source["audio"]))
        source_event_start = int(event["start_sample"])
        source_event_end = int(event["end_sample"])
        mapped_start = int(placement["mapped_start_sample"])
        mapped_end = int(placement["mapped_end_sample"])
        if str(placement["role"]) == "control":
            source_occurrence_start = source_event_start
            source_occurrence_end = source_event_end
        else:
            mapping_id = str(placement.get("mapping_id") or "")
            mapping = dependency_mappings_by_id.get(mapping_id)
            if mapping is None:
                raise ValueError(f"Scorer audit dependency mapping is missing: {mapping_id}")
            intervals = [
                interval
                for interval in mapping["mapped_intervals"]
                if int(interval["tile_index"]) == int(placement["tile_index"])
                and int(interval["mapped_start_sample"]) == mapped_start
                and int(interval["mapped_end_sample"]) == mapped_end
            ]
            if len(intervals) != 1:
                raise ValueError("Scorer audit occurrence does not match its source mapping")
            source_occurrence_start = int(intervals[0]["source_start_sample"])
            source_occurrence_end = int(intervals[0]["source_end_sample"])
        source_occurrence_samples = source_occurrence_end - source_occurrence_start
        rendered_occurrence_samples = mapped_end - mapped_start
        if source_occurrence_samples != rendered_occurrence_samples:
            raise ValueError("Scorer audit source and target occurrence lengths differ")
        payload.append(
            {
                **placement,
                "schema": ITEM_SCHEMA,
                "item_index": index,
                "item_id": str(placement["placement_id"]),
                "target_audio": copy_audio("target", target_id, target_path),
                "source_audio": copy_audio("source", source_id, source_path),
                "source_clip_audio": clip_audio(
                    source_path, source_event_start, source_event_end
                ),
                "source_occurrence_clip_audio": clip_audio(
                    source_path, source_occurrence_start, source_occurrence_end
                ),
                "rendered_clip_audio": clip_audio(
                    target_path, mapped_start, mapped_end
                ),
                "source_clip_duration_s": (
                    source_event_end - source_event_start
                ) / int(source["sample_rate"]),
                "rendered_clip_duration_s": (
                    mapped_end - mapped_start
                ) / int(target["sample_rate"]),
                "source_event_sample_count": source_event_end - source_event_start,
                "source_occurrence_sample_count": source_occurrence_samples,
                "source_occurrence_start_s": source_occurrence_start
                / int(source["sample_rate"]),
                "source_occurrence_end_s": source_occurrence_end
                / int(source["sample_rate"]),
                "source_occurrence_duration_s": source_occurrence_samples
                / int(source["sample_rate"]),
                "target_duration_s": float(target["duration_s"]),
                "source_event_start_s": source_event_start / int(source["sample_rate"]),
                "source_event_end_s": source_event_end / int(source["sample_rate"]),
                "canonical_spans": [
                    {
                        **span,
                        "start_s": int(span["start_sample"]) / int(target["sample_rate"]),
                        "end_s": int(span["end_sample"]) / int(target["sample_rate"]),
                        "clip_audio": clip_audio(
                            target_path,
                            int(span["start_sample"]),
                            int(span["end_sample"]),
                        ),
                    }
                    for span in target["canonical_spans"]
                ],
                "background_label_change_ranges": [
                    {
                        **span,
                        "start_s": int(span["start_sample"]) / int(target["sample_rate"]),
                        "end_s": int(span["end_sample"]) / int(target["sample_rate"]),
                        "clip_audio": clip_audio(
                            target_path,
                            int(span["start_sample"]),
                            int(span["end_sample"]),
                        ),
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
        "dependency_mappings": str(dependency_mappings_path),
        "dependency_mappings_sha256": _sha256(dependency_mappings_path),
        "repair_events": str(events_path),
        "repair_events_sha256": _sha256(events_path),
        "audit_manifest": str(manifest),
        "audit_manifest_sha256": _sha256(manifest),
        "review_item_count": len(payload),
        "registered_core_item_count": sum(bool(row["core_registered"]) for row in payload),
        "no_label_change_item_count": sum(not bool(row["core_registered"]) for row in payload),
        "standalone_exact_clip_playback": True,
        "standalone_clip_count": len(clips),
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
