#!/usr/bin/env python3
"""Generate exact-span repair audit for failed Scorer v10 canonical sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


SOURCE_VERDICT_SCHEMA = "speech_scorer_v10_canonical_manual_verdict_v1"
SPAN_VERDICT_SCHEMA = "speech_scorer_v10_canonical_span_manual_verdict_v1"
SUMMARY_SCHEMA = "speech_scorer_v10_canonical_span_repair_audit_summary_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_audit(
    *, canonical_sources: Path, source_verdicts: Path, output_dir: Path
) -> Path:
    sources = {str(row["source_id"]): row for row in _rows(canonical_sources)}
    verdicts: dict[str, dict[str, Any]] = {}
    for row in _rows(source_verdicts):
        if row.get("schema") != SOURCE_VERDICT_SCHEMA:
            raise ValueError("invalid Scorer v10 canonical source verdict schema")
        source_id = str(row.get("source_id") or "")
        if source_id not in sources or source_id in verdicts:
            raise ValueError(f"invalid or duplicate source verdict: {source_id}")
        verdicts[source_id] = row

    repair_source_ids = sorted(
        source_id
        for source_id, verdict in verdicts.items()
        if verdict.get("verdict") != "correct"
        and sources[source_id].get("row_role") == "speech"
    )
    quarantined_background_ids = sorted(
        str(sources[source_id]["background_id"])
        for source_id, verdict in verdicts.items()
        if verdict.get("verdict") == "contains_target_speech"
        and sources[source_id].get("row_role") == "all_background"
    )
    if not repair_source_ids:
        raise ValueError("no failed speech sources require span repair")

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for source_index, source_id in enumerate(repair_source_ids):
        source = sources[source_id]
        audio_source = Path(str(source["audio"]))
        if not audio_source.is_file():
            raise ValueError(f"span repair audio is missing: {audio_source}")
        audio_target = audio_dir / f"source-{source_index:02d}{audio_source.suffix.lower()}"
        shutil.copy2(audio_source, audio_target)
        spans: list[dict[str, Any]] = []
        for span_index, raw_span in enumerate(source["canonical_spans"]):
            span = {
                **raw_span,
                "span_id": f"{source_id}::span{span_index:02d}",
                "span_index": span_index,
                "start_s": int(raw_span["start_sample"]) / int(source["sample_rate"]),
                "end_s": int(raw_span["end_sample"]) / int(source["sample_rate"]),
            }
            spans.append(span)
            manifest_rows.append(
                {
                    "schema": "speech_scorer_v10_canonical_span_repair_item_v1",
                    "span_id": span["span_id"],
                    "source_id": source_id,
                    "partition": source["partition"],
                    "parent_verdict": verdicts[source_id]["verdict"],
                    "parent_note": str(verdicts[source_id].get("note") or ""),
                    "original_label": span["label"],
                    "start_sample": span["start_sample"],
                    "end_sample": span["end_sample"],
                    "start_s": span["start_s"],
                    "end_s": span["end_s"],
                    "background_id": str(span.get("background_id") or ""),
                    "core_id": str(span.get("core_id") or ""),
                }
            )
        payload.append(
            {
                "source_id": source_id,
                "partition": source["partition"],
                "duration_s": source["duration_s"],
                "audio": audio_target.relative_to(output_dir).as_posix(),
                "parent_verdict": verdicts[source_id]["verdict"],
                "parent_note": str(verdicts[source_id].get("note") or ""),
                "spans": spans,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_manifest = output_dir / "audit_manifest.jsonl"
    audit_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scorer v10 span repair audit</title>
<style>body{{margin:0;background:#f3f5f7;color:#20242a;font-family:Segoe UI,Arial,sans-serif}}header{{position:sticky;top:0;z-index:4;background:#fff;border-bottom:1px solid #c9d0d8;padding:12px 18px}}main{{max-width:1050px;margin:18px auto;padding:0 14px}}section{{background:#fff;border:1px solid #c9d0d8;border-radius:8px;padding:14px;margin-bottom:14px}}audio{{width:100%}}.span-row{{display:grid;grid-template-columns:minmax(260px,1fr) minmax(310px,auto);gap:10px;align-items:center;padding:9px 0;border-top:1px solid #d7dde3}}.play{{width:100%;margin:0;border:0;padding:9px;text-align:left;cursor:pointer}}.speech{{background:#58aa70}}.background{{background:#d0a14b}}.play.playing{{outline:3px solid #1769aa;outline-offset:-3px;color:#fff}}button,textarea{{font:inherit}}.choice{{padding:6px 9px;margin:2px}}.choice.active{{background:#1769aa;color:#fff}}textarea{{width:100%;box-sizing:border-box;min-height:52px}}small{{color:#5c6570}}@media(max-width:760px){{.span-row{{grid-template-columns:1fr}}}}</style></head><body>
<header><strong>1.7B Scorer v10 canonical span repair</strong> <button id="save">保存逐 span 裁决</button> <span id="status"></span></header><main><section><b>只裁当前精确 span：</b>点击绿色/黄色条后只从该 span 起点播放到终点并立即停止，无任何上下文。每条独立选择它实际应是 speech、background 或 unsure。不要根据原颜色迁就旧标签。已确认含目标对白的 3 个 all-background 资产已进入隔离清单，无需在本页重复试听。</section><div id="list"></div></main>
<script>const rows={encoded};const key='scorer-v10-canonical-span-repair-v1:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let activeAudio=null,activeButton=null,activeStop=null;function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function ensure(span){{ann[span.span_id]??={{verdict:'',note:''}};return ann[span.span_id];}}function stopPlayback(){{if(activeAudio&&activeStop)activeAudio.removeEventListener('timeupdate',activeStop);if(activeAudio)activeAudio.pause();if(activeButton)activeButton.classList.remove('playing');activeAudio=null;activeButton=null;activeStop=null;}}function playExact(audio,button,start,end){{if(activeAudio===audio&&activeButton===button&&!audio.paused){{stopPlayback();return;}}stopPlayback();activeAudio=audio;activeButton=button;button.classList.add('playing');const begin=()=>{{audio.currentTime=start;activeStop=()=>{{if(audio.currentTime>=end)stopPlayback();}};audio.addEventListener('timeupdate',activeStop);audio.play();}};if(audio.readyState<1){{audio.addEventListener('loadedmetadata',begin,{{once:true}});audio.load();}}else begin();}}function choices(span,a){{return ['speech','background','unsure'].map(v=>`<button class="choice ${{a.verdict===v?'active':''}}" data-v="${{v}}">${{v}}</button>`).join('');}}function persist(){{localStorage.setItem(key,JSON.stringify(ann));render();}}function render(){{stopPlayback();const root=document.getElementById('list');root.innerHTML='';for(const row of rows){{const section=document.createElement('section');section.innerHTML=`<h2>${{esc(row.source_id)}}</h2><small>${{esc(row.partition)}} / 原裁决=${{esc(row.parent_verdict)}} / 备注=${{esc(row.parent_note||'无')}}</small><audio controls preload="metadata" src="${{esc(row.audio)}}"></audio><div class="spans"></div>`;const audio=section.querySelector('audio'),spans=section.querySelector('.spans');for(const span of row.spans){{const a=ensure(span),line=document.createElement('div');line.className='span-row';line.innerHTML=`<button type="button" class="play ${{span.label}}" data-start="${{span.start_s}}" data-end="${{span.end_s}}">原标签 ${{esc(span.label)}} / ${{Number(span.start_s).toFixed(3)}}–${{Number(span.end_s).toFixed(3)}}s</button><div>${{choices(span,a)}} <input type="text" placeholder="备注" value="${{esc(a.note)}}"></div>`;const play=line.querySelector('.play');play.onclick=()=>playExact(audio,play,Number(play.dataset.start),Number(play.dataset.end));line.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{{a.verdict=b.dataset.v;a.updated_at=new Date().toISOString();persist();}});line.querySelector('input').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};spans.appendChild(line);}}root.appendChild(section);}}const all=rows.flatMap(r=>r.spans);document.getElementById('status').textContent=`已裁决 ${{all.filter(s=>ensure(s).verdict).length}}/${{all.length}}`;}}document.getElementById('save').onclick=async()=>{{const content=rows.flatMap(r=>r.spans.map(s=>{{const a=ensure(s);return JSON.stringify({{schema:'{SPAN_VERDICT_SCHEMA}',span_id:s.span_id,source_id:r.source_id,original_label:s.label,verdict:a.verdict||'unreviewed',note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}})).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();</script></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "title": "Scorer v10 canonical span repair",
        "canonical_sources": str(canonical_sources),
        "canonical_sources_sha256": hashlib.sha256(canonical_sources.read_bytes()).hexdigest(),
        "source_verdicts": str(source_verdicts),
        "source_verdicts_sha256": hashlib.sha256(source_verdicts.read_bytes()).hexdigest(),
        "repair_source_count": len(payload),
        "review_item_count": len(manifest_rows),
        "audit_manifest": str(audit_manifest),
        "quarantined_background_ids": quarantined_background_ids,
        "manual_gate_status": "pending",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    update_audit_entrypoints(latest_html=index, title=summary["title"])
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-sources", required=True)
    parser.add_argument("--source-verdicts", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        build_audit(
            canonical_sources=Path(args.canonical_sources),
            source_verdicts=Path(args.source_verdicts),
            output_dir=Path(args.output_dir),
        )
    )
