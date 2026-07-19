#!/usr/bin/env python3
"""Generate a playable audit for Scorer v10 canonical source labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


PARTITIONS = ("train", "val", "test")


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _even_pick(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            bool(row.get("additive_overlay")),
            float(row.get("duration_s") or 0.0),
            str(row["source_id"]),
        ),
    )
    if count <= 0 or len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indexes = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indexes]


def _background_bucket(row: dict[str, Any]) -> str:
    value = str(row.get("background_type") or "").lower()
    if "speech_fragment" in value:
        return "semantic_leakage_risk"
    if any(token in value for token in ("music", "impact", "mechan", "vehicle", "noise")):
        return "music_impact_noise"
    if "kiss" in value:
        return "kiss"
    if any(token in value for token in ("moan", "groan", "cry", "sob", "vocal")):
        return "moan_cry_vocal"
    if "breath" in value:
        return "breathing"
    if any(token in value for token in ("silence", "pause", "empty")):
        return "silence"
    return "other"


def _diverse_background_pick(
    rows: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    priorities = (
        "semantic_leakage_risk",
        "music_impact_noise",
        "kiss",
        "moan_cry_vocal",
        "breathing",
        "silence",
        "other",
    )
    selected: list[dict[str, Any]] = []
    for bucket in priorities:
        candidates = [row for row in rows if _background_bucket(row) == bucket]
        if candidates:
            selected.append(
                max(
                    candidates,
                    key=lambda row: (
                        float(row.get("duration_s") or 0.0), str(row["source_id"])
                    ),
                )
            )
        if len(selected) >= count:
            return selected
    selected_ids = {str(row["source_id"]) for row in selected}
    remaining = [row for row in rows if str(row["source_id"]) not in selected_ids]
    selected.extend(_even_pick(remaining, count - len(selected)))
    return selected


def select_audit_rows(
    rows: Sequence[dict[str, Any]], *, per_role_partition: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        for role in ("speech", "all_background"):
            pool = [
                row
                for row in rows
                if row.get("partition") == partition and row.get("row_role") == role
            ]
            if not pool:
                raise ValueError(f"canonical audit has no {partition}/{role} rows")
            selected.extend(
                _diverse_background_pick(pool, per_role_partition)
                if role == "all_background"
                else _even_pick(pool, per_role_partition)
            )
    identities = [str(row["source_id"]) for row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("canonical audit selection contains duplicate sources")
    return selected


def build_audit(
    *,
    canonical_sources: Path,
    output_dir: Path,
    per_role_partition: int = 4,
    source_ids: Sequence[str] | None = None,
) -> Path:
    canonical_rows = _rows(canonical_sources)
    if source_ids:
        by_id = {str(row["source_id"]): row for row in canonical_rows}
        missing = [source_id for source_id in source_ids if source_id not in by_id]
        if missing:
            raise ValueError(f"canonical audit source ids are missing: {missing[:3]}")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("canonical audit source ids must be unique")
        selected = [by_id[source_id] for source_id in source_ids]
    else:
        selected = select_audit_rows(
            canonical_rows, per_role_partition=per_role_partition
        )
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        source = Path(str(row["audio"]))
        if not source.is_file():
            raise ValueError(f"canonical audit audio is missing: {source}")
        destination = audio_dir / f"item-{index:03d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        spans = []
        for span in row["canonical_spans"]:
            spans.append(
                {
                    **span,
                    "start_s": int(span["start_sample"]) / int(row["sample_rate"]),
                    "end_s": int(span["end_sample"]) / int(row["sample_rate"]),
                }
            )
        payload.append(
            {
                "source_id": row["source_id"],
                "partition": row["partition"],
                "row_role": row["row_role"],
                "duration_s": row["duration_s"],
                "audio": destination.relative_to(output_dir).as_posix(),
                "spans": spans,
                "core_ids": list(row.get("core_ids") or ()),
                "background_type": str(row.get("background_type") or "mixed_drop_assets"),
                "additive_overlay": row.get("additive_overlay") is not None,
            }
        )

    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scorer v10 canonical data audit</title>
<style>body{{margin:0;background:#f3f5f7;color:#20242a;font-family:Segoe UI,Arial,sans-serif}}header{{position:sticky;top:0;z-index:4;background:#fff;border-bottom:1px solid #c9d0d8;padding:12px 18px}}main{{max-width:1080px;margin:18px auto;padding:0 14px}}section,article{{background:#fff;border:1px solid #c9d0d8;border-radius:8px;padding:14px;margin-bottom:14px}}article.done{{border-left:6px solid #267443}}audio{{width:100%}}.track{{height:40px;display:flex;margin:10px 0;background:#e1e5e9}}.span{{min-width:2px;height:40px;margin:0;padding:7px 4px;border:0;border-right:1px solid rgba(0,0,0,.2);box-sizing:border-box;overflow:hidden;white-space:nowrap;font-size:12px;text-align:left;cursor:pointer}}.span.playing{{outline:3px solid #1769aa;outline-offset:-3px;color:#fff}}.speech{{background:#58aa70}}.background{{background:#d0a14b}}button,textarea{{font:inherit}}button{{padding:6px 9px;margin:3px}}button.active{{background:#1769aa;color:#fff}}button.risk.active{{background:#a52f2f}}textarea{{width:100%;box-sizing:border-box;min-height:56px}}small{{color:#5c6570}}@media(max-width:700px){{.span{{font-size:10px}}}}</style></head><body>
<header><strong>1.7B Scorer v10 canonical data audit</strong> <button id="save">保存裁决</button> <span id="status"></span></header><main><section><b>职责：</b>绿色 Galgame core 是清楚可辨、具字幕价值的目标 speech；黄色来自严格 CueQC/Omni definite-drop，包括呻吟、喘息、呼吸、亲吻、音乐与 impact。点击任一绿色或黄色播放条，只播放该条的精确区间并在终点立即停止；再次点击同一条会立即停止，不添加任何前后上下文。听到黄色中存在清楚语义对白，或绿色 core 中存在不应作为目标 speech 的长段内容时必须标风险；不确定就选 unsure。此页不代表已通过人工 gate。</section><div id="list"></div></main>
<script>const rows={encoded};const key='scorer-v10-canonical-data-audit-v1:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let activeAudio=null,activeButton=null,activeStop=null;function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function ensure(r){{ann[r.source_id]??={{verdict:'',note:''}};return ann[r.source_id];}}function stopPlayback(){{if(activeAudio&&activeStop)activeAudio.removeEventListener('timeupdate',activeStop);if(activeAudio)activeAudio.pause();if(activeButton)activeButton.classList.remove('playing');activeAudio=null;activeButton=null;activeStop=null;}}function playExact(audio,button,start,end){{if(activeAudio===audio&&activeButton===button&&!audio.paused){{stopPlayback();return;}}stopPlayback();activeAudio=audio;activeButton=button;button.classList.add('playing');const begin=()=>{{audio.currentTime=start;activeStop=()=>{{if(audio.currentTime>=end)stopPlayback();}};audio.addEventListener('timeupdate',activeStop);audio.play();}};if(audio.readyState<1){{audio.addEventListener('loadedmetadata',begin,{{once:true}});audio.load();}}else begin();}}function persist(){{localStorage.setItem(key,JSON.stringify(ann));render();}}function buttons(r,a){{const values=r.row_role==='speech'?[['correct','标签正确'],['speech_in_background','黄色含目标对白'],['background_in_speech','绿色含长段非目标'],['unsure','不确定']]:[['correct','全段可 drop'],['contains_target_speech','含目标对白'],['unsure','不确定']];return values.map(v=>`<button data-v="${{v[0]}}" class="${{v[0]!=='correct'?'risk ':''}}${{a.verdict===v[0]?'active':''}}">${{v[1]}}</button>`).join('');}}function render(){{stopPlayback();const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r),card=document.createElement('article');if(a.verdict==='correct')card.classList.add('done');const spans=r.spans.map(s=>`<button type="button" class="span ${{s.label}}" style="width:${{Math.max(.2,100*(s.end_s-s.start_s)/r.duration_s)}}%" data-play-start="${{s.start_s}}" data-play-end="${{s.end_s}}" title="${{esc(s.label_source)}} ${{Number(s.start_s).toFixed(3)}}-${{Number(s.end_s).toFixed(3)}}s">${{esc(s.label)}} ${{Number(s.start_s).toFixed(2)}}–${{Number(s.end_s).toFixed(2)}}s</button>`).join('');card.innerHTML=`<h2>${{esc(r.source_id)}}</h2><small>${{esc(r.partition)}} / ${{esc(r.row_role)}} / ${{Number(r.duration_s).toFixed(3)}}s / ${{esc(r.background_type)}}${{r.additive_overlay?' / overlay':''}}</small><audio controls preload="metadata" src="${{esc(r.audio)}}"></audio><div class="track">${{spans}}</div><div>${{buttons(r,a)}}</div><textarea placeholder="记录具体时间和原因">${{esc(a.note)}}</textarea>`;const audio=card.querySelector('audio');card.querySelectorAll('[data-play-start]').forEach(b=>b.onclick=()=>playExact(audio,b,Number(b.dataset.playStart),Number(b.dataset.playEnd)));audio.addEventListener('ended',stopPlayback);card.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{{a.verdict=b.dataset.v;a.updated_at=new Date().toISOString();persist();}});card.querySelector('textarea').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};root.appendChild(card);}}document.getElementById('status').textContent=`已裁决 ${{rows.filter(r=>ensure(r).verdict).length}}/${{rows.length}}`;}}document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r);return JSON.stringify({{schema:'speech_scorer_v10_canonical_manual_verdict_v1',source_id:r.source_id,partition:r.partition,row_role:r.row_role,verdict:a.verdict||'unreviewed',note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();</script></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    audit_manifest = output_dir / "audit_manifest.jsonl"
    audit_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload),
        encoding="utf-8",
    )
    summary = {
        "schema": "speech_scorer_v10_canonical_data_audit_summary_v1",
        "title": "Scorer v10 canonical data audit",
        "review_item_count": len(payload),
        "canonical_sources": str(canonical_sources),
        "canonical_sources_sha256": hashlib.sha256(canonical_sources.read_bytes()).hexdigest(),
        "audit_manifest": str(audit_manifest),
        "partition_role_counts": {
            f"{partition}/{role}": sum(
                row["partition"] == partition and row["row_role"] == role
                for row in payload
            )
            for partition in PARTITIONS
            for role in ("speech", "all_background")
        },
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-role-partition", type=int, default=4)
    parser.add_argument("--source-id", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        build_audit(
            canonical_sources=Path(args.canonical_sources),
            output_dir=Path(args.output_dir),
            per_role_partition=args.per_role_partition,
            source_ids=args.source_id or None,
        )
    )
