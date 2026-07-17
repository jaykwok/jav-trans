#!/usr/bin/env python3
"""Build an audio-only CueQC audit over provisional sub-islands before Inner."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints


ITEM_SCHEMA = "pre_inner_cueqc_audit_item_v1"
SUMMARY_SCHEMA = "pre_inner_cueqc_audit_summary_v1"
VERDICT_SCHEMA = "pre_inner_cueqc_manual_verdict_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_partition(rows: list[dict[str, Any]]) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["sample_id"])].append(row)
    for sample_id, group in by_source.items():
        ordered = sorted(group, key=lambda row: float(row["start_s"]))
        for left, right in zip(ordered, ordered[1:]):
            if float(right["start_s"]) + 1e-6 < float(left["end_s"]):
                raise ValueError(f"{sample_id}: provisional sub-islands must not overlap")
        if any(float(row["end_s"]) <= float(row["start_s"]) for row in ordered):
            raise ValueError(f"{sample_id}: provisional sub-island must have positive duration")


def build_items(*, subislands: Path, output_dir: Path) -> list[dict[str, Any]]:
    rows = _rows(subislands)
    if not rows:
        raise ValueError("provisional sub-islands are empty")
    validate_partition(rows)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    result: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in copied:
            source = Path(str(row["audio"]))
            target = audio_dir / f"{sample_id}{source.suffix.lower() or '.wav'}"
            shutil.copyfile(source, target)
            copied[sample_id] = target.relative_to(output_dir).as_posix()
        result.append(
            {
                "schema": ITEM_SCHEMA,
                "sample_id": sample_id,
                "subisland_id": str(row["subisland_id"]),
                "audio": copied[sample_id],
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "duration_s": float(row["duration_s"]),
                "left_event_id": row.get("left_event_id"),
                "right_event_id": row.get("right_event_id"),
                "decision_contract": "whole_provisional_subisland_content_before_inner_v1",
            }
        )
    return result


def build_page(*, rows: list[dict[str, Any]], output_dir: Path, update_latest: bool = True) -> Path:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["sample_id"])].append(row)
    payload = json.dumps(
        [
            {
                "sample_id": sample_id,
                "audio": group[0]["audio"],
                "subislands": sorted(group, key=lambda row: float(row["start_s"])),
            }
            for sample_id, group in sorted(by_source.items())
        ],
        ensure_ascii=False,
    ).replace("</", "<\\/")
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Pre-Inner CueQC fixed audit</title><style>
body{{margin:0;background:#0d1117;color:#e6edf3;font-family:system-ui}}header{{position:sticky;top:0;background:#161b22;padding:12px 18px;border-bottom:1px solid #30363d}}main{{max-width:1100px;margin:auto;padding:16px}}article{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin:14px 0}}.help{{background:#10233f;padding:14px;border-radius:10px}}audio{{width:100%}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #30363d;text-align:left}}button{{margin:3px;padding:6px 10px;background:#21262d;color:#e6edf3;border:1px solid #484f58;border-radius:6px}}button.active{{background:#1f6feb}}button.keep.active{{background:#238636}}button.drop.active{{background:#da3633}}button.unsure.active{{background:#8250df}}input{{width:95%;background:#0d1117;color:#e6edf3}}small{{color:#8b949e}}
</style></head><body><header><strong>Pre-Inner CueQC · provisional sub-islands</strong> <button id="save">保存 CueQC 标签</button> <span id="status"></span></header><main><section class="help"><h2>判断整块是否保留；不要判断边缘是否精确</h2><p><b>keep：</b>只要包含任何需要保留的真实语音，即使前后静音很宽、夹杂噪声或边缘不准，也必须保留，之后交给 Inner 修边。<b>drop：</b>整块没有目标语音，只有静音、BGM、环境噪声、嘈杂背景人声、喘息、呻吟、亲吻、哭声等非语义声音。<b>unsure：</b>混合、重叠或听不清；只保留在 teacher/data 审计层并从二分类训练、loss、metrics 排除，runtime 模型不会输出 unsure。</p><p>每个 source 的 provisional sub-islands 按时间排列且互不重叠；不同 Scorer island 之间允许存在未输出区间。本页只决定整块去留，不评价字幕、ASR 或 Inner edge。</p></section><div id="list"></div></main><script>
const rows={payload};const key='pre-inner-cueqc-audit-v1:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let active=null,timer=null;function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function ensure(id){{ann[id]??={{label:'',note:''}};return ann[id];}}function persist(){{localStorage.setItem(key,JSON.stringify(ann));status();}}function status(){{const all=rows.flatMap(r=>r.subislands),done=all.filter(x=>ensure(x.subisland_id).label).length;document.getElementById('status').textContent=`完成 ${{done}}/${{all.length}}`;}}function play(audio,start,end){{if(active&&active!==audio)active.pause();active=audio;if(timer)clearTimeout(timer);audio.currentTime=Number(start);audio.play();timer=setTimeout(()=>audio.pause(),Math.max(1,(Number(end)-Number(start))*1000));}}function choice(a,x,label,text){{return `<button data-id="${{esc(x.subisland_id)}}" data-label="${{label}}" class="${{label}} ${{a.label===label?'active':''}}">${{text}}</button>`;}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const card=document.createElement('article'),body=r.subislands.map(x=>{{const a=ensure(x.subisland_id);return `<tr><td><b>${{esc(x.subisland_id.split('__').pop())}}</b><br>${{Number(x.start_s).toFixed(3)}}–${{Number(x.end_s).toFixed(3)}}s<br><small>${{Number(x.duration_s).toFixed(3)}}s</small></td><td><button data-start="${{x.start_s}}" data-end="${{x.end_s}}">播放完整 sub-island</button></td><td>${{choice(a,x,'keep','keep')}}${{choice(a,x,'drop','drop')}}${{choice(a,x,'unsure','unsure')}}<br><input data-note="${{esc(x.subisland_id)}}" placeholder="备注" value="${{esc(a.note)}}"></td></tr>`;}}).join('');card.innerHTML=`<h2>${{esc(r.sample_id)}}</h2><audio controls preload="metadata" src="${{esc(r.audio)}}"></audio><table><thead><tr><th>provisional sub-island</th><th>试听</th><th>CueQC</th></tr></thead><tbody>${{body}}</tbody></table>`;const audio=card.querySelector('audio');card.querySelectorAll('[data-start]').forEach(b=>b.onclick=()=>play(audio,b.dataset.start,b.dataset.end));card.querySelectorAll('[data-id]').forEach(b=>b.onclick=()=>{{const a=ensure(b.dataset.id);a.label=b.dataset.label;a.updated_at=new Date().toISOString();persist();render();}});card.querySelectorAll('[data-note]').forEach(i=>i.onchange=()=>{{const a=ensure(i.dataset.note);a.note=i.value;a.updated_at=new Date().toISOString();persist();}});root.appendChild(card);}}status();}}
document.getElementById('save').onclick=async()=>{{const content=rows.flatMap(r=>r.subislands.map(x=>{{const a=ensure(x.subisland_id);return JSON.stringify({{schema:'{VERDICT_SCHEMA}',sample_id:x.sample_id,subisland_id:x.subisland_id,start_s:x.start_s,end_s:x.end_s,label:a.label||'unreviewed',note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}})).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();
</script></body></html>"""
    output_dir.mkdir(parents=True, exist_ok=True)
    page = output_dir / "index.html"
    page.write_text(html, encoding="utf-8")
    if update_latest:
        update_audit_entrypoints(latest_html=page, title="Pre-Inner CueQC fixed audit")
    return page


def build(*, subislands: Path, output_dir: Path, update_latest: bool = True) -> dict[str, Any]:
    rows = build_items(subislands=subislands, output_dir=output_dir)
    items_path = output_dir / "cueqc_items.jsonl"
    _write(items_path, rows)
    page = build_page(rows=rows, output_dir=output_dir, update_latest=update_latest)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "source_count": len({row["sample_id"] for row in rows}),
        "subisland_count": len(rows),
        "labels": ["keep", "drop", "unsure"],
        "input_stage": "after_acoustic_split_before_inner",
        "items": str(items_path),
        "page": str(page),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pre-Inner CueQC audit.")
    parser.add_argument("--subislands", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build(
                subislands=Path(args.subislands),
                output_dir=Path(args.output_dir),
                update_latest=not args.no_update_latest,
            ),
            ensure_ascii=False,
        )
    )
