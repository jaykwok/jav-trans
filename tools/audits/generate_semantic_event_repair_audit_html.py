#!/usr/bin/env python3
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


DECISIONS = ("semantic_split", "acoustic_continue", "outer_only", "unsure")


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_repair_rows(
    *, events: Path, verdicts: Path, specs: Path
) -> list[dict[str, Any]]:
    event_rows = {str(row["event_key"]): row for row in _rows(events)}
    verdict_rows = {str(row["event_key"]): row for row in _rows(verdicts)}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in _rows(specs):
        repair_id = str(spec["repair_id"])
        if repair_id in seen:
            raise ValueError(f"duplicate repair_id: {repair_id}")
        seen.add(repair_id)
        event_key = str(spec["event_key"])
        candidate_id = str(spec["candidate_id"])
        event = event_rows[event_key]
        verdict = verdict_rows[event_key]
        candidates = {
            str(row["candidate_id"]): row for row in event.get("candidates") or []
        }
        candidate_labels = {
            str(row["candidate_id"]): str(row.get("label") or "")
            for row in verdict.get("candidates") or []
        }
        if candidate_id not in candidates:
            raise ValueError(f"unknown candidate {event_key}/{candidate_id}")
        if candidate_labels.get(candidate_id) != "safe":
            raise ValueError(
                f"repair candidate must have an approved safe label: {event_key}/{candidate_id}"
            )
        candidate = candidates[candidate_id]
        result.append(
            {
                "schema": "semantic_event_repair_candidate_v1",
                "repair_id": repair_id,
                "sample_id": str(event["sample_id"]),
                "source_event_key": event_key,
                "candidate_id": candidate_id,
                "time_s": float(candidate["time_s"]),
                "left_text": str(spec["left_text"]),
                "right_text": str(spec["right_text"]),
                "scope_hint": str(spec.get("scope_hint") or ""),
                "reason": str(spec.get("reason") or ""),
                "full_audio": str(event["audio"]),
                "left_audio": str(candidate["left_audio"]),
                "right_audio": str(candidate["right_audio"]),
                "tick_audio": str(candidate["tick_audio"]),
                "proposer_probability": float(candidate["proposer_probability"]),
                "projection_file_sha256": str(event["projection_file_sha256"]),
                "proposer_sha256": str(event["proposer_sha256"]),
            }
        )
    if len(result) != 5:
        raise ValueError("semantic event repair smoke must contain exactly five items")
    return result


def _copy(path: Path, destination: Path, root: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return destination.relative_to(root).as_posix()


def build_audit(
    *, rows: list[dict[str, Any]], output_dir: Path, update_latest: bool = True
) -> Path:
    payload_rows: list[dict[str, Any]] = []
    for row in rows:
        item_dir = output_dir / "audio" / str(row["repair_id"])
        copied = dict(row)
        copied["full_audio"] = _copy(
            Path(row["full_audio"]),
            item_dir / f"full{Path(row['full_audio']).suffix}",
            output_dir,
        )
        for field, filename in (
            ("left_audio", "left.wav"),
            ("right_audio", "right.wav"),
            ("tick_audio", "tick.wav"),
        ):
            copied[field] = _copy(Path(row[field]), item_dir / filename, output_dir)
        payload_rows.append(copied)

    payload = json.dumps(payload_rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Semantic Event Repair Smoke5</title>
<style>body{{margin:0;background:#f3f5f7;color:#20242a;font-family:Segoe UI,Arial,sans-serif}}header{{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid #cbd2da;padding:12px 18px}}main{{max-width:1050px;margin:18px auto;padding:0 14px}}article,.help{{background:#fff;border:1px solid #cbd2da;border-radius:8px;padding:15px;margin-bottom:16px}}article.done{{border-left:6px solid #1b7a3a}}.help{{border-left:6px solid #1769aa}}.texts{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.text{{background:#f7f9fb;border:1px solid #d6dce3;border-radius:6px;padding:10px}}.audio-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}audio{{width:100%}}button,textarea{{font:inherit}}button{{padding:7px 10px;margin:3px}}button.active{{background:#1769aa;color:#fff}}button.split.active{{background:#1b7a3a}}button.outer.active{{background:#8a4b08}}textarea{{box-sizing:border-box;width:100%;min-height:58px}}small{{color:#59616a}}@media(max-width:760px){{.texts,.audio-grid{{grid-template-columns:1fr}}}}</style></head><body>
<header><strong>Multi-safe-run · Semantic Event Repair Smoke5</strong>　<button id="save">保存全部裁决</button> <span id="status"></span></header><main><section class="help"><h2>裁决对象是“这个 safe 点属于哪种边界”</h2><p><b>新增/保留 Semantic Split：</b>左右文本可作为两个独立自然字幕单元。<br><b>声学安全但语义继续：</b>这里没截字，但左右不应拆成两条字幕。<br><b>仅 Outer 边缘：</b>位于首个语义单元之前或最后单元之后，不是内部 Split。<br><b>不确定：</b>当前证据不足。</p><p>左/右试听仍是精确 hard cut；tick 只定位，不插入时间。</p></section><div id="list"></div></main><script>
const rows={payload};const key='semantic-event-repair-smoke5-v1:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let active=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function ensure(r){{ann[r.repair_id]??={{decision:'',note:''}};return ann[r.repair_id];}}function persist(){{localStorage.setItem(key,JSON.stringify(ann));status();}}function status(){{document.getElementById('status').textContent=`已裁决 ${{rows.filter(r=>ensure(r).decision).length}} / ${{rows.length}}`;}}function play(a){{if(active&&active!==a)active.pause();active=a;}}function button(a,value,label,cls=''){{return `<button data-decision="${{value}}" class="${{cls}} ${{a.decision===value?'active':''}}">${{label}}</button>`;}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r),card=document.createElement('article');if(a.decision)card.classList.add('done');card.innerHTML=`<h2>${{esc(r.repair_id)}} · ${{Number(r.time_s).toFixed(3)}}s</h2><p><b>范围提示：</b>${{esc(r.scope_hint)}}<br><small>${{esc(r.reason)}}</small></p><div class="texts"><div class="text"><b>候选左侧文本</b><br>${{esc(r.left_text)}}</div><div class="text"><b>候选右侧文本</b><br>${{esc(r.right_text)}}</div></div><p><b>完整 source</b></p><audio controls preload="metadata" src="${{esc(r.full_audio)}}"></audio><div class="audio-grid"><div><b>左侧硬截断</b><audio controls preload="metadata" src="${{esc(r.left_audio)}}"></audio></div><div><b>右侧硬起播</b><audio controls preload="metadata" src="${{esc(r.right_audio)}}"></audio></div></div><p><b>定位 tick（可不听）</b></p><audio controls preload="metadata" src="${{esc(r.tick_audio)}}"></audio><div>${{button(a,'semantic_split','新增/保留 Semantic Split','split')}}${{button(a,'acoustic_continue','声学安全但语义继续')}}${{button(a,'outer_only','仅 Outer 边缘','outer')}}${{button(a,'unsure','不确定')}}</div><label><b>备注</b><textarea>${{esc(a.note)}}</textarea></label>`;card.querySelectorAll('audio').forEach(x=>x.onplay=()=>play(x));card.querySelectorAll('[data-decision]').forEach(b=>b.onclick=()=>{{a.decision=b.dataset.decision;a.updated_at=new Date().toISOString();persist();render();}});card.querySelector('textarea').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};root.appendChild(card);}}status();}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r);return JSON.stringify({{schema:'semantic_event_repair_manual_verdict_v1',repair_id:r.repair_id,sample_id:r.sample_id,source_event_key:r.source_event_key,candidate_id:r.candidate_id,time_s:r.time_s,left_text:r.left_text,right_text:r.right_text,decision:a.decision||'unreviewed',note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();
</script></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    if update_latest:
        update_audit_entrypoints(
            latest_html=index, title="Semantic Event Repair Smoke5"
        )
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed-five semantic event repair audit from approved safe candidates."
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--specs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = build_repair_rows(
        events=Path(args.events),
        verdicts=Path(args.verdicts),
        specs=Path(args.specs),
    )
    print(
        build_audit(
            rows=rows,
            output_dir=Path(args.output_dir),
            update_latest=not args.no_update_latest,
        )
    )
