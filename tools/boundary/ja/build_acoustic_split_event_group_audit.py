#!/usr/bin/env python3
"""Build a minimal manual audit for grouping sparse Split candidates into events."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints


ITEM_SCHEMA = "acoustic_split_event_group_audit_v1"
SUMMARY_SCHEMA = "acoustic_split_event_group_audit_summary_v1"
VERDICT_SCHEMA = "acoustic_split_event_group_manual_verdict_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_runs(
    candidates: list[dict[str, Any]], verdict_candidates: list[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    label_by_id = {
        str(row["candidate_id"]): str(row.get("label") or "")
        for row in verdict_candidates
    }
    runs: list[list[dict[str, Any]]] = []
    active: list[dict[str, Any]] = []
    for candidate in candidates:
        if label_by_id.get(str(candidate["candidate_id"])) == "split":
            active.append(candidate)
        elif active:
            runs.append(active)
            active = []
    if active:
        runs.append(active)
    return runs


def build_items(*, audit_items: Path, verdicts: Path, output_dir: Path) -> list[dict[str, Any]]:
    items = _rows(audit_items)
    verdict_by_id = {str(row["sample_id"]): row for row in _rows(verdicts)}
    rows: list[dict[str, Any]] = []
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        sample_id = str(item["sample_id"])
        verdict = verdict_by_id.get(sample_id)
        if verdict is None or not verdict.get("complete"):
            raise ValueError(f"{sample_id}: complete acoustic Split verdict required")
        runs = split_runs(item["candidates"], verdict["candidates"])
        links: list[dict[str, Any]] = []
        serialized_runs: list[dict[str, Any]] = []
        for run_index, run in enumerate(runs):
            serialized_runs.append(
                {
                    "run_id": f"r{run_index:02d}",
                    "candidate_ids": [str(row["candidate_id"]) for row in run],
                    "candidates": [dict(row) for row in run],
                }
            )
            for left, right in zip(run, run[1:]):
                links.append(
                    {
                        "link_id": f"{left['candidate_id']}__{right['candidate_id']}",
                        "run_id": f"r{run_index:02d}",
                        "left_candidate_id": str(left["candidate_id"]),
                        "right_candidate_id": str(right["candidate_id"]),
                        "left_time_s": float(left["time_s"]),
                        "right_time_s": float(right["time_s"]),
                        "left_probability": float(left["proposer_probability"]),
                        "right_probability": float(right["proposer_probability"]),
                        "play_start_s": float(left["context_start_s"]),
                        "play_end_s": float(right["context_end_s"]),
                    }
                )
        source = Path(audit_items).parent / str(item["audio"])
        target = audio_dir / f"{sample_id}{source.suffix.lower() or '.wav'}"
        shutil.copyfile(source, target)
        rows.append(
            {
                "schema": ITEM_SCHEMA,
                "sample_id": sample_id,
                "audio": target.relative_to(output_dir).as_posix(),
                "duration_s": float(item["duration_s"]),
                "runs": serialized_runs,
                "links": links,
                "split_candidate_count": sum(len(run) for run in runs),
                "automatic_singleton_event_count": sum(len(run) == 1 for run in runs),
            }
        )
    return rows


def build_page(*, rows: list[dict[str, Any]], output_dir: Path, update_latest: bool = True) -> Path:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Acoustic Split event grouping</title><style>
body{{margin:0;background:#0d1117;color:#e6edf3;font-family:system-ui}}header{{position:sticky;top:0;background:#161b22;padding:12px 18px;border-bottom:1px solid #30363d}}main{{max-width:1100px;margin:auto;padding:16px}}article{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin:14px 0}}article.done{{border-color:#2ea043}}.help{{background:#10233f;padding:14px;border-radius:10px}}audio{{width:100%}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #30363d;text-align:left}}button{{margin:3px;padding:6px 10px;background:#21262d;color:#e6edf3;border:1px solid #484f58;border-radius:6px}}button.active{{background:#1f6feb}}textarea{{width:100%;min-height:55px;background:#0d1117;color:#e6edf3}}small{{color:#8b949e}}
</style></head><body><header><strong>Acoustic Split v3 · event grouping</strong> <button id="save">保存分组</button> <span id="status"></span></header><main><section class="help"><h2>只补稀疏 split candidates 的 event 分组</h2><p><b>同一 gap：</b>左右候选落在同一段静音/杂音 basin，最终只形成一个 event。<b>新 gap：</b>右候选属于后面的另一个停顿，应开始新 event。<b>不确定：</b>无法稳定分组；该 run 不进入训练。</p><p>被 continue 隔开的 split runs 已自动分开；只有一个候选的 run 自动成为单 event。本页不重审 Split、CueQC 或 Inner。</p></section><div id="list"></div></main><script>
const rows={payload};const key='acoustic-split-event-group-audit-v1:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let timer=null,active=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function ensure(r){{ann[r.sample_id]??={{links:{{}},note:''}};return ann[r.sample_id];}}function complete(r,a){{return r.links.every(x=>a.links[x.link_id]);}}function persist(){{localStorage.setItem(key,JSON.stringify(ann));status();}}function status(){{document.getElementById('status').textContent=`完成 ${{rows.filter(r=>complete(r,ensure(r))).length}}/${{rows.length}}`;}}function play(audio,start,end){{if(active&&active!==audio)active.pause();active=audio;if(timer)clearTimeout(timer);audio.currentTime=Number(start);audio.play();timer=setTimeout(()=>audio.pause(),Math.max(1,(Number(end)-Number(start))*1000));}}function choice(a,x,label,text){{return `<button data-link="${{esc(x.link_id)}}" data-label="${{label}}" class="${{a.links[x.link_id]===label?'active':''}}">${{text}}</button>`;}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r),card=document.createElement('article');if(complete(r,a))card.classList.add('done');const body=r.links.length?r.links.map(x=>`<tr><td>${{esc(x.left_candidate_id)}} ${{Number(x.left_time_s).toFixed(3)}}s<br>↔<br>${{esc(x.right_candidate_id)}} ${{Number(x.right_time_s).toFixed(3)}}s</td><td><button data-start="${{x.play_start_s}}" data-end="${{x.play_end_s}}">播放两候选覆盖区</button><br><small>p=${{Number(x.left_probability).toFixed(3)}} / ${{Number(x.right_probability).toFixed(3)}}</small></td><td>${{choice(a,x,'same_event','同一 gap')}}${{choice(a,x,'new_event','新 gap')}}${{choice(a,x,'unsure','不确定')}}</td></tr>`).join(''):'<tr><td colspan="3">全部 split run 都是 singleton，自动完成。</td></tr>';card.innerHTML=`<h2>${{esc(r.sample_id)}}</h2><small>split candidates=${{r.split_candidate_count}} · singleton events=${{r.automatic_singleton_event_count}}</small><audio controls preload="metadata" src="${{esc(r.audio)}}"></audio><table><thead><tr><th>相邻 split 候选</th><th>试听</th><th>是否同一 event</th></tr></thead><tbody>${{body}}</tbody></table><textarea placeholder="分组备注">${{esc(a.note)}}</textarea>`;const audio=card.querySelector('audio');card.querySelectorAll('[data-start]').forEach(b=>b.onclick=()=>play(audio,b.dataset.start,b.dataset.end));card.querySelectorAll('[data-link]').forEach(b=>b.onclick=()=>{{a.links[b.dataset.link]=b.dataset.label;a.updated_at=new Date().toISOString();persist();render();}});card.querySelector('textarea').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};root.appendChild(card);}}status();}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r);return JSON.stringify({{schema:'{VERDICT_SCHEMA}',sample_id:r.sample_id,links:r.links.map(x=>({{link_id:x.link_id,left_candidate_id:x.left_candidate_id,right_candidate_id:x.right_candidate_id,decision:a.links[x.link_id]||'unreviewed'}})),complete:complete(r,a),note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();
</script></body></html>"""
    output_dir.mkdir(parents=True, exist_ok=True)
    page = output_dir / "index.html"
    page.write_text(html, encoding="utf-8")
    if update_latest:
        update_audit_entrypoints(latest_html=page, title="Acoustic Split event grouping")
    return page


def build(*, audit_items: Path, verdicts: Path, output_dir: Path, update_latest: bool = True) -> dict[str, Any]:
    rows = build_items(audit_items=audit_items, verdicts=verdicts, output_dir=output_dir)
    items_path = output_dir / "group_items.jsonl"
    _write(items_path, rows)
    page = build_page(rows=rows, output_dir=output_dir, update_latest=update_latest)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "source_count": len(rows),
        "split_candidate_count": sum(row["split_candidate_count"] for row in rows),
        "manual_link_count": sum(len(row["links"]) for row in rows),
        "automatic_singleton_event_count": sum(
            row["automatic_singleton_event_count"] for row in rows
        ),
        "items": str(items_path),
        "page": str(page),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build acoustic Split event grouping audit.")
    parser.add_argument("--audit-items", required=True)
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build(
                audit_items=Path(args.audit_items),
                verdicts=Path(args.verdicts),
                output_dir=Path(args.output_dir),
                update_latest=not args.no_update_latest,
            ),
            ensure_ascii=False,
        )
    )
