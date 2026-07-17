#!/usr/bin/env python3
"""Generate candidate-level correction audit for manually confirmed Split v4 residual misses."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def build(*, source_dir: Path, verdict_paths: list[Path], output_dir: Path) -> dict:
    verdicts: dict[str, dict] = {}
    for path in verdict_paths:
        for row in _rows(path):
            if row.get("verdict") not in {None, "", "unreviewed"}:
                verdicts[str(row["audit_id"])] = row
    selected = [
        row
        for row in _rows(source_dir / "audit_manifest.jsonl")
        if verdicts.get(str(row["audit_id"]), {}).get("verdict") == "missing_cut"
    ]
    if not selected:
        raise ValueError("no manually confirmed missing-cut residuals found")
    if any(not row.get("residual_candidates") for row in selected):
        raise ValueError("missing-cut residual has no eligible binary candidates")

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    copied: set[str] = set()
    for row in selected:
        filename = Path(str(row["audio_src"])).name
        if filename not in copied:
            shutil.copyfile(source_dir / "audio" / filename, audio_dir / filename)
            copied.add(filename)
    (output_dir / "candidate_items.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        "utf-8",
    )
    payload = json.dumps(selected, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Split v4 missing-cut candidate audit</title><style>
body{{margin:0;background:#0d1117;color:#e6edf3;font-family:system-ui}}header{{position:sticky;top:0;z-index:2;background:#161b22;padding:12px 18px;border-bottom:1px solid #30363d}}main{{max-width:1100px;margin:auto;padding:16px}}article{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin:14px 0}}article.done{{border-color:#2ea043}}.help{{background:#10233f;padding:14px;border-radius:10px}}audio{{width:100%}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #30363d;text-align:left}}button,input{{margin:3px;padding:6px 10px;background:#21262d;color:#e6edf3;border:1px solid #484f58;border-radius:6px}}button.active{{background:#1f6feb}}button.cut.active{{background:#da3633}}button.continue.active{{background:#238636}}button.unsure.active{{background:#8250df}}input{{width:95%}}small{{color:#8b949e}}</style></head><body>
<header><strong>Acoustic Split v4 · Missing-cut 候选补标</strong> <button id="save">保存候选标签</button> <span id="status"></span></header><main><section class="help"><h2>为已确认漏切的 residual 指定具体 cut candidate</h2><p>候选按时间顺序逐段试听：第一个候选从 residual 起点播放；之后每个候选都从上一个候选点播放到当前候选点。当前候选始终是播放终点。真实句界标 cut；句内或无须切分标 continue；听不清标 unsure。</p></section><div id="list"></div></main><script>
const rows={payload},key='split-v4-missing-cut-candidate-audit-v1:'+location.pathname,ann=JSON.parse(localStorage.getItem(key)||'{{}}');let active=null,timer=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function ensure(r){{ann[r.audit_id]??={{labels:{{}},note:''}};return ann[r.audit_id]}}function complete(r,a){{return r.residual_candidates.every(c=>a.labels[c.candidate_id])}}function persist(){{localStorage.setItem(key,JSON.stringify(ann));status()}}function status(){{document.getElementById('status').textContent=`完成 ${{rows.filter(r=>complete(r,ensure(r))).length}}/${{rows.length}}`}}function play(audio,start,end){{if(active&&active!==audio)active.pause();active=audio;if(timer)clearTimeout(timer);audio.currentTime=Math.max(0,Number(start));audio.play();timer=setTimeout(()=>audio.pause(),Math.max(1,(Number(end)-Number(start))*1000))}}function choice(a,c,label,text){{return `<button data-id="${{esc(c.candidate_id)}}" data-label="${{label}}" class="${{label}} ${{a.labels[c.candidate_id]===label?'active':''}}">${{text}}</button>`}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r),card=document.createElement('article');if(complete(r,a))card.classList.add('done');const body=r.residual_candidates.map((c,i)=>{{const start=i===0?Number(r.start_s):Number(r.residual_candidates[i-1].time_s),end=Number(c.time_s);return `<tr><td>${{esc(c.candidate_id)}}<br><b>候选 cut ${{end.toFixed(3)}}s</b><br><small>p_cut=${{Number(c.p_cut).toFixed(4)}} · model=${{esc(c.model_label)}}</small></td><td><button data-start="${{start}}" data-end="${{end}}">播放候选间 sub · ${{start.toFixed(3)}}–${{end.toFixed(3)}}s</button></td><td>${{choice(a,c,'cut','cut')}}${{choice(a,c,'continue','continue')}}${{choice(a,c,'unsure','unsure')}}</td></tr>`}}).join('');card.innerHTML=`<h2>${{esc(r.audit_id)}} · ${{esc(r.audio_id)}}</h2><small>residual sub-island ${{Number(r.start_s).toFixed(3)}}–${{Number(r.end_s).toFixed(3)}}s · ${{Number(r.duration_s).toFixed(3)}}s</small><audio controls preload="metadata" src="${{esc(r.audio_src)}}"></audio><div><button data-start="${{r.core_start}}" data-end="${{r.core_end}}">播放完整 chunk</button><button data-start="${{r.start_s}}" data-end="${{r.end_s}}">播放完整 missing residual</button></div><table><thead><tr><th>候选</th><th>上一个候选 → 当前候选</th><th>人工标签</th></tr></thead><tbody>${{body}}</tbody></table><input placeholder="补标备注" value="${{esc(a.note)}}">`;const audio=card.querySelector('audio');card.querySelectorAll('[data-start]').forEach(b=>b.onclick=()=>play(audio,b.dataset.start,b.dataset.end));card.querySelectorAll('[data-id]').forEach(b=>b.onclick=()=>{{a.labels[b.dataset.id]=b.dataset.label;a.updated_at=new Date().toISOString();persist();render()}});card.querySelector('input').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist()}};root.appendChild(card)}}status()}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r);return JSON.stringify({{schema:'split_v4_missing_cut_candidate_manual_verdict_v1',audit_id:r.audit_id,audio_id:r.audio_id,candidates:r.residual_candidates.map(c=>({{candidate_id:c.candidate_id,time_s:c.time_s,p_cut:c.p_cut,manual_label:a.labels[c.candidate_id]||'unreviewed'}})),complete:complete(r,a),note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}})}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error}};render();
</script></body></html>"""
    (output_dir / "index.html").write_text(page, "utf-8")
    summary = {
        "schema": "split_v4_missing_cut_candidate_audit_v1",
        "residual_count": len(selected),
        "candidate_count": sum(len(row["residual_candidates"]) for row in selected),
        "manual_verdicts": str(output_dir / "manual_verdicts.jsonl"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")
    update_audit_entrypoints(
        latest_html=output_dir / "index.html",
        title="Acoustic Split v4 Missing-cut Candidate Audit",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit-dir", required=True)
    parser.add_argument("--verdicts", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build(
        source_dir=Path(args.source_audit_dir),
        verdict_paths=[Path(path) for path in args.verdicts],
        output_dir=Path(args.output_dir),
    ), ensure_ascii=False))
