#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _segmented(
    source: Path, output: Path, cuts_s: list[float], duration_s: float
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    bounds = [0.0, *cuts_s, float(duration_s)]
    filters = []
    inputs = []
    for index, (start, end) in enumerate(zip(bounds, bounds[1:])):
        filters.append(
            f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[p{index}]"
        )
        inputs.append(f"[p{index}]")
        if index + 1 < len(bounds) - 1:
            filters.append(f"anullsrc=r=16000:cl=mono:d=1[s{index}]")
            inputs.append(f"[s{index}]")
    filters.append(f"{''.join(inputs)}concat=n={len(inputs)}:v=0:a=1[out]")
    filter_graph = ";".join(filters)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-filter_complex", filter_graph, "-map", "[out]", "-ac", "1", "-ar", "16000", str(output)],
        check=True,
    )


def build_audit(
    *,
    labels: Path,
    semantic_labels: Path,
    request_audio_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    semantic_by_island = {
        str(row["island_id"]): row for row in _read_jsonl(semantic_labels)
    }
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in _read_jsonl(labels):
        audio_name = str(row["island_id"]).replace("#", "__") + ".wav"
        source = request_audio_dir / audio_name
        stem = str(row["boundary_id"]).replace("#", "__")
        original = output_dir / "audio" / f"{stem}.wav"
        original.parent.mkdir(parents=True, exist_ok=True)
        if not original.exists():
            original.write_bytes(source.read_bytes())
        segmented = output_dir / "audio" / f"{stem}__timing-safe.wav"
        semantic_cuts = [
            float(cut["time_s"])
            for cut in semantic_by_island[str(row["island_id"])].get("cuts") or []
        ]
        boundary_index = int(str(row["boundary_id"]).rsplit("#b", 1)[1])
        if boundary_index >= len(semantic_cuts):
            raise ValueError(f"boundary index is outside semantic cuts: {row['boundary_id']}")
        audited_cuts = list(semantic_cuts)
        audited_cuts[boundary_index] = float(row["safe_cut_time_s"])
        if audited_cuts != sorted(audited_cuts):
            raise ValueError(f"safe timing crosses an adjacent semantic cut: {row['boundary_id']}")
        _segmented(source, segmented, audited_cuts, float(row["duration_s"]))
        rows.append(
            {
                **row,
                "audio": original.relative_to(output_dir).as_posix(),
                "segmented_audio": segmented.relative_to(output_dir).as_posix(),
                "semantic_cut_count": len(semantic_cuts),
                "audited_cuts": audited_cuts,
                "target_boundary_index": boundary_index,
            }
        )
    (output_dir / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Split Boundary Timing Smoke</title><style>body{{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f5f6f8;color:#20242a}}header{{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #d9dde3;padding:12px 20px}}main{{max-width:1080px;margin:18px auto;padding:0 16px}}article{{background:#fff;border:1px solid #d9dde3;border-radius:6px;padding:14px;margin-bottom:14px}}article.done{{border-left:5px solid #197642}}audio{{width:100%;margin:6px 0 10px}}button,input{{font:inherit}}button{{border:1px solid #b9c0c9;background:#fff;padding:7px 10px;border-radius:4px;cursor:pointer}}button.active{{background:#1769aa;color:#fff}}.meta,.legend,.verdict{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}.meta{{font-size:13px;color:#59616c}}.timeline{{height:36px;position:relative;background:#eef1f4;border:1px solid #b9c0c9;border-radius:3px;margin:8px 0}}.mark{{position:absolute;top:0;bottom:0;width:3px;background:#35a853;border:0;padding:0;cursor:pointer}}.mark.target{{width:7px;background:#08752d;transform:translateX(-2px)}}.swatch{{display:inline-block;width:12px;height:3px;margin-right:4px}}.note{{flex:1;min-width:240px;padding:7px}}pre{{white-space:pre-wrap;background:#f6f7f9;padding:8px;border-radius:4px;font-size:12px}}</style></head><body><header><strong>Split Boundary Timing · 5 samples</strong> <span>只显示当前最终切点集合；深绿色粗线为本卡 Timing target</span> <button id="save">保存裁决</button> <span id="status"></span></header><main id="list"></main><script>const rows={payload};const key='split-boundary-timing-smoke-v2-current-only';const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let active=null;function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function playAt(a,t){{if(active&&active!==a)active.pause();active=a;a.currentTime=Math.max(0,t-1);const end=Math.min(a.duration||1e9,t+1);const stop=()=>{{if(a.currentTime>=end){{a.pause();a.removeEventListener('timeupdate',stop)}}}};a.addEventListener('timeupdate',stop);a.play();}}function render(){{const list=document.getElementById('list');list.innerHTML='';for(const row of rows){{const a=ann[row.boundary_id]||{{verdict:'',note:''}};const el=document.createElement('article');if(a.verdict)el.classList.add('done');const dur=Number(row.duration_s);const marks=(row.audited_cuts||[]).map((t,index)=>`<button class="mark ${{index===row.target_boundary_index?'target':''}}" data-t="${{Number(t)}}" title="当前切点 #${{index}} ${{Number(t).toFixed(3)}}s" style="left:${{Number(t)/dur*100}}%"></button>`).join('');el.innerHTML=`<strong>${{esc(row.boundary_id)}}</strong><div class="meta"><span>island ${{dur.toFixed(2)}}s</span><span>当前切点总数 ${{row.audited_cuts.length}}</span><span>目标 safe=${{Number(row.safe_cut_time_s).toFixed(3)}}s</span></div><strong>原始 island</strong><audio class="original" controls preload="metadata" src="${{esc(row.audio)}}"></audio><strong>按下方全部绿色切点切开（块间静音 1 秒）</strong><audio controls preload="metadata" src="${{esc(row.segmented_audio)}}"></audio><div class="timeline">${{marks}}</div><div class="legend"><span><i class="swatch" style="background:#35a853"></i>当前全部切点</span><span><i class="swatch" style="background:#08752d;height:7px"></i>本卡目标 safe cut</span><span>点击任一绿线跨点试听 ±1s</span></div><div class="verdict"><button data-v="correct">当前切割正确</button><button data-v="early">目标仍然偏早</button><button data-v="late">目标偏晚</button><button data-v="wrong_boundary">目标跳到相邻边界</button><button data-v="unsure">无法判断</button><input class="note" placeholder="备注" value="${{esc(a.note)}}"></div><pre>${{esc(JSON.stringify({{target_index:row.target_boundary_index,target_safe_time_s:row.safe_cut_time_s,current_cuts:row.audited_cuts,confidence:row.confidence,reason:row.reason}},null,2))}}</pre>`;const original=el.querySelector('.original');el.querySelectorAll('audio').forEach(p=>p.onplay=()=>{{if(active&&active!==p)active.pause();active=p}});el.querySelectorAll('.mark').forEach(b=>b.onclick=()=>playAt(original,Number(b.dataset.t)));el.querySelectorAll('[data-v]').forEach(b=>{{if(b.dataset.v===a.verdict)b.classList.add('active');b.onclick=()=>{{ann[row.boundary_id]={{verdict:b.dataset.v,note:el.querySelector('.note').value,updated_at:new Date().toISOString()}};localStorage.setItem(key,JSON.stringify(ann));el.querySelectorAll('[data-v]').forEach(x=>x.classList.toggle('active',x===b));el.classList.add('done');document.getElementById('status').textContent=`已审 ${{Object.values(ann).filter(x=>x.verdict).length}} / ${{rows.length}}`;}}}});el.querySelector('.note').onchange=e=>{{ann[row.boundary_id]={{...(ann[row.boundary_id]||{{verdict:'',note:''}}),note:e.target.value,updated_at:new Date().toISOString()}};localStorage.setItem(key,JSON.stringify(ann))}};list.appendChild(el)}}document.getElementById('status').textContent=`已审 ${{Object.values(ann).filter(x=>x.verdict).length}} / ${{rows.length}}`;}}document.getElementById('save').onclick=async()=>{{const content=rows.filter(r=>ann[r.boundary_id]?.verdict).map(r=>JSON.stringify({{schema:'semantic_split_v3_boundary_timing_manual_verdict_v2',boundary_id:r.boundary_id,verdict:ann[r.boundary_id].verdict,note:ann[r.boundary_id].note||'',updated_at:ann[r.boundary_id].updated_at}})).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error}};render();</script></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    summary = {"schema": "semantic_split_v3_boundary_timing_audit_summary_v1", "item_count": len(rows), "manual_verdicts": str(output_dir / "manual_verdicts.jsonl")}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_audit_entrypoints(latest_html=index, title="Split Boundary Timing Smoke")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--semantic-labels", required=True)
    parser.add_argument("--request-audio-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_audit(labels=Path(args.labels), semantic_labels=Path(args.semantic_labels), request_audio_dir=Path(args.request_audio_dir), output_dir=Path(args.output_dir)), ensure_ascii=False))


if __name__ == "__main__":
    main()
