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


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_audit(*, labels: Path, output_dir: Path) -> Path:
    rows = _rows(labels)
    checkpoint_sha256 = str(
        ((rows[0].get("outer_prediction") or {}).get("checkpoint_sha256") if rows else "")
        or "unknown"
    )
    storage_key = f"outer-v2-actual-source-{checkpoint_sha256[:16]}"
    media_dir = output_dir / "audio"
    media_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        source = Path(row["source_audio"])
        refined = Path(row["audio"])
        source_copy = media_dir / f"{sample_id}__source{source.suffix}"
        refined_copy = media_dir / f"{sample_id}__outer.wav"
        shutil.copy2(source, source_copy)
        shutil.copy2(refined, refined_copy)
        payload.append(
            {
                "sample_id": sample_id,
                "reference_text": str(row.get("reference_text") or ""),
                "source_audio": source_copy.relative_to(output_dir).as_posix(),
                "outer_audio": refined_copy.relative_to(output_dir).as_posix(),
                "source_duration_s": float(row["source_duration_s"]),
                "outer_duration_s": float(row["duration_s"]),
                "source_span": row["source_span"],
                "outer_prediction": row["outer_prediction"],
                "outer_alignment_violations": row["outer_alignment_violations"],
                "semantic_core_span": row.get("semantic_core_span"),
                "edge_noise": row.get("edge_noise"),
            }
        )
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<" + "\\/")
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Outer v2 actual-island fixed5</title>
<style>body{{font-family:system-ui;margin:24px;background:#f5f7fa;color:#18202a}}article{{background:white;padding:18px;margin:18px 0;border-radius:12px}}article.done{{outline:3px solid #36a269}}audio{{width:100%;margin:6px 0 12px}}.notice{{background:#eaf3ff;border-left:5px solid #1769aa;padding:12px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.edge{{border:1px solid #ccd4dc;border-radius:8px;padding:12px}}select,textarea,button{{font:inherit}}select{{width:100%;padding:8px}}textarea{{width:100%;min-height:58px;margin-top:12px}}small{{color:#5b6673}}</style></head><body>
<h1>Outer v2 · 实际工作流 source fixed-5</h1><p class="notice">只审核学习型 Outer 的实际输出，不审核 Split/Inner，也不显示旧切点或 Omni coarse timeline。原音频与 Outer 输出是同一 source 的前后裁边对照。若卡片显示“已拼接首尾 definite-drop”，这些呻吟/喘息/噪声就是本轮要求模型移除的内容。请分别判断前缘 start 和后缘 end：是否完整保留所需台词，并排除亲吻声、喘息、呻吟、嘈杂人声、BGM 等不需要声音。当前 checkpoint：<code>{checkpoint_sha256[:16]}</code>。</p>
<button id="save">保存裁决</button><span id="status"></span><main id="list"></main>
<script>const rows={data};const key={json.dumps(storage_key)};let state=JSON.parse(localStorage.getItem(key)||'{{}}');
function esc(x){{return String(x??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));}}
function ensure(id){{state[id]=state[id]||{{start_model:'',end_model:'',note:''}};return state[id];}}
function persist(){{localStorage.setItem(key,JSON.stringify(state));}}
const options=[['','请选择'],['correct','正确'],['clipped_semantic','截掉了要保留的语音'],['too_wide_nonsemantic','保留了不需要的声音'],['unsure','不确定']];
function optionHtml(value){{return options.map(([v,t])=>`<option value="${{v}}" ${{v===value?'selected':''}}>${{t}}</option>`).join('');}}
function overall(a){{if(a.start_model==='clipped_semantic'||a.end_model==='clipped_semantic')return 'clipped_semantic';if(a.start_model==='too_wide_nonsemantic'||a.end_model==='too_wide_nonsemantic')return 'too_wide_nonsemantic';if(a.start_model==='correct'&&a.end_model==='correct')return 'correct';return 'unsure';}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r.sample_id),card=document.createElement('article');if(a.start_model&&a.end_model)card.classList.add('done');const p=r.outer_prediction,span=r.source_span,n=r.edge_noise;const noise=n?`<p><b>已拼接首尾 definite-drop：</b>前侧 ${{esc(n.leading.kind)}} / ${{esc(n.leading.background_type)}} / ${{Number(n.leading.duration_s).toFixed(3)}}s；后侧 ${{esc(n.trailing.kind)}} / ${{esc(n.trailing.background_type)}} / ${{Number(n.trailing.duration_s).toFixed(3)}}s</p>`:'';const inputTitle=n?'Outer 输入（semantic source + 首尾 definite-drop）':'原始 semantic source';card.innerHTML=`<h2>${{esc(r.sample_id)}}</h2><p><b>参考文本：</b>${{esc(r.reference_text)}}</p>${{noise}}<small>原 source ${{r.source_duration_s.toFixed(3)}}s → Outer 保留 ${{Number(span.start_s).toFixed(3)}}–${{Number(span.end_s).toFixed(3)}}s（输出 ${{r.outer_duration_s.toFixed(3)}}s）<br>start=${{esc(p.start_action)}} · end=${{esc(p.end_action)}}</small><h3>${{inputTitle}}</h3><audio controls preload="metadata" src="${{esc(r.source_audio)}}"></audio><h3>学习型 Outer v2 输出</h3><audio controls preload="metadata" src="${{esc(r.outer_audio)}}"></audio><div class="grid"><section class="edge"><h3>前缘 start</h3><p>模型 start 是否完整保留台词开头，并排除前导非语义声音？</p><select data-kind="start_model">${{optionHtml(a.start_model)}}</select></section><section class="edge"><h3>后缘 end</h3><p>模型 end 是否完整保留句尾，并排除尾随非语义声音？</p><select data-kind="end_model">${{optionHtml(a.end_model)}}</select></section></div><textarea placeholder="备注">${{esc(a.note)}}</textarea>`;card.querySelectorAll('select').forEach(s=>s.onchange=()=>{{a[s.dataset.kind]=s.value;a.updated_at=new Date().toISOString();persist();render();}});card.querySelector('textarea').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};root.appendChild(card);}}}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r.sample_id);return JSON.stringify({{schema:'outer_refined_source_manual_verdict_v2',sample_id:r.sample_id,start_model:a.start_model||'unreviewed',end_model:a.end_model||'unreviewed',verdict:overall(a),note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?' 已保存到 '+out.path:' 保存失败: '+out.error;}};render();</script></body></html>"""
    path = output_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    update_audit_entrypoints(
        latest_html=path,
        title="Outer v2 实际 island fixed-5",
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Outer-refined source fixed audit.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(build_audit(labels=Path(args.labels), output_dir=Path(args.output_dir)))
