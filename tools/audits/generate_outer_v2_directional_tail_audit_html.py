#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_tail_rows(rows: list[dict[str, Any]], *, count: int = 5) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_ranked(values: list[dict[str, Any]], limit: int, reason: str) -> None:
        rank = 0
        for row in values:
            audio_id = str(row["audio_id"])
            if audio_id in selected_ids:
                continue
            rank += 1
            selected.append({**row, "selection_reason": f"{reason}_{rank}"})
            selected_ids.add(audio_id)
            if rank >= limit or len(selected) >= count:
                return

    inward = sorted(
        rows,
        key=lambda row: max(
            float(row.get("start_inward_s", 0.0)),
            float(row.get("end_inward_s", 0.0)),
        ),
        reverse=True,
    )
    outward = sorted(
        rows,
        key=lambda row: max(
            float(row.get("start_outward_s", 0.0)),
            float(row.get("end_outward_s", 0.0)),
        ),
        reverse=True,
    )
    add_ranked(inward, min(3, count), "inward")
    add_ranked(outward, min(2, count - len(selected)), "outward")
    if len(selected) < count:
        remaining = sorted(
            rows,
            key=lambda row: max(
                float(row.get("start_absolute_s", 0.0)),
                float(row.get("end_absolute_s", 0.0)),
            ),
            reverse=True,
        )
        add_ranked(remaining, count - len(selected), "absolute")
    return selected[:count]


def _speech_text(row: dict[str, Any]) -> str:
    parts = [
        str(source.get("source_text") or "")
        for source in row.get("sources") or []
        if source.get("source_audio_id")
    ]
    return " / ".join(part for part in parts if part)


def _write_crop(
    audio: np.ndarray,
    sample_rate: int,
    *,
    start_s: float,
    end_s: float,
    output: Path,
) -> None:
    start = max(0, min(len(audio), int(round(start_s * sample_rate))))
    end = max(start, min(len(audio), int(round(end_s * sample_rate))))
    sf.write(str(output), audio[start:end], sample_rate, subtype="PCM_16")


def build_audit(
    *,
    evaluation_details: Path,
    synthetic_details: Path,
    output_dir: Path,
    count: int = 5,
) -> Path:
    details = select_tail_rows(_rows(evaluation_details), count=count)
    synthetic = {str(row["audio_id"]): row for row in _rows(synthetic_details)}
    media_dir = output_dir / "audio"
    media_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    for row in details:
        audio_id = str(row["audio_id"])
        source = Path(row["source_audio"])
        source_copy = media_dir / f"{audio_id}__complete.wav"
        predicted_copy = media_dir / f"{audio_id}__model-kept.wav"
        target_copy = media_dir / f"{audio_id}__training-target.wav"
        shutil.copy2(source, source_copy)
        audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        _write_crop(
            audio,
            sample_rate,
            start_s=float(row["predicted_start_s"]),
            end_s=float(row["predicted_end_s"]),
            output=predicted_copy,
        )
        _write_crop(
            audio,
            sample_rate,
            start_s=float(row["truth_start_s"]),
            end_s=float(row["truth_end_s"]),
            output=target_copy,
        )
        source_meta = synthetic[audio_id]
        payload.append(
            {
                **row,
                "reference_text": _speech_text(source_meta),
                "timeline_pattern": source_meta.get("timeline_pattern"),
                "background_mix": bool(source_meta.get("background_mix")),
                "source_duration_s": float(
                    source_meta.get("duration_s") or len(audio) / sample_rate
                ),
                "complete_audio": source_copy.relative_to(output_dir).as_posix(),
                "predicted_audio": predicted_copy.relative_to(output_dir).as_posix(),
                "target_audio": target_copy.relative_to(output_dir).as_posix(),
            }
        )
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<" + "\\/")
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Outer v2 directional tail fixed5</title>
<style>body{{font-family:system-ui;margin:24px;background:#f5f7fa;color:#17202a}}article{{background:#fff;padding:20px;margin:20px 0;border-radius:12px}}article.done{{outline:3px solid #35a269}}audio{{width:100%;margin:6px 0 12px}}.notice{{background:#fff6d8;border-left:5px solid #e6a700;padding:12px}}.timeline{{height:58px;position:relative;background:#e8edf2;border-radius:8px;margin:12px 0}}.bar{{height:18px;position:absolute;border-radius:5px;color:#111;font-size:12px;line-height:18px;padding-left:4px;box-sizing:border-box;overflow:hidden}}.target{{top:8px;background:#f2c94c}}.model{{top:32px;background:#6fcf97}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.edge{{border:1px solid #ccd4dc;border-radius:8px;padding:12px}}select,textarea,button{{font:inherit}}select{{width:100%;padding:7px;margin:4px 0 10px}}textarea{{width:100%;min-height:64px}}button{{margin:4px;padding:7px 10px}}small{{color:#56616d}}code{{background:#eef1f4;padding:1px 4px}}</style></head><body>
<h1>Outer v2 · 方向性长尾 fixed-5</h1>
<p class="notice"><b>目的：</b>区分“模型边缘错”与“synthetic training target 过宽/过窄”。黄色是训练标签，绿色是模型预测；它们是两套替代边缘，<b>不是相邻区间</b>，音频内容重叠是刻意的对照。不要按数值大小裁决，只按是否完整保留所需台词、是否排除亲吻声/喘息/呻吟/嘈杂人声/BGM 等不需要声音裁决。</p>
<p><b>每一侧分别做两件事：</b>①模型边缘是否正确；②训练标签边缘是否正确。若二者都可接受就分别选择“正确”。</p>
<button id="save">保存裁决</button><span id="status"></span><main id="list"></main>
<script>const rows={data};const key='outer-v2-directional-tail-fixed5-v1';let state=JSON.parse(localStorage.getItem(key)||'{{}}');let stopAt=null;
function esc(x){{return String(x??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));}}
function ensure(id){{state[id]=state[id]||{{start:{{}},end:{{}},note:''}};return state[id];}}
function persist(){{localStorage.setItem(key,JSON.stringify(state));}}
function playRange(audio,start,end){{document.querySelectorAll('audio').forEach(a=>{{if(a!==audio)a.pause();}});stopAt=end;audio.currentTime=Math.max(0,start);audio.play();}}
document.addEventListener('timeupdate',e=>{{if(e.target.tagName==='AUDIO'&&stopAt!==null&&e.target.currentTime>=stopAt){{e.target.pause();stopAt=null;}}}},true);
const modelOptions=[['','请选择模型边缘'],['correct','正确'],['clipped_semantic','截掉了要保留的语音'],['too_wide_nonsemantic','保留了不需要的声音'],['unsure','不确定']];
const targetOptions=[['','请选择训练标签边缘'],['correct','正确'],['includes_nonsemantic','标签包含不需要的声音'],['clips_semantic','标签截掉要保留的语音'],['unsure','不确定']];
function options(items,value){{return items.map(([v,t])=>`<option value="${{v}}" ${{v===value?'selected':''}}>${{t}}</option>`).join('');}}
function complete(a){{return ['start','end'].every(edge=>a[edge].model&&a[edge].target);}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r.audio_id),duration=Number(r.source_duration_s),card=document.createElement('article');if(complete(a))card.classList.add('done');const targetLeft=100*Number(r.truth_start_s)/duration,targetWidth=100*(Number(r.truth_end_s)-Number(r.truth_start_s))/duration,modelLeft=100*Number(r.predicted_start_s)/duration,modelWidth=100*(Number(r.predicted_end_s)-Number(r.predicted_start_s))/duration;card.innerHTML=`<h2>${{esc(r.audio_id)}} · ${{esc(r.selection_reason)}}</h2><p><b>参考文本：</b>${{esc(r.reference_text)||'(空)'}}</p><small>完整 island ${{duration.toFixed(3)}}s · pattern=${{esc(r.timeline_pattern)}} · background_mix=${{r.background_mix}}<br>训练标签 ${{Number(r.truth_start_s).toFixed(3)}}–${{Number(r.truth_end_s).toFixed(3)}}s；模型 ${{Number(r.predicted_start_s).toFixed(3)}}–${{Number(r.predicted_end_s).toFixed(3)}}s<br>start signed=${{Number(r.start_signed_s).toFixed(3)}}s（正=向内） · end signed=${{Number(r.end_signed_s).toFixed(3)}}s（负=向内）</small><div class="timeline"><div class="bar target" style="left:${{targetLeft}}%;width:${{targetWidth}}%">训练标签</div><div class="bar model" style="left:${{modelLeft}}%;width:${{modelWidth}}%">模型预测</div></div><h3>完整 synthetic island</h3><audio class="complete" controls preload="metadata" src="${{esc(r.complete_audio)}}"></audio><div><button data-play="model-start">从模型 start 播 2.5s</button><button data-play="target-start">从标签 start 播 2.5s</button><button data-play="model-end">播放模型 end 前 2.5s</button><button data-play="target-end">播放标签 end 前 2.5s</button></div><div class="grid"><div><h3>模型保留结果</h3><audio controls preload="metadata" src="${{esc(r.predicted_audio)}}"></audio></div><div><h3>训练标签保留结果</h3><audio controls preload="metadata" src="${{esc(r.target_audio)}}"></audio></div></div><div class="grid"><section class="edge"><h3>前缘 start</h3><label><b>模型 start</b>：它有没有截语音或留杂声？</label><select data-edge="start" data-kind="model">${{options(modelOptions,a.start.model)}}</select><label><b>训练标签 start</b>：黄色边缘本身准不准？</label><select data-edge="start" data-kind="target">${{options(targetOptions,a.start.target)}}</select></section><section class="edge"><h3>后缘 end</h3><label><b>模型 end</b>：它有没有截语音或留杂声？</label><select data-edge="end" data-kind="model">${{options(modelOptions,a.end.model)}}</select><label><b>训练标签 end</b>：黄色边缘本身准不准？</label><select data-edge="end" data-kind="target">${{options(targetOptions,a.end.target)}}</select></section></div><textarea placeholder="备注：例如‘标签把前导喘息算进语义；模型正确排除’">${{esc(a.note)}}</textarea>`;const audio=card.querySelector('audio.complete');card.querySelector('[data-play="model-start"]').onclick=()=>playRange(audio,Number(r.predicted_start_s),Math.min(duration,Number(r.predicted_start_s)+2.5));card.querySelector('[data-play="target-start"]').onclick=()=>playRange(audio,Number(r.truth_start_s),Math.min(duration,Number(r.truth_start_s)+2.5));card.querySelector('[data-play="model-end"]').onclick=()=>playRange(audio,Math.max(0,Number(r.predicted_end_s)-2.5),Math.min(duration,Number(r.predicted_end_s)));card.querySelector('[data-play="target-end"]').onclick=()=>playRange(audio,Math.max(0,Number(r.truth_end_s)-2.5),Math.min(duration,Number(r.truth_end_s)));card.querySelectorAll('select').forEach(s=>s.onchange=()=>{{a[s.dataset.edge][s.dataset.kind]=s.value;a.updated_at=new Date().toISOString();persist();render();}});card.querySelector('textarea').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};root.appendChild(card);}}}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r.audio_id);return JSON.stringify({{schema:'outer_v2_directional_tail_manual_verdict_v1',audio_id:r.audio_id,selection_reason:r.selection_reason,start_model:a.start.model||'unreviewed',start_target:a.start.target||'unreviewed',end_model:a.end.model||'unreviewed',end_target:a.end.target||'unreviewed',note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?' 已保存到 '+out.path:' 保存失败: '+out.error;}};render();</script></body></html>"""
    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    summary = {
        "schema": "outer_v2_directional_tail_audit_summary_v1",
        "item_count": len(payload),
        "selection_policy": "top3_max_inward_then_top2_max_outward_unique",
        "audio_ids": [row["audio_id"] for row in payload],
        "manual_verdicts": str(output_dir / "manual_verdicts.jsonl"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    update_audit_entrypoints(
        latest_html=index_path,
        title="Outer v2 方向性长尾 fixed-5",
    )
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Outer v2 directional tail fixed audit.")
    parser.add_argument("--evaluation-details", required=True)
    parser.add_argument("--synthetic-details", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        build_audit(
            evaluation_details=Path(args.evaluation_details),
            synthetic_details=Path(args.synthetic_details),
            output_dir=Path(args.output_dir),
            count=args.count,
        )
    )
