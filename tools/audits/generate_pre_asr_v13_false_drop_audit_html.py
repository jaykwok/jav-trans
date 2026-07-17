#!/usr/bin/env python3
"""Generate a precise whole-subisland audit page for CueQC v13 false drops."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.asr.cueqc.label_pre_asr_with_omni import slice_audio_clip  # noqa: E402
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "item"


def build(
    *, false_drop_manifest: Path, output_dir: Path, update_latest: bool = True
) -> dict[str, Any]:
    source_rows = _rows(false_drop_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in source_rows:
        row_id = str(source.get("row_id") or "")
        if not row_id or row_id in seen:
            raise ValueError("false-drop row_id values must be non-empty and unique")
        seen.add(row_id)
        audio = Path(str(source.get("audio") or ""))
        if not audio.is_file():
            raise FileNotFoundError(f"false-drop source audio not found: {audio}")
        start = float(source.get("start_s") or 0.0)
        end = float(source.get("end_s") or 0.0)
        if end <= start:
            raise ValueError(f"false-drop interval must be positive: {row_id}")
        clip = media_dir / f"{_safe_name(row_id)}.mp3"
        slice_audio_clip(
            source_audio=audio,
            row={"start": start, "end": end, "duration_s": end - start},
            output_path=clip,
            fmt="mp3",
            bitrate="64k",
            sample_rate=16000,
            force=False,
        )
        rows.append(
            {
                **source,
                "audio_src": clip.relative_to(output_dir).as_posix(),
                "clip_duration_s": end - start,
            }
        )
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    page = output_dir / "index.html"
    page.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>CueQC v13 false-drop audit</title>
<style>body{{margin:0;background:#0d1117;color:#e6edf3;font-family:Segoe UI,Arial,sans-serif}}header{{position:sticky;top:0;z-index:2;background:#161b22;padding:12px 18px;border-bottom:1px solid #30363d}}main{{max-width:1000px;margin:auto;padding:16px}}article,.help{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin:14px 0}}article.done{{border-color:#2ea043}}audio{{width:100%;margin:10px 0}}button,input{{margin:3px;padding:7px 11px;background:#21262d;color:#e6edf3;border:1px solid #484f58;border-radius:6px}}button.active{{background:#1f6feb}}button.true_speech.active{{background:#da3633}}button.safe_drop.active{{background:#238636}}button.unsure.active{{background:#8250df}}input{{width:60%}}small{{color:#8b949e}}</style></head><body>
<header><strong>CueQC v13 · 全量 false-drop 人工 gate</strong> <button id="save">保存审计结果</button> <span id="status"></span></header><main><section class="help"><h2>每条播放器就是模型将删除的完整 sub-island</h2><p>不截取 1.5 秒窗口，播放器从该 sub-island 的精确起点播放到精确终点。只判断整块是否含任何应进入 ASR 的真实词语或有词义发声。</p><p><b>安全删除：</b>整块只有静音、音乐、噪声、喘息、呻吟等非语义声音。<b>真语音误删：</b>存在任何需字幕的真实语音。<b>不确定：</b>听不清；不能晋升。</p></section><div id="list"></div></main><script>
const rows={payload},key='cueqc-v13-false-drop-audit-v1:'+location.pathname,ann=JSON.parse(localStorage.getItem(key)||'{{}}');
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}function ensure(r){{ann[r.row_id]??={{verdict:'',note:''}};return ann[r.row_id]}}function persist(){{localStorage.setItem(key,JSON.stringify(ann));status()}}function status(){{document.getElementById('status').textContent=`完成 ${{rows.filter(r=>ensure(r).verdict).length}}/${{rows.length}}`}}function choice(a,v,t){{return `<button data-v="${{v}}" class="${{v}} ${{a.verdict===v?'active':''}}">${{t}}</button>`}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r),card=document.createElement('article');if(a.verdict)card.classList.add('done');card.innerHTML=`<h2>${{esc(r.row_id)}}</h2><small>${{esc(r.source_partition)}} · ${{Number(r.start_s).toFixed(3)}}–${{Number(r.end_s).toFixed(3)}}s · p(drop)=${{Number(r.prob_drop).toFixed(4)}} · teacher=${{esc(r.teacher_label)}} · exact=${{esc(r.exact_core_label||'-')}}</small><audio controls preload="metadata" src="${{esc(r.audio_src)}}"></audio><div>${{choice(a,'safe_drop','安全删除')}}${{choice(a,'true_speech','真语音误删')}}${{choice(a,'unsure','不确定')}} <input placeholder="备注" value="${{esc(a.note)}}"></div>`;card.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{{a.verdict=b.dataset.v;a.updated_at=new Date().toISOString();persist();render()}});card.querySelector('input').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist()}};root.appendChild(card)}}status()}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r);return JSON.stringify({{schema:'cueqc_v13_false_drop_manual_verdict_v1',row_id:r.row_id,verdict:a.verdict||'unreviewed',note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}})}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error}};render();
</script></body></html>""",
        encoding="utf-8",
    )
    summary = {
        "schema": "cueqc_v13_false_drop_audit_page_summary_v1",
        "item_count": len(rows),
        "playback_contract": "exact_whole_subisland_v1",
        "false_drop_manifest": str(false_drop_manifest),
        "manual_verdicts": str(output_dir / "manual_verdicts.jsonl"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if update_latest:
        update_audit_entrypoints(
            latest_html=page, title="CueQC v13 False-drop Audit"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--false-drop-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build(
                false_drop_manifest=Path(args.false_drop_manifest),
                output_dir=Path(args.output_dir),
                update_latest=not args.no_update_latest,
            ),
            ensure_ascii=False,
        )
    )
