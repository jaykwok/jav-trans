#!/usr/bin/env python3
"""Build an Inner-edge audit over CueQC-retained provisional sub-islands."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.outer_refiner_v2 import load_outer_edge_refiner_v2  # noqa: E402
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


ITEM_SCHEMA = "inner_subisland_edge_audit_item_v1"
SUMMARY_SCHEMA = "inner_subisland_edge_audit_summary_v1"
VERDICT_SCHEMA = "inner_subisland_edge_manual_verdict_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def edge_features(ptm: np.ndarray, mfcc: np.ndarray) -> np.ndarray:
    total = min(int(ptm.shape[0]), int(mfcc.shape[0]))
    if total <= 0:
        return np.zeros((0, 2089), dtype=np.float32)
    position = (
        np.arange(total, dtype=np.float32) / max(1, total - 1)
    ).reshape(-1, 1)
    return np.concatenate(
        (
            np.asarray(ptm[:total], dtype=np.float32),
            np.asarray(mfcc[:total], dtype=np.float32),
            position,
        ),
        axis=1,
    )


def edge_ownership(*, start_s: float, end_s: float, source_duration_s: float) -> dict[str, bool]:
    return {
        "start_requires_inner": float(start_s) > 1e-9,
        "end_requires_inner": float(end_s) < float(source_duration_s) - 1e-9,
    }


def build_items(
    *,
    inner_inputs: Path,
    outer_labels: Path,
    feature_manifest: Path,
    checkpoint: Path,
    ptm_repo_id: str,
    device: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    inputs = _rows(inner_inputs)
    outer_by_id = {str(row["sample_id"]): row for row in _rows(outer_labels)}
    feature_by_id = {str(row["sample_id"]): row for row in _rows(feature_manifest)}
    model = load_outer_edge_refiner_v2(
        checkpoint, device=device, expected_ptm_repo_id=ptm_repo_id
    )
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    result: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inputs:
        by_source[str(row["sample_id"])].append(row)
    for sample_id, group in by_source.items():
        outer = outer_by_id[sample_id]
        feature_row = feature_by_id[sample_id]
        with np.load(PROJECT_ROOT / str(feature_row["feature_path"]), allow_pickle=False) as payload:
            ptm = np.asarray(payload["ptm"], dtype=np.float32)
            mfcc = np.asarray(payload["mfcc"], dtype=np.float32)
        frame_hop_s = float(feature_row.get("frame_hop_s") or 0.02)
        source_offset_s = float(outer["source_span"]["start_s"])
        source_duration_s = float(outer["duration_s"])
        source_audio = Path(str(group[0]["audio"]))
        target_audio = audio_dir / f"{sample_id}{source_audio.suffix.lower() or '.wav'}"
        shutil.copyfile(source_audio, target_audio)
        copied[sample_id] = target_audio.relative_to(output_dir).as_posix()
        for row in sorted(group, key=lambda item: float(item["start_s"])):
            start_s = float(row["start_s"])
            end_s = float(row["end_s"])
            source_start_frame = max(
                0, int(round((source_offset_s + start_s) / frame_hop_s))
            )
            source_end_frame = min(
                ptm.shape[0], int(round((source_offset_s + end_s) / frame_hop_s))
            )
            features = edge_features(
                ptm[source_start_frame:source_end_frame],
                mfcc[source_start_frame:source_end_frame],
            )
            raw_duration_s = float(features.shape[0]) * frame_hop_s
            if features.shape[0] < 2:
                prediction = None
                refined_start_s = start_s
                refined_end_s = end_s
            else:
                prediction = model.predict_islands(
                    frame_feature_groups=[features],
                    raw_spans=[(0.0, raw_duration_s)],
                    frame_hop_s=frame_hop_s,
                )[0]
                refined_start_s = min(end_s, start_s + float(prediction.start_s))
                refined_end_s = min(end_s, start_s + float(prediction.end_s))
            ownership = edge_ownership(
                start_s=start_s,
                end_s=end_s,
                source_duration_s=source_duration_s,
            )
            result.append(
                {
                    "schema": ITEM_SCHEMA,
                    "sample_id": sample_id,
                    "subisland_id": str(row["subisland_id"]),
                    "audio": copied[sample_id],
                    "source_duration_s": source_duration_s,
                    "raw_start_s": start_s,
                    "raw_end_s": end_s,
                    "refined_start_s": refined_start_s,
                    "refined_end_s": refined_end_s,
                    **ownership,
                    "bootstrap_prediction": (
                        {
                            **asdict(prediction),
                            "class_probabilities": None,
                        }
                        if prediction is not None
                        else {
                            "start_action": "abstain",
                            "end_action": "abstain",
                            "abstain_reason": "subisland_too_short",
                        }
                    ),
                    "bootstrap_checkpoint_sha256": model.sha256,
                    "teacher_usage": "bootstrap_preview_only_not_training_truth",
                }
            )
    return result, model.sha256


def build_page(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    update_latest: bool = True,
    noisy_edge_mode: bool = False,
) -> Path:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["sample_id"])].append(row)
    payload = json.dumps(
        [
            {
                "sample_id": sample_id,
                "audio": group[0]["audio"],
                "subislands": sorted(group, key=lambda row: float(row["raw_start_s"])),
            }
            for sample_id, group in sorted(by_source.items())
        ],
        ensure_ascii=False,
    ).replace("</", "<\\/")
    page_title = (
        "Inner noisy-edge discriminative audit"
        if noisy_edge_mode
        else "Inner sub-island edge audit"
    )
    help_html = (
        "<h2>判别性 Inner gate：输入确实包含已知前后非语义污染</h2>"
        "<p>每条都是 CueQC 已决定 keep 的 provisional sub-island 形态；前后附加真实 definite-drop 喘息、亲吻声、哭声或环境噪声，两个边缘都归 Inner。比较 raw 与 refined，判断是否保留完整台词并清掉可移除的边缘污染。</p>"
        "<p>构造时的 core 精确区间只用于离线核验，不作为模型输入；bootstrap 仍只是现役 Outer v2 同架构预览，不是正式 Inner checkpoint 或训练真值。</p>"
        if noisy_edge_mode
        else "<h2>只审内部朝向边缘；全局最外侧边缘仍归 Outer v2</h2>"
        "<p>bootstrap 使用现役 Outer v2 同架构权重，只验证“在 post-Split sub-island 上是否能像 Outer 一样修边”，不是正式 Inner checkpoint 或训练真值。</p>"
    )
    storage_key = (
        "inner-noisy-edge-audit-v1" if noisy_edge_mode else "inner-subisland-edge-audit-v1"
    )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{page_title}</title><style>
body{{margin:0;background:#0d1117;color:#e6edf3;font-family:system-ui}}header{{position:sticky;top:0;background:#161b22;padding:12px 18px;border-bottom:1px solid #30363d}}main{{max-width:1200px;margin:auto;padding:16px}}article{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin:14px 0}}.help{{background:#10233f;padding:14px;border-radius:10px}}audio{{width:100%}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #30363d;text-align:left}}button{{margin:3px;padding:6px 10px;background:#21262d;color:#e6edf3;border:1px solid #484f58;border-radius:6px}}button.active{{background:#1f6feb}}input{{width:95%;background:#0d1117;color:#e6edf3}}small{{color:#8b949e}}
</style></head><body><header><strong>{page_title}</strong> <button id="save">保存边缘裁决</button> <span id="status"></span></header><main><section class="help">{help_html}<p><b>正确：</b>没有截掉语音，也没有明显残留静音/杂音。<b>截语音：</b>丢失开头、句尾 mora、拖音或真实 speech。<b>残留杂音：</b>保留了本可从该边缘去掉的静音/非语义声音。<b>不确定：</b>重叠、连续或听不清。</p></section><div id="list"></div></main><script>
const rows={payload};const key='{storage_key}:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let active=null,timer=null;function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function ensure(id){{ann[id]??={{start:'',end:'',note:''}};return ann[id];}}function complete(x,a){{return (!x.start_requires_inner||a.start)&&(!x.end_requires_inner||a.end);}}function persist(){{localStorage.setItem(key,JSON.stringify(ann));status();}}function status(){{const all=rows.flatMap(r=>r.subislands),done=all.filter(x=>complete(x,ensure(x.subisland_id))).length;document.getElementById('status').textContent=`完成 ${{done}}/${{all.length}}`;}}function play(audio,start,end){{if(active&&active!==audio)active.pause();active=audio;if(timer)clearTimeout(timer);audio.currentTime=Number(start);audio.play();timer=setTimeout(()=>audio.pause(),Math.max(1,(Number(end)-Number(start))*1000));}}function choices(a,id,edge){{return ['correct','clipped','too_wide','unsure'].map(v=>`<button data-id="${{esc(id)}}" data-edge="${{edge}}" data-label="${{v}}" class="${{a[edge]===v?'active':''}}">${{{{correct:'正确',clipped:'截语音',too_wide:'残留杂音',unsure:'不确定'}}[v]}}</button>`).join('');}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const card=document.createElement('article'),body=r.subislands.map(x=>{{const a=ensure(x.subisland_id),p=x.bootstrap_prediction||{{}},n=x.edge_noise||null,meta=n?`<br><small>前侧 ${{esc(n.leading?.background_type||n.leading?.kind||'')}}；后侧 ${{esc(n.trailing?.background_type||n.trailing?.kind||'')}}</small>`:'';return `<tr><td><b>${{esc(x.subisland_id.split('__').pop())}}</b><br>raw ${{Number(x.raw_start_s).toFixed(3)}}–${{Number(x.raw_end_s).toFixed(3)}}s<br>refined ${{Number(x.refined_start_s).toFixed(3)}}–${{Number(x.refined_end_s).toFixed(3)}}s${{meta}}<br><small>${{esc(p.start_action)}} / ${{esc(p.end_action)}} ${{esc(p.abstain_reason||'')}}</small></td><td><button data-start="${{x.raw_start_s}}" data-end="${{x.raw_end_s}}">播放 raw</button><button data-start="${{x.refined_start_s}}" data-end="${{x.refined_end_s}}">播放 refined</button></td><td>${{x.start_requires_inner?choices(a,x.subisland_id,'start'):'Outer owned'}}</td><td>${{x.end_requires_inner?choices(a,x.subisland_id,'end'):'Outer owned'}}</td><td><input data-note="${{esc(x.subisland_id)}}" value="${{esc(a.note)}}" placeholder="备注"></td></tr>`;}}).join('');const text=r.subislands[0]?.reference_text?`<p><b>参考文本：</b>${{esc(r.subislands[0].reference_text)}}</p>`:'';card.innerHTML=`<h2>${{esc(r.sample_id)}}</h2>${{text}}<audio controls preload="metadata" src="${{esc(r.audio)}}"></audio><table><thead><tr><th>sub-island</th><th>试听</th><th>start</th><th>end</th><th>备注</th></tr></thead><tbody>${{body}}</tbody></table>`;const audio=card.querySelector('audio');card.querySelectorAll('[data-start]').forEach(b=>b.onclick=()=>play(audio,b.dataset.start,b.dataset.end));card.querySelectorAll('[data-id]').forEach(b=>b.onclick=()=>{{const a=ensure(b.dataset.id);a[b.dataset.edge]=b.dataset.label;a.updated_at=new Date().toISOString();persist();render();}});card.querySelectorAll('[data-note]').forEach(i=>i.onchange=()=>{{const a=ensure(i.dataset.note);a.note=i.value;a.updated_at=new Date().toISOString();persist();}});root.appendChild(card);}}status();}}
document.getElementById('save').onclick=async()=>{{const content=rows.flatMap(r=>r.subislands.map(x=>{{const a=ensure(x.subisland_id);return JSON.stringify({{schema:'{VERDICT_SCHEMA}',sample_id:x.sample_id,subisland_id:x.subisland_id,start_verdict:x.start_requires_inner?(a.start||'unreviewed'):'outer_owned',end_verdict:x.end_requires_inner?(a.end||'unreviewed'):'outer_owned',note:a.note||'',bootstrap_prediction:x.bootstrap_prediction,updated_at:a.updated_at||new Date().toISOString()}});}})).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();
</script></body></html>"""
    output_dir.mkdir(parents=True, exist_ok=True)
    page = output_dir / "index.html"
    page.write_text(html, encoding="utf-8")
    if update_latest:
        update_audit_entrypoints(latest_html=page, title="Inner sub-island edge fixed audit")
    return page


def build(
    *,
    inner_inputs: Path,
    outer_labels: Path,
    feature_manifest: Path,
    checkpoint: Path,
    ptm_repo_id: str,
    device: str,
    output_dir: Path,
    update_latest: bool = True,
) -> dict[str, Any]:
    rows, checkpoint_sha = build_items(
        inner_inputs=inner_inputs,
        outer_labels=outer_labels,
        feature_manifest=feature_manifest,
        checkpoint=checkpoint,
        ptm_repo_id=ptm_repo_id,
        device=device,
        output_dir=output_dir,
    )
    items_path = output_dir / "inner_items.jsonl"
    _write(items_path, rows)
    page = build_page(rows=rows, output_dir=output_dir, update_latest=update_latest)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "source_count": len({row["sample_id"] for row in rows}),
        "subisland_count": len(rows),
        "owned_start_count": sum(bool(row["start_requires_inner"]) for row in rows),
        "owned_end_count": sum(bool(row["end_requires_inner"]) for row in rows),
        "bootstrap_checkpoint_sha256": checkpoint_sha,
        "bootstrap_usage": "preview_only_not_training_truth",
        "items": str(items_path),
        "page": str(page),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Inner sub-island edge audit.")
    parser.add_argument("--inner-inputs", required=True)
    parser.add_argument("--outer-labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ptm-repo-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build(
                inner_inputs=Path(args.inner_inputs),
                outer_labels=Path(args.outer_labels),
                feature_manifest=Path(args.feature_manifest),
                checkpoint=Path(args.checkpoint),
                ptm_repo_id=str(args.ptm_repo_id),
                device=str(args.device),
                output_dir=Path(args.output_dir),
                update_latest=not args.no_update_latest,
            ),
            ensure_ascii=False,
        )
    )
