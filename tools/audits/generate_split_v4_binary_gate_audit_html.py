#!/usr/bin/env python3
"""Build a listenable audit for Split v4 false cuts and >8s residuals."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.split_model import load_acoustic_split_v4_planner  # noqa: E402
from tools.boundary.ja.train_acoustic_split_v4_model import (  # noqa: E402
    _pad_batch,
    apply_manual_label_overrides,
    island_batches,
    partition_group_names,
)
from tools.boundary.ja.acoustic_split_v4_dataset import (  # noqa: E402
    load_island_dataset,
)
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _manifest_sources(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text("utf-8"))
    return {str(row["audio_id"]): dict(row) for row in payload}


def _cut_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    position = 0
    while position < values.size:
        if int(values[position]) != 0:
            position += 1
            continue
        start = position
        while position + 1 < values.size and int(values[position + 1]) == 0:
            position += 1
        runs.append((start, position))
        position += 1
    return runs


def build(args: argparse.Namespace) -> dict:
    import torch

    planner = load_acoustic_split_v4_planner(args.checkpoint, device=args.device)
    data = load_island_dataset(Path(args.dataset))
    manual_override_summary = None
    if args.label_overrides:
        manual_override_summary = apply_manual_label_overrides(
            data,
            metadata_path=Path(args.metadata),
            overrides_path=Path(args.label_overrides),
        )
    metadata = _read_jsonl(Path(args.metadata))
    if len(metadata) != int(data["labels"].shape[0]):
        raise ValueError("metadata row count does not match dataset")
    source_by_id = _manifest_sources(Path(args.source_manifest))
    normalization = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in planner.normalization.items()
    }
    partitions = partition_group_names(data)
    selected = partitions["val"] + partitions["test"]
    audit_rows: list[dict] = []
    false_cut_count = 0
    residual_count = 0
    planner.model.eval()
    with torch.inference_mode():
        for batch in island_batches(
            selected,
            data["groups"],
            batch_islands=args.batch_islands,
            max_batch_candidates=args.max_batch_candidates,
        ):
            frames, scalars, mask, labels, *_ = _pad_batch(
                data,
                batch,
                frame_mean=normalization["frame_mean"],
                frame_std=normalization["frame_std"],
                scalar_mean=normalization["scalar_mean"],
                scalar_std=normalization["scalar_std"],
            )
            probabilities = torch.softmax(
                planner.model(frames.to(planner.device), scalars.to(planner.device), mask.to(planner.device))["label"],
                dim=-1,
            ).cpu().numpy()
            for batch_index, group_name in enumerate(batch):
                indexes = data["groups"][group_name]
                count = int(indexes.size)
                rows = [metadata[int(index)] for index in indexes]
                predicted = probabilities[batch_index, :count].argmax(axis=-1)
                p_cut = probabilities[batch_index, :count, 0]
                truth = labels[batch_index, :count].numpy()
                first = rows[0]
                audio_id = str(first["audio_id"])
                source_row = source_by_id.get(audio_id) or {}
                audio = str(source_row.get("audio") or "")
                source_duration_s = float(
                    source_row.get("duration_s") or first["core_end"]
                )
                partition = str(first["partition"])
                truth_for_events = truth.copy()
                predicted_for_events = predicted.copy()
                predicted_for_events[~np.isin(truth_for_events, (0, 1))] = -100
                truth_runs = _cut_runs(truth_for_events)
                predicted_runs = _cut_runs(predicted_for_events)
                predicted_representatives = [
                    max(range(start, end + 1), key=lambda index: p_cut[index])
                    for start, end in predicted_runs
                ]
                predicted_event_times = [
                    float(rows[index]["time_s"])
                    for index in predicted_representatives
                ]
                predicted_boundaries = [
                    float(first["core_start"]),
                    *predicted_event_times,
                    float(first["core_end"]),
                ]
                predicted_subislands = [
                    {
                        "subisland_index": index,
                        "start_s": start,
                        "end_s": end,
                        "duration_s": end - start,
                    }
                    for index, (start, end) in enumerate(
                        zip(predicted_boundaries, predicted_boundaries[1:])
                    )
                ]
                unmatched_truth = set(range(len(truth_runs)))
                unmatched_predicted: list[tuple[int, int]] = []
                for predicted_run in predicted_runs:
                    predicted_indexes = set(range(predicted_run[0], predicted_run[1] + 1))
                    matches = [
                        index
                        for index in unmatched_truth
                        if predicted_indexes & set(range(truth_runs[index][0], truth_runs[index][1] + 1))
                    ]
                    if matches:
                        unmatched_truth.remove(matches[0])
                    else:
                        unmatched_predicted.append(predicted_run)
                for start_index, end_index in unmatched_predicted:
                    representative = max(
                        range(start_index, end_index + 1), key=lambda index: p_cut[index]
                    )
                    row = rows[representative]
                    event_index = predicted_runs.index((start_index, end_index))
                    subisland_start = float(row["time_s"])
                    subisland_end = (
                        predicted_event_times[event_index + 1]
                        if event_index + 1 < len(predicted_event_times)
                        else float(first["core_end"])
                    )
                    false_cut_count += 1
                    audit_rows.append({
                            "audit_id": f"false-cut-event-{false_cut_count:04d}",
                            "category": "unmatched_predicted_cut_event",
                            "partition": partition,
                            "audio_id": audio_id,
                            "audio": audio,
                            "source_duration_s": source_duration_s,
                            "time_s": float(row["time_s"]),
                            "core_start": float(row["core_start"]),
                            "core_end": float(row["core_end"]),
                            "subisland_start_s": subisland_start,
                            "subisland_end_s": subisland_end,
                            "subisland_duration_s": subisland_end - subisland_start,
                            "predicted_event_times_s": predicted_event_times,
                            "predicted_subislands": predicted_subislands,
                            "p_cut": float(p_cut[representative]),
                            "candidate_start_index": start_index,
                            "candidate_end_index": end_index,
                            "truth": "no_cut_event",
                            "prediction": "cut",
                        })
                boundaries = predicted_boundaries
                for start, end in zip(boundaries, boundaries[1:]):
                    if end - start <= args.residual_threshold_s:
                        continue
                    residual_candidates = [
                        {
                            "candidate_id": (
                                f"{audio_id}__s{int(row.get('segment_index') or 0):02d}"
                                f"__t{float(row['time_s']):.3f}"
                            ),
                            "candidate_index": position,
                            "time_s": float(row["time_s"]),
                            "p_cut": float(p_cut[position]),
                            "p_continue": float(1.0 - p_cut[position]),
                            "model_label": (
                                "cut" if int(predicted[position]) == 0 else "continue"
                            ),
                        }
                        for position, row in enumerate(rows)
                        if start < float(row["time_s"]) < end
                        and int(truth[position]) in (0, 1)
                    ]
                    residual_count += 1
                    audit_rows.append({
                        "audit_id": f"long-residual-{residual_count:04d}",
                        "category": "long_residual",
                        "partition": partition,
                        "audio_id": audio_id,
                        "audio": audio,
                        "source_duration_s": source_duration_s,
                        "start_s": start,
                        "end_s": end,
                        "duration_s": end - start,
                        "core_start": float(first["core_start"]),
                        "core_end": float(first["core_end"]),
                        "subisland_start_s": start,
                        "subisland_end_s": end,
                        "subisland_duration_s": end - start,
                        "predicted_event_times_s": predicted_event_times,
                        "predicted_subislands": predicted_subislands,
                        "residual_candidates": residual_candidates,
                    })
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for row in audit_rows:
        audio_id = str(row["audio_id"])
        if audio_id not in copied:
            source = PROJECT_ROOT / row["audio"]
            target = audio_dir / f"{audio_id}{source.suffix.lower() or '.wav'}"
            shutil.copyfile(source, target)
            copied[audio_id] = target.relative_to(output).as_posix()
        row["audio_src"] = copied[audio_id]
    manifest = output / "audit_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows), "utf-8")
    summary = {
        "schema": "split_v4_binary_gate_audit_v1",
        "checkpoint": planner.signature(),
        "partitions": ["val", "test"],
        "false_cut_count": false_cut_count,
        "long_residual_count": residual_count,
        "residual_threshold_s": args.residual_threshold_s,
        "audit_item_count": len(audit_rows),
        "manual_gate": "pending",
        "manual_label_overrides": manual_override_summary,
        "manifest": str(manifest),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")
    payload = json.dumps(audit_rows, ensure_ascii=False).replace("</", "<\\/")
    page = output / "index.html"
    page.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset=utf-8><title>Split v4 binary gate audit</title>
<style>body{{margin:0;background:#0d1117;color:#e6edf3;font-family:system-ui}}header{{position:sticky;top:0;z-index:2;background:#161b22;padding:12px 18px;border-bottom:1px solid #30363d}}main{{max-width:1100px;margin:auto;padding:16px}}article{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin:14px 0}}article.done{{border-color:#2ea043}}.help{{background:#10233f;padding:14px;border-radius:10px}}audio{{width:100%}}button,input{{margin:3px;padding:6px 10px;background:#21262d;color:#e6edf3;border:1px solid #484f58;border-radius:6px}}button.active{{background:#1f6feb}}button.false_cut.active,button.missing_cut.active{{background:#da3633}}button.valid_cut.active,button.acceptable.active{{background:#238636}}button.unsure.active{{background:#8250df}}input{{width:65%}}small{{color:#8b949e}}pre{{white-space:pre-wrap;background:#0d1117;padding:8px}}</style></head><body>
<header><strong>Acoustic Split v4 · 二分类晋升人工 gate</strong> <button id="save">保存审计结果</button> <span id="status"></span></header><main><section class="help"><h2>只审核 Split event，不审核 CueQC / Inner / 字幕</h2><p><b>误切 event：</b>重点听红点前后是否属于同一句真实 speech。若这里确实应切，选“真实边界”；若切断句内语音，选“句内误切”。</p><p><b>&gt;8s residual：</b>完整听 residual，判断是否漏掉必须切开的真实句界；纯长句或合理连续语音选“可接受”，存在漏切选“存在漏切”。</p></section><div id="list"></div></main><script>
const rows={payload},key='split-v4-binary-gate-audit-v1:'+location.pathname,ann=JSON.parse(localStorage.getItem(key)||'{{}}');let timer=null,active=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function ensure(r){{ann[r.audit_id]??={{verdict:'',note:''}};return ann[r.audit_id]}}function persist(){{localStorage.setItem(key,JSON.stringify(ann));status()}}function status(){{document.getElementById('status').textContent=`完成 ${{rows.filter(r=>ensure(r).verdict).length}}/${{rows.length}}`}}function play(audio,start,end){{if(active&&active!==audio)active.pause();active=audio;if(timer)clearTimeout(timer);audio.currentTime=Math.max(0,Number(start));audio.play();timer=setTimeout(()=>audio.pause(),Math.max(1,(Number(end)-Number(start))*1000))}}function choice(a,label,text){{return `<button data-label="${{label}}" class="${{label}} ${{a.verdict===label?'active':''}}">${{text}}</button>`}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r),card=document.createElement('article'),coreStart=Number(r.core_start),coreEnd=Number(r.core_end),subs=r.predicted_subislands||[];if(a.verdict)card.classList.add('done');const subButtons=subs.map(s=>`<button data-start="${{s.start_s}}" data-end="${{s.end_s}}">sub ${{Number(s.subisland_index)+1}} · ${{Number(s.start_s).toFixed(3)}}–${{Number(s.end_s).toFixed(3)}}s</button>`).join('');const controls=`<button data-start="${{coreStart}}" data-end="${{coreEnd}}">播放完整 chunk</button><div>${{subButtons}}</div>`;const verdicts=r.category==='unmatched_predicted_cut_event'?choice(a,'valid_cut','真实边界')+choice(a,'false_cut','句内误切')+choice(a,'unsure','听不清'):choice(a,'acceptable','可接受')+choice(a,'missing_cut','存在漏切')+choice(a,'unsure','听不清');card.innerHTML=`<h2>${{esc(r.audit_id)}} · ${{esc(r.audio_id)}}</h2><small>${{esc(r.partition)}} · ${{esc(r.category)}} · chunk ${{coreStart.toFixed(3)}}–${{coreEnd.toFixed(3)}}s · ${{r.predicted_event_times_s.length}} cuts → ${{subs.length}} subs · p_cut=${{r.p_cut==null?'-':Number(r.p_cut).toFixed(4)}}</small><audio controls preload="metadata" src="${{esc(r.audio_src)}}"></audio><div>${{controls}}</div><div>${{verdicts}} <input placeholder="备注" value="${{esc(a.note)}}"></div><pre>${{esc(JSON.stringify(r,null,2))}}</pre>`;const audio=card.querySelector('audio');card.querySelectorAll('[data-start]').forEach(b=>b.onclick=()=>play(audio,b.dataset.start,b.dataset.end));card.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>{{a.verdict=b.dataset.label;a.updated_at=new Date().toISOString();persist();render()}});card.querySelector('input').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist()}};root.appendChild(card)}}status()}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r);return JSON.stringify({{schema:'split_v4_binary_gate_manual_verdict_v1',audit_id:r.audit_id,category:r.category,audio_id:r.audio_id,verdict:a.verdict||'unreviewed',note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}})}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error}};render();
</script></body></html>""", "utf-8")
    if not args.no_update_latest:
        update_audit_entrypoints(
            latest_html=page,
            title="Acoustic Split v4 Binary Gate Audit",
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--label-overrides", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-islands", type=int, default=4)
    parser.add_argument("--max-batch-candidates", type=int, default=256)
    parser.add_argument("--residual-threshold-s", type=float, default=8.0)
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False))
