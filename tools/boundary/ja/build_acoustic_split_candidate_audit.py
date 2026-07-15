#!/usr/bin/env python3
"""Build an audio-only fixed-N audit for conservative acoustic Split v3.

The learned proposer only enumerates high-recall boundary evidence.  Reviewers
decide whether each candidate pause should logically separate two provisional
speech sub-islands; exact speech edges are intentionally deferred to the Inner
Edge Refiner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.ja.model import (  # noqa: E402
    checkpoint_sha256,
    score_speech_island_probabilities_batch,
)
from boundary.ja.proposal import load_boundary_proposal_checkpoint  # noqa: E402
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.boundary.ja.build_semantic_anchor_proposer_audit import (  # noqa: E402
    select_stratified_proposer_frames,
)
from tools.boundary.ja.label_semantic_source_candidates_with_omni import (  # noqa: E402
    learned_frame_embeddings,
    load_task_aware_projection,
)


SCHEMA = "acoustic_split_candidate_audit_v1"
SUMMARY_SCHEMA = "acoustic_split_candidate_audit_summary_v1"
VERDICT_SCHEMA = "acoustic_split_candidate_manual_verdict_v1"
OUTER_AUDIO_CONTRACT = "learned_outer_refined_island_v1"
ALLOWED_LABELS = ("split", "continue", "unsure")


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def adaptive_candidate_count(frame_count: int) -> int:
    """Scale the 5..9 audit budget with available acoustic resolution."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    return max(5, min(9, int(math.ceil(math.sqrt(frame_count) / 3.0))))


def adaptive_internal_frame_region(
    frame_count: int, *, candidate_count: int
) -> tuple[int, int]:
    """Exclude Outer-edge peaks with a duration-proportional interior band."""

    if frame_count <= candidate_count + 2:
        raise ValueError("not enough frames for internal Split candidates")
    margin = max(1, int(math.floor(frame_count / (candidate_count + 2))))
    start = margin
    end = frame_count - margin
    if end - start < candidate_count:
        start, end = 1, frame_count - 1
    if end - start < candidate_count:
        raise ValueError("adaptive internal region has too few frames")
    return start, end


def candidate_contexts(times_s: list[float], *, duration_s: float) -> list[dict[str, float]]:
    """Partition an island at adjacent-candidate midpoints without overlap."""

    if not times_s or times_s != sorted(times_s) or len(set(times_s)) != len(times_s):
        raise ValueError("candidate times must be unique and sorted")
    if not all(0.0 < value < duration_s for value in times_s):
        raise ValueError("candidate times must be strictly inside the island")
    contexts: list[dict[str, float]] = []
    for index, value in enumerate(times_s):
        start = 0.0 if index == 0 else (times_s[index - 1] + value) / 2.0
        end = duration_s if index + 1 == len(times_s) else (value + times_s[index + 1]) / 2.0
        contexts.append(
            {
                "context_start_s": float(start),
                "candidate_time_s": float(value),
                "context_end_s": float(end),
            }
        )
    return contexts


def select_fixed_sources(scored_rows: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    """Cover duration quantiles and prefer strong learned-proposer evidence."""

    if count <= 0 or len(scored_rows) < count:
        raise ValueError("not enough scored sources for fixed audit")
    ordered = sorted(scored_rows, key=lambda row: (float(row["duration_s"]), str(row["sample_id"])))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    edges = np.linspace(0, len(ordered), count + 1, dtype=np.int64)
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        bucket = ordered[int(left) : max(int(left) + 1, int(right))]
        winner = max(
            bucket,
            key=lambda row: (
                float(row["max_proposer_probability"]),
                float(row["proposer_probability_std"]),
                str(row["sample_id"]),
            ),
        )
        selected.append(winner)
        selected_ids.add(str(winner["sample_id"]))
    if len(selected) < count:
        remaining = sorted(
            (row for row in scored_rows if str(row["sample_id"]) not in selected_ids),
            key=lambda row: (-float(row["max_proposer_probability"]), str(row["sample_id"])),
        )
        selected.extend(remaining[: count - len(selected)])
    return sorted(selected, key=lambda row: float(row["duration_s"]))


def _score_rows(
    *,
    outer_rows: list[dict[str, Any]],
    feature_by_id: dict[str, dict[str, Any]],
    projection_path: Path,
    proposer_path: Path,
    device: str,
) -> list[dict[str, Any]]:
    mean, components, projection_digest = load_task_aware_projection(projection_path)
    if mean.size != 2048 or components.shape != (128, 2048):
        raise ValueError("audit requires learned Linear(2048->128) projection")
    proposer = load_boundary_proposal_checkpoint(proposer_path, device=device)
    if proposer.ptm_dim != 128:
        raise ValueError("proposer must consume the learned PTM128 representation")

    prepared: list[tuple[dict[str, Any], np.ndarray, np.ndarray, float, float]] = []
    for row in outer_rows:
        sample_id = str(row["sample_id"])
        if row.get("audio_contract") != OUTER_AUDIO_CONTRACT:
            raise ValueError(f"{sample_id}: expected learned Outer v2 audio contract")
        feature_row = feature_by_id.get(sample_id)
        if feature_row is None:
            raise ValueError(f"{sample_id}: feature manifest row missing")
        payload = np.load(PROJECT_ROOT / str(feature_row["feature_path"]), allow_pickle=False)
        projected = learned_frame_embeddings(
            ptm=np.asarray(payload["ptm"], dtype=np.float32),
            projection_mean=mean,
            projection_components=components,
        )
        mfcc = np.asarray(payload["mfcc"], dtype=np.float32)
        span = dict(row["source_span"])
        prepared.append((row, projected, mfcc, float(span["start_s"]), float(span["end_s"])))

    scored: list[dict[str, Any]] = []
    batch_size = 16
    for batch_start in range(0, len(prepared), batch_size):
        batch = prepared[batch_start : batch_start + batch_size]
        probabilities = score_speech_island_probabilities_batch(
            proposer,
            feature_pairs=[(item[1], item[2]) for item in batch],
        )
        for (row, _projected, _mfcc, source_start_s, source_end_s), values in zip(
            batch, probabilities, strict=True
        ):
            frame_hop_s = 0.02
            start_frame = max(0, int(math.ceil(source_start_s / frame_hop_s)))
            end_frame = min(values.size, int(math.floor(source_end_s / frame_hop_s)))
            cropped = np.asarray(values[start_frame:end_frame], dtype=np.float32)
            if cropped.size < 25:
                continue
            count = adaptive_candidate_count(int(cropped.size))
            internal_start, internal_end = adaptive_internal_frame_region(
                int(cropped.size), candidate_count=count
            )
            frames = select_stratified_proposer_frames(
                cropped,
                region_start_s=float(internal_start) * frame_hop_s,
                region_end_s=float(internal_end) * frame_hop_s,
                frame_hop_s=frame_hop_s,
                candidate_count=count,
            )
            duration_s = float(row["duration_s"])
            times_s = [
                min(duration_s - 1e-6, max(1e-6, float(frame) * frame_hop_s))
                for frame in frames
            ]
            contexts = candidate_contexts(times_s, duration_s=duration_s)
            candidates = []
            for index, (frame, context) in enumerate(zip(frames, contexts, strict=True)):
                candidates.append(
                    {
                        "candidate_id": f"c{index:02d}",
                        "time_s": float(context["candidate_time_s"]),
                        "context_start_s": float(context["context_start_s"]),
                        "context_end_s": float(context["context_end_s"]),
                        "proposer_probability": float(cropped[int(frame)]),
                    }
                )
            scored.append(
                {
                    "schema": SCHEMA,
                    "sample_id": str(row["sample_id"]),
                    "audio": str(row["audio"]),
                    "duration_s": duration_s,
                    "audio_contract": OUTER_AUDIO_CONTRACT,
                    "outer_checkpoint_sha256": str(row["outer_prediction"]["checkpoint_sha256"]),
                    "projection_digest": projection_digest,
                    "projection_file_sha256": _file_sha256(projection_path),
                    "proposer_sha256": checkpoint_sha256(proposer_path),
                    "candidate_count": len(candidates),
                    "max_proposer_probability": float(np.max(cropped)),
                    "proposer_probability_std": float(np.std(cropped)),
                    "candidates": candidates,
                }
            )
    return scored


def _copy_audio(rows: list[dict[str, Any]], output_dir: Path) -> None:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = PROJECT_ROOT / str(row["audio"])
        suffix = source.suffix.lower() or ".wav"
        target = audio_dir / f"{row['sample_id']}{suffix}"
        shutil.copyfile(source, target)
        row["audio"] = target.relative_to(output_dir).as_posix()


def build_audit_html(
    *, rows: list[dict[str, Any]], output_dir: Path, update_latest: bool = True
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Acoustic Split v3 fixed audit</title><style>
body{{margin:0;background:#0d1117;color:#e6edf3;font-family:system-ui}}header{{position:sticky;top:0;z-index:2;padding:12px 18px;background:#161b22;border-bottom:1px solid #30363d}}main{{max-width:1200px;margin:auto;padding:16px}}article{{padding:16px;margin:14px 0;background:#161b22;border:1px solid #30363d;border-radius:10px}}article.done{{border-color:#2ea043}}.help{{background:#10233f;padding:14px;border-radius:10px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #30363d;text-align:left}}button{{margin:3px;padding:6px 10px;background:#21262d;color:#e6edf3;border:1px solid #484f58;border-radius:6px}}button.active{{background:#1f6feb;border-color:#58a6ff}}button.split.active{{background:#238636}}button.continue.active{{background:#9e6a03}}button.unsure.active{{background:#8250df}}audio{{width:100%}}small{{color:#8b949e}}textarea{{width:100%;min-height:60px;background:#0d1117;color:#e6edf3}}.coverage{{margin-top:10px;padding-top:10px;border-top:1px solid #30363d}}
</style></head><body><header><strong>Acoustic Split v3 · actual Outer v2 island fixed-{len(rows)}</strong> <button id="save">保存全部裁决</button> <span id="status"></span></header><main><section class="help"><h2>只判断是否值得拆成两个 provisional speech sub-island</h2><p>本页不提供文本、Omni 时间轴或旧切点。候选来自独立学习型 BoundaryProposalScorer；它只负责高召回枚举，不是最终切点。</p><p><b>split：</b>此处存在足够明确的 utterance/韵律停顿，值得逻辑拆开；不要求候选秒数正好落在安全边缘。<b>continue：</b>同一句/同一连续 utterance 内的短停顿、犹豫或换气，不应拆。<b>unsure：</b>听不清、连续重叠或难以稳定判断。</p><p>每个候选的前侧和后侧播放区间由相邻候选中点划分，彼此不重叠。Inner Refiner 后续仍从原 island 上精修相邻 sub-island 的 end/start，本页不裁精确边缘。</p></section><div id="list"></div></main><script>
const rows={payload};const key='acoustic-split-candidate-audit-v1:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let timer=null,active=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function ensure(r){{ann[r.sample_id]??={{labels:{{}},coverage:'',note:''}};return ann[r.sample_id];}}
function complete(r,a){{return r.candidates.every(c=>a.labels[c.candidate_id])&&Boolean(a.coverage);}}
function persist(){{localStorage.setItem(key,JSON.stringify(ann));status();}}
function status(){{const done=rows.filter(r=>complete(r,ensure(r))).length;document.getElementById('status').textContent=`完成 ${{done}}/${{rows.length}}`;}}
function play(audio,start,end){{if(active&&active!==audio)active.pause();active=audio;if(timer)clearTimeout(timer);audio.currentTime=Math.max(0,Number(start));audio.play();timer=setTimeout(()=>audio.pause(),Math.max(1,(Number(end)-Number(start))*1000));}}
function choice(a,c,label,text){{return `<button data-cid="${{esc(c.candidate_id)}}" data-label="${{label}}" class="${{label}} ${{a.labels[c.candidate_id]===label?'active':''}}">${{text}}</button>`;}}
function coverage(a,value,text){{return `<button data-coverage="${{value}}" class="${{a.coverage===value?'active':''}}">${{text}}</button>`;}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r),card=document.createElement('article');if(complete(r,a))card.classList.add('done');const audioId='audio-'+r.sample_id;const body=r.candidates.map(c=>`<tr><td><b>${{esc(c.candidate_id)}}</b><br>${{Number(c.time_s).toFixed(3)}}s<br><small>p=${{Number(c.proposer_probability).toFixed(4)}}</small></td><td><button data-play-start="${{c.context_start_s}}" data-play-end="${{c.time_s}}">播放前侧</button><button data-play-start="${{c.time_s}}" data-play-end="${{c.context_end_s}}">播放后侧</button><button data-play-start="${{c.context_start_s}}" data-play-end="${{c.context_end_s}}">连续播放</button><br><small>${{Number(c.context_start_s).toFixed(3)}}–${{Number(c.time_s).toFixed(3)}}–${{Number(c.context_end_s).toFixed(3)}}s</small></td><td>${{choice(a,c,'split','split')}}${{choice(a,c,'continue','continue')}}${{choice(a,c,'unsure','unsure')}}</td></tr>`).join('');card.innerHTML=`<h2>${{esc(r.sample_id)}} · ${{Number(r.duration_s).toFixed(3)}}s</h2><small>Outer v2=${{esc(r.outer_checkpoint_sha256.slice(0,12))}} · proposer=${{esc(r.proposer_sha256.slice(0,12))}} · learned projection=${{esc(r.projection_digest.slice(0,12))}}</small><p><b>完整 Outer v2 island</b></p><audio id="${{esc(audioId)}}" controls preload="metadata" src="${{esc(r.audio)}}"></audio><table><thead><tr><th>候选</th><th>不重叠试听区间</th><th>是否逻辑拆分</th></tr></thead><tbody>${{body}}</tbody></table><div class="coverage"><b>候选覆盖：</b>${{coverage(a,'complete','所有实际可分停顿均有候选')}}${{coverage(a,'missed','存在漏掉的可分停顿')}}${{coverage(a,'unsure','不确定')}}</div><label><b>备注</b><textarea placeholder="指出漏候选、句内短暂停顿或其他问题">${{esc(a.note)}}</textarea></label>`;const audio=card.querySelector('audio');card.querySelectorAll('[data-play-start]').forEach(b=>b.onclick=()=>play(audio,b.dataset.playStart,b.dataset.playEnd));card.querySelectorAll('[data-cid]').forEach(b=>b.onclick=()=>{{a.labels[b.dataset.cid]=b.dataset.label;a.updated_at=new Date().toISOString();persist();render();}});card.querySelectorAll('[data-coverage]').forEach(b=>b.onclick=()=>{{a.coverage=b.dataset.coverage;a.updated_at=new Date().toISOString();persist();render();}});card.querySelector('textarea').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};root.appendChild(card);}}status();}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r);return JSON.stringify({{schema:'{VERDICT_SCHEMA}',sample_id:r.sample_id,coverage:a.coverage||'unreviewed',candidates:r.candidates.map(c=>({{candidate_id:c.candidate_id,time_s:c.time_s,proposer_probability:c.proposer_probability,label:a.labels[c.candidate_id]||'unreviewed'}})),complete:complete(r,a),note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();
</script></body></html>"""
    page = output_dir / "index.html"
    page.write_text(html, encoding="utf-8")
    if update_latest:
        update_audit_entrypoints(
            latest_html=page, title="Acoustic Split v3 fixed audit"
        )
    return page


def build_audit(
    *,
    outer_labels: Path,
    feature_manifest: Path,
    projection_path: Path,
    proposer_path: Path,
    output_dir: Path,
    count: int,
    device: str,
    update_latest: bool = True,
) -> dict[str, Any]:
    outer_rows = _rows(outer_labels)
    feature_by_id = {str(row["sample_id"]): row for row in _rows(feature_manifest)}
    scored = _score_rows(
        outer_rows=outer_rows,
        feature_by_id=feature_by_id,
        projection_path=projection_path,
        proposer_path=proposer_path,
        device=device,
    )
    selected = select_fixed_sources(scored, count=count)
    _copy_audio(selected, output_dir)
    items_path = output_dir / "audit_items.jsonl"
    _write_jsonl(items_path, selected)
    page = build_audit_html(rows=selected, output_dir=output_dir, update_latest=update_latest)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "source_count": len(outer_rows),
        "scored_source_count": len(scored),
        "selected_count": len(selected),
        "candidate_count": sum(len(row["candidates"]) for row in selected),
        "labels": list(ALLOWED_LABELS),
        "text_or_timeline_visible": False,
        "decision_contract": "conservative_acoustic_utterance_boundary_not_exact_edge_v1",
        "projection_path": str(projection_path),
        "projection_sha256": _file_sha256(projection_path),
        "proposer_path": str(proposer_path),
        "proposer_sha256": checkpoint_sha256(proposer_path),
        "items": str(items_path),
        "page": str(page),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audio-only acoustic Split v3 fixed audit.")
    parser.add_argument("--outer-labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--proposer-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-update-latest", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.count <= 15:
        parser.error("--count must be in 1..15")
    return args


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build_audit(
                outer_labels=Path(args.outer_labels),
                feature_manifest=Path(args.feature_manifest),
                projection_path=Path(args.projection),
                proposer_path=Path(args.proposer_checkpoint),
                output_dir=Path(args.output_dir),
                count=int(args.count),
                device=str(args.device),
                update_latest=not args.no_update_latest,
            ),
            ensure_ascii=False,
        )
    )
