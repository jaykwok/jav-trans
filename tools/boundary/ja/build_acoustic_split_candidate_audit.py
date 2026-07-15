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
from boundary.outer_refiner_v2 import load_outer_edge_refiner_v2  # noqa: E402
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
INNER_REVIEW_LABELS = ("correct", "clipped", "too_wide", "abstain", "unsure")


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


def _edge_frame_features(ptm: np.ndarray, mfcc: np.ndarray) -> np.ndarray:
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


def bootstrap_inner_edges(
    *,
    rows: list[dict[str, Any]],
    checkpoint: Path,
    ptm_repo_id: str,
    device: str,
) -> str:
    """Apply Outer v2 independently to both provisional candidate sides.

    The checkpoint is only a bootstrap visualization for manual review.  It is
    not treated as an Inner training target.
    """

    model = load_outer_edge_refiner_v2(
        checkpoint,
        device=device,
        expected_ptm_repo_id=ptm_repo_id,
    )
    for row in rows:
        feature_path = PROJECT_ROOT / str(row["feature_path"])
        with np.load(feature_path, allow_pickle=False) as payload:
            ptm = np.asarray(payload["ptm"], dtype=np.float32)
            mfcc = np.asarray(payload["mfcc"], dtype=np.float32)
        frame_hop_s = float(row.get("frame_hop_s") or 0.02)
        source_span = dict(row["source_span"])
        outer_start_frame = max(0, int(round(float(source_span["start_s"]) / frame_hop_s)))
        outer_end_frame = min(
            ptm.shape[0], int(round(float(source_span["end_s"]) / frame_hop_s))
        )
        for candidate in row["candidates"]:
            candidate_frame = max(
                outer_start_frame + 1,
                min(
                    outer_end_frame - 1,
                    outer_start_frame
                    + int(round(float(candidate["time_s"]) / frame_hop_s)),
                ),
            )
            left_features = _edge_frame_features(
                ptm[outer_start_frame:candidate_frame],
                mfcc[outer_start_frame:candidate_frame],
            )
            right_features = _edge_frame_features(
                ptm[candidate_frame:outer_end_frame],
                mfcc[candidate_frame:outer_end_frame],
            )
            if left_features.shape[0] < 2 or right_features.shape[0] < 2:
                candidate["bootstrap_inner"] = {
                    "status": "abstain",
                    "reason": "provisional_side_too_short",
                    "checkpoint_sha256": model.sha256,
                }
                continue
            left_duration = float(left_features.shape[0]) * frame_hop_s
            right_duration = float(right_features.shape[0]) * frame_hop_s
            left_prediction, right_prediction = model.predict_islands(
                frame_feature_groups=[left_features, right_features],
                raw_spans=[(0.0, left_duration), (0.0, right_duration)],
                frame_hop_s=frame_hop_s,
            )
            candidate_time = float(candidate["time_s"])
            left_end = min(candidate_time, float(left_prediction.end_s))
            right_start = min(
                float(row["duration_s"]),
                candidate_time + float(right_prediction.start_s),
            )
            abstain_reasons = [
                reason
                for reason in (
                    str(left_prediction.abstain_reason or ""),
                    str(right_prediction.abstain_reason or ""),
                )
                if reason
            ]
            candidate["bootstrap_inner"] = {
                "status": "abstain" if abstain_reasons else "refined",
                "reason": ";".join(abstain_reasons),
                "left_end_s": left_end,
                "right_start_s": right_start,
                "removed_gap_start_s": left_end,
                "removed_gap_end_s": right_start,
                "removed_gap_duration_s": max(0.0, right_start - left_end),
                "left_end_action": str(left_prediction.end_action),
                "right_start_action": str(right_prediction.start_action),
                "checkpoint_sha256": model.sha256,
                "teacher_usage": "bootstrap_preview_only_not_training_truth",
            }
    return model.sha256


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
                    "feature_path": str(feature_row["feature_path"]),
                    "frame_hop_s": float(feature_row.get("frame_hop_s") or 0.02),
                    "source_span": dict(row["source_span"]),
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
</style></head><body><header><strong>Acoustic Split v3 · actual Outer v2 island fixed-{len(rows)}</strong> <button id="save">保存全部裁决</button> <span id="status"></span></header><main><section class="help"><h2>先闭合 Split；bootstrap Inner 只作可选诊断</h2><p>本页不提供文本、Omni 时间轴或旧切点。候选来自独立学习型 BoundaryProposalScorer；它只负责高召回枚举，不是最终切点。</p><p><b>split：</b>此处存在足够明确的 utterance/韵律停顿，值得逻辑拆开。<b>continue：</b>同一句/同一连续 utterance 内的短停顿、犹豫或换气，不应拆。<b>unsure：</b>听不清、连续重叠或难以稳定判断。</p><p><b>bootstrap Inner：</b>现役 Outer v2 分别对候选左右 provisional sub-island 向内精修，仅用于可选预览，不是 Inner 训练真值，也不是本页完成条件。正式顺序是 Split run 闭合后先由 CueQC v13 对完整 provisional sub-island 做 keep/drop/unsure，再只对保留项运行 Inner。若 Split 应成立但当前预览截语音或残留杂音，Split 仍标 <code>split</code>；bootstrap Inner 质量可以选填。</p><p><b>静音/杂音块：</b>若两侧都有真实 speech 且应成为两个 sub-island，则落在同一静音块里的多个候选全部标 <code>split</code>，连续 split run 最终只形成一个 event；noise-only sub-island 后续由 CueQC 整块 drop，removed span 不会成为 chunk。若两侧仍是同一句的短暂停顿则标 <code>continue</code>。若只有一侧有 speech、另一侧只是尾部/前导杂音，这是 Scorer/Outer 问题，标 <code>unsure</code> 并备注“one-sided speech”。</p></section><div id="list"></div></main><script>
const rows={payload};const key='acoustic-split-candidate-audit-v1:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let timer=null,active=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function ensure(r){{ann[r.sample_id]??={{labels:{{}},inner:{{}},coverage:'',note:''}};ann[r.sample_id].labels??={{}};ann[r.sample_id].inner??={{}};return ann[r.sample_id];}}
function hasBootstrapAttempt(c){{return Boolean(c.bootstrap_inner);}}
function hasBootstrap(c){{return Boolean(c.bootstrap_inner&&c.bootstrap_inner.left_end_s!=null&&c.bootstrap_inner.right_start_s!=null&&Number.isFinite(Number(c.bootstrap_inner.left_end_s))&&Number.isFinite(Number(c.bootstrap_inner.right_start_s)));}}
function complete(r,a){{return r.candidates.every(c=>a.labels[c.candidate_id])&&Boolean(a.coverage);}}
function persist(){{localStorage.setItem(key,JSON.stringify(ann));status();}}
function status(){{const done=rows.filter(r=>complete(r,ensure(r))).length;document.getElementById('status').textContent=`完成 ${{done}}/${{rows.length}}`;}}
function play(audio,start,end){{if(active&&active!==audio)active.pause();active=audio;if(timer)clearTimeout(timer);audio.currentTime=Math.max(0,Number(start));audio.play();timer=setTimeout(()=>audio.pause(),Math.max(1,(Number(end)-Number(start))*1000));}}
function choice(a,c,label,text){{return `<button data-cid="${{esc(c.candidate_id)}}" data-label="${{label}}" class="${{label}} ${{a.labels[c.candidate_id]===label?'active':''}}">${{text}}</button>`;}}
function innerChoice(a,c,label,text){{return `<button data-inner-cid="${{esc(c.candidate_id)}}" data-inner-label="${{label}}" class="${{a.inner[c.candidate_id]===label?'active':''}}">${{text}}</button>`;}}
function coverage(a,value,text){{return `<button data-coverage="${{value}}" class="${{a.coverage===value?'active':''}}">${{text}}</button>`;}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r),card=document.createElement('article');if(complete(r,a))card.classList.add('done');const audioId='audio-'+r.sample_id;const body=r.candidates.map(c=>{{const b=c.bootstrap_inner||{{}},innerButtons=hasBootstrapAttempt(c)?`<div><b>bootstrap Inner：</b>${{innerChoice(a,c,'correct','正确')}}${{innerChoice(a,c,'clipped','截语音')}}${{innerChoice(a,c,'too_wide','残留杂音')}}${{innerChoice(a,c,'abstain','失败')}}${{innerChoice(a,c,'unsure','不确定')}}</div>`:'',preview=hasBootstrap(c)?`<button data-play-start="0" data-play-end="${{b.left_end_s}}">精修左 sub-island</button><button data-play-start="${{b.removed_gap_start_s}}" data-play-end="${{b.removed_gap_end_s}}">removed gap</button><button data-play-start="${{b.right_start_s}}" data-play-end="${{r.duration_s}}">精修右 sub-island</button><br><small>left.end=${{Number(b.left_end_s).toFixed(3)}}s · right.start=${{Number(b.right_start_s).toFixed(3)}}s · gap=${{Number(b.removed_gap_duration_s).toFixed(3)}}s · ${{esc(b.status)}}</small>${{innerButtons}}`:`<small>bootstrap Inner: ${{esc(b.status||'disabled')}} ${{esc(b.reason||'')}}</small>${{innerButtons}}`;return `<tr><td><b>${{esc(c.candidate_id)}}</b><br>${{Number(c.time_s).toFixed(3)}}s<br><small>p=${{Number(c.proposer_probability).toFixed(4)}}</small></td><td><button data-play-start="${{c.context_start_s}}" data-play-end="${{c.time_s}}">候选前侧</button><button data-play-start="${{c.time_s}}" data-play-end="${{c.context_end_s}}">候选后侧</button><button data-play-start="${{c.context_start_s}}" data-play-end="${{c.context_end_s}}">候选连续</button><br><small>${{Number(c.context_start_s).toFixed(3)}}–${{Number(c.time_s).toFixed(3)}}–${{Number(c.context_end_s).toFixed(3)}}s</small></td><td>${{preview}}</td><td>${{choice(a,c,'split','split')}}${{choice(a,c,'continue','continue')}}${{choice(a,c,'unsure','unsure')}}</td></tr>`;}}).join('');card.innerHTML=`<h2>${{esc(r.sample_id)}} · ${{Number(r.duration_s).toFixed(3)}}s</h2><small>Outer v2=${{esc(r.outer_checkpoint_sha256.slice(0,12))}} · proposer=${{esc(r.proposer_sha256.slice(0,12))}} · learned projection=${{esc(r.projection_digest.slice(0,12))}}</small><p><b>完整 Outer v2 island</b></p><audio id="${{esc(audioId)}}" controls preload="metadata" src="${{esc(r.audio)}}"></audio><table><thead><tr><th>候选</th><th>原候选上下文</th><th>bootstrap Inner 实际结果</th><th>Split 标签</th></tr></thead><tbody>${{body}}</tbody></table><div class="coverage"><b>候选覆盖：</b>${{coverage(a,'complete','所有实际可分停顿均有候选')}}${{coverage(a,'missed','存在漏掉的可分停顿')}}${{coverage(a,'unsure','不确定')}}</div><label><b>备注</b><textarea placeholder="指出漏候选、句内短暂停顿或 Inner 精修问题">${{esc(a.note)}}</textarea></label>`;const audio=card.querySelector('audio');card.querySelectorAll('[data-play-start]').forEach(b=>b.onclick=()=>play(audio,b.dataset.playStart,b.dataset.playEnd));card.querySelectorAll('[data-cid]').forEach(b=>b.onclick=()=>{{a.labels[b.dataset.cid]=b.dataset.label;a.updated_at=new Date().toISOString();persist();render();}});card.querySelectorAll('[data-inner-cid]').forEach(b=>b.onclick=()=>{{a.inner[b.dataset.innerCid]=b.dataset.innerLabel;a.updated_at=new Date().toISOString();persist();render();}});card.querySelectorAll('[data-coverage]').forEach(b=>b.onclick=()=>{{a.coverage=b.dataset.coverage;a.updated_at=new Date().toISOString();persist();render();}});card.querySelector('textarea').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};root.appendChild(card);}}status();}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r);return JSON.stringify({{schema:'{VERDICT_SCHEMA}',sample_id:r.sample_id,coverage:a.coverage||'unreviewed',candidates:r.candidates.map(c=>({{candidate_id:c.candidate_id,time_s:c.time_s,proposer_probability:c.proposer_probability,label:a.labels[c.candidate_id]||'unreviewed',inner_verdict:a.inner[c.candidate_id]||'not_reviewed',bootstrap_inner:c.bootstrap_inner||null}})),complete:complete(r,a),note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();
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
    inner_bootstrap_checkpoint: Path | None,
    ptm_repo_id: str,
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
    bootstrap_sha256 = ""
    if inner_bootstrap_checkpoint is not None:
        bootstrap_sha256 = bootstrap_inner_edges(
            rows=selected,
            checkpoint=inner_bootstrap_checkpoint,
            ptm_repo_id=ptm_repo_id,
            device=device,
        )
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
        "inner_bootstrap_checkpoint": (
            str(inner_bootstrap_checkpoint) if inner_bootstrap_checkpoint else ""
        ),
        "inner_bootstrap_sha256": bootstrap_sha256,
        "inner_bootstrap_usage": (
            "preview_only_not_training_truth" if inner_bootstrap_checkpoint else "disabled"
        ),
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
    parser.add_argument("--inner-bootstrap-checkpoint", default="")
    parser.add_argument(
        "--ptm-repo-id",
        default="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
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
                inner_bootstrap_checkpoint=(
                    Path(args.inner_bootstrap_checkpoint)
                    if args.inner_bootstrap_checkpoint
                    else None
                ),
                ptm_repo_id=str(args.ptm_repo_id),
                output_dir=Path(args.output_dir),
                count=int(args.count),
                device=str(args.device),
                update_latest=not args.no_update_latest,
            ),
            ensure_ascii=False,
        )
    )
