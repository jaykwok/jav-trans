#!/usr/bin/env python3
"""Build a fixed semantic-anchor audit from a learned projected-PTM proposer."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from audio.loading import load_audio_16k_mono  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.features import (  # noqa: E402
    FeatureConfig,
    build_ptm_feature_extractor,
    extract_mfcc,
)
from boundary.ja.model import (  # noqa: E402
    checkpoint_sha256,
    score_speech_island_probabilities_batch,
)
from boundary.ja.proposal import load_boundary_proposal_checkpoint  # noqa: E402
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.boundary.ja.build_feature_cache import (  # noqa: E402
    _combine_workflow_window_features,
    _extract_ptm_window_features,
    _workflow_window_starts,
)
from tools.boundary.ja.label_semantic_source_candidates_with_omni import (  # noqa: E402
    learned_frame_embeddings,
    load_task_aware_projection,
)


SCHEMA = "semantic_anchor_learned_proposer_audit_v2"
SUMMARY_SCHEMA = "semantic_anchor_learned_proposer_audit_summary_v2"
DEFAULT_PROJECTION = PROJECT_ROOT / (
    "src/checkpoints/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf/"
    "semantic_split_model_v2.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf."
    "ptm_projection.93f60750a9a0f88d.npz"
)
LABELS = ("left_clipped", "safe", "right_clipped", "unsure")


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


def adaptive_event_regions(row: dict[str, Any]) -> list[dict[str, Any]]:
    events = [
        event
        for event in row.get("semantic_events") or []
        if event.get("status") == "matched"
    ]
    anchors = [
        float(event["coarse_anchor_s"])
        if event.get("coarse_anchor_s") is not None
        else (
            float(event["interval_start_s"]) + float(event["interval_end_s"])
        )
        / 2.0
        for event in events
    ]
    blockers = sorted(
        float(item["time_s"]) for item in row.get("region_blocker_anchors") or []
    )
    all_anchors = sorted([*anchors, *blockers])
    duration_s = float(row["duration_s"])
    result: list[dict[str, Any]] = []
    for index, (event, anchor_s) in enumerate(zip(events, anchors, strict=True)):
        previous = max((value for value in all_anchors if value < anchor_s), default=None)
        following = min((value for value in all_anchors if value > anchor_s), default=None)
        region_start_s = 0.0 if previous is None else (previous + anchor_s) / 2.0
        region_end_s = duration_s if following is None else (anchor_s + following) / 2.0
        if not 0.0 <= region_start_s < region_end_s <= duration_s:
            raise ValueError(f"invalid adaptive event region for {row['sample_id']}")
        result.append(
            {
                **event,
                "coarse_anchor_s": anchor_s,
                "region_start_s": region_start_s,
                "region_end_s": region_end_s,
                "region_contract": "adjacent_semantic_or_blocker_anchor_midpoints_with_source_edges_v2",
            }
        )
    return result


def select_stratified_proposer_frames(
    probabilities: np.ndarray,
    *,
    region_start_s: float,
    region_end_s: float,
    frame_hop_s: float,
    candidate_count: int,
) -> list[int]:
    values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if not 5 <= candidate_count <= 9:
        raise ValueError("candidate_count must be in 5..9")
    start = max(0, int(np.floor(region_start_s / frame_hop_s)))
    end = min(values.size, int(np.ceil(region_end_s / frame_hop_s)))
    if end - start < candidate_count:
        raise ValueError("adaptive region has fewer frames than candidates")
    edges = np.linspace(start, end, candidate_count + 1, dtype=np.int64)
    selected: list[int] = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        right = max(int(left) + 1, int(right))
        local = values[int(left) : right]
        selected.append(int(left) + int(np.argmax(local)))
    if len(set(selected)) != candidate_count:
        raise ValueError("stratified proposer selection produced duplicate frames")
    return sorted(selected)


def _extract_features(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    ptm_repo: str,
    model_path: str,
    device: str,
    dtype: str,
) -> list[dict[str, Any]]:
    config = FeatureConfig(
        ptm=ptm_repo,
        frame_hop_s=0.02,
        window_s=30.0,
        overlap_s=5.0,
        n_mfcc=40,
        n_fft=400,
        feature_dim=2048,
        device=device,
        dtype=dtype,
        model_path=model_path,
        download=False,
        attention="sdpa",
        language="Japanese",
    )
    extractor = build_ptm_feature_extractor(config)
    manifest: list[dict[str, Any]] = []
    try:
        for position, row in enumerate(rows, start=1):
            audio, sample_rate = load_audio_16k_mono(str(row["audio"]))
            window_samples = max(1, int(round(config.window_s * sample_rate)))
            windows: list[dict[str, Any]] = []
            for window_index, start_sample in enumerate(
                _workflow_window_starts(
                    sample_count=len(audio),
                    sample_rate=sample_rate,
                    window_s=config.window_s,
                    overlap_s=config.overlap_s,
                )
            ):
                end_sample = min(len(audio), start_sample + window_samples)
                chunk = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
                windows.append(
                    {
                        "window_index": window_index,
                        "start_sample": int(start_sample),
                        "audio": chunk,
                        "mfcc": extract_mfcc(chunk, sample_rate=sample_rate, config=config),
                    }
                )
            ptm_features, batch_count = _extract_ptm_window_features(
                ptm_extractor=extractor,
                window_audios=[window["audio"] for window in windows],
                sample_rate=sample_rate,
                ptm_window_batch_size=1,
            )
            bundle = _combine_workflow_window_features(
                windows=windows,
                ptm_features=ptm_features,
                duration_s=len(audio) / sample_rate,
                sample_rate=sample_rate,
                config=config,
            )
            feature_path = output_dir / "features" / f"{row['sample_id']}.npz"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                feature_path,
                ptm2048=np.asarray(bundle["ptm"], dtype=np.float32),
                mfcc=np.asarray(bundle["mfcc"], dtype=np.float32),
                frame_hop_s=np.asarray([config.frame_hop_s], dtype=np.float32),
            )
            manifest.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "audio": str(row["audio"]),
                    "feature_path": str(feature_path),
                    "frame_count": int(bundle["ptm"].shape[0]),
                    "ptm2048_dim": int(bundle["ptm"].shape[1]),
                    "mfcc_dim": int(bundle["mfcc"].shape[1]),
                    "frame_hop_s": config.frame_hop_s,
                    "ptm_batch_count": batch_count,
                }
            )
            print(
                f"anchor_features={position}/{len(rows)} sample_id={row['sample_id']} "
                f"frames={bundle['ptm'].shape[0]} ptm_dim={bundle['ptm'].shape[1]}",
                flush=True,
            )
    finally:
        extractor.close()
    return manifest


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    values = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = np.rint(values * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _materialize_audio(rows: list[dict[str, Any]], output_dir: Path) -> None:
    for row in rows:
        source, sample_rate = load_audio_16k_mono(row["audio"])
        event_dir = output_dir / "candidate_audio" / str(row["event_key"])
        region_start = int(round(float(row["region_start_s"]) * sample_rate))
        region_end = int(round(float(row["region_end_s"]) * sample_rate))
        region = np.asarray(source[region_start:region_end], dtype=np.float32)
        original = event_dir / "region.wav"
        _write_wav(original, region, sample_rate)
        row["region_audio"] = str(original)
        for candidate in row["candidates"]:
            marker = int(round(float(candidate["time_s"]) * sample_rate)) - region_start
            marker = max(0, min(marker, region.size))
            left_path = event_dir / f"{candidate['candidate_id']}__left.wav"
            right_path = event_dir / f"{candidate['candidate_id']}__right.wav"
            tick_path = event_dir / f"{candidate['candidate_id']}__tick.wav"
            _write_wav(left_path, region[:marker], sample_rate)
            _write_wav(right_path, region[marker:], sample_rate)

            # The tick locates the candidate without shifting the waveform.
            # Hard left/right previews remain the actual label evidence.
            tick_audio = np.array(region, copy=True)
            tick_samples = max(1, int(round(0.012 * sample_rate)))
            tick_start = max(0, min(marker - tick_samples // 2, tick_audio.size))
            tick_end = min(tick_audio.size, tick_start + tick_samples)
            if tick_end > tick_start:
                phase = np.arange(tick_end - tick_start, dtype=np.float32) / sample_rate
                envelope = np.hanning((tick_end - tick_start) * 2)[
                    : tick_end - tick_start
                ].astype(np.float32)
                tick = 0.18 * np.sin(2.0 * np.pi * 1800.0 * phase) * envelope
                tick_audio[tick_start:tick_end] = np.clip(
                    tick_audio[tick_start:tick_end] + tick, -1.0, 1.0
                )
            _write_wav(tick_path, tick_audio, sample_rate)
            candidate["left_audio"] = str(left_path)
            candidate["right_audio"] = str(right_path)
            candidate["tick_audio"] = str(tick_path)
            candidate["preview_contract"] = (
                "hard_left_end_plus_hard_right_start_with_noninserting_tick_v1"
            )


def _relative_copy(path: Path, *, audit_dir: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return destination.relative_to(audit_dir).as_posix()


def build_audit_html(
    *, rows: list[dict[str, Any]], audit_dir: Path, update_latest: bool = True
) -> Path:
    payload_rows: list[dict[str, Any]] = []
    for row in rows:
        event_key = str(row["event_key"])
        copied = dict(row)
        copied["full_audio"] = _relative_copy(
            Path(row["audio"]),
            audit_dir=audit_dir,
            destination=audit_dir / "audio" / event_key / f"full{Path(row['audio']).suffix}",
        )
        copied["region_audio"] = _relative_copy(
            Path(row["region_audio"]),
            audit_dir=audit_dir,
            destination=audit_dir / "audio" / event_key / "region.wav",
        )
        copied["candidates"] = []
        for candidate in row["candidates"]:
            copied_candidate = dict(candidate)
            for field, suffix in (
                ("left_audio", "left.wav"),
                ("right_audio", "right.wav"),
                ("tick_audio", "tick.wav"),
            ):
                copied_candidate[field] = _relative_copy(
                    Path(candidate[field]),
                    audit_dir=audit_dir,
                    destination=(
                        audit_dir
                        / "audio"
                        / event_key
                        / f"{candidate['candidate_id']}__{suffix}"
                    ),
                )
            copied["candidates"].append(copied_candidate)
        payload_rows.append(copied)

    payload = json.dumps(payload_rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Semantic Anchor Learned-Proposer Audit</title>
<style>body{{margin:0;background:#f3f5f7;color:#20242a;font-family:Segoe UI,Arial,sans-serif}}header{{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid #cbd2da;padding:12px 18px}}main{{max-width:1180px;margin:18px auto;padding:0 14px}}article,.help{{background:#fff;border:1px solid #cbd2da;border-radius:8px;padding:15px;margin:0 0 16px}}article.done{{border-left:6px solid #1b7a3a}}.help{{border-left:6px solid #1769aa}}.texts{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.text{{background:#f6f8fa;border:1px solid #d7dde3;border-radius:6px;padding:10px}}audio{{width:100%}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #d6dce3;padding:7px;vertical-align:top}}button,textarea{{font:inherit}}button{{padding:6px 9px;margin:2px}}button.active{{background:#1769aa;color:#fff}}button.safe.active{{background:#1b7a3a}}button.unsure.active{{background:#9a6700}}textarea{{width:100%;min-height:55px;box-sizing:border-box}}code{{background:#eef1f4;padding:2px 4px;border-radius:3px}}small{{color:#59616a}}@media(max-width:760px){{.texts{{grid-template-columns:1fr}}table{{font-size:12px}}}}</style></head><body>
<header><strong>Semantic Anchor → Learned Proposer 候选审计</strong>　<button id="save">保存全部裁决</button> <span id="status"></span></header><main><section class="help"><h2>这页只验证候选覆盖，不审 Omni 浮点秒数</h2><p>Omni 时间只作为粗锚点。候选来自 full PTM2048 经 repo-bound 任务学习型 <code>Linear(2048→128)</code> 投影后训练的 Proposer；不是 PCA、不是前 128 截断、不是能量最低点规则。</p><p>试听不再插入 1 秒静音：<b>左侧硬截断</b>精确结束在候选点，<b>右侧硬起播</b>精确从候选点开始；<b>定位 tick</b>只在原邻域叠加 12ms 短音，不改变时间轴，也不作为裁决依据。</p><p><b>切早：</b>左侧语义还没说完；<b>安全：</b>左侧已经完整结束且右侧尚未开始；<b>切晚：</b>右侧语义已经开始；<b>不确定：</b>重叠、连续语音或听不清。一个 event 至少出现一个人工 safe，才算候选覆盖。</p></section><div id="list"></div></main><script>
const rows={payload};const key='semantic-anchor-learned-proposer-audit-v2:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let active=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function ensure(r){{ann[r.event_key]??={{labels:{{}},note:''}};return ann[r.event_key];}}function persist(){{localStorage.setItem(key,JSON.stringify(ann));status();}}function complete(r,a){{return r.candidates.every(c=>a.labels[c.candidate_id]);}}function covered(r,a){{return Object.values(a.labels).includes('safe');}}function status(){{const done=rows.filter(r=>complete(r,ensure(r))).length,cover=rows.filter(r=>covered(r,ensure(r))).length;document.getElementById('status').textContent=`逐候选完成 ${{done}}/${{rows.length}} · safe覆盖 ${{cover}}/${{rows.length}}`;}}function play(el){{if(active&&active!==el)active.pause();active=el;}}
function choice(a,c,label,text,cls=''){{return `<button data-cid="${{c.candidate_id}}" data-label="${{label}}" class="${{cls}} ${{a.labels[c.candidate_id]===label?'active':''}}">${{text}}</button>`;}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r),card=document.createElement('article');if(complete(r,a))card.classList.add('done');const candidates=r.candidates.map(c=>`<tr><td><b>${{esc(c.candidate_id)}}</b><br>${{Number(c.time_s).toFixed(3)}}s<br><small>p=${{Number(c.proposer_probability).toFixed(4)}}</small></td><td><small>左侧硬截断</small><audio controls preload="metadata" src="${{esc(c.left_audio)}}"></audio><small>右侧硬起播</small><audio controls preload="metadata" src="${{esc(c.right_audio)}}"></audio><small>原邻域 + 12ms 定位 tick</small><audio controls preload="metadata" src="${{esc(c.tick_audio)}}"></audio></td><td>${{choice(a,c,'left_clipped','切早')}}${{choice(a,c,'safe','安全','safe')}}${{choice(a,c,'right_clipped','切晚')}}${{choice(a,c,'unsure','不确定','unsure')}}</td></tr>`).join('');card.innerHTML=`<h2>${{esc(r.event_key)}}</h2><div class="texts"><div class="text"><b>左侧应完整保留</b><br>${{esc(r.left_text)}}</div><div class="text"><b>右侧应完整保留</b><br>${{esc(r.right_text)}}</div></div><p><b>粗锚点：</b>${{Number(r.coarse_anchor_s).toFixed(3)}}s；<b>自适应邻域：</b>${{Number(r.region_start_s).toFixed(3)}}–${{Number(r.region_end_s).toFixed(3)}}s</p><small>projection SHA256=${{esc(r.projection_file_sha256)}}<br>proposer SHA256=${{esc(r.proposer_sha256)}}</small><p><b>完整 source</b></p><audio controls preload="metadata" src="${{esc(r.full_audio)}}"></audio><p><b>当前邻域原音频</b></p><audio controls preload="metadata" src="${{esc(r.region_audio)}}"></audio><table><thead><tr><th>候选</th><th>无时移边界试听</th><th>人工标签</th></tr></thead><tbody>${{candidates}}</tbody></table><label><b>备注</b><textarea>${{esc(a.note)}}</textarea></label>`;card.querySelectorAll('audio').forEach(x=>x.onplay=()=>play(x));card.querySelectorAll('[data-cid]').forEach(b=>b.onclick=()=>{{a.labels[b.dataset.cid]=b.dataset.label;a.updated_at=new Date().toISOString();persist();render();}});card.querySelector('textarea').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};root.appendChild(card);}}status();}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r),labels=r.candidates.map(c=>({{candidate_id:c.candidate_id,time_s:c.time_s,proposer_probability:c.proposer_probability,label:a.labels[c.candidate_id]||'unreviewed'}}));return JSON.stringify({{schema:'semantic_anchor_learned_proposer_manual_verdict_v2',event_key:r.event_key,sample_id:r.sample_id,event_id:r.event_id,coarse_anchor_s:r.coarse_anchor_s,region_start_s:r.region_start_s,region_end_s:r.region_end_s,candidates:labels,complete:complete(r,a),safe_covered:covered(r,a),note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();
</script></body></html>"""
    index = audit_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    if update_latest:
        update_audit_entrypoints(
            latest_html=index, title="Semantic Anchor Learned-Proposer Audit"
        )
    return index


def run(args: argparse.Namespace) -> dict[str, Any]:
    vram_safety_ratio = apply_vram_safety_cap()
    labels = _rows(Path(args.labels))
    selected_ids = list(args.sample_id or [])
    if selected_ids:
        by_id = {str(row["sample_id"]): row for row in labels}
        labels = [by_id[sample_id] for sample_id in selected_ids]
    if not labels:
        raise ValueError("no semantic timeline labels selected")
    requested_event_keys = set(args.event_key or [])
    if requested_event_keys:
        labels = [
            row
            for row in labels
            if any(
                f"{row['sample_id']}__{event['event_id']}" in requested_event_keys
                for event in row.get("semantic_events") or []
            )
        ]
        if not labels:
            raise ValueError("no requested semantic events found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    projection_path = Path(args.ptm_projection)
    projection_mean, projection_components, projection_digest = (
        load_task_aware_projection(projection_path)
    )
    if projection_mean.shape != (2048,) or projection_components.shape != (128, 2048):
        raise ValueError("audit requires learned Linear(2048->128) projection")

    feature_manifest = _extract_features(
        rows=labels,
        output_dir=output_dir,
        ptm_repo=args.ptm,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
    )
    _write_jsonl(output_dir / "feature_manifest.jsonl", feature_manifest)
    feature_by_id = {str(row["sample_id"]): row for row in feature_manifest}

    proposer_path = Path(args.proposer_checkpoint)
    proposer = load_boundary_proposal_checkpoint(proposer_path, device=args.device)
    event_rows: list[dict[str, Any]] = []
    for row in labels:
        feature_row = feature_by_id[str(row["sample_id"])]
        payload = np.load(feature_row["feature_path"], allow_pickle=False)
        projected = learned_frame_embeddings(
            ptm=np.asarray(payload["ptm2048"], dtype=np.float32),
            projection_mean=projection_mean,
            projection_components=projection_components,
        )
        mfcc = np.asarray(payload["mfcc"], dtype=np.float32)
        probabilities = score_speech_island_probabilities_batch(
            proposer, feature_pairs=[(projected, mfcc)]
        )[0]
        units = {item["unit_id"]: item for item in row["text_units"]}
        for event in adaptive_event_regions(row):
            event_key = f"{row['sample_id']}__{event['event_id']}"
            if requested_event_keys and event_key not in requested_event_keys:
                continue
            frames = select_stratified_proposer_frames(
                probabilities,
                region_start_s=float(event["region_start_s"]),
                region_end_s=float(event["region_end_s"]),
                frame_hop_s=0.02,
                candidate_count=int(args.candidate_count),
            )
            candidates = [
                {
                    "candidate_id": f"c{index:02d}",
                    "feature_index": int(frame),
                    "time_s": min(float(row["duration_s"]), (int(frame) + 0.5) * 0.02),
                    "proposer_probability": float(probabilities[int(frame)]),
                    "selection_contract": "per_adaptive_stratum_argmax_without_probability_threshold_v1",
                }
                for index, frame in enumerate(frames)
            ]
            event_rows.append(
                {
                    "schema": SCHEMA,
                    "event_key": event_key,
                    "sample_id": str(row["sample_id"]),
                    "event_id": str(event["event_id"]),
                    "audio": str(row["audio"]),
                    "duration_s": float(row["duration_s"]),
                    "reference_text": str(row["reference_text"]),
                    "left_unit_id": str(event["left_unit_id"]),
                    "right_unit_id": str(event["right_unit_id"]),
                    "left_text": str(units[event["left_unit_id"]]["text"]),
                    "right_text": str(units[event["right_unit_id"]]["text"]),
                    "coarse_anchor_s": float(event["coarse_anchor_s"]),
                    "region_start_s": float(event["region_start_s"]),
                    "region_end_s": float(event["region_end_s"]),
                    "region_contract": str(event["region_contract"]),
                    "ptm_contract": "full_ptm2048_then_task_aware_linear_projection_to_128",
                    "projection_path": str(projection_path),
                    "projection_digest": projection_digest,
                    "projection_file_sha256": _file_sha256(projection_path),
                    "proposer_checkpoint": str(proposer_path),
                    "proposer_sha256": checkpoint_sha256(proposer_path),
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                }
            )
    _materialize_audio(event_rows, output_dir)
    _write_jsonl(output_dir / "events.jsonl", event_rows)
    audit_path = build_audit_html(
        rows=event_rows,
        audit_dir=Path(args.audit_dir),
        update_latest=not args.no_update_latest,
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "sample_count": len(labels),
        "event_count": len(event_rows),
        "candidate_count": sum(len(row["candidates"]) for row in event_rows),
        "candidate_count_per_event": int(args.candidate_count),
        "ptm_repo_id": args.ptm,
        "ptm_input_dim": 2048,
        "ptm_projected_dim": 128,
        "ptm_projection_contract": "task_aware_linear_2048_to_128",
        "projection_path": str(projection_path),
        "projection_digest": projection_digest,
        "projection_file_sha256": _file_sha256(projection_path),
        "proposer_checkpoint": str(proposer_path),
        "proposer_sha256": checkpoint_sha256(proposer_path),
        "vram_safety_ratio": vram_safety_ratio,
        "shared_vram_budget": False,
        "audit": str(audit_path),
        "training_ready": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a semantic-anchor candidate audit with learned projected PTM128 proposer scores."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--proposer-checkpoint", required=True)
    parser.add_argument("--ptm-projection", default=str(DEFAULT_PROJECTION))
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--event-key", action="append", default=[])
    parser.add_argument("--candidate-count", type=int, default=9)
    parser.add_argument("--ptm", default=QWEN_ASR_17B_REPO_ID)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-update-latest", action="store_true")
    args = parser.parse_args()
    if not 5 <= args.candidate_count <= 9:
        parser.error("--candidate-count must be in 5..9")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
