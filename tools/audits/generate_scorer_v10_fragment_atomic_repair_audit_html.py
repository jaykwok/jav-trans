#!/usr/bin/env python3
"""Generate the minimal atomic-label audit after Scorer fragmentation review."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


FRAGMENT_VERDICT_SCHEMA = "speech_scorer_v10_fragmentation_gap_manual_verdict_v3"
ATOMIC_ITEM_SCHEMA = "speech_scorer_v10_fragment_atomic_repair_item_v1"
RELATION_SCHEMA = "speech_scorer_v10_fragment_atomic_relation_v1"
MANUAL_VERDICT_SCHEMA = "speech_scorer_v10_fragment_atomic_manual_verdict_v1"
SUMMARY_SCHEMA = "speech_scorer_v10_fragment_atomic_repair_audit_summary_v1"
ALLOWED_FRAGMENT_VERDICTS = {
    "same_asr_unit_keep_continuous",
    "separate_drop_nonsemantic",
    "separate_keep_both_speech",
    "cluster_not_speech_core",
    "unsure",
    "unreviewed",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_id(cluster_id: str, kind: str, span: dict[str, Any]) -> str:
    return (
        f"{cluster_id}:{kind}:"
        f"{int(span['start_frame'])}-{int(span['end_frame'])}"
    )


def infer_atomic_units(
    *,
    audit_rows: Iterable[dict[str, Any]],
    verdict_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Infer exact atomic labels without guessing separate-drop sides."""

    targets: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        audit_id = str(row.get("audit_id") or "")
        if not audit_id or audit_id in targets:
            raise ValueError("fragmentation manifest requires unique audit_id values")
        targets[audit_id] = row

    verdicts: dict[str, dict[str, Any]] = {}
    for row in verdict_rows:
        if row.get("schema") != FRAGMENT_VERDICT_SCHEMA:
            raise ValueError("invalid Scorer fragmentation verdict schema")
        audit_id = str(row.get("audit_id") or "")
        if audit_id not in targets or audit_id in verdicts:
            raise ValueError(f"invalid or duplicate fragmentation verdict: {audit_id}")
        verdict = str(row.get("verdict") or "unreviewed")
        if verdict not in ALLOWED_FRAGMENT_VERDICTS:
            raise ValueError(f"invalid fragmentation verdict: {verdict}")
        verdicts[audit_id] = row
    if set(verdicts) != set(targets):
        raise ValueError("fragmentation verdicts must cover the complete audit manifest")

    units: dict[str, dict[str, Any]] = {}
    labels: dict[str, set[str]] = defaultdict(set)
    reasons: dict[str, set[str]] = defaultdict(set)
    relations: list[dict[str, Any]] = []

    def register(
        manifest_row: dict[str, Any], kind: str, span: dict[str, Any]
    ) -> str:
        cluster_id = str(manifest_row["cluster_id"])
        atomic_id = _atomic_id(cluster_id, kind, span)
        candidate = {
            "schema": ATOMIC_ITEM_SCHEMA,
            "atomic_id": atomic_id,
            "cluster_id": cluster_id,
            "source_id": str(manifest_row["source_id"]),
            "partition": str(manifest_row["partition"]),
            "truth_run_index": int(manifest_row["truth_run_index"]),
            "audio": str(manifest_row["audio"]),
            "kind": kind,
            "start_frame": int(span["start_frame"]),
            "end_frame": int(span["end_frame"]),
            "start_s": float(span["start_s"]),
            "end_s": float(span["end_s"]),
        }
        existing = units.get(atomic_id)
        if existing is not None and existing != candidate:
            raise ValueError(f"atomic fragment identity is inconsistent: {atomic_id}")
        units[atomic_id] = candidate
        return atomic_id

    def constrain(atomic_id: str, label: str, reason: str) -> None:
        labels[atomic_id].add(label)
        reasons[atomic_id].add(reason)
        if len(labels[atomic_id]) > 1:
            raise ValueError(
                f"conflicting inferred atomic labels for {atomic_id}: "
                f"{sorted(labels[atomic_id])}"
            )

    for audit_id, manifest_row in targets.items():
        verdict = str(verdicts[audit_id].get("verdict") or "unreviewed")
        left_id = register(manifest_row, "model_speech", manifest_row["left_span"])
        gap_id = register(
            manifest_row, "model_background_gap", manifest_row["gap_span"]
        )
        right_id = register(manifest_row, "model_speech", manifest_row["right_span"])
        if verdict == "same_asr_unit_keep_continuous":
            for atomic_id in (left_id, gap_id, right_id):
                constrain(atomic_id, "speech", f"{audit_id}:{verdict}")
        elif verdict == "separate_keep_both_speech":
            constrain(left_id, "speech", f"{audit_id}:{verdict}:left")
            constrain(gap_id, "background", f"{audit_id}:{verdict}:gap")
            constrain(right_id, "speech", f"{audit_id}:{verdict}:right")
        elif verdict == "cluster_not_speech_core":
            # Saved v3 behavior shows this was used for the local left/gap/right
            # triplet, not necessarily every gap in the containing truth run.
            for atomic_id in (left_id, gap_id, right_id):
                constrain(atomic_id, "background", f"{audit_id}:local_triplet_noncore")
        elif verdict == "separate_drop_nonsemantic":
            constrain(gap_id, "background", f"{audit_id}:{verdict}:gap")
            relations.append(
                {
                    "schema": RELATION_SCHEMA,
                    "relation_id": f"{audit_id}:at_least_one_background_side",
                    "audit_id": audit_id,
                    "cluster_id": str(manifest_row["cluster_id"]),
                    "source_id": str(manifest_row["source_id"]),
                    "left_atomic_id": left_id,
                    "right_atomic_id": right_id,
                    "constraint": "at_least_one_side_background",
                }
            )

    changed = True
    while changed:
        changed = False
        for relation in relations:
            left_id = str(relation["left_atomic_id"])
            right_id = str(relation["right_atomic_id"])
            left_label = next(iter(labels[left_id])) if len(labels[left_id]) == 1 else ""
            right_label = (
                next(iter(labels[right_id])) if len(labels[right_id]) == 1 else ""
            )
            if left_label == "speech" and not right_label:
                constrain(
                    right_id,
                    "background",
                    f"{relation['relation_id']}:left_is_speech",
                )
                changed = True
            elif right_label == "speech" and not left_label:
                constrain(
                    left_id,
                    "background",
                    f"{relation['relation_id']}:right_is_speech",
                )
                changed = True
            elif left_label == "speech" and right_label == "speech":
                raise ValueError(
                    "separate-drop relation has two inferred speech sides: "
                    f"{relation['relation_id']}"
                )

    atomic_rows: list[dict[str, Any]] = []
    for atomic_id, unit in units.items():
        inferred = next(iter(labels[atomic_id])) if len(labels[atomic_id]) == 1 else ""
        atomic_rows.append(
            {
                **unit,
                "inferred_label": inferred,
                "inference_reasons": sorted(reasons[atomic_id]),
                "review_required": not bool(inferred),
            }
        )
    partition_order = {"val": 0, "test": 1, "train": 2}
    atomic_rows.sort(
        key=lambda row: (
            partition_order.get(str(row["partition"]), 3),
            str(row["source_id"]),
            int(row["truth_run_index"]),
            int(row["start_frame"]),
            str(row["kind"]),
        )
    )
    relations.sort(key=lambda row: str(row["relation_id"]))
    return atomic_rows, relations


def bind_atomic_units_to_canonical(
    *, atomic_rows: Iterable[dict[str, Any]], canonical_rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Bind every frame atom to exactly one corrected canonical speech span."""

    sources: dict[str, dict[str, Any]] = {}
    for row in canonical_rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in sources:
            raise ValueError("canonical binding requires unique source_id values")
        sources[source_id] = row

    bound: list[dict[str, Any]] = []
    for row in atomic_rows:
        source_id = str(row["source_id"])
        source = sources.get(source_id)
        if source is None:
            raise ValueError(f"atomic repair source is absent from canonical: {source_id}")
        sample_rate = int(source["sample_rate"])
        sample_count = int(source["sample_count"])
        if sample_rate % 50:
            raise ValueError("20ms atomic binding requires sample rate divisible by 50")
        samples_per_frame = sample_rate // 50
        start_sample = int(row["start_frame"]) * samples_per_frame
        end_sample = min(
            sample_count, int(row["end_frame"]) * samples_per_frame
        )
        matches = [
            (span_index, span)
            for span_index, span in enumerate(source.get("canonical_spans") or ())
            if str(span.get("label") or "") == "speech"
            and start_sample >= int(span["start_sample"])
            and end_sample <= int(span["end_sample"])
        ]
        if len(matches) != 1:
            raise ValueError(
                "atomic unit must map to exactly one canonical speech span: "
                f"{row['atomic_id']} matches={len(matches)}"
            )
        span_index, span = matches[0]
        bound.append(
            {
                **row,
                "sample_rate": sample_rate,
                "sample_count": sample_count,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "canonical_span_index": span_index,
                "core_id": str(span.get("core_id") or ""),
                "original_canonical_label": "speech",
            }
        )
    return bound


def _render_page(
    *, clusters: list[dict[str, Any]], review_count: int, auto_count: int
) -> str:
    encoded = (
        json.dumps(clusters, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return (
        """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scorer v10 fragment atomic repair</title>
<style>
:root{--border:#c9d0d8;--speech:#58aa70;--background:#efc75e;--unknown:#417fc2;--ok:#267443;--risk:#a52f2f}
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#20242a;font-family:Segoe UI,Arial,sans-serif}
header{position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:#fff;border-bottom:1px solid var(--border);padding:10px 18px}header strong{margin-right:auto}
main{max-width:1080px;margin:18px auto;padding:0 14px}section,article{background:#fff;border:1px solid var(--border);border-radius:6px;padding:14px;margin-bottom:14px}article.done{border-left:6px solid var(--ok)}
audio{display:none}small{color:#5c6570}button{font:inherit;border:1px solid #69737e;border-radius:5px;background:#fff;padding:7px 10px;cursor:pointer}
.workflow{padding:9px;border-left:4px solid var(--unknown);background:#eef4fb;margin:10px 0}.full{width:100%;margin:8px 0 12px;background:#eef4fb;border-color:var(--unknown)}
.unit{display:grid;grid-template-columns:minmax(260px,1fr) minmax(330px,auto);gap:10px;align-items:center;padding:9px 0;border-top:1px solid #e1e5e9}.unit.done{background:#f4faf6}.play{width:100%;min-height:54px}.play.speech{background:var(--speech)}.play.background{background:var(--background)}.play.unresolved{background:var(--unknown);color:#fff}.play.playing{outline:3px solid #111;outline-offset:-3px}
.badge{display:inline-block;padding:6px 9px;border-radius:5px;background:#e8ecef}.choices{display:flex;gap:6px;flex-wrap:wrap}.choices button.active{background:#1769aa;color:#fff}.choices button.risk.active{background:var(--risk)}
@media(max-width:760px){.unit{grid-template-columns:1fr}header strong{width:100%}}
</style></head><body>
<header><strong>1.7B Scorer v10 · fragment atomic repair</strong><span id="status"></span><button id="stop" type="button">停止播放</button><button id="save" type="button">保存裁决</button></header>
<main><section>61 条拓扑裁决已自动确定 __AUTO_COUNT__ 个原子段；这里只补审无法判断左右归属的 __REVIEW_COUNT__ 个 model_speech island。每个蓝条只播放自身精确区间；完整 island 串播放仅用于听上下文，不代表运行时合并。请选择该蓝条本身是 speech core、非语义 background 或 unsure。</section><div id="list"></div></main>
<script>
const clusters=__ROWS__;const key='scorer-v10-fragment-atomic-repair-v1:'+location.pathname;let ann={};try{ann=JSON.parse(localStorage.getItem(key)||'{}');}catch(_error){ann={};}
let activeAudio=null,activeButton=null,activeTimer=null,activeFrame=null,activeCheck=null,activeLoad=null;
function esc(value){return String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));}
function ensure(unit){ann[unit.atomic_id]??={verdict:''};return ann[unit.atomic_id];}
function stopPlayback(){if(activeAudio&&activeCheck){activeAudio.removeEventListener('timeupdate',activeCheck);activeAudio.removeEventListener('ended',activeCheck);}if(activeAudio&&activeLoad)activeAudio.removeEventListener('loadedmetadata',activeLoad);if(activeTimer!==null)clearTimeout(activeTimer);if(activeFrame!==null)cancelAnimationFrame(activeFrame);if(activeAudio)activeAudio.pause();if(activeButton)activeButton.classList.remove('playing');activeAudio=null;activeButton=null;activeTimer=null;activeFrame=null;activeCheck=null;activeLoad=null;}
function playExact(audio,button,start,end){if(activeAudio===audio&&activeButton===button&&!audio.paused){stopPlayback();return;}stopPlayback();activeAudio=audio;activeButton=button;button.classList.add('playing');const begin=async()=>{activeLoad=null;if(activeAudio!==audio||activeButton!==button)return;audio.currentTime=start;activeCheck=()=>{if(audio.ended||audio.currentTime>=end)stopPlayback();};audio.addEventListener('timeupdate',activeCheck);audio.addEventListener('ended',activeCheck);const watch=()=>{if(activeAudio!==audio)return;if(audio.currentTime>=end){stopPlayback();return;}activeFrame=requestAnimationFrame(watch);};try{await audio.play();if(activeAudio!==audio){audio.pause();return;}activeFrame=requestAnimationFrame(watch);activeTimer=setTimeout(stopPlayback,Math.max(100,(end-start)*1000+120));}catch(error){stopPlayback();document.getElementById('status').textContent='播放失败: '+error.message;}};if(audio.readyState<1){activeLoad=begin;audio.addEventListener('loadedmetadata',begin,{once:true});audio.load();}else begin();}
function playButton(unit,label){const duration=Math.round((unit.end_s-unit.start_s)*1000);const kind=unit.review_required?'unresolved':unit.inferred_label;return '<button type="button" class="play '+kind+'" data-start="'+unit.start_s+'" data-end="'+unit.end_s+'">'+esc(label)+' · '+duration+'ms<br>'+Number(unit.start_s).toFixed(2)+'–'+Number(unit.end_s).toFixed(2)+'s</button>';}
function choices(unit){const verdict=ensure(unit).verdict;return '<div class="choices"><button type="button" data-v="speech" class="'+(verdict==='speech'?'active':'')+'">speech core</button><button type="button" data-v="background" class="'+(verdict==='background'?'active':'')+'">非语义 / background</button><button type="button" data-v="unsure" class="risk '+(verdict==='unsure'?'active':'')+'">不确定</button></div>';}
function updateStatus(){const units=clusters.flatMap(cluster=>cluster.units.filter(unit=>unit.review_required));document.getElementById('status').textContent='已裁决 '+units.filter(unit=>ensure(unit).verdict).length+'/'+units.length;}
function persist(){localStorage.setItem(key,JSON.stringify(ann));updateStatus();}
function sync(line,unit){const verdict=ensure(unit).verdict;line.classList.toggle('done',Boolean(verdict));line.querySelectorAll('[data-v]').forEach(button=>button.classList.toggle('active',button.dataset.v===verdict));}
function render(){stopPlayback();const root=document.getElementById('list');root.innerHTML='';for(const cluster of clusters){const card=document.createElement('article');card.innerHTML='<h2>'+esc(cluster.source_id)+'</h2><small>'+esc(cluster.partition)+' / truth run '+cluster.truth_run_index+' / unresolved '+cluster.review_count+' / '+Number(cluster.start_s).toFixed(2)+'–'+Number(cluster.end_s).toFixed(2)+'s</small><div class="workflow">完整播放包含内部 gap，仅供判断当前蓝条是否属于 speech core；实际 Scorer island 仍保持独立。</div><button type="button" class="full" data-start="'+cluster.start_s+'" data-end="'+cluster.end_s+'">完整 island 串 · 审计上下文</button><audio preload="none" src="'+esc(cluster.audio)+'"></audio><div class="units"></div>';const audio=card.querySelector('audio');const full=card.querySelector('.full');full.onclick=()=>playExact(audio,full,Number(full.dataset.start),Number(full.dataset.end));const container=card.querySelector('.units');for(const unit of cluster.units){const line=document.createElement('div');line.className='unit';const original=unit.kind==='model_speech'?'model_speech':'model_background gap';line.innerHTML=playButton(unit,original)+(unit.review_required?choices(unit):'<span class="badge">自动确定：'+esc(unit.inferred_label)+'</span>');const play=line.querySelector('[data-start]');play.onclick=()=>playExact(audio,play,Number(play.dataset.start),Number(play.dataset.end));if(unit.review_required){line.querySelectorAll('[data-v]').forEach(button=>button.onclick=()=>{const a=ensure(unit);a.verdict=button.dataset.v;a.updated_at=new Date().toISOString();sync(line,unit);card.classList.toggle('done',cluster.units.filter(item=>item.review_required).every(item=>ensure(item).verdict));persist();});sync(line,unit);}container.appendChild(line);}card.classList.toggle('done',cluster.units.filter(item=>item.review_required).every(item=>ensure(item).verdict));root.appendChild(card);}updateStatus();}
document.getElementById('stop').onclick=stopPlayback;document.getElementById('save').onclick=async()=>{const units=clusters.flatMap(cluster=>cluster.units.filter(unit=>unit.review_required));const content=units.map(unit=>{const a=ensure(unit);return JSON.stringify({schema:'__VERDICT_SCHEMA__',atomic_id:unit.atomic_id,cluster_id:unit.cluster_id,source_id:unit.source_id,partition:unit.partition,start_frame:unit.start_frame,end_frame:unit.end_frame,start_s:unit.start_s,end_s:unit.end_s,verdict:a.verdict||'unreviewed',updated_at:a.updated_at||new Date().toISOString()});}).join('\\n')+'\\n';try{const response=await fetch('/__audit_api__/save-labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({href:location.pathname,filename:'manual_verdicts.jsonl',content})});const output=await response.json();document.getElementById('status').textContent=response.ok&&output.ok?'已保存到 '+output.path:'保存失败: '+(output.error||response.status);}catch(error){document.getElementById('status').textContent='保存失败: '+error.message;}};render();
</script></body></html>
"""
        .replace("__ROWS__", encoded)
        .replace("__AUTO_COUNT__", str(auto_count))
        .replace("__REVIEW_COUNT__", str(review_count))
        .replace("__VERDICT_SCHEMA__", MANUAL_VERDICT_SCHEMA)
    )


def build_audit(
    *,
    canonical_sources: Path,
    fragmentation_audit_manifest: Path,
    fragmentation_manual_verdicts: Path,
    output_dir: Path,
) -> Path:
    inferred_rows, relations = infer_atomic_units(
        audit_rows=_rows(fragmentation_audit_manifest),
        verdict_rows=_rows(fragmentation_manual_verdicts),
    )
    atomic_rows = bind_atomic_units_to_canonical(
        atomic_rows=inferred_rows,
        canonical_rows=_rows(canonical_sources),
    )
    review_rows = [row for row in atomic_rows if row["review_required"]]
    if not review_rows:
        raise ValueError("fragment atomic repair has no unresolved units")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in atomic_rows:
        grouped[str(row["cluster_id"])].append(row)
    review_clusters = {str(row["cluster_id"]) for row in review_rows}
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[Path, str] = {}
    payload: list[dict[str, Any]] = []
    for cluster_id in sorted(review_clusters):
        units = sorted(grouped[cluster_id], key=lambda row: int(row["start_frame"]))
        raw_audio = Path(str(units[0]["audio"]))
        source_audio = (
            raw_audio
            if raw_audio.is_absolute()
            else fragmentation_audit_manifest.parent / raw_audio
        ).resolve()
        if not source_audio.is_file():
            raise ValueError(f"fragment atomic audio is missing: {source_audio}")
        if source_audio not in copied:
            target = audio_dir / f"item-{len(copied):03d}{source_audio.suffix.lower()}"
            shutil.copy2(source_audio, target)
            copied[source_audio] = target.relative_to(output_dir).as_posix()
        payload.append(
            {
                "cluster_id": cluster_id,
                "source_id": str(units[0]["source_id"]),
                "partition": str(units[0]["partition"]),
                "truth_run_index": int(units[0]["truth_run_index"]),
                "audio": copied[source_audio],
                "start_s": min(float(row["start_s"]) for row in units),
                "end_s": max(float(row["end_s"]) for row in units),
                "review_count": sum(bool(row["review_required"]) for row in units),
                "units": units,
            }
        )
    partition_order = {"val": 0, "test": 1, "train": 2}
    payload.sort(
        key=lambda row: (
            partition_order.get(str(row["partition"]), 3),
            -int(row["review_count"]),
            str(row["source_id"]),
            int(row["truth_run_index"]),
        )
    )

    atomic_manifest = output_dir / "atomic_manifest.jsonl"
    atomic_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in atomic_rows),
        encoding="utf-8",
    )
    relation_manifest = output_dir / "relation_manifest.jsonl"
    relation_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in relations),
        encoding="utf-8",
    )
    index = output_dir / "index.html"
    index.write_text(
        _render_page(
            clusters=payload,
            review_count=len(review_rows),
            auto_count=len(atomic_rows) - len(review_rows),
        ),
        encoding="utf-8",
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "title": "Scorer v10 fragment atomic repair",
        "canonical_sources": str(canonical_sources),
        "canonical_sources_sha256": hashlib.sha256(
            canonical_sources.read_bytes()
        ).hexdigest(),
        "fragmentation_audit_manifest": str(fragmentation_audit_manifest),
        "fragmentation_audit_manifest_sha256": hashlib.sha256(
            fragmentation_audit_manifest.read_bytes()
        ).hexdigest(),
        "fragmentation_manual_verdicts": str(fragmentation_manual_verdicts),
        "fragmentation_manual_verdicts_sha256": hashlib.sha256(
            fragmentation_manual_verdicts.read_bytes()
        ).hexdigest(),
        "atomic_unit_count": len(atomic_rows),
        "auto_resolved_count": len(atomic_rows) - len(review_rows),
        "review_item_count": len(review_rows),
        "review_cluster_count": len(payload),
        "relation_count": len(relations),
        "inference_conflict_count": 0,
        "cluster_not_speech_core_semantics": "local_left_gap_right_triplet_all_nonsemantic",
        "atomic_manifest": str(atomic_manifest),
        "relation_manifest": str(relation_manifest),
        "playback_context_s": 0.0,
        "full_island_cluster_audit_playback": True,
        "runtime_merge_effect": "none_audit_only",
        "manual_gate_status": "pending",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_audit_entrypoints(latest_html=index, title=summary["title"])
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-sources", required=True)
    parser.add_argument("--fragmentation-audit-manifest", required=True)
    parser.add_argument("--fragmentation-manual-verdicts", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        build_audit(
            canonical_sources=Path(args.canonical_sources),
            fragmentation_audit_manifest=Path(args.fragmentation_audit_manifest),
            fragmentation_manual_verdicts=Path(args.fragmentation_manual_verdicts),
            output_dir=Path(args.output_dir),
        )
    )
