#!/usr/bin/env python3
"""Generate a source-level Scorer v12 Teacher review page."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.ja.vocal_envelope_v12 import (  # noqa: E402
    VOCAL_ENVELOPE_SCORER_V12_AUDIT_ITEM_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_ENVELOPE_STRUCTURE_OPTIONS,
    VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_NONVOCAL_SAFETY_OPTIONS,
    VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_VOCAL_COVERAGE_OPTIONS,
)
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.audits.review_page_core import (  # noqa: E402
    AuditOptionAxis,
    AuditReviewPageSpec,
    render_audit_review_page,
    validate_audit_option_contract,
)
from tools.boundary.ja.vocal_envelope_scorer_v12_calibration import (  # noqa: E402
    CALIBRATION_ARTIFACT_SHA256,
    evidence_span_signature,
    load_approved_calibration,
)


CONTRACT_ID = "boundary_acoustic_binary_v12"
SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_teacher_audit_summary_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index(
    rows: Sequence[Mapping[str, Any]], *, name: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in result:
            raise ValueError(f"{name} requires unique non-empty source_id")
        result[source_id] = dict(row)
    return result


def _script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _validate_option_contract() -> None:
    axes = (
        AuditOptionAxis(
            field="vocal_coverage",
            options=VOCAL_ENVELOPE_SCORER_V12_VOCAL_COVERAGE_OPTIONS,
        ),
        AuditOptionAxis(
            field="non_vocal_safety",
            options=VOCAL_ENVELOPE_SCORER_V12_NONVOCAL_SAFETY_OPTIONS,
        ),
        AuditOptionAxis(
            field="envelope_structure",
            options=VOCAL_ENVELOPE_SCORER_V12_ENVELOPE_STRUCTURE_OPTIONS,
        ),
    )
    results: dict[tuple[str, ...], str] = {}
    for vocal in axes[0].options:
        for nonvocal in axes[1].options:
            for structure in axes[2].options:
                combination = (vocal, nonvocal, structure)
                if "unsure" in combination:
                    result = "unsure"
                elif combination == (
                    "definite_vocal_complete",
                    "definite_non_vocal_clean",
                    "event_envelopes_continuous",
                ):
                    result = "approved"
                else:
                    result = "rejected"
                results[combination] = result
    validate_audit_option_contract(axes=axes, combination_results=results)


def _page(payload: list[dict[str, Any]], *, manifest_sha: str, preaudit_sha: str) -> str:
    _validate_option_contract()
    intro = """
<section class="audit-help">
  <h2>审计合同</h2>
  <p><b>绿色 vocal：</b>所有人类声道、口腔或呼吸系统产生的连续发声事件包络；同一事件中的短停顿和呼吸应连续保留。</p>
  <p><b>黄色 non-vocal：</b>明确不含任何人类发声的机械、动作、衣物/床体、水声、纯音乐、静音或环境噪声。</p>
  <p><b>灰色 unsure：</b>单次三态 Teacher 无法可靠判断人声重叠或安全边界；训练时为 -100，不代表 vocal 或 non-vocal 真值。</p>
  <p>每条必须完整听完，再分别判断：vocal 是否漏声/截边、黄色是否混入任何人声、同一发声事件是否被切碎或跨独立长背景过度合并。颜色条点击后只播放自身精确区间，不添加上下文。</p>
</section>
<div id="cards"></div>
"""
    body = ""
    css = """
.audit-help,.audit-card{background:#fff;border:1px solid #d7dde4;border-radius:10px;padding:14px;margin:0 0 14px}
.audit-card h3{margin:0 0 4px}.audit-meta{margin-bottom:8px}.audit-full-row{display:flex;gap:8px;align-items:center;margin:8px 0}.audit-full-row audio{margin:0;flex:1}
.vocal{background:#2db66f;color:#062d19}.nonvocal{background:#f0c84b;color:#3b2d00}.unsure{background:#9aa6b2;color:#15202a}.conflict{background:#df6c68;color:#3b0807}
.audit-verdict{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:10px;margin-top:12px}.audit-verdict label{display:flex;flex-direction:column;gap:4px}.audit-verdict select,.audit-notes{padding:7px}.audit-reviewed{display:flex!important;flex-direction:row!important;align-items:center;gap:7px!important;margin-top:10px}.audit-notes{width:100%;box-sizing:border-box;margin-top:8px}
@media(max-width:1000px){.audit-verdict{grid-template-columns:1fr}}
"""
    js = f"""
const entries={_script_json(payload)};
const sourceManifestSha={_script_json(manifest_sha)};
const preauditSha={_script_json(preaudit_sha)};
const labels={{
  vocal_coverage:[
    ['','请选择'],['definite_vocal_complete','vocal 完整，无漏声/截边'],['definite_vocal_missing_or_clipped','存在漏声、截边或真发声落在黄色'],['unsure','不确定']
  ],
  non_vocal_safety:[
    ['','请选择'],['definite_non_vocal_clean','黄色均不含人类发声'],['definite_non_vocal_contains_vocal','黄色含对白/呼吸/呻吟等人类发声'],['unsure','不确定']
  ],
  envelope_structure:[
    ['','请选择'],['event_envelopes_continuous','同一发声事件包络连续且未跨独立长背景'],['same_event_fragmented','同一发声事件被切碎'],['overmerged_independent_nonvocal','跨越独立长 non-vocal 过度合并'],['both_fragmented_and_overmerged','同时存在切碎和过度合并'],['unsure','不确定']
  ]
}};
function approved(state){{return Boolean(state.reviewed_full_source)&&state.vocal_coverage==='definite_vocal_complete'&&state.non_vocal_safety==='definite_non_vocal_clean'&&state.envelope_structure==='event_envelopes_continuous';}}
const core=createAuditReviewCore({{
  entries,
  storageKey:'vocal-envelope-scorer-v12-teacher-audit-v1',
  statusLabel:'已完成',
  entryId:entry=>entry.source_id,
  defaultState:()=>({{vocal_coverage:'',non_vocal_safety:'',envelope_structure:'',reviewed_full_source:false,notes:'',updated_at:''}}),
  isComplete:state=>Boolean(state.reviewed_full_source&&state.vocal_coverage&&state.non_vocal_safety&&state.envelope_structure),
  shouldSerialize:state=>Boolean(state.reviewed_full_source&&state.vocal_coverage&&state.non_vocal_safety&&state.envelope_structure),
  serialize:(entry,state)=>({{
    schema:{_script_json(VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA)},
    boundary_serialization_contract_id:{_script_json(CONTRACT_ID)},
    source_id:entry.source_id,video_id:entry.video_id,partition:entry.partition,
    audio_sha256:entry.audio_sha256,duration_s:entry.duration_s,frame_count:entry.frame_count,
    source_manifest_sha256:sourceManifestSha,preaudit_sha256:preauditSha,
    reviewed_full_source:Boolean(state.reviewed_full_source),vocal_coverage:state.vocal_coverage,
    non_vocal_safety:state.non_vocal_safety,envelope_structure:state.envelope_structure,
    approved:approved(state),notes:String(state.notes||''),updated_at:state.updated_at||new Date().toISOString(),
    training_manifest_allowed:approved(state)
  }})
}});
function optionHtml(values,current){{return values.map(([value,text])=>`<option value="${{escapeAuditHtml(value)}}" ${{value===current?'selected':''}}>${{escapeAuditHtml(text)}}</option>`).join('');}}
function render(){{
  const root=document.getElementById('cards');root.innerHTML='';
  for(const entry of entries){{
    const state=core.ensure(entry),card=document.createElement('section');card.className='audit-card';
    card.innerHTML=`<h3>${{escapeAuditHtml(entry.source_id)}}</h3><div class="audit-meta"><small>${{escapeAuditHtml(entry.partition)}} / ${{escapeAuditHtml(entry.video_id)}} / ${{entry.frame_count}} frames / ${{Number(entry.duration_s).toFixed(3)}}s</small></div><div class="audit-full-row"><button type="button" class="full-play">播放完整 source</button><audio controls preload="metadata" src="${{escapeAuditHtml(entry.audio)}}"></audio></div><div class="lanes"></div><div class="audit-verdict"><label>1. Vocal 覆盖<select data-field="vocal_coverage">${{optionHtml(labels.vocal_coverage,state.vocal_coverage)}}</select></label><label>2. Non-vocal 安全<select data-field="non_vocal_safety">${{optionHtml(labels.non_vocal_safety,state.non_vocal_safety)}}</select></label><label>3. 包络结构<select data-field="envelope_structure">${{optionHtml(labels.envelope_structure,state.envelope_structure)}}</select></label></div><label class="audit-reviewed"><input type="checkbox" data-field="reviewed_full_source" ${{state.reviewed_full_source?'checked':''}}>已完整听完本条 source</label><input class="audit-notes" data-field="notes" placeholder="可选备注" value="${{escapeAuditHtml(state.notes||'')}}"><small class="approval">当前：${{approved(state)?'可进入 canonical':'不可进入 canonical'}}</small>`;
    const audio=card.querySelector('audio'),lanes=card.querySelector('.lanes');
    card.querySelector('.full-play').onclick=event=>play(audio,event.currentTarget,0,Number(entry.duration_s));
    appendAuditSpanLane({{container:lanes,audio,durationS:Number(entry.duration_s),label:'canonical vocal',metric:`${{entry.vocal_spans.length}} spans`,spans:entry.vocal_spans,className:'vocal',title:(span,start,end)=>`vocal ${{formatAuditSpan(start,end)}} · ${{span.reason||''}}`}});
    appendAuditSpanLane({{container:lanes,audio,durationS:Number(entry.duration_s),label:'canonical non-vocal',metric:`${{entry.non_vocal_spans.length}} spans`,spans:entry.non_vocal_spans,className:'nonvocal',title:(span,start,end)=>`non-vocal ${{formatAuditSpan(start,end)}} · ${{span.category||''}} · ${{span.reason||''}}`}});
    appendAuditSpanLane({{container:lanes,audio,durationS:Number(entry.duration_s),label:'canonical unsure',metric:`${{entry.unsure_spans.length}} spans`,spans:entry.unsure_spans,className:span=>span.conflict?'conflict':'unsure',title:(span,start,end)=>`${{span.conflict?'conflict':'unsure'}} ${{formatAuditSpan(start,end)}}`}});
    for(const element of card.querySelectorAll('[data-field]')){{
      const field=element.dataset.field;
      element.onchange=()=>{{state[field]=element.type==='checkbox'?element.checked:element.value;state.updated_at=new Date().toISOString();core.persist();render();}};
    }}
    root.appendChild(card);
  }}
  core.updateStatus();
}}
document.getElementById('stop').onclick=stop;document.getElementById('save').onclick=()=>core.save();render();
"""
    return render_audit_review_page(
        AuditReviewPageSpec(
            title="Scorer v12 vocal-envelope Teacher review",
            intro_html=intro,
            body_html=body,
            adapter_css=css,
            adapter_js=js,
        )
    )


def build(
    *,
    source_manifest: Path,
    preaudit: Path,
    output_dir: Path,
    partitions: Sequence[str] = (),
    calibration_manifest: Path | None = None,
    calibration_preaudit: Path | None = None,
    calibration_verdicts: Path | None = None,
) -> dict[str, Any]:
    source_manifest = source_manifest.resolve()
    preaudit = preaudit.resolve()
    sources = _index(_rows(source_manifest), name="v12 source manifest")
    evidence = _index(_rows(preaudit), name="v12 preaudit")
    if set(sources) != set(evidence):
        raise ValueError("v12 audit requires exact source/preaudit identity coverage")
    selected_partitions = set(partitions)
    if selected_partitions - {"train", "val", "test"}:
        raise ValueError(f"invalid v12 audit partitions: {sorted(selected_partitions)}")
    calibration_paths = (
        calibration_manifest,
        calibration_preaudit,
        calibration_verdicts,
    )
    if any(path is not None for path in calibration_paths) and not all(
        path is not None for path in calibration_paths
    ):
        raise ValueError(
            "v12 audit calibration exclusion requires all three calibration files"
        )
    calibration: dict[str, Any] | None = None
    if all(path is not None for path in calibration_paths):
        assert calibration_manifest is not None
        assert calibration_preaudit is not None
        assert calibration_verdicts is not None
        calibration = load_approved_calibration(
            manifest=calibration_manifest,
            preaudit=calibration_preaudit,
            verdicts=calibration_verdicts,
            expected_hashes=CALIBRATION_ARTIFACT_SHA256,
        )
        if not set(calibration["sources"]).issubset(sources):
            raise ValueError("v12 audit manifest omits calibrated pilot sources")
    manifest_sha = _sha256(source_manifest)
    preaudit_sha = _sha256(preaudit)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    skipped_calibration_ids: list[str] = []
    for index, source_id in enumerate(sorted(sources)):
        source = sources[source_id]
        row = evidence[source_id]
        if row.get("schema") != VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA:
            raise ValueError(f"wrong v12 preaudit schema: {source_id}")
        if row.get("boundary_serialization_contract_id") != CONTRACT_ID:
            raise ValueError(f"wrong v12 central contract: {source_id}")
        for field in ("partition", "video_id", "audio_sha256", "frame_count"):
            if row.get(field) != source.get(field):
                raise ValueError(f"v12 audit {field} mismatch: {source_id}")
        if row.get("source_manifest_sha256") != manifest_sha:
            raise ValueError(f"v12 audit source manifest binding mismatch: {source_id}")
        partition = str(source.get("partition") or "")
        if selected_partitions and partition not in selected_partitions:
            continue
        if calibration is not None and source_id in calibration["sources"]:
            calibration_source = calibration["sources"][source_id]
            for field in (
                "video_id",
                "partition",
                "audio_sha256",
                "duration_s",
                "frame_count",
                "sample_rate",
                "sample_count",
            ):
                if source.get(field) != calibration_source.get(field):
                    raise ValueError(
                        f"v12 audit calibrated source {field} drift: {source_id}"
                    )
            if evidence_span_signature(
                row,
                frame_count=int(source["frame_count"]),
                source_id=source_id,
            ) != calibration["signatures"][source_id]:
                raise ValueError(
                    f"v12 audit calibrated evidence changed after approval: {source_id}"
                )
            skipped_calibration_ids.append(source_id)
            continue
        audio = Path(str(source.get("audio") or ""))
        if not audio.is_absolute():
            audio = (source_manifest.parent / audio).resolve()
        if not audio.is_file() or _sha256(audio) != str(source.get("audio_sha256") or ""):
            raise ValueError(f"v12 audit audio SHA mismatch: {source_id}")
        target = audio_dir / f"source-{index:03d}.wav"
        shutil.copy2(audio, target)
        conflicts = {
            (int(span["start_frame"]), int(span["end_frame"]))
            for span in row.get("conflict_spans") or ()
        }
        unsure = []
        for span in row.get("unsure_spans") or ():
            copied = dict(span)
            copied["conflict"] = (
                int(span["start_frame"]), int(span["end_frame"])
            ) in conflicts
            unsure.append(copied)
        payload.append(
            {
                "schema": VOCAL_ENVELOPE_SCORER_V12_AUDIT_ITEM_SCHEMA,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "source_id": source_id,
                "video_id": str(source["video_id"]),
                "partition": partition,
                "audio": target.relative_to(output_dir).as_posix(),
                "audio_sha256": str(source["audio_sha256"]),
                "duration_s": float(source["duration_s"]),
                "frame_count": int(source["frame_count"]),
                "vocal_spans": list(row.get("vocal_spans") or ()),
                "non_vocal_spans": list(row.get("non_vocal_spans") or ()),
                "unsure_spans": unsure,
            }
        )
    manifest = output_dir / "audit_manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in payload
        ),
        encoding="utf-8",
    )
    index = output_dir / "index.html"
    index.write_text(
        _page(payload, manifest_sha=manifest_sha, preaudit_sha=preaudit_sha),
        encoding="utf-8",
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": manifest_sha,
        "preaudit": str(preaudit),
        "preaudit_sha256": preaudit_sha,
        "source_count": len(payload),
        "selected_partitions": sorted(selected_partitions),
        "skipped_calibration_source_count": len(skipped_calibration_ids),
        "skipped_calibration_source_ids": skipped_calibration_ids,
        "calibration_id": calibration["calibration_id"] if calibration else None,
        "calibration_manifest_sha256": (
            calibration["hashes"]["manifest"] if calibration else None
        ),
        "calibration_preaudit_sha256": (
            calibration["hashes"]["preaudit"] if calibration else None
        ),
        "calibration_verdicts_sha256": (
            calibration["hashes"]["verdicts"] if calibration else None
        ),
        "audit_manifest": str(manifest),
        "audit_manifest_sha256": _sha256(manifest),
        "manual_verdict_schema": VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
        "manual_gate_status": "pending",
        "training_manifest_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_audit_entrypoints(latest_html=index, title="Scorer v12 Teacher review")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--preaudit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--partition",
        action="append",
        choices=("train", "val", "test"),
        default=[],
    )
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--calibration-preaudit")
    parser.add_argument("--calibration-verdicts")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build(
                source_manifest=Path(args.source_manifest),
                preaudit=Path(args.preaudit),
                output_dir=Path(args.output_dir),
                partitions=args.partition,
                calibration_manifest=(
                    Path(args.calibration_manifest)
                    if args.calibration_manifest
                    else None
                ),
                calibration_preaudit=(
                    Path(args.calibration_preaudit)
                    if args.calibration_preaudit
                    else None
                ),
                calibration_verdicts=(
                    Path(args.calibration_verdicts)
                    if args.calibration_verdicts
                    else None
                ),
            ),
            ensure_ascii=False,
        )
    )
