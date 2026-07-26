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
    VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
    VOCAL_ENVELOPE_SCORER_V12_VOCAL_COVERAGE_OPTIONS,
    VOCAL_ENVELOPE_SCORER_V12_VOCAL_PURITY_OPTIONS,
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
SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_teacher_audit_summary_v2"


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
            field="vocal_purity",
            options=VOCAL_ENVELOPE_SCORER_V12_VOCAL_PURITY_OPTIONS,
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
        for purity in axes[1].options:
            for nonvocal in axes[2].options:
                for structure in axes[3].options:
                    combination = (vocal, purity, nonvocal, structure)
                    if "unsure" in combination:
                        result = "unsure"
                    elif combination == (
                        "definite_vocal_complete",
                        "definite_vocal_excludes_separable_background",
                        "definite_non_vocal_clean",
                        "event_envelopes_continuous",
                    ):
                        result = "approved"
                    else:
                        result = "rejected"
                    results[combination] = result
    validate_audit_option_contract(axes=axes, combination_results=results)


def _page(
    payload: list[dict[str, Any]],
    *,
    source_manifest_sha: str,
    preaudit_sha: str,
    audit_manifest_sha: str,
) -> str:
    _validate_option_contract()
    intro = """
<section class="audit-help">
  <h2>Human Vocal Envelope 审计合同（可编辑三态分区）</h2>
  <p><b>绿色 vocal：</b>任何人类声道、口腔或呼吸系统产生的发声证据，包括对白、耳语、呻吟、喘息、吸呼气、哭笑、咳嗽、亲吻/唾液/口腔声、歌唱和背景人声。</p>
  <p><b>黄色 non-vocal：</b>确认没有人类发声的纯机械、撞击/拍打、动作、衣物/床体、水声、纯音乐、静音、底噪和环境噪声。肉体撞击虽由人体产生，但不是人类发声。</p>
  <p><b>灰色 unsure：</b>单次三态 Teacher 无法可靠判断人声重叠或安全边界；训练时为 -100，不代表 vocal 或 non-vocal 真值。</p>
  <p>上方三轨是 Teacher 原始证据；下方“可编辑最终分区”才是你要保存的人工修订。可点击任一区间选中并播放，转换标签、修改起止、拆分、合并相邻同标签、删除或新增区间。删除/修改时允许暂时出现空洞，但点击“确认本条”或保存前必须做到从 <code>0</code> 到完整 duration 连续覆盖、无重叠、无空洞。</p>
  <p>同一发声事件内部的短停顿、吸气、释气和非语义过渡可随绿色包络保留；只有声学上独立、可安全分离的纯背景被绿色跨越时，才算 purity 问题。</p>
  <p>每条必须完整听完，再分别判断：vocal 是否漏声/截边、绿色是否吞入可独立纯背景、黄色是否混入任何人类发声、同一发声事件是否被切碎或跨独立 non-vocal 过度合并。颜色条点击后只播放自身精确区间，不添加上下文；所有编辑后的区间也遵守同一规则。</p>
</section>
<div id="cards"></div>
"""
    body = ""
    css = """
.audit-help,.audit-card{background:#fff;border:1px solid #d7dde4;border-radius:10px;padding:14px;margin:0 0 14px}
.audit-card h3{margin:0 0 4px}.audit-meta{margin-bottom:8px}.audit-full-row{display:flex;gap:8px;align-items:center;margin:8px 0}.audit-full-row audio{margin:0;flex:1}
.vocal{background:#2db66f;color:#062d19}.nonvocal{background:#f0c84b;color:#3b2d00}.unsure{background:#9aa6b2;color:#15202a}.conflict{background:#df6c68;color:#3b0807}
.audit-verdict{display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));gap:10px;margin-top:12px}.audit-verdict label{display:flex;flex-direction:column;gap:4px}.audit-verdict select,.audit-notes{padding:7px}.audit-reviewed{display:flex!important;flex-direction:row!important;align-items:center;gap:7px!important;margin-top:10px}.audit-notes{width:100%;box-sizing:border-box;margin-top:8px}
.editable-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:14px}.editable-head b{margin-right:auto}.editable-track{position:relative;height:46px;background:#e7ebef;border-radius:5px;overflow:hidden;margin:8px 0}.editable-span{position:absolute;top:0;height:100%;border:0;border-right:1px solid rgba(0,0,0,.28);padding:4px 3px;font-size:10px;overflow:hidden;white-space:nowrap;cursor:pointer}.editable-span.selected{outline:3px solid #111;z-index:2}.editable-rows{display:flex;flex-direction:column;gap:7px}.editable-row{display:grid;grid-template-columns:minmax(145px,1.2fr) 90px 90px minmax(260px,2fr);gap:6px;align-items:center;border-top:1px solid #d7dde4;padding:7px 0}.editable-row.selected{background:#eef5ff;border-radius:5px;padding-left:5px}.editable-row input,.editable-row select{width:100%;box-sizing:border-box;padding:6px}.editable-actions{display:flex;gap:4px;flex-wrap:wrap}.editable-actions button{padding:5px 7px}.editable-add{display:grid;grid-template-columns:90px 90px 150px auto;gap:6px;align-items:end;margin-top:9px}.editable-add label{display:grid;gap:3px;font-size:12px}.editable-add input,.editable-add select{padding:6px}.partition-error{color:#a52f2f;font-weight:600;margin-top:7px}.partition-ok{color:#267443;font-weight:600;margin-top:7px}
@media(max-width:1000px){.audit-verdict{grid-template-columns:1fr}}
@media(max-width:850px){.editable-row{grid-template-columns:1fr 1fr}.editable-actions{grid-column:1/-1}.editable-add{grid-template-columns:1fr 1fr}.editable-add button{grid-column:1/-1}}
"""
    js = f"""
const entries={_script_json(payload)};
const sourceManifestSha={_script_json(source_manifest_sha)};
const preauditSha={_script_json(preaudit_sha)};
const auditManifestSha={_script_json(audit_manifest_sha)};
const labels={{
  vocal_coverage:[
    ['','请选择'],['definite_vocal_complete','真正人声完整，无漏声/截边'],['definite_vocal_missing_or_clipped','存在漏声、截边或真正人声落在黄色'],['unsure','不确定']
  ],
  vocal_purity:[
    ['','请选择'],['definite_vocal_excludes_separable_background','绿色未吞入可独立纯背景'],['definite_vocal_contains_separable_background','绿色跨越可独立删除的纯机械/环境背景'],['unsure','不确定']
  ],
  non_vocal_safety:[
    ['','请选择'],['definite_non_vocal_clean','黄色均不含真正人声'],['definite_non_vocal_contains_vocal','黄色含对白、带声呻吟或其他真正人声'],['unsure','不确定']
  ],
  envelope_structure:[
    ['','请选择'],['event_envelopes_continuous','同一真正人声事件连续且未吞入独立 non-vocal'],['same_event_fragmented','同一真正人声事件被切碎'],['overmerged_independent_nonvocal','跨越独立 non-vocal 过度合并'],['both_fragmented_and_overmerged','同时存在切碎和过度合并'],['unsure','不确定']
  ]
}};
function approved(state,entry){{return Boolean(state.reviewed_full_source&&validateAuditPartition(state.segments,entry.frame_count).ok)&&state.vocal_coverage==='definite_vocal_complete'&&state.vocal_purity==='definite_vocal_excludes_separable_background'&&state.non_vocal_safety==='definite_non_vocal_clean'&&state.envelope_structure==='event_envelopes_continuous';}}
const auditIdentity=[location.pathname,sourceManifestSha,preauditSha,auditManifestSha].join('|');
const core=createAuditReviewCore({{
  entries,
  storageKey:'vocal-envelope-scorer-v12-teacher-audit-v2-editable:'+auditIdentity,
  statusLabel:'已完成',
  entryId:entry=>entry.source_id,
  defaultState:()=>({{segments:[],selected_id:'',vocal_coverage:'',vocal_purity:'',non_vocal_safety:'',envelope_structure:'',reviewed_full_source:false,notes:'',updated_at:''}}),
  isComplete:(state,entry)=>Boolean(state.reviewed_full_source&&state.vocal_coverage&&state.vocal_purity&&state.non_vocal_safety&&state.envelope_structure&&validateAuditPartition(state.segments,entry.frame_count).ok),
  shouldSerialize:(state,entry)=>Boolean(state.reviewed_full_source&&state.vocal_coverage&&state.vocal_purity&&state.non_vocal_safety&&state.envelope_structure&&validateAuditPartition(state.segments,entry.frame_count).ok),
  serialize:async(entry,state)=>{{
    const checked=validateAuditPartition(state.segments,entry.frame_count);
    if(!checked.ok)throw new Error(`${{entry.source_id}}: ${{checked.error}}`);
    const corrected_spans=checked.segments.map(segment=>{{const times=auditPartitionSeconds(segment,0.02);return {{label:segment.label,start_frame:segment.start_frame,end_frame:segment.end_frame,start_s:times.start_s,end_s:times.end_s}};}});
    return {{
    schema:{_script_json(VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA)},
    boundary_serialization_contract_id:{_script_json(CONTRACT_ID)},
    task_semantics:{_script_json(VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS)},
    source_id:entry.source_id,video_id:entry.video_id,partition:entry.partition,
    audio_sha256:entry.audio_sha256,duration_s:entry.duration_s,frame_count:entry.frame_count,
    source_manifest_sha256:sourceManifestSha,preaudit_sha256:preauditSha,
    audit_manifest_sha256:auditManifestSha,evidence_span_signature:entry.evidence_span_signature,
    corrected_spans,corrected_span_signature:await auditPartitionSha256(checked.segments),
    reviewed_full_source:Boolean(state.reviewed_full_source),vocal_coverage:state.vocal_coverage,
    vocal_purity:state.vocal_purity,non_vocal_safety:state.non_vocal_safety,
    envelope_structure:state.envelope_structure,
    approved:approved(state,entry),notes:String(state.notes||''),updated_at:state.updated_at||new Date().toISOString(),
    training_manifest_allowed:approved(state,entry)
  }};
}}}});
function optionHtml(values,current){{return values.map(([value,text])=>`<option value="${{escapeAuditHtml(value)}}" ${{value===current?'selected':''}}>${{escapeAuditHtml(text)}}</option>`).join('');}}
function labelText(label){{return {{vocal_candidate:'vocal',non_vocal_candidate:'non-vocal',unsure:'unsure'}}[label]||label;}}
function labelClass(label){{return {{vocal_candidate:'vocal',non_vocal_candidate:'nonvocal',unsure:'unsure'}}[label]||'unsure';}}
function segmentId(){{return `s-${{Date.now()}}-${{Math.random().toString(16).slice(2)}}`;}}
function teacherSegments(entry){{
  const source=[];
  for(const [field,label] of [['vocal_spans','vocal_candidate'],['non_vocal_spans','non_vocal_candidate'],['unsure_spans','unsure']])for(const span of (entry[field]||[]))source.push({{id:segmentId(),label,start_frame:Number(span.start_frame),end_frame:Number(span.end_frame),category:span.category||'',reason:span.reason||''}});
  source.sort((a,b)=>a.start_frame-b.start_frame||a.end_frame-b.end_frame);
  const output=[];let cursor=0;
  for(const raw of source){{let start=Math.max(0,Math.min(entry.frame_count,Math.round(raw.start_frame))),end=Math.max(0,Math.min(entry.frame_count,Math.round(raw.end_frame)));if(end<=start)continue;if(start>cursor)output.push({{id:segmentId(),label:'unsure',start_frame:cursor,end_frame:start,category:'uncertain',reason:'Teacher evidence gap'}});if(start<cursor)start=cursor;if(end<=start)continue;output.push({{...raw,start_frame:start,end_frame:end}});cursor=end;}}
  if(cursor<entry.frame_count)output.push({{id:segmentId(),label:'unsure',start_frame:cursor,end_frame:entry.frame_count,category:'uncertain',reason:'Teacher evidence gap'}});
  if(!output.length)output.push({{id:segmentId(),label:'unsure',start_frame:0,end_frame:entry.frame_count,category:'uncertain',reason:'No Teacher spans'}});
  return normalizeAuditPartition(output).map((segment,index)=>({{...segment,id:segment.id||`teacher-${{index}}`}}));
}}
function ensureState(entry){{
  const state=core.ensure(entry);
  if(!Array.isArray(state.segments)||!state.segments.length){{state.segments=teacherSegments(entry);state.selected_id=state.segments[0]?.id||'';state.updated_at=state.updated_at||'';}}
  return state;
}}
function loosePartitionError(segments,frameCount){{
  const count=Number(frameCount),normalized=normalizeAuditPartition(segments);if(!Number.isInteger(count)||count<=0)return '无效的 frame_count';let cursor=0;
  for(const segment of normalized){{if(!AUDIT_PARTITION_LABELS.has(segment.label))return `未知标签：${{segment.label}}`;if(!Number.isInteger(segment.start_frame)||!Number.isInteger(segment.end_frame))return '区间边界必须对齐到 20ms frame';if(segment.start_frame<0||segment.end_frame>count||segment.end_frame<=segment.start_frame)return '存在越界或空区间';if(segment.start_frame<cursor)return '区间重叠';cursor=segment.end_frame;}}
  return '';
}}
function changeSegments(entry,state,segments,selectedId=''){{const error=loosePartitionError(segments,entry.frame_count);if(error){{setAuditMessage(error,true);return false;}}state.segments=normalizeAuditPartition(segments);state.selected_id=selectedId||state.selected_id||state.segments[0]?.id||'';state.reviewed_full_source=false;state.updated_at=new Date().toISOString();core.persist();render();return true;}}
function setAuditMessage(message,isError=false){{const node=document.getElementById('status');node.textContent=message;node.classList.toggle('error',isError);}}
function updateSegment(entry,state,id,startInput,endInput){{const segments=cloneAuditPartition(state.segments),index=segments.findIndex(segment=>segment.id===id);if(index<0)return;const start=Math.round(Number(startInput.value)/0.02),end=Math.round(Number(endInput.value)/0.02);if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start){{setAuditMessage('区间必须满足 end > start',true);return;}}segments[index]={{...segments[index],start_frame:start,end_frame:end}};changeSegments(entry,state,segments,id);}}
function convertSegment(entry,state,id,label){{const segments=cloneAuditPartition(state.segments),segment=segments.find(item=>item.id===id);if(!segment)return;segment.label=label;changeSegments(entry,state,segments,id);}}
function removeSegment(entry,state,id){{const segments=cloneAuditPartition(state.segments).filter(segment=>segment.id!==id);state.selected_id='';changeSegments(entry,state,segments,'');}}
function splitSegment(entry,state,id,splitInput){{const segments=cloneAuditPartition(state.segments),index=segments.findIndex(segment=>segment.id===id);if(index<0)return;const segment=segments[index],at=Math.round(Number(splitInput.value)/0.02);if(!Number.isFinite(at)||at<=segment.start_frame||at>=segment.end_frame){{setAuditMessage('拆分点必须在当前区间内部',true);return;}}segments.splice(index,1,{{...segment,id:segment.id,end_frame:at}},{{...segment,id:segmentId(),start_frame:at}});changeSegments(entry,state,segments,segment.id);}}
function mergeSegment(entry,state,id){{const segments=cloneAuditPartition(state.segments),index=segments.findIndex(segment=>segment.id===id);if(index<0)return;let changed=false;if(index>0&&segments[index-1].label===segments[index].label&&segments[index-1].end_frame===segments[index].start_frame){{segments[index-1].end_frame=segments[index].end_frame;segments.splice(index,1);changed=true;}}else if(index+1<segments.length&&segments[index+1].label===segments[index].label&&segments[index].end_frame===segments[index+1].start_frame){{segments[index].end_frame=segments[index+1].end_frame;segments.splice(index+1,1);changed=true;}}if(!changed){{setAuditMessage('当前区间没有相邻且同标签的区间',true);return;}}changeSegments(entry,state,segments,id);}}
function mergeAll(entry,state){{const normalized=normalizeAuditPartition(state.segments);if(normalized.length===state.segments.length){{setAuditMessage('没有可合并的相邻同标签区间');return;}}changeSegments(entry,state,normalized,state.selected_id);}}
function addSegment(entry,state,startInput,endInput,labelSelect){{const start=Math.round(Number(startInput.value)/0.02),end=Math.round(Number(endInput.value)/0.02),label=labelSelect.value;if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start){{setAuditMessage('新增区间必须满足 end > start',true);return;}}const segments=cloneAuditPartition(state.segments);if(segments.some(segment=>start<segment.end_frame&&end>segment.start_frame)){{setAuditMessage('新增区间与已有区间重叠；请先删除或缩短原区间',true);return;}}segments.push({{id:segmentId(),label,start_frame:start,end_frame:end,category:'manual',reason:'human edit'}});changeSegments(entry,state,segments,segments[segments.length-1].id);}}
function relabelRange(entry,state,startInput,endInput,labelSelect){{const start=Math.round(Number(startInput.value)/0.02),end=Math.round(Number(endInput.value)/0.02),label=labelSelect.value;if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start||start<0||end>entry.frame_count){{setAuditMessage('选区必须位于 source 内且满足 end > start',true);return;}}const result=[];let touched=false,selectedId='';for(const original of normalizeAuditPartition(state.segments)){{if(original.end_frame<=start||original.start_frame>=end){{result.push(original);continue;}}touched=true;if(original.start_frame<start)result.push({{...original,end_frame:start}});const innerStart=Math.max(original.start_frame,start),innerEnd=Math.min(original.end_frame,end);if(innerEnd>innerStart){{const replacement={{...original,id:segmentId(),label,start_frame:innerStart,end_frame:innerEnd,category:'manual',reason:'human relabel'}};selectedId=selectedId||replacement.id;result.push(replacement);}}if(original.end_frame>end)result.push({{...original,id:segmentId(),start_frame:end}});}}if(!touched){{const replacement={{id:segmentId(),label,start_frame:start,end_frame:end,category:'manual',reason:'human relabel'}};selectedId=replacement.id;result.push(replacement);}}changeSegments(entry,state,result,selectedId);}}
function partitionMessage(entry,state){{const checked=validateAuditPartition(state.segments,entry.frame_count);return checked.ok?'完整覆盖，无重叠或空洞':checked.error;}}
function render(){{
  const root=document.getElementById('cards');root.innerHTML='';
  for(const entry of entries){{
    const state=ensureState(entry),card=document.createElement('section');card.className='audit-card';
    card.innerHTML=`<h3>${{escapeAuditHtml(entry.source_id)}}</h3><div class="audit-meta"><small>${{escapeAuditHtml(entry.partition)}} / ${{escapeAuditHtml(entry.video_id)}} / ${{entry.frame_count}} frames / ${{Number(entry.duration_s).toFixed(3)}}s / 当前：${{approved(state,entry)?'可进入 canonical':'待修订或待审核'}}</small></div><div class="audit-full-row"><button type="button" class="full-play">播放完整 source</button><audio controls preload="metadata" src="${{escapeAuditHtml(entry.audio)}}"></audio></div><div class="lanes"></div><div class="editable-head"><b>可编辑最终分区（初始值 = Teacher 三态 segments）</b><button type="button" class="merge-all">合并全部相邻同标签</button></div><div class="editable-track"></div><div class="editable-rows"></div><div class="editable-add"><label>选区开始（秒）<input class="add-start" type="number" min="0" max="${{entry.duration_s}}" step="0.02" value="0.00"></label><label>选区结束（秒）<input class="add-end" type="number" min="0" max="${{entry.duration_s}}" step="0.02" value="${{Number(entry.duration_s).toFixed(2)}}"></label><label>标签<select class="add-label"><option value="vocal_candidate">vocal</option><option value="non_vocal_candidate">non-vocal</option><option value="unsure">unsure</option></select></label><button type="button" class="add-segment">空洞中新增</button><button type="button" class="relabel-range">选区转换/覆盖</button></div><div class="partition-state ${{validateAuditPartition(state.segments,entry.frame_count).ok?'partition-ok':'partition-error'}}">最终分区：${{escapeAuditHtml(partitionMessage(entry,state))}}</div><div class="audit-verdict"><label>1. Vocal 覆盖<select data-field="vocal_coverage">${{optionHtml(labels.vocal_coverage,state.vocal_coverage)}}</select></label><label>2. Vocal 纯度<select data-field="vocal_purity">${{optionHtml(labels.vocal_purity,state.vocal_purity)}}</select></label><label>3. Non-vocal 安全<select data-field="non_vocal_safety">${{optionHtml(labels.non_vocal_safety,state.non_vocal_safety)}}</select></label><label>4. 包络结构<select data-field="envelope_structure">${{optionHtml(labels.envelope_structure,state.envelope_structure)}}</select></label></div><label class="audit-reviewed"><input type="checkbox" data-field="reviewed_full_source" ${{state.reviewed_full_source?'checked':''}}>已完整听完本条 source 并确认最终分区</label><input class="audit-notes" data-field="notes" placeholder="可选备注" value="${{escapeAuditHtml(state.notes||'')}}"><small class="approval">保存内容会同时写入 Teacher 原始 evidence SHA 与 corrected_span_signature</small>`;
    const audio=card.querySelector('audio'),lanes=card.querySelector('.lanes');
    card.querySelector('.full-play').onclick=event=>play(audio,event.currentTarget,0,Number(entry.duration_s));
    appendAuditSpanLane({{container:lanes,audio,durationS:Number(entry.duration_s),label:'canonical vocal',metric:`${{entry.vocal_spans.length}} spans`,spans:entry.vocal_spans,className:'vocal',title:(span,start,end)=>`vocal ${{formatAuditSpan(start,end)}} · ${{span.category||''}} · ${{span.reason||''}}`}});
    appendAuditSpanLane({{container:lanes,audio,durationS:Number(entry.duration_s),label:'canonical non-vocal',metric:`${{entry.non_vocal_spans.length}} spans`,spans:entry.non_vocal_spans,className:'nonvocal',title:(span,start,end)=>`non-vocal ${{formatAuditSpan(start,end)}} · ${{span.category||''}} · ${{span.reason||''}}`}});
    appendAuditSpanLane({{container:lanes,audio,durationS:Number(entry.duration_s),label:'canonical unsure',metric:`${{entry.unsure_spans.length}} spans`,spans:entry.unsure_spans,className:span=>span.conflict?'conflict':'unsure',title:(span,start,end)=>`${{span.conflict?'conflict':'unsure'}} ${{formatAuditSpan(start,end)}}`}});
    const editableTrack=card.querySelector('.editable-track'),editableRows=card.querySelector('.editable-rows');
    for(const segment of normalizeAuditPartition(state.segments)){{
      const start=segment.start_frame*0.02,end=segment.end_frame*0.02,button=document.createElement('button');button.type='button';button.className=`editable-span ${{labelClass(segment.label)}} ${{state.selected_id===segment.id?'selected':''}}`;button.style.left=`${{100*segment.start_frame/entry.frame_count}}%`;button.style.width=`${{Math.max(.12,100*(segment.end_frame-segment.start_frame)/entry.frame_count)}}%`;button.textContent=labelText(segment.label);button.title=`${{labelText(segment.label)}} ${{formatAuditSpan(start,end)}}`;button.onclick=()=>{{state.selected_id=segment.id;core.persist();render();}};editableTrack.appendChild(button);
      const row=document.createElement('div');row.className=`editable-row ${{state.selected_id===segment.id?'selected':''}}`;row.innerHTML=`<button type="button" class="editable-play ${{labelClass(segment.label)}}">播放 ${{labelText(segment.label)}} ${{formatAuditSpan(start,end)}}</button><select class="edit-label"><option value="vocal_candidate" ${{segment.label==='vocal_candidate'?'selected':''}}>vocal</option><option value="non_vocal_candidate" ${{segment.label==='non_vocal_candidate'?'selected':''}}>non-vocal</option><option value="unsure" ${{segment.label==='unsure'?'selected':''}}>unsure</option></select><label><small>开始</small><input class="edit-start" type="number" min="0" max="${{entry.duration_s}}" step="0.02" value="${{start.toFixed(2)}}"></label><label><small>结束</small><input class="edit-end" type="number" min="0" max="${{entry.duration_s}}" step="0.02" value="${{end.toFixed(2)}}"></label><div class="editable-actions"><button type="button" class="update-segment">更新起止</button><input class="split-at" type="number" min="${{start.toFixed(2)}}" max="${{end.toFixed(2)}}" step="0.02" placeholder="拆分点(s)"><button type="button" class="split-segment">拆分</button><button type="button" class="merge-segment">合并相邻同标签</button><button type="button" class="remove-segment risk">删除</button></div>`;
      row.querySelector('.editable-play').onclick=event=>play(audio,event.currentTarget,start,end);row.querySelector('.edit-label').onchange=event=>convertSegment(entry,state,segment.id,event.target.value);row.querySelector('.update-segment').onclick=()=>updateSegment(entry,state,segment.id,row.querySelector('.edit-start'),row.querySelector('.edit-end'));row.querySelector('.split-segment').onclick=()=>splitSegment(entry,state,segment.id,row.querySelector('.split-at'));row.querySelector('.merge-segment').onclick=()=>mergeSegment(entry,state,segment.id);row.querySelector('.remove-segment').onclick=()=>removeSegment(entry,state,segment.id);editableRows.appendChild(row);
    }}
    card.querySelector('.merge-all').onclick=()=>mergeAll(entry,state);card.querySelector('.add-segment').onclick=()=>addSegment(entry,state,card.querySelector('.add-start'),card.querySelector('.add-end'),card.querySelector('.add-label'));card.querySelector('.relabel-range').onclick=()=>relabelRange(entry,state,card.querySelector('.add-start'),card.querySelector('.add-end'),card.querySelector('.add-label'));
    for(const element of card.querySelectorAll('[data-field]')){{
      const field=element.dataset.field;
      element.onchange=()=>{{if(field==='reviewed_full_source'){{if(element.checked){{const checked=validateAuditPartition(state.segments,entry.frame_count);if(!checked.ok){{element.checked=false;setAuditMessage(`不能确认：${{checked.error}}`,true);return;}}state.segments=checked.segments;state.reviewed_full_source=true;}}else state.reviewed_full_source=false;}}else state[field]=element.type==='checkbox'?element.checked:element.value;state.updated_at=new Date().toISOString();core.persist();render();}};
    }}
    root.appendChild(card);
  }}
  core.updateStatus();
}}
document.getElementById('stop').onclick=stop;document.getElementById('save').onclick=async()=>{{const pending=entries.find(entry=>!core.isComplete?.(ensureState(entry),entry));if(pending){{setAuditMessage(`尚未完成：${{pending.source_id}}`,true);return;}}await core.save();}};render();
"""
    return render_audit_review_page(
        AuditReviewPageSpec(
            title="Scorer v12 Human Vocal Envelope Teacher review",
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
        if row.get("task_semantics") != VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS:
            raise ValueError(f"wrong v12 task semantics: {source_id}")
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
                "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
                "source_id": source_id,
                "video_id": str(source["video_id"]),
                "partition": partition,
                "audio": target.relative_to(output_dir).as_posix(),
                "audio_sha256": str(source["audio_sha256"]),
                "duration_s": float(source["duration_s"]),
                "frame_count": int(source["frame_count"]),
                "source_manifest_sha256": manifest_sha,
                "preaudit_sha256": preaudit_sha,
                "evidence_span_signature": evidence_span_signature(
                    row,
                    frame_count=int(source["frame_count"]),
                    source_id=source_id,
                ),
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
    audit_manifest_sha = _sha256(manifest)
    index = output_dir / "index.html"
    index.write_text(
        _page(
            payload,
            source_manifest_sha=manifest_sha,
            preaudit_sha=preaudit_sha,
            audit_manifest_sha=audit_manifest_sha,
        ),
        encoding="utf-8",
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
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
        "audit_manifest_sha256": audit_manifest_sha,
        "manual_verdict_schema": VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
        "manual_gate_status": "pending",
        "training_manifest_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_audit_entrypoints(
        latest_html=index,
        title="Scorer v12 Human Vocal Envelope Teacher review",
    )
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
