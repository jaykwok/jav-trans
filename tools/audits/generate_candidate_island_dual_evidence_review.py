#!/usr/bin/env python3
"""Compare Scorer v11 dual-evidence preaudit with existing human full truth."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.audits.audit_prompt import (  # noqa: E402
    ResolvedAuditPrompt,
    resolve_audit_prompt,
)
from tools.audits.compare_candidate_island_preaudits import (  # noqa: E402
    _audio_url,
    _frame_bounds,
    _index,
    _label_runs,
    _labels,
    _rows,
    _sha256,
)
from tools.audits.review_page_core import (  # noqa: E402
    AuditOptionAxis,
    AuditReviewPageSpec,
    render_audit_review_page,
    validate_audit_option_contract,
)


SUMMARY_SCHEMA = "candidate_island_dual_evidence_review_summary_v1"
DETAIL_SCHEMA = "candidate_island_dual_evidence_review_item_v1"
BRIDGE_VERDICT_SCHEMA = (
    "candidate_island_scorer_v11_bridge_gap_manual_verdict_v3"
)
DEFAULT_REVIEW_PROMPT = """先播放完整 source 确认前后是否属于同一轮对话，再播放精确 gap、Protect 覆盖片段和未覆盖片段。先判断 gap 是否含语言，再判断语言是否被 Protect 完整覆盖，最后判断 Protect 保留的非语义部分是否造成过度合并。不要按固定时长、声音类别或 ASR 能否识别自动裁决。"""
BRIDGE_AUDIT_AXES = (
    AuditOptionAxis(
        field="content_verdict",
        options=(
            "contains_semantic_dialogue",
            "no_semantic_dialogue",
            "content_unsure",
        ),
    ),
    AuditOptionAxis(
        field="semantic_coverage_verdict",
        options=(
            "semantic_fully_protected",
            "semantic_missed_or_clipped",
            "semantic_coverage_unsure",
            "not_applicable_no_semantic",
        ),
    ),
    AuditOptionAxis(
        field="envelope_verdict",
        options=(
            "acceptable_continuous_envelope",
            "overmerged_independent_background",
            "envelope_unsure",
        ),
    ),
)


def _is_valid_bridge_combination(combination: tuple[str, ...]) -> bool:
    content, coverage, _envelope = combination
    if content == "contains_semantic_dialogue":
        return coverage in {
            "semantic_fully_protected",
            "semantic_missed_or_clipped",
            "semantic_coverage_unsure",
        }
    if content == "no_semantic_dialogue":
        return coverage == "not_applicable_no_semantic"
    if content == "content_unsure":
        return coverage == "semantic_coverage_unsure"
    return False


def _bridge_combination_results() -> dict[tuple[str, str, str], str]:
    results: dict[tuple[str, str, str], str] = {}
    envelopes = BRIDGE_AUDIT_AXES[2].options
    for envelope in envelopes:
        results[
            ("contains_semantic_dialogue", "semantic_coverage_unsure", envelope)
        ] = "unsure"
        results[("content_unsure", "semantic_coverage_unsure", envelope)] = (
            "unsure"
        )
    results[
        (
            "contains_semantic_dialogue",
            "semantic_fully_protected",
            "acceptable_continuous_envelope",
        )
    ] = "human_background_contains_semantic_dialogue"
    results[
        (
            "contains_semantic_dialogue",
            "semantic_fully_protected",
            "overmerged_independent_background",
        )
    ] = "semantic_present_and_background_overmerged"
    results[
        (
            "contains_semantic_dialogue",
            "semantic_fully_protected",
            "envelope_unsure",
        )
    ] = "unsure"
    results[
        (
            "contains_semantic_dialogue",
            "semantic_missed_or_clipped",
            "acceptable_continuous_envelope",
        )
    ] = "semantic_missed_or_clipped"
    results[
        (
            "contains_semantic_dialogue",
            "semantic_missed_or_clipped",
            "overmerged_independent_background",
        )
    ] = "semantic_missed_and_background_overmerged"
    results[
        (
            "contains_semantic_dialogue",
            "semantic_missed_or_clipped",
            "envelope_unsure",
        )
    ] = "unsure"
    results[
        (
            "no_semantic_dialogue",
            "not_applicable_no_semantic",
            "acceptable_continuous_envelope",
        )
    ] = "acceptable_nonsemantic_bridge"
    results[
        (
            "no_semantic_dialogue",
            "not_applicable_no_semantic",
            "overmerged_independent_background",
        )
    ] = "teacher_overmerged_independent_background"
    results[
        (
            "no_semantic_dialogue",
            "not_applicable_no_semantic",
            "envelope_unsure",
        )
    ] = "unsure"
    return results


BRIDGE_COMBINATION_RESULTS = _bridge_combination_results()
validate_audit_option_contract(
    axes=BRIDGE_AUDIT_AXES,
    combination_results=BRIDGE_COMBINATION_RESULTS,
    is_valid_combination=_is_valid_bridge_combination,
)


def _human_labels(row: dict[str, Any], *, frame_count: int) -> list[str]:
    labels = ["__unlabeled__"] * frame_count
    for index, span in enumerate(row.get("spans") or ()):
        label = str(span.get("label") or "")
        if label not in {"inside_candidate", "outside_candidate", "unsure"}:
            raise ValueError(f"unsupported human label at span {index}: {label}")
        start, end = _frame_bounds(span)
        if not 0 <= start < end <= frame_count:
            raise ValueError(
                f"invalid human span for {row.get('source_id')}: {start}..{end}"
            )
        if any(value != "__unlabeled__" for value in labels[start:end]):
            raise ValueError(f"overlapping human spans for {row.get('source_id')}")
        labels[start:end] = [label] * (end - start)
    if "__unlabeled__" in labels:
        raise ValueError(f"human truth must cover full source: {row.get('source_id')}")
    return labels


def _evidence_mask(
    row: dict[str, Any],
    *,
    field: str,
    frame_count: int,
) -> list[bool]:
    result = [False] * frame_count
    for span in row.get(field) or ():
        start, end = _frame_bounds(span)
        if not 0 <= start < end <= frame_count:
            raise ValueError(
                f"invalid {field} for {row.get('source_id')}: {start}..{end}"
            )
        result[start:end] = [True] * (end - start)
    return result


def _all_runs(labels: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label in ("inside_candidate", "outside_candidate", "unsure"):
        result.extend(_label_runs(labels, label=label))
    return sorted(
        result,
        key=lambda span: (span["start_s"], span["end_s"], span["label"]),
    )


def _boolean_runs(values: list[bool], *, label: str) -> list[dict[str, Any]]:
    pseudo = [label if value else "__none__" for value in values]
    return _label_runs(pseudo, label=label)


def _human_span_coverage(
    human_labels: list[str],
    protect: list[bool],
) -> tuple[int, int, list[dict[str, Any]]]:
    spans = _label_runs(human_labels, label="inside_candidate")
    fully_covered = 0
    details: list[dict[str, Any]] = []
    for span in spans:
        start, end = _frame_bounds(span)
        covered = sum(protect[start:end])
        total = end - start
        fully_covered += int(covered == total)
        details.append(
            {
                **span,
                "covered_frames": covered,
                "total_frames": total,
                "coverage_ratio": covered / max(total, 1),
                "fully_covered": covered == total,
            }
        )
    return len(spans), fully_covered, details


def _mask_runs_with_offset(
    values: list[bool],
    *,
    start_frame: int,
    selected: bool,
    label: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    run_start: int | None = None
    for index, value in enumerate([*values, not selected]):
        if value == selected and run_start is None:
            run_start = index
        elif value != selected and run_start is not None:
            absolute_start = start_frame + run_start
            absolute_end = start_frame + index
            result.append(
                {
                    "label": label,
                    "start_frame": absolute_start,
                    "end_frame": absolute_end,
                    "start_s": absolute_start * 0.02,
                    "end_s": absolute_end * 0.02,
                }
            )
            run_start = None
    return result


def _bridged_background_gaps(
    human_labels: list[str],
    protect: list[bool],
    *,
    source_id: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    frame_count = len(human_labels)
    for span in _label_runs(human_labels, label="outside_candidate"):
        start, end = _frame_bounds(span)
        enclosed = (
            start > 0
            and end < frame_count
            and human_labels[start - 1] == "inside_candidate"
            and human_labels[end] == "inside_candidate"
        )
        if not enclosed:
            continue
        protected_frames = sum(protect[start:end])
        if protected_frames <= 0:
            continue
        result.append(
            {
                "gap_id": (
                    f"{source_id}::bridge-gap::{start:06d}-{end:06d}"
                ),
                "label": "bridge",
                "start_s": start * 0.02,
                "end_s": end * 0.02,
                "start_frame": start,
                "end_frame": end,
                "duration_s": (end - start) * 0.02,
                "protected_frames": protected_frames,
                "protected_ratio": protected_frames / max(end - start, 1),
                "fully_bridged": protected_frames == end - start,
                "protected_overlap_spans": _mask_runs_with_offset(
                    protect[start:end],
                    start_frame=start,
                    selected=True,
                    label="protected_overlap",
                ),
                "unprotected_overlap_spans": _mask_runs_with_offset(
                    protect[start:end],
                    start_frame=start,
                    selected=False,
                    label="unprotected_overlap",
                ),
            }
        )
    return result


def _page(rows: list[dict[str, Any]], *, review_prompt: str) -> str:
    encoded = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    adapter_css = r"""
article,.contract{background:#fff;border:1px solid #ccd6df;border-radius:10px;padding:14px;margin-bottom:14px}.prompt{background:#eef6ff;border-left:5px solid #315f9d;padding:10px 12px;margin:10px 0}.contract table{width:100%;border-collapse:collapse;margin:8px 0 14px}.contract th,.contract td{border:1px solid #c9d3dc;padding:7px;text-align:left;vertical-align:top}.contract th{background:#edf1f5}.human-inside{background:#315f9d;color:#fff}.human-outside{background:#8d98a5;color:#fff}.human-unsure{background:#725190;color:#fff}.protect{background:#27a2c2;color:#fff}.remove{background:#e5bb2c;color:#1d1d1d}.final-inside{background:#258b57;color:#fff}.final-outside{background:#f2cf45;color:#1d1d1d}.final-unsure{background:#d87800;color:#fff}.conflict{background:#8d3db7;color:#fff}.bridge{background:#6f8734;color:#fff}.unsafe{background:#d32626;color:#fff}.metrics{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;font-size:12px}.good{color:#087443}.bad{color:#b3261e;font-weight:700}.legend{display:flex;gap:13px;flex-wrap:wrap}.swatch{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:-1px}.gap-reviews{margin-top:14px;border-top:1px solid #d7e0e8;padding-top:10px}.gap-review{display:grid;grid-template-columns:minmax(240px,340px) minmax(460px,1fr);gap:12px;padding:11px;margin:9px 0;background:#f7f9f3;border:1px solid #d5ddbd;border-radius:8px}.gap-play{width:100%;min-height:42px;text-align:left;background:#6f8734;color:#fff;border:0;border-radius:5px;padding:7px 9px;cursor:pointer}.clip-protected{background:#27a2c2;color:#fff}.clip-unprotected{background:#8d98a5;color:#fff}.axis{border:1px solid #d5dde4;border-radius:6px;padding:8px;margin-bottom:8px;background:#fff}.axis-title{font-weight:700;margin-bottom:5px}.gap-controls{display:flex;gap:6px;flex-wrap:wrap;align-items:center}.choice{border:1px solid #8d99a5;border-radius:5px;background:#fff;padding:7px 9px;cursor:pointer}.choice:disabled{opacity:.42;cursor:not-allowed}.choice.active{outline:3px solid #18212b;outline-offset:-2px}.choice[data-value="contains_semantic_dialogue"].active,.choice[data-value="semantic_fully_protected"].active{background:#b8d7f2}.choice[data-value="no_semantic_dialogue"].active,.choice[data-value="acceptable_continuous_envelope"].active{background:#bfe5cc}.choice[data-value="semantic_missed_or_clipped"].active,.choice[data-value="overmerged_independent_background"].active{background:#f4b8b4}.choice[data-value$="unsure"].active{background:#f3d49d}.combined{font-weight:700;padding:7px 9px;background:#edf1f5;border-radius:5px}.gap-note{width:100%;min-height:42px;margin-top:7px;box-sizing:border-box}.empty{color:#607080;font-style:italic}@media(max-width:900px){.gap-review{grid-template-columns:1fr}.contract{overflow-x:auto}}
"""
    prompt_html = html.escape(review_prompt).replace("\n", "<br>")
    intro_html = f"""<section class="contract"><h2>审计结构与选项意义</h2><div class="prompt"><b>本页审计提示</b><p>{prompt_html}</p><small>提示是 Adapter 配置；Core 固定负责播放器、状态、localStorage 和保存 API。<code>--prompt</code> / <code>--prompt-file</code> 只替换任务说明，不改变三轴 schema。</small></div><p>人工蓝段是既有 Split 级语音锚点，不是 Scorer 的逐帧 outside 真值。一个 gap 可能同时“含对白”又“保留了过多独立背景”，所以不能使用单一互斥标签；每条 gap 按三个维度保存。</p><table><thead><tr><th>审计轴</th><th>选项</th><th>含义</th><th>结果用途</th></tr></thead><tbody><tr><td rowspan="3">A. gap 内容</td><td><code>contains_semantic_dialogue</code></td><td>听到明确词语、句尾、语言结构或交流性发声。</td><td>人工 background 含语言，需要后续精确修正；不能当 Scorer outside。</td></tr><tr><td><code>no_semantic_dialogue</code></td><td>确认整段没有语言或交流性发声。</td><td>继续只评价非语义背景是否允许桥接。</td></tr><tr><td><code>content_unsure</code></td><td>无法可靠区分语言与呻吟、呼吸或噪声。</td><td>保持 unresolved，不进入自动训练真值。</td></tr><tr><td rowspan="3">B. 语义覆盖</td><td><code>semantic_fully_protected</code></td><td>发现的全部语义都落在 Protect 覆盖内，边缘无截断。</td><td>Teacher 没有漏掉该 gap 内新发现的语言。</td></tr><tr><td><code>semantic_missed_or_clipped</code></td><td>至少有一部分语义位于 Protect 未覆盖区，或词头词尾被截。</td><td>高优先级漏保护/截断错误。</td></tr><tr><td><code>semantic_coverage_unsure</code></td><td>有疑似语言，但无法确认覆盖是否完整。</td><td>保持 unresolved；不得用数值推断替代听感。</td></tr><tr><td rowspan="3">C. 非语义包络</td><td><code>acceptable_continuous_envelope</code></td><td>Protect 保留的非语义部分很短、与同一轮对话紧密相连，或删除会破坏连续波形。</td><td>允许桥接，不算 Scorer false keep。</td></tr><tr><td><code>overmerged_independent_background</code></td><td>Protect 保留了声学独立、明显过长且可安全删除的非语义背景。</td><td>Teacher 过度合并；后续需缩小候选包络。</td></tr><tr><td><code>envelope_unsure</code></td><td>无法判断该背景是否应在 Scorer 阶段独立删除。</td><td>保持 unresolved。</td></tr></tbody></table><h3>组合结果</h3><table><thead><tr><th>组合</th><th>解释</th></tr></thead><tbody><tr><td>含语义 + 完整保护 + 包络合理</td><td>Teacher 正确；人工 background 需要修正。</td></tr><tr><td>含语义 + 漏保护/截断</td><td>即使其余包络合理，也属于高优先级 Scorer recall 风险。</td></tr><tr><td>含语义 + 过度合并</td><td>混合问题：人工 background 内有语言，同时 Teacher 还保留了可独立删除的长背景。</td></tr><tr><td>无语义 + 包络合理</td><td>允许的短背景桥接。</td></tr><tr><td>无语义 + 过度合并</td><td>纯 Teacher overmerge。</td></tr><tr><td>任一轴不确定</td><td>保存为 unsure，不直接编译训练标签。</td></tr></tbody></table><p>“长”只由完整 source 上下文、声学独立性和删除后的连续性共同判断，不设固定秒数阈值。下方保留完整 source 播放；精确 gap、Protect 覆盖片段和未覆盖片段均不添加上下文。</p><p class="legend"><span><i class="swatch protect"></i>Protect evidence</span><span><i class="swatch remove"></i>Remove evidence</span><span><i class="swatch bridge"></i>被 Protect 覆盖的锚点间 BG gap</span><span><i class="swatch conflict"></i>冲突</span><span><i class="swatch unsafe"></i>真语音误删</span></p></section>"""
    adapter_js = r"""
const rows=__ROWS__;
const verdictSchema=__VERDICT_SCHEMA__;
const boundaryContract=__BOUNDARY_CONTRACT__;
const combinationResults=__COMBINATION_RESULTS__;
const storageKey='candidate-island-scorer-v11-bridge-gap-review-v3:'+location.pathname;
const esc=escapeAuditHtml;
function allGaps(){return rows.flatMap(row=>row.bridged_background_gaps.map(gap=>({row,gap})));}
function combinationKey(state){return [state.content_verdict||'',state.semantic_coverage_verdict||'',state.envelope_verdict||''].join('|');}
function complete(state){return Object.prototype.hasOwnProperty.call(combinationResults,combinationKey(state));}
function combinedVerdict(state){return combinationResults[combinationKey(state)]||'unreviewed';}
const combinedLabels={unreviewed:'未完成三个审计维度',unsure:'不确定：保持 unresolved',semantic_missed_and_background_overmerged:'混合错误：语义漏保护且背景过度合并',semantic_missed_or_clipped:'高优先级错误：语义漏保护或截断',semantic_present_and_background_overmerged:'混合问题：人工 BG 含语义且 Teacher 过度合并',human_background_contains_semantic_dialogue:'人工 BG 含语义；Teacher 保护完整且包络合理',teacher_overmerged_independent_background:'纯 Teacher overmerge：独立背景应删除',acceptable_nonsemantic_bridge:'可接受的无语义短背景桥接'};
const gapEntries=allGaps(),entryByGapId=new Map(gapEntries.map(entry=>[entry.gap.gap_id,entry]));
const reviewCore=createAuditReviewCore({storageKey,entries:gapEntries,entryId:entry=>entry.gap.gap_id,defaultState:()=>({content_verdict:'',semantic_coverage_verdict:'',envelope_verdict:'',note:''}),isComplete:state=>complete(state),statusLabel:'桥接裁决',statusExtra:()=>`unsafe ${rows.reduce((n,row)=>n+row.unsafe_outside_frames,0)} frames`,filename:'manual_verdicts.jsonl',serialize:(entry,state)=>{const {row,gap}=entry,combined=combinedVerdict(state);return {schema:verdictSchema,boundary_serialization_contract_id:boundaryContract,gap_id:gap.gap_id,source_id:row.source_id,partition:row.partition,start_frame:gap.start_frame,end_frame:gap.end_frame,start_s:gap.start_s,end_s:gap.end_s,duration_s:gap.duration_s,protected_frames:gap.protected_frames,protected_ratio:gap.protected_ratio,fully_bridged:gap.fully_bridged,content_verdict:state.content_verdict||'unreviewed',semantic_coverage_verdict:state.semantic_coverage_verdict||'unreviewed',envelope_verdict:state.envelope_verdict||'unreviewed',combined_verdict:combined,verdict:combined,complete:complete(state),note:state.note||'',updated_at:state.updated_at||new Date().toISOString()};}});
function ensure(gap){return reviewCore.ensure(entryByGapId.get(gap.gap_id));}
function persist(){reviewCore.persist();}
function lane(card,audio,row,label,spans,kind,metric=''){appendAuditSpanLane({container:card,audio,durationS:row.duration_s,label,metric,spans,className:span=>{const suffix=span.label==='inside_candidate'?'inside':span.label==='outside_candidate'?'outside':'unsure';return kind==='human'?`human-${suffix}`:kind==='final'?`final-${suffix}`:kind;},title:(span,start,end)=>`${label} ${span.label||kind} ${formatAuditSpan(start,end)}`,text:(_span,start,end)=>formatAuditSpan(start,end)});}
function clipButtons(container,audio,spans,label,className){appendAuditClipButtons({container,audio,spans,label,className});}
function syncReview(item,state){item.querySelectorAll('[data-field]').forEach(button=>button.classList.toggle('active',state[button.dataset.field]===button.dataset.value));const coverageButtons=item.querySelectorAll('[data-field="semantic_coverage_verdict"]');const needsCoverage=state.content_verdict==='contains_semantic_dialogue';coverageButtons.forEach(button=>button.disabled=!needsCoverage);const hint=item.querySelector('.coverage-hint');hint.textContent=needsCoverage?'请判断全部语义是否被 Protect 覆盖':state.content_verdict==='no_semantic_dialogue'?'已自动记为不适用（确认无语义）':state.content_verdict==='content_unsure'?'已自动记为覆盖不确定':'先完成 A 轴';const combined=combinedVerdict(state);item.querySelector('.combined').textContent='组合结果：'+combinedLabels[combined];}
function renderGapReviews(card,audio,row){const section=document.createElement('div');section.className='gap-reviews';section.innerHTML='<b>锚点间 background gap 人工裁决</b><small> 按 A 内容 → B 语义覆盖 → C 非语义包络依次判断；B 只在 A=含语义时手动选择。</small>';if(!row.bridged_background_gaps.length){const empty=document.createElement('p');empty.className='empty';empty.textContent='本条没有被 Protect 覆盖的锚点间 background gap。';section.appendChild(empty);}for(const gap of row.bridged_background_gaps){const state=ensure(gap),item=document.createElement('div');item.className='gap-review';item.innerHTML=`<div class="gap-audio"><button type="button" class="gap-play">播放精确 gap ${formatAuditSpan(gap.start_s,gap.end_s)}</button><small>${esc(gap.gap_id)}<br>duration=${formatAuditTimestamp(gap.duration_s)} · protected=${(100*Number(gap.protected_ratio)).toFixed(1)}% · ${gap.fully_bridged?'完整桥接':'部分覆盖'}</small></div><div><div class="axis"><div class="axis-title">A. gap 内容</div><div class="gap-controls"><button type="button" class="choice" data-field="content_verdict" data-value="contains_semantic_dialogue">含语义对白</button><button type="button" class="choice" data-field="content_verdict" data-value="no_semantic_dialogue">确认无语义</button><button type="button" class="choice" data-field="content_verdict" data-value="content_unsure">内容不确定</button></div></div><div class="axis"><div class="axis-title">B. 语义是否被 Protect 完整覆盖</div><div class="gap-controls"><button type="button" class="choice" data-field="semantic_coverage_verdict" data-value="semantic_fully_protected">全部覆盖且无截断</button><button type="button" class="choice" data-field="semantic_coverage_verdict" data-value="semantic_missed_or_clipped">存在漏保护或截断</button><button type="button" class="choice" data-field="semantic_coverage_verdict" data-value="semantic_coverage_unsure">覆盖不确定</button></div><small class="coverage-hint"></small></div><div class="axis"><div class="axis-title">C. Protect 保留的非语义包络</div><div class="gap-controls"><button type="button" class="choice" data-field="envelope_verdict" data-value="acceptable_continuous_envelope">可接受连续包络</button><button type="button" class="choice" data-field="envelope_verdict" data-value="overmerged_independent_background">过度合并独立长背景</button><button type="button" class="choice" data-field="envelope_verdict" data-value="envelope_unsure">包络不确定</button></div></div><div class="combined"></div><textarea class="gap-note" placeholder="可选备注；混合情况可注明语义和长背景的大致位置">${esc(state.note||'')}</textarea></div>`;const audioBox=item.querySelector('.gap-audio'),playButton=item.querySelector('.gap-play');playButton.onclick=()=>play(audio,playButton,Number(gap.start_s),Number(gap.end_s));clipButtons(audioBox,audio,gap.protected_overlap_spans,'Protect 覆盖片段','clip-protected');clipButtons(audioBox,audio,gap.unprotected_overlap_spans,'Protect 未覆盖片段','clip-unprotected');item.querySelectorAll('[data-field]').forEach(button=>button.onclick=()=>{const field=button.dataset.field,value=button.dataset.value;state[field]=value;if(field==='content_verdict'){if(value==='no_semantic_dialogue')state.semantic_coverage_verdict='not_applicable_no_semantic';else if(value==='content_unsure')state.semantic_coverage_verdict='semantic_coverage_unsure';else if(['not_applicable_no_semantic','semantic_coverage_unsure'].includes(state.semantic_coverage_verdict))state.semantic_coverage_verdict='';}state.updated_at=new Date().toISOString();syncReview(item,state);persist();});item.querySelector('.gap-note').onchange=event=>{state.note=event.target.value;state.updated_at=new Date().toISOString();persist();};syncReview(item,state);section.appendChild(item);}card.appendChild(section);}
const root=document.getElementById('list');for(const row of rows){const card=document.createElement('article');card.innerHTML=`<h2>${esc(row.source_id)}</h2><small>${esc(row.partition)} · ${Number(row.duration_s).toFixed(2)}s${row.failed_closed?' · ⚠ teacher failed closed':''}</small><audio controls preload="metadata" src="${esc(row.audio)}"></audio>`;const audio=card.querySelector('audio');lane(card,audio,row,'人工语音锚点 / BG',row.human_spans,'human',`inside ${(100*row.human_inside_ratio).toFixed(1)}%`);lane(card,audio,row,'Protect evidence',row.protect_spans,'protect',`anchor coverage ${(100*row.protect_recall).toFixed(1)}% · full spans ${row.fully_protected_human_span_count}/${row.human_inside_span_count}`);lane(card,audio,row,'Remove evidence',row.remove_spans,'remove',`anchor hits ${row.remove_human_inside_frames} frames`);lane(card,audio,row,'最终三态标签',row.final_spans,'final',`inside ${(100*row.final_inside_ratio).toFixed(1)}% · outside ${(100*row.final_outside_ratio).toFixed(1)}% · unsure ${(100*row.final_unsure_ratio).toFixed(1)}%`);lane(card,audio,row,'被 Protect 覆盖的锚点间 BG gap',row.bridged_background_gaps,'bridge',`${row.bridged_gap_count} gaps · max ${row.max_bridged_gap_s.toFixed(2)}s`);lane(card,audio,row,'Protect / Remove 冲突',row.conflict_spans,'conflict',`${row.conflict_frames} frames`);lane(card,audio,row,'真语音被 outside 命中',row.unsafe_outside_spans,'unsafe',`${row.unsafe_outside_frames} frames / ${row.unsafe_outside_s.toFixed(2)}s`);const metrics=document.createElement('div');metrics.className='metrics';metrics.innerHTML=`<span>supervised ${(100*row.supervised_ratio).toFixed(1)}%</span><span>Protect recall（诊断） ${(100*row.protect_recall).toFixed(1)}%</span><span class="${row.final_outside_precision>=.95?'good':'bad'}">outside precision ${(100*row.final_outside_precision).toFixed(1)}%</span><span class="${row.true_speech_retention>=.95?'good':'bad'}">true-speech retention ${(100*row.true_speech_retention).toFixed(1)}%</span>`;card.appendChild(metrics);renderGapReviews(card,audio,row);root.appendChild(card);}
document.getElementById('stop').onclick=()=>{stop();reviewCore.updateStatus('已停止');};
document.getElementById('save').onclick=()=>reviewCore.save();
reviewCore.updateStatus();
"""
    adapter_js = (
        adapter_js.replace("__ROWS__", encoded)
        .replace("__VERDICT_SCHEMA__", json.dumps(BRIDGE_VERDICT_SCHEMA))
        .replace(
            "__COMBINATION_RESULTS__",
            json.dumps(
                {
                    "|".join(combination): result
                    for combination, result in BRIDGE_COMBINATION_RESULTS.items()
                },
                ensure_ascii=False,
            ),
        )
        .replace(
            "__BOUNDARY_CONTRACT__",
            json.dumps(ACOUSTIC_BINARY_V12_CONTRACT.contract_id),
        )
    )
    return render_audit_review_page(
        AuditReviewPageSpec(
            title="Scorer v11 · Protect × Remove 双证据 held-out 对照",
            intro_html=intro_html,
            body_html='<div id="list"></div>',
            adapter_css=adapter_css,
            adapter_js=adapter_js,
        )
    )


def generate(
    *,
    manifest: Path,
    human_verdicts: Path,
    candidate: Path,
    output_dir: Path,
    update_nav: bool = True,
    review_prompt: ResolvedAuditPrompt | None = None,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    human_verdicts = human_verdicts.resolve()
    candidate = candidate.resolve()
    output_dir = output_dir.resolve()
    resolved_prompt = review_prompt or resolve_audit_prompt(
        default_prompt=DEFAULT_REVIEW_PROMPT,
    )
    sources = _index(manifest, name="source manifest")
    human = _index(human_verdicts, name="human verdicts")
    candidate_rows = _rows(candidate)
    if not candidate_rows:
        raise ValueError("dual-evidence candidate preaudit is empty")

    details: list[dict[str, Any]] = []
    totals = {
        "frame_count": 0,
        "human_inside_frames": 0,
        "human_outside_frames": 0,
        "protect_frames": 0,
        "protect_true_inside_frames": 0,
        "remove_frames": 0,
        "remove_true_outside_frames": 0,
        "final_inside_frames": 0,
        "final_outside_frames": 0,
        "final_unsure_frames": 0,
        "final_inside_true_frames": 0,
        "final_outside_true_frames": 0,
        "unsafe_outside_frames": 0,
        "conflict_frames": 0,
        "human_inside_span_count": 0,
        "fully_protected_human_span_count": 0,
        "bridged_gap_count": 0,
        "fully_bridged_gap_count": 0,
        "bridged_gap_frames": 0,
        "max_bridged_gap_frames": 0,
        "remove_human_inside_frames": 0,
        "all_outside_protect_frames": 0,
        "failed_closed_count": 0,
    }
    seen: set[str] = set()
    for candidate_row in candidate_rows:
        source_id = str(candidate_row.get("source_id") or "")
        if not source_id or source_id in seen:
            raise ValueError("candidate preaudit requires unique source_id")
        seen.add(source_id)
        if source_id not in sources or source_id not in human:
            raise ValueError(f"missing manifest or human truth for {source_id}")
        source = sources[source_id]
        truth = human[source_id]
        frame_count = int(source["frame_count"])
        if int(candidate_row.get("frame_count") or 0) != frame_count:
            raise ValueError(f"candidate frame geometry mismatch: {source_id}")
        if str(candidate_row.get("boundary_serialization_contract_id") or "") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError(f"candidate boundary contract mismatch: {source_id}")
        human_labels = _human_labels(truth, frame_count=frame_count)
        final_labels = _labels(candidate_row, frame_count=frame_count)
        protect = _evidence_mask(
            candidate_row,
            field="protected_evidence_spans",
            frame_count=frame_count,
        )
        remove = _evidence_mask(
            candidate_row,
            field="remove_evidence_spans",
            frame_count=frame_count,
        )
        conflict = [left and right for left, right in zip(protect, remove)]
        unsafe = [
            predicted == "outside_candidate" and actual == "inside_candidate"
            for predicted, actual in zip(final_labels, human_labels)
        ]
        human_span_count, fully_protected_span_count, human_span_coverage = (
            _human_span_coverage(human_labels, protect)
        )
        bridged_gaps = _bridged_background_gaps(
            human_labels,
            protect,
            source_id=source_id,
        )
        bridged_gap_frames = sum(
            int(span["end_frame"]) - int(span["start_frame"])
            for span in bridged_gaps
        )
        max_bridged_gap_frames = max(
            (
                int(span["end_frame"]) - int(span["start_frame"])
                for span in bridged_gaps
            ),
            default=0,
        )
        remove_human_inside_frames = sum(
            flag and actual == "inside_candidate"
            for flag, actual in zip(remove, human_labels)
        )
        counts = {
            "frame_count": frame_count,
            "human_inside_frames": human_labels.count("inside_candidate"),
            "human_outside_frames": human_labels.count("outside_candidate"),
            "protect_frames": sum(protect),
            "protect_true_inside_frames": sum(
                flag and actual == "inside_candidate"
                for flag, actual in zip(protect, human_labels)
            ),
            "remove_frames": sum(remove),
            "remove_true_outside_frames": sum(
                flag and actual == "outside_candidate"
                for flag, actual in zip(remove, human_labels)
            ),
            "final_inside_frames": final_labels.count("inside_candidate"),
            "final_outside_frames": final_labels.count("outside_candidate"),
            "final_unsure_frames": final_labels.count("unsure"),
            "final_inside_true_frames": sum(
                predicted == actual == "inside_candidate"
                for predicted, actual in zip(final_labels, human_labels)
            ),
            "final_outside_true_frames": sum(
                predicted == actual == "outside_candidate"
                for predicted, actual in zip(final_labels, human_labels)
            ),
            "unsafe_outside_frames": sum(unsafe),
            "conflict_frames": sum(conflict),
            "human_inside_span_count": human_span_count,
            "fully_protected_human_span_count": fully_protected_span_count,
            "bridged_gap_count": len(bridged_gaps),
            "fully_bridged_gap_count": sum(
                bool(span["fully_bridged"]) for span in bridged_gaps
            ),
            "bridged_gap_frames": bridged_gap_frames,
            "max_bridged_gap_frames": max_bridged_gap_frames,
            "remove_human_inside_frames": remove_human_inside_frames,
            "all_outside_protect_frames": (
                sum(protect)
                if human_labels.count("inside_candidate") == 0
                else 0
            ),
            "failed_closed_count": int(bool(candidate_row.get("teacher_failed_closed"))),
        }
        for key, value in counts.items():
            if key == "max_bridged_gap_frames":
                totals[key] = max(totals[key], value)
            else:
                totals[key] += value
        human_inside = max(counts["human_inside_frames"], 1)
        protect_frames = max(counts["protect_frames"], 1)
        remove_frames = max(counts["remove_frames"], 1)
        final_inside = max(counts["final_inside_frames"], 1)
        final_outside = max(counts["final_outside_frames"], 1)
        details.append(
            {
                "schema": DETAIL_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "source_id": source_id,
                "partition": str(source.get("partition") or ""),
                "duration_s": float(source["duration_s"]),
                "frame_count": frame_count,
                "audio": _audio_url(
                    str(candidate_row.get("audio") or source["audio"]),
                    manifest=candidate,
                ),
                "human_spans": _all_runs(human_labels),
                "protect_spans": [
                    {**span, "label": "protect"}
                    for span in candidate_row.get("protected_evidence_spans") or ()
                ],
                "remove_spans": [
                    {**span, "label": "remove"}
                    for span in candidate_row.get("remove_evidence_spans") or ()
                ],
                "final_spans": _all_runs(final_labels),
                "human_span_coverage": human_span_coverage,
                "bridged_background_gaps": bridged_gaps,
                "conflict_spans": _boolean_runs(conflict, label="conflict"),
                "unsafe_outside_spans": _boolean_runs(unsafe, label="unsafe"),
                **counts,
                "human_inside_ratio": counts["human_inside_frames"] / frame_count,
                "protect_recall": counts["protect_true_inside_frames"] / human_inside,
                "protect_precision": counts["protect_true_inside_frames"] / protect_frames,
                "remove_precision": counts["remove_true_outside_frames"] / remove_frames,
                "final_inside_ratio": counts["final_inside_frames"] / frame_count,
                "final_outside_ratio": counts["final_outside_frames"] / frame_count,
                "final_unsure_ratio": counts["final_unsure_frames"] / frame_count,
                "supervised_ratio": (
                    counts["final_inside_frames"] + counts["final_outside_frames"]
                )
                / frame_count,
                "final_inside_precision": counts["final_inside_true_frames"] / final_inside,
                "final_outside_precision": counts["final_outside_true_frames"] / final_outside,
                "true_speech_retention": (
                    1.0
                    - counts["unsafe_outside_frames"]
                    / max(counts["human_inside_frames"], 1)
                ),
                "unsafe_outside_s": counts["unsafe_outside_frames"] * 0.02,
                "max_bridged_gap_s": counts["max_bridged_gap_frames"] * 0.02,
                "failed_closed": bool(candidate_row.get("teacher_failed_closed")),
            }
        )

    details.sort(
        key=lambda row: (
            -int(row["unsafe_outside_frames"]),
            -float(row["max_bridged_gap_s"]),
            float(row["protect_recall"]),
            str(row["source_id"]),
        )
    )
    frame_count = max(totals["frame_count"], 1)
    human_inside = max(totals["human_inside_frames"], 1)
    protect_frames = max(totals["protect_frames"], 1)
    remove_frames = max(totals["remove_frames"], 1)
    final_inside = max(totals["final_inside_frames"], 1)
    final_outside = max(totals["final_outside_frames"], 1)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "human_verdicts": str(human_verdicts),
        "human_verdicts_sha256": _sha256(human_verdicts),
        "candidate": str(candidate),
        "candidate_sha256": _sha256(candidate),
        "source_count": len(details),
        **totals,
        "protect_recall": totals["protect_true_inside_frames"] / human_inside,
        "protect_precision": totals["protect_true_inside_frames"] / protect_frames,
        "remove_precision": totals["remove_true_outside_frames"] / remove_frames,
        "final_inside_precision": totals["final_inside_true_frames"] / final_inside,
        "final_outside_precision": totals["final_outside_true_frames"] / final_outside,
        "final_inside_ratio": totals["final_inside_frames"] / frame_count,
        "final_outside_ratio": totals["final_outside_frames"] / frame_count,
        "final_unsure_ratio": totals["final_unsure_frames"] / frame_count,
        "supervised_ratio": (
            totals["final_inside_frames"] + totals["final_outside_frames"]
        )
        / frame_count,
        "conflict_ratio": totals["conflict_frames"] / frame_count,
        "true_speech_retention": (
            1.0
            - totals["unsafe_outside_frames"]
            / max(totals["human_inside_frames"], 1)
        ),
        "unsafe_outside_s": totals["unsafe_outside_frames"] * 0.02,
        "max_bridged_gap_s": totals["max_bridged_gap_frames"] * 0.02,
        "true_speech_retention_gate": 0.95,
        "final_outside_precision_gate": 0.95,
        "protect_recall_is_diagnostic_only": True,
        "training_manifest_allowed": False,
        "manual_verdicts": str(output_dir / "manual_verdicts.jsonl"),
        "manual_verdict_schema": BRIDGE_VERDICT_SCHEMA,
        "review_prompt_source": resolved_prompt.source,
        "review_prompt_sha256": resolved_prompt.sha256,
        "audit_navigation_updated": update_nav,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "per_source.jsonl"
    detail_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in details
        ),
        encoding="utf-8",
    )
    summary["per_source"] = str(detail_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    index = output_dir / "index.html"
    index.write_text(
        _page(details, review_prompt=resolved_prompt.text),
        encoding="utf-8",
    )
    if update_nav:
        update_audit_entrypoints(
            latest_html=index,
            title="Scorer v11 dual-evidence held-out review",
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--human-verdicts", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument(
        "--update-nav",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    prompt = resolve_audit_prompt(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        default_prompt=DEFAULT_REVIEW_PROMPT,
    )
    print(
        json.dumps(
            generate(
                manifest=Path(args.manifest),
                human_verdicts=Path(args.human_verdicts),
                candidate=Path(args.candidate),
                output_dir=Path(args.output_dir),
                update_nav=args.update_nav,
                review_prompt=prompt,
            ),
            ensure_ascii=False,
        )
    )
