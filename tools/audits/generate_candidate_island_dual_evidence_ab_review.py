#!/usr/bin/env python3
"""Render a shared-core A/B review for two normalized Scorer dual-evidence runs."""
from __future__ import annotations

import argparse
import hashlib
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
from tools.audits.review_page_core import (  # noqa: E402
    AuditOptionAxis,
    AuditReviewPageSpec,
    render_audit_review_page,
    validate_audit_option_contract,
)


SUMMARY_SCHEMA = "candidate_island_dual_evidence_ab_review_summary_v1"
DETAIL_SCHEMA = "candidate_island_dual_evidence_ab_review_item_v1"
VERDICT_SCHEMA = "candidate_island_dual_evidence_ab_manual_verdict_v1"
COMPARISON_OPTIONS = (
    "base_better",
    "candidate_better",
    "equivalent_both_acceptable",
    "equivalent_both_unacceptable",
    "tradeoff_no_clear_winner",
    "comparison_unsure",
)

validate_audit_option_contract(
    axes=(AuditOptionAxis(field="comparison_verdict", options=COMPARISON_OPTIONS),),
    combination_results={(option,): option for option in COMPARISON_OPTIONS},
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _index(path: Path, *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _rows(path):
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in result:
            raise ValueError(f"{name} requires unique source_id")
        result[source_id] = row
    return result


def _summary_path(value: Path) -> Path:
    result = value.resolve()
    if result.is_dir():
        result = result / "summary.json"
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _resolved_child(value: str, *, parent: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _load_review(value: Path, *, name: str) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    summary_path = _summary_path(value)
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if str(summary.get("schema") or "") != "candidate_island_dual_evidence_review_summary_v1":
        raise ValueError(f"{name} is not a normalized dual-evidence review")
    if str(summary.get("boundary_serialization_contract_id") or "") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError(f"{name} boundary contract mismatch")
    details_path = _resolved_child(str(summary.get("per_source") or ""), parent=summary_path.parent)
    if not details_path.is_file():
        raise FileNotFoundError(details_path)
    return summary_path, summary, _index(details_path, name=f"{name} per-source review")


def _frame_bounds(span: dict[str, Any]) -> tuple[int, int]:
    if "start_frame" in span and "end_frame" in span:
        return int(span["start_frame"]), int(span["end_frame"])
    return round(float(span["start_s"]) / 0.02), round(float(span["end_s"]) / 0.02)


def _labels(spans: list[dict[str, Any]], *, frame_count: int, source_id: str) -> list[str]:
    labels = ["__unlabeled__"] * frame_count
    for span in spans:
        label = str(span.get("label") or "")
        if label not in {"inside_candidate", "outside_candidate", "unsure"}:
            raise ValueError(f"unsupported final label for {source_id}: {label}")
        start, end = _frame_bounds(span)
        if not 0 <= start < end <= frame_count:
            raise ValueError(f"invalid final span for {source_id}: {start}..{end}")
        if any(value != "__unlabeled__" for value in labels[start:end]):
            raise ValueError(f"overlapping final spans for {source_id}")
        labels[start:end] = [label] * (end - start)
    if "__unlabeled__" in labels:
        raise ValueError(f"final labels must cover the full source: {source_id}")
    return labels


def _difference_runs(left: list[str], right: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    active: tuple[str, str] | None = None
    start = 0
    for index, pair in enumerate([*zip(left, right), ("__end__", "__end__")]):
        changed = pair[0] != pair[1]
        if changed and active is None:
            active = pair
            start = index
        elif changed and pair != active:
            result.append(
                {
                    "label": f"{active[0]} -> {active[1]}",
                    "start_frame": start,
                    "end_frame": index,
                    "start_s": start * 0.02,
                    "end_s": index * 0.02,
                }
            )
            active = pair
            start = index
        elif not changed and active is not None:
            result.append(
                {
                    "label": f"{active[0]} -> {active[1]}",
                    "start_frame": start,
                    "end_frame": index,
                    "start_s": start * 0.02,
                    "end_s": index * 0.02,
                }
            )
            active = None
    return result


def _arm_metrics(row: dict[str, Any]) -> dict[str, Any]:
    human_inside = int(row.get("human_inside_frames") or 0)
    unsafe = int(row.get("unsafe_outside_frames") or 0)
    return {
        "protect_recall": float(row.get("protect_recall") or 0.0),
        "final_outside_precision": float(row.get("final_outside_precision") or 0.0),
        "supervised_ratio": float(row.get("supervised_ratio") or 0.0),
        "conflict_frames": int(row.get("conflict_frames") or 0),
        "unsafe_outside_frames": unsafe,
        "true_speech_retention": (
            1.0 if human_inside == 0 else 1.0 - unsafe / human_inside
        ),
    }


def _aggregate_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    human_inside = int(summary.get("human_inside_frames") or 0)
    unsafe = int(summary.get("unsafe_outside_frames") or 0)
    return {
        "protect_recall": float(summary.get("protect_recall") or 0.0),
        "final_outside_precision": float(summary.get("final_outside_precision") or 0.0),
        "supervised_ratio": float(summary.get("supervised_ratio") or 0.0),
        "conflict_ratio": float(summary.get("conflict_ratio") or 0.0),
        "unsafe_outside_frames": unsafe,
        "unsafe_outside_s": float(summary.get("unsafe_outside_s") or 0.0),
        "true_speech_retention": (
            1.0 if human_inside == 0 else 1.0 - unsafe / human_inside
        ),
    }


def _page(
    rows: list[dict[str, Any]],
    *,
    base_name: str,
    candidate_name: str,
    base_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> str:
    encoded = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    base_json = json.dumps(base_name, ensure_ascii=False)
    candidate_json = json.dumps(candidate_name, ensure_ascii=False)
    verdict_schema = json.dumps(VERDICT_SCHEMA)
    contract = json.dumps(ACOUSTIC_BINARY_V12_CONTRACT.contract_id)

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    intro_html = f"""<section class="contract"><h2>High / Medium 对照合同</h2><p>本页复用 Human Audit Page Core；两臂先由同一个 dual-evidence Adapter 规范化，再在同一完整 source、同一人工真值和同一 20ms 帧网格上对照。点击任一色块只播放该精确区间；完整播放器用于上下文判断。</p><table><thead><tr><th>运行</th><th>人工真语音保留率（gate ≥95%）</th><th>最终 outside precision（gate ≥95%）</th><th>Protect recall（诊断）</th><th>监督覆盖</th><th>冲突</th></tr></thead><tbody><tr><td>{html.escape(base_name)}</td><td>{pct(base_metrics['true_speech_retention'])}</td><td>{pct(base_metrics['final_outside_precision'])}</td><td>{pct(base_metrics['protect_recall'])}</td><td>{pct(base_metrics['supervised_ratio'])}</td><td>{pct(base_metrics['conflict_ratio'])}</td></tr><tr><td>{html.escape(candidate_name)}</td><td>{pct(candidate_metrics['true_speech_retention'])}</td><td>{pct(candidate_metrics['final_outside_precision'])}</td><td>{pct(candidate_metrics['protect_recall'])}</td><td>{pct(candidate_metrics['supervised_ratio'])}</td><td>{pct(candidate_metrics['conflict_ratio'])}</td></tr></tbody></table><p><b>裁决选项：</b><code>base_better</code> / <code>candidate_better</code> 表示一臂在语音安全和可用监督之间整体更好；<code>equivalent_both_acceptable</code> 表示两者差异不影响职责且都可接受；<code>equivalent_both_unacceptable</code> 表示两者均有不可接受问题；<code>tradeoff_no_clear_winner</code> 表示各有明显优劣；<code>comparison_unsure</code> 表示听感不足以判断。六项覆盖 A/B 的全部有效结论。</p><p>人工真语音保留率和 outside precision 是当前发布门槛；Protect recall 只帮助定位为什么最终三态发生变化，不单独否决。斜纹轨为 {html.escape(candidate_name)}，纯色轨为 {html.escape(base_name)}。</p></section><div id="list"></div>"""
    adapter_css = r"""
.contract,article{background:#fff;border:1px solid #ccd6df;border-radius:10px;padding:14px;margin-bottom:14px}.contract{overflow-x:auto}.contract table{width:100%;border-collapse:collapse}.contract th,.contract td{border:1px solid #c9d3dc;padding:7px;text-align:left}.human-inside{background:#315f9d;color:#fff}.human-outside{background:#9ca6af;color:#fff}.human-unsure{background:#725190;color:#fff}.base-protect{background:#27a2c2;color:#fff}.base-remove{background:#e5bb2c;color:#1d1d1d}.candidate-protect{background:repeating-linear-gradient(135deg,#7652ad 0,#7652ad 7px,#9b79cf 7px,#9b79cf 14px);color:#fff}.candidate-remove{background:repeating-linear-gradient(135deg,#d85f8c 0,#d85f8c 7px,#ee96b5 7px,#ee96b5 14px);color:#fff}.base-final-inside{background:#258b57;color:#fff}.base-final-outside{background:#f2cf45;color:#1d1d1d}.base-final-unsure{background:#d87800;color:#fff}.candidate-final-inside{background:repeating-linear-gradient(135deg,#258b57 0,#258b57 7px,#5bad7d 7px,#5bad7d 14px);color:#fff}.candidate-final-outside{background:repeating-linear-gradient(135deg,#d9ae1d 0,#d9ae1d 7px,#f2cf45 7px,#f2cf45 14px);color:#1d1d1d}.candidate-final-unsure{background:repeating-linear-gradient(135deg,#b45d00 0,#b45d00 7px,#e69235 7px,#e69235 14px);color:#fff}.unsafe{background:#d32626;color:#fff}.candidate-unsafe{background:repeating-linear-gradient(135deg,#a90000 0,#a90000 7px,#e33e3e 7px,#e33e3e 14px);color:#fff}.difference{background:#222;color:#fff}.metrics{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0;font-size:12px}.verdicts{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}.choice{border:1px solid #8d99a5;border-radius:5px;background:#fff;padding:7px 9px;cursor:pointer}.choice.active{outline:3px solid #18212b;outline-offset:-2px;background:#cfe4f7}.note{width:100%;min-height:44px;margin-top:8px;box-sizing:border-box}.good{color:#087443}.bad{color:#b3261e;font-weight:700}
"""
    adapter_js = r"""
const rows=__ROWS__,baseName=__BASE_NAME__,candidateName=__CANDIDATE_NAME__,verdictSchema=__VERDICT_SCHEMA__,boundaryContract=__CONTRACT__;
const optionLabels={base_better:`${baseName} 更好`,candidate_better:`${candidateName} 更好`,equivalent_both_acceptable:'等价且两者可接受',equivalent_both_unacceptable:'等价但两者均不可接受',tradeoff_no_clear_winner:'各有优劣，无明确胜者',comparison_unsure:'不确定'};
const review=createAuditReviewCore({storageKey:'scorer-v11-dual-evidence-ab::'+location.pathname,entries:rows,entryId:row=>row.source_id,defaultState:()=>({comparison_verdict:'',note:'',updated_at:''}),isComplete:state=>Boolean(state.comparison_verdict),shouldSerialize:state=>Boolean(state.comparison_verdict||state.note),serialize:(row,state)=>({schema:verdictSchema,boundary_serialization_contract_id:boundaryContract,source_id:row.source_id,base_name:baseName,candidate_name:candidateName,comparison_verdict:state.comparison_verdict||'unreviewed',note:state.note||'',updated_at:state.updated_at||new Date().toISOString()}),filename:'manual_verdicts.jsonl',statusLabel:'A/B 裁决'});
function lane(card,audio,row,label,spans,className,metric=''){appendAuditSpanLane({container:card,audio,durationS:row.duration_s,label,metric,spans,className:span=>typeof className==='function'?className(span):className,title:(span,start,end)=>`${label} ${span.label||''} ${formatAuditSpan(start,end)}`,text:(span,start,end)=>`${span.label||''} ${formatAuditSpan(start,end)}`});}
function finalClass(prefix,span){return `${prefix}-final-${span.label==='inside_candidate'?'inside':span.label==='outside_candidate'?'outside':'unsure'}`;}
const root=document.getElementById('list');
for(const row of rows){const state=review.ensure(row),card=document.createElement('article');card.innerHTML=`<h2>${escapeAuditHtml(row.source_id)}</h2><small>${escapeAuditHtml(row.partition)} · ${formatAuditTimestamp(row.duration_s)} · final changed ${(100*row.changed_ratio).toFixed(2)}%</small><audio controls preload="metadata" src="${escapeAuditHtml(row.audio)}"></audio>`;const audio=card.querySelector('audio');lane(card,audio,row,'人工 truth',row.human_spans,span=>`human-${span.label==='inside_candidate'?'inside':span.label==='outside_candidate'?'outside':'unsure'}`,'蓝=真语音，灰=background');lane(card,audio,row,`${baseName} Protect`,row.base.protect_spans,'base-protect',`recall ${(100*row.base.metrics.protect_recall).toFixed(2)}%`);lane(card,audio,row,`${baseName} Remove`,row.base.remove_spans,'base-remove');lane(card,audio,row,`${baseName} 最终三态`,row.base.final_spans,span=>finalClass('base',span),`speech retain ${(100*row.base.metrics.true_speech_retention).toFixed(2)}% · outside P ${(100*row.base.metrics.final_outside_precision).toFixed(2)}%`);lane(card,audio,row,`${candidateName} Protect`,row.candidate.protect_spans,'candidate-protect',`recall ${(100*row.candidate.metrics.protect_recall).toFixed(2)}%`);lane(card,audio,row,`${candidateName} Remove`,row.candidate.remove_spans,'candidate-remove');lane(card,audio,row,`${candidateName} 最终三态`,row.candidate.final_spans,span=>finalClass('candidate',span),`speech retain ${(100*row.candidate.metrics.true_speech_retention).toFixed(2)}% · outside P ${(100*row.candidate.metrics.final_outside_precision).toFixed(2)}%`);lane(card,audio,row,`${baseName} 真语音被 outside`,row.base.unsafe_outside_spans,'unsafe',`${row.base.metrics.unsafe_outside_frames} frames`);lane(card,audio,row,`${candidateName} 真语音被 outside`,row.candidate.unsafe_outside_spans,'candidate-unsafe',`${row.candidate.metrics.unsafe_outside_frames} frames`);lane(card,audio,row,'最终标签差异',row.difference_spans,'difference',`${row.changed_frames} frames`);const metrics=document.createElement('div');metrics.className='metrics';metrics.innerHTML=`<span>${baseName}: supervised ${(100*row.base.metrics.supervised_ratio).toFixed(1)}%, conflicts ${row.base.metrics.conflict_frames}</span><span>${candidateName}: supervised ${(100*row.candidate.metrics.supervised_ratio).toFixed(1)}%, conflicts ${row.candidate.metrics.conflict_frames}</span>`;card.appendChild(metrics);const controls=document.createElement('div');controls.className='verdicts';for(const [value,label] of Object.entries(optionLabels)){const button=document.createElement('button');button.type='button';button.className='choice';button.dataset.value=value;button.textContent=label;button.onclick=()=>{state.comparison_verdict=value;state.updated_at=new Date().toISOString();sync();review.persist();};controls.appendChild(button);}card.appendChild(controls);const note=document.createElement('textarea');note.className='note';note.placeholder='可选备注：指出更安全/更有用的差异区间';note.value=state.note||'';note.onchange=event=>{state.note=event.target.value;state.updated_at=new Date().toISOString();review.persist();};card.appendChild(note);function sync(){controls.querySelectorAll('.choice').forEach(button=>button.classList.toggle('active',button.dataset.value===state.comparison_verdict));}sync();root.appendChild(card);}
document.getElementById('stop').onclick=stop;document.getElementById('save').onclick=review.save;review.updateStatus();
"""
    adapter_js = (
        adapter_js.replace("__ROWS__", encoded)
        .replace("__BASE_NAME__", base_json)
        .replace("__CANDIDATE_NAME__", candidate_json)
        .replace("__VERDICT_SCHEMA__", verdict_schema)
        .replace("__CONTRACT__", contract)
    )
    return render_audit_review_page(
        AuditReviewPageSpec(
            title=f"Scorer v11 dual-evidence A/B · {base_name} vs {candidate_name}",
            intro_html=intro_html,
            body_html="",
            adapter_css=adapter_css,
            adapter_js=adapter_js,
        )
    )


def generate(
    *,
    base_review: Path,
    candidate_review: Path,
    output_dir: Path,
    base_name: str = "Medium",
    candidate_name: str = "High",
    update_nav: bool = True,
) -> dict[str, Any]:
    base_path, base_summary, base_rows = _load_review(base_review, name=base_name)
    candidate_path, candidate_summary, candidate_rows = _load_review(
        candidate_review,
        name=candidate_name,
    )
    for field in ("manifest_sha256", "human_verdicts_sha256"):
        if str(base_summary.get(field) or "") != str(candidate_summary.get(field) or ""):
            raise ValueError(f"A/B reviews use different {field}")
    if set(base_rows) != set(candidate_rows):
        raise ValueError("A/B reviews must contain the same source ids")

    details: list[dict[str, Any]] = []
    changed_frames_total = 0
    for source_id, base in base_rows.items():
        candidate = candidate_rows[source_id]
        frame_count = int(base["frame_count"])
        if int(candidate["frame_count"]) != frame_count:
            raise ValueError(f"A/B frame geometry mismatch: {source_id}")
        if abs(float(base["duration_s"]) - float(candidate["duration_s"])) > 1e-6:
            raise ValueError(f"A/B duration mismatch: {source_id}")
        if str(base.get("audio") or "") != str(candidate.get("audio") or ""):
            raise ValueError(f"A/B audio identity mismatch: {source_id}")
        if base.get("human_spans") != candidate.get("human_spans"):
            raise ValueError(f"A/B human truth mismatch: {source_id}")
        left = _labels(
            list(base.get("final_spans") or ()),
            frame_count=frame_count,
            source_id=source_id,
        )
        right = _labels(
            list(candidate.get("final_spans") or ()),
            frame_count=frame_count,
            source_id=source_id,
        )
        difference_spans = _difference_runs(left, right)
        changed_frames = sum(left_label != right_label for left_label, right_label in zip(left, right))
        changed_frames_total += changed_frames
        details.append(
            {
                "schema": DETAIL_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "partition": str(base.get("partition") or ""),
                "duration_s": float(base["duration_s"]),
                "frame_count": frame_count,
                "audio": str(base["audio"]),
                "human_spans": list(base.get("human_spans") or ()),
                "base": {
                    "protect_spans": list(base.get("protect_spans") or ()),
                    "remove_spans": list(base.get("remove_spans") or ()),
                    "final_spans": list(base.get("final_spans") or ()),
                    "unsafe_outside_spans": list(base.get("unsafe_outside_spans") or ()),
                    "metrics": _arm_metrics(base),
                },
                "candidate": {
                    "protect_spans": list(candidate.get("protect_spans") or ()),
                    "remove_spans": list(candidate.get("remove_spans") or ()),
                    "final_spans": list(candidate.get("final_spans") or ()),
                    "unsafe_outside_spans": list(candidate.get("unsafe_outside_spans") or ()),
                    "metrics": _arm_metrics(candidate),
                },
                "difference_spans": difference_spans,
                "changed_frames": changed_frames,
                "changed_ratio": changed_frames / max(frame_count, 1),
            }
        )
    details.sort(
        key=lambda row: (
            -int(row["changed_frames"]),
            -int(row["candidate"]["metrics"]["unsafe_outside_frames"]),
            str(row["source_id"]),
        )
    )
    frame_count = sum(int(row["frame_count"]) for row in details)
    base_metrics = _aggregate_metrics(base_summary)
    candidate_metrics = _aggregate_metrics(candidate_summary)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "per_source.jsonl"
    detail_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in details
        ),
        encoding="utf-8",
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "base_name": base_name,
        "candidate_name": candidate_name,
        "base_review": str(base_path),
        "base_review_sha256": _sha256(base_path),
        "candidate_review": str(candidate_path),
        "candidate_review_sha256": _sha256(candidate_path),
        "manifest_sha256": str(base_summary.get("manifest_sha256") or ""),
        "human_verdicts_sha256": str(base_summary.get("human_verdicts_sha256") or ""),
        "source_count": len(details),
        "frame_count": frame_count,
        "changed_frames": changed_frames_total,
        "changed_ratio": changed_frames_total / max(frame_count, 1),
        "changed_source_count": sum(bool(row["changed_frames"]) for row in details),
        "base_metrics": base_metrics,
        "candidate_metrics": candidate_metrics,
        "true_speech_retention_gate": 0.95,
        "final_outside_precision_gate": 0.95,
        "protect_recall_is_diagnostic_only": True,
        "manual_verdicts": str(output_dir / "manual_verdicts.jsonl"),
        "manual_verdict_schema": VERDICT_SCHEMA,
        "per_source": str(detail_path),
        "training_manifest_allowed": False,
        "audit_navigation_updated": update_nav,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    index = output_dir / "index.html"
    index.write_text(
        _page(
            details,
            base_name=base_name,
            candidate_name=candidate_name,
            base_metrics=base_metrics,
            candidate_metrics=candidate_metrics,
        ),
        encoding="utf-8",
    )
    if update_nav:
        update_audit_entrypoints(
            latest_html=index,
            title=f"Scorer v11 dual-evidence A/B · {base_name} vs {candidate_name}",
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-review", required=True)
    parser.add_argument("--candidate-review", required=True)
    parser.add_argument("--base-name", default="Medium")
    parser.add_argument("--candidate-name", default="High")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--update-nav",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            generate(
                base_review=Path(args.base_review),
                candidate_review=Path(args.candidate_review),
                base_name=args.base_name,
                candidate_name=args.candidate_name,
                output_dir=Path(args.output_dir),
                update_nav=args.update_nav,
            ),
            ensure_ascii=False,
        )
    )
