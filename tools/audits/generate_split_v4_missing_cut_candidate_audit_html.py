#!/usr/bin/env python3
"""Generate candidate-level correction audit for manually confirmed Split v4 residual misses."""
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.audits.audit_prompt import (  # noqa: E402
    ResolvedAuditPrompt,
    resolve_audit_prompt,
)
from tools.audits.review_page_core import (  # noqa: E402
    AuditOptionAxis,
    AuditReviewPageSpec,
    render_audit_review_page,
    validate_audit_option_contract,
)


SUMMARY_SCHEMA = "split_v4_missing_cut_candidate_audit_v2"
MANUAL_VERDICT_SCHEMA = "split_v4_missing_cut_candidate_manual_verdict_v1"
DEFAULT_REVIEW_PROMPT = """这些 residual 已由上一层人工审计确认至少存在一个漏切。现在只判断每个真实 candidate 查询点应为 cut、continue 还是 unsure。先听完整 residual，再分别听 candidate 左侧、右侧和左右合并波形；如果 candidate 位于两个不同目标事件之间且切开不会截断任何一侧，标 cut。同一目标事件内部的停顿、呼吸、呻吟或动作声仍标 continue。无法可靠判断时标 unsure，并在训练中映射为 ignore=-100。"""
SPLIT_CANDIDATE_AXES = (
    AuditOptionAxis(
        field="manual_label",
        options=("cut", "continue", "unsure"),
    ),
)
SPLIT_CANDIDATE_RESULTS = {
    ("cut",): "cut",
    ("continue",): "continue",
    ("unsure",): "unsure",
}
validate_audit_option_contract(
    axes=SPLIT_CANDIDATE_AXES,
    combination_results=SPLIT_CANDIDATE_RESULTS,
)


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _page(rows: list[dict[str, Any]], *, review_prompt: str) -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    prompt_html = html.escape(review_prompt).replace("\n", "<br>")
    intro_html = f"""<section class="contract"><h2>审计结构与选项意义</h2><div class="prompt"><b>本页审计提示</b><p>{prompt_html}</p></div><p>本页只处理已经确认存在漏切的 residual，并为其中每个真实 Proposal candidate 生成 Split 二分类监督。它不新增 candidate，也不使用固定时长或概率阈值决定标签。</p><table><thead><tr><th>选项</th><th>完整含义</th><th>训练映射</th></tr></thead><tbody><tr><td><code>cut</code></td><td>candidate 两侧属于应独立送往下游的不同目标事件，且当前切点不会截断词头、词尾或同一事件。</td><td><code>cut</code></td></tr><tr><td><code>continue</code></td><td>candidate 位于同一目标事件内部；中间即使有短停顿、呼吸、呻吟或动作声，也不应由 Split 切开。</td><td><code>continue</code></td></tr><tr><td><code>unsure</code></td><td>无法可靠判断事件关系或边界安全性。</td><td><code>ignore=-100</code></td></tr></tbody></table><p>三项覆盖单个 candidate 查询的全部结果。每行提供左侧、右侧和两侧合并播放；这些区间由相邻真实 candidate 与 residual 边界决定，不向模型或人工偷偷加入固定秒数上下文。</p></section>"""
    adapter_css = """
.contract,article{background:#fff;border:1px solid #ccd6df;border-radius:10px;padding:14px;margin-bottom:14px}.prompt{background:#eef6ff;border-left:5px solid #315f9d;padding:10px 12px;margin:10px 0}.contract table,.candidate-table{width:100%;border-collapse:collapse}.contract th,.contract td,.candidate-table th,.candidate-table td{border:1px solid #c9d3dc;padding:7px;text-align:left;vertical-align:top}.contract th,.candidate-table th{background:#edf1f5}article.done{border-left:6px solid #258b57}.meta{color:#607080}.play-controls,.choices{display:flex;gap:6px;flex-wrap:wrap}.play-controls button,.choice{border:1px solid #8d99a5;border-radius:5px;background:#fff;padding:6px 9px;cursor:pointer}.choice.active{outline:3px solid #18212b;outline-offset:-2px}.choice[data-value="cut"].active{background:#f4b8b4}.choice[data-value="continue"].active{background:#bfe5cc}.choice[data-value="unsure"].active{background:#f3d49d}.note{width:100%;min-height:48px;margin-top:8px;box-sizing:border-box}@media(max-width:850px){.candidate-table,.candidate-table tbody,.candidate-table tr,.candidate-table td{display:block}.candidate-table thead{display:none}.candidate-table tr{border:1px solid #c9d3dc;margin:8px 0}.candidate-table td{border:0}}
"""
    adapter_js = r"""
const rows=__ROWS__,verdictSchema=__VERDICT_SCHEMA__,boundaryContract=__BOUNDARY_CONTRACT__;
const allowedLabels=new Set(['cut','continue','unsure']);
function complete(row,state){return row.residual_candidates.every(candidate=>allowedLabels.has(state.labels[candidate.candidate_id]));}
const reviewCore=createAuditReviewCore({storageKey:'split-v4-missing-cut-candidate-audit-v2:'+location.pathname,entries:rows,entryId:row=>row.audit_id,defaultState:()=>({labels:{},note:''}),isComplete:(state,row)=>complete(row,state),statusLabel:'Split residual 补标',filename:'manual_verdicts.jsonl',serialize:(row,state)=>({schema:verdictSchema,boundary_serialization_contract_id:boundaryContract,audit_id:row.audit_id,audio_id:row.audio_id,candidates:row.residual_candidates.map(candidate=>({candidate_id:candidate.candidate_id,time_s:candidate.time_s,p_cut:candidate.p_cut,manual_label:state.labels[candidate.candidate_id]||'unreviewed'})),complete:complete(row,state),note:state.note||'',updated_at:state.updated_at||new Date().toISOString()})});
function sync(card,row,state){card.classList.toggle('done',complete(row,state));card.querySelectorAll('[data-candidate-id][data-value]').forEach(button=>button.classList.toggle('active',state.labels[button.dataset.candidateId]===button.dataset.value));}
function playButton(label,start,end,className=''){return `<button type="button" class="${className}" data-play-start="${start}" data-play-end="${end}">${label} ${Number(start).toFixed(3)}–${Number(end).toFixed(3)}s</button>`;}
const root=document.getElementById('list');for(const row of rows){const state=reviewCore.ensure(row),card=document.createElement('article'),candidates=row.residual_candidates||[];const candidateRows=candidates.map((candidate,index)=>{const previous=index===0?Number(row.start_s):Number(candidates[index-1].time_s),time=Number(candidate.time_s),next=index+1<candidates.length?Number(candidates[index+1].time_s):Number(row.end_s);return `<tr><td><b>${escapeAuditHtml(candidate.candidate_id)}</b><br><span class="meta">t=${time.toFixed(3)}s · p_cut=${Number(candidate.p_cut).toFixed(4)} · model=${escapeAuditHtml(candidate.model_label||'-')}</span></td><td><div class="play-controls">${playButton('左侧',previous,time)}${playButton('右侧',time,next)}${playButton('左右合并',previous,next)}</div></td><td><div class="choices"><button type="button" class="choice" data-candidate-id="${escapeAuditHtml(candidate.candidate_id)}" data-value="cut">cut：不同目标事件</button><button type="button" class="choice" data-candidate-id="${escapeAuditHtml(candidate.candidate_id)}" data-value="continue">continue：同一目标事件</button><button type="button" class="choice" data-candidate-id="${escapeAuditHtml(candidate.candidate_id)}" data-value="unsure">unsure</button></div></td></tr>`;}).join('');card.innerHTML=`<h2>${escapeAuditHtml(row.audit_id)} · ${escapeAuditHtml(row.audio_id)}</h2><div class="meta">residual ${Number(row.start_s).toFixed(3)}–${Number(row.end_s).toFixed(3)}s · ${Number(row.duration_s).toFixed(3)}s · ${candidates.length} candidates</div><audio controls preload="metadata" src="${escapeAuditHtml(row.audio_src)}"></audio><div class="play-controls">${playButton('播放完整 chunk',row.core_start,row.core_end,'play-full')}${playButton('播放完整 missing residual',row.start_s,row.end_s,'play-residual')}</div><table class="candidate-table"><thead><tr><th>candidate query</th><th>边界两侧精确播放</th><th>人工二分类标签</th></tr></thead><tbody>${candidateRows}</tbody></table><textarea class="note" placeholder="可选：记录具体对白关系、切点风险或 unsure 原因">${escapeAuditHtml(state.note||'')}</textarea>`;const audio=card.querySelector('audio');card.querySelectorAll('[data-play-start]').forEach(button=>button.onclick=()=>play(audio,button,Number(button.dataset.playStart),Number(button.dataset.playEnd)));card.querySelectorAll('[data-candidate-id][data-value]').forEach(button=>button.onclick=()=>{state.labels[button.dataset.candidateId]=button.dataset.value;state.updated_at=new Date().toISOString();sync(card,row,state);reviewCore.persist();});card.querySelector('.note').onchange=event=>{state.note=event.target.value;state.updated_at=new Date().toISOString();reviewCore.persist();};sync(card,row,state);root.appendChild(card);}
document.getElementById('stop').onclick=()=>{stop();reviewCore.updateStatus('已停止');};document.getElementById('save').onclick=()=>reviewCore.save();reviewCore.updateStatus();
"""
    adapter_js = (
        adapter_js.replace("__ROWS__", payload)
        .replace("__VERDICT_SCHEMA__", json.dumps(MANUAL_VERDICT_SCHEMA))
        .replace(
            "__BOUNDARY_CONTRACT__",
            json.dumps(ACOUSTIC_BINARY_V12_CONTRACT.contract_id),
        )
    )
    return render_audit_review_page(
        AuditReviewPageSpec(
            title="Acoustic Split v4 · Missing-cut candidate 补标",
            intro_html=intro_html,
            body_html='<div id="list"></div>',
            adapter_css=adapter_css,
            adapter_js=adapter_js,
        )
    )


def build(
    *,
    source_dir: Path,
    verdict_paths: list[Path],
    output_dir: Path,
    review_prompt: ResolvedAuditPrompt | None = None,
    update_latest: bool = True,
) -> dict:
    resolved_prompt = review_prompt or resolve_audit_prompt(
        default_prompt=DEFAULT_REVIEW_PROMPT,
    )
    verdicts: dict[str, dict] = {}
    for path in verdict_paths:
        for row in _rows(path):
            if row.get("verdict") not in {None, "", "unreviewed"}:
                verdicts[str(row["audit_id"])] = row
    selected = [
        row
        for row in _rows(source_dir / "audit_manifest.jsonl")
        if verdicts.get(str(row["audit_id"]), {}).get("verdict") == "missing_cut"
    ]
    if not selected:
        raise ValueError("no manually confirmed missing-cut residuals found")
    missing_candidate_ids = [
        str(row["audit_id"])
        for row in selected
        if not row.get("residual_candidates")
    ]
    if missing_candidate_ids:
        raise ValueError(
            "missing-cut residual has no eligible binary candidates; "
            "classify it as a Proposal candidate-coverage failure instead: "
            f"{missing_candidate_ids}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    copied: set[str] = set()
    for row in selected:
        filename = Path(str(row["audio_src"])).name
        if filename not in copied:
            shutil.copyfile(source_dir / "audio" / filename, audio_dir / filename)
            copied.add(filename)
    (output_dir / "candidate_items.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        "utf-8",
    )
    (output_dir / "index.html").write_text(
        _page(selected, review_prompt=resolved_prompt.text),
        "utf-8",
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "residual_count": len(selected),
        "candidate_count": sum(len(row["residual_candidates"]) for row in selected),
        "manual_verdicts": str(output_dir / "manual_verdicts.jsonl"),
        "manual_verdict_schema": MANUAL_VERDICT_SCHEMA,
        "training_manifest_allowed": False,
        "review_prompt_source": resolved_prompt.source,
        "review_prompt_sha256": resolved_prompt.sha256,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")
    if update_latest:
        update_audit_entrypoints(
            latest_html=output_dir / "index.html",
            title="Acoustic Split v4 Missing-cut Candidate Audit",
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit-dir", required=True)
    parser.add_argument("--verdicts", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    prompt = resolve_audit_prompt(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        default_prompt=DEFAULT_REVIEW_PROMPT,
    )
    print(json.dumps(build(
        source_dir=Path(args.source_audit_dir),
        verdict_paths=[Path(path) for path in args.verdicts],
        output_dir=Path(args.output_dir),
        review_prompt=prompt,
        update_latest=not args.no_update_latest,
    ), ensure_ascii=False))
