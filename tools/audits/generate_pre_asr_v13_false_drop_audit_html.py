#!/usr/bin/env python3
"""Generate a precise whole-subisland audit page for CueQC v13 false drops."""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from tools.asr.cueqc.label_pre_asr_with_omni import slice_audio_clip  # noqa: E402
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


SUMMARY_SCHEMA = "cueqc_v13_false_drop_audit_page_summary_v2"
MANUAL_VERDICT_SCHEMA = "cueqc_v13_false_drop_manual_verdict_v1"
DEFAULT_REVIEW_PROMPT = """每条音频是 CueQC 模型准备整块删除的完整 provisional sub-island。判断整块是否含任何应继续送 Inner 和 ASR 的 acoustic semantic core；不要按 ASR 是否能转写、片段时长或声音类别自动裁决。只要包含真实词语、词语残片、交流性应答或有语义发声，就属于 true_speech；确认全块只有非语义呻吟、喘息、呼吸、噪声、音乐、动作声或静音时，才属于 safe_drop。听不清或职责边界不确定时选择 unsure。"""
CUEQC_FALSE_DROP_AXES = (
    AuditOptionAxis(
        field="verdict",
        options=("safe_drop", "true_speech", "unsure"),
    ),
)
CUEQC_FALSE_DROP_RESULTS = {
    ("safe_drop",): "safe_drop",
    ("true_speech",): "true_speech",
    ("unsure",): "unsure",
}
validate_audit_option_contract(
    axes=CUEQC_FALSE_DROP_AXES,
    combination_results=CUEQC_FALSE_DROP_RESULTS,
)


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "item"


def _page(rows: list[dict[str, Any]], *, review_prompt: str) -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    prompt_html = html.escape(review_prompt).replace("\n", "<br>")
    intro_html = f"""<section class="contract"><h2>审计结构与选项意义</h2><div class="prompt"><b>本页审计提示</b><p>{prompt_html}</p></div><p>这是 <b>prediction drop &amp; truth keep</b> 的零误删 gate。每条播放器都是模型将删除的完整 provisional sub-island；CueQC 只判断整块 keep/drop，不负责 Split 切句，也不负责 Inner 的声学首尾裁边。</p><table><thead><tr><th>选项</th><th>完整含义</th><th>Gate 结果</th></tr></thead><tbody><tr><td><code>safe_drop</code></td><td>确认整块不含任何应送 Inner/ASR 的 acoustic semantic core；可以包含呻吟、喘息、呼吸、静音、音乐、噪声或动作声。</td><td>模型删除正确。</td></tr><tr><td><code>true_speech</code></td><td>存在任意真实词语、词语残片、交流性应答、句尾或其他应进入 ASR 的有语义发声。</td><td>真语音误删，zero-deletion gate 失败。</td></tr><tr><td><code>unsure</code></td><td>无法可靠判断是否存在 semantic core，或职责边界仍有歧义。</td><td>保持未闭合，禁止晋升。</td></tr></tbody></table><p>三项覆盖该审计目标的全部结果，且互斥：确认无 semantic core、确认存在 semantic core、无法确认。页面不根据概率、时长、ASR 文本或声音类别自动选择。</p></section>"""
    adapter_css = """
.contract,article{background:#fff;border:1px solid #ccd6df;border-radius:10px;padding:14px;margin-bottom:14px}.prompt{background:#eef6ff;border-left:5px solid #315f9d;padding:10px 12px;margin:10px 0}.contract table{width:100%;border-collapse:collapse}.contract th,.contract td{border:1px solid #c9d3dc;padding:7px;text-align:left;vertical-align:top}.contract th{background:#edf1f5}article.done{border-left:6px solid #258b57}.meta{color:#607080;overflow-wrap:anywhere}.controls{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin:8px 0}.choice,.play-whole{border:1px solid #8d99a5;border-radius:5px;background:#fff;padding:7px 10px;cursor:pointer}.choice.active{outline:3px solid #18212b;outline-offset:-2px}.choice[data-value="safe_drop"].active{background:#bfe5cc}.choice[data-value="true_speech"].active{background:#f4b8b4}.choice[data-value="unsure"].active{background:#f3d49d}.note{width:100%;min-height:48px;box-sizing:border-box}
"""
    adapter_js = r"""
const rows=__ROWS__,verdictSchema=__VERDICT_SCHEMA__,boundaryContract=__BOUNDARY_CONTRACT__;
const allowedVerdicts=new Set(['safe_drop','true_speech','unsure']);
const reviewCore=createAuditReviewCore({storageKey:'cueqc-v13-false-drop-audit-v2:'+location.pathname,entries:rows,entryId:row=>row.row_id,defaultState:()=>({verdict:'',note:''}),isComplete:state=>allowedVerdicts.has(state.verdict),statusLabel:'CueQC false-drop 裁决',filename:'manual_verdicts.jsonl',serialize:(row,state)=>({schema:verdictSchema,boundary_serialization_contract_id:boundaryContract,row_id:row.row_id,source_partition:row.source_partition||'',start_s:Number(row.start_s),end_s:Number(row.end_s),verdict:state.verdict||'unreviewed',note:state.note||'',updated_at:state.updated_at||new Date().toISOString()})});
function sync(card,state){card.classList.toggle('done',allowedVerdicts.has(state.verdict));card.querySelectorAll('[data-value]').forEach(button=>button.classList.toggle('active',button.dataset.value===state.verdict));}
const root=document.getElementById('list');for(const row of rows){const state=reviewCore.ensure(row),card=document.createElement('article');card.innerHTML=`<h2>${escapeAuditHtml(row.row_id)}</h2><div class="meta">${escapeAuditHtml(row.source_partition||'')} · source ${Number(row.start_s).toFixed(3)}–${Number(row.end_s).toFixed(3)}s · clip ${Number(row.clip_duration_s).toFixed(3)}s · p(drop)=${Number(row.prob_drop).toFixed(4)} · teacher=${escapeAuditHtml(row.teacher_label||'-')} · exact=${escapeAuditHtml(row.exact_core_label||'-')}</div><audio controls preload="metadata" src="${escapeAuditHtml(row.audio_src)}"></audio><div class="controls"><button type="button" class="play-whole">直接播放完整 sub-island</button><button type="button" class="choice" data-value="safe_drop">安全删除：无 semantic core</button><button type="button" class="choice" data-value="true_speech">真语音误删：含 semantic core</button><button type="button" class="choice" data-value="unsure">不确定</button></div><textarea class="note" placeholder="可选：记录听到的词语、应答或不确定原因">${escapeAuditHtml(state.note||'')}</textarea>`;const audio=card.querySelector('audio'),playButton=card.querySelector('.play-whole');playButton.onclick=()=>play(audio,playButton,0,Number(row.clip_duration_s));card.querySelectorAll('[data-value]').forEach(button=>button.onclick=()=>{state.verdict=button.dataset.value;state.updated_at=new Date().toISOString();sync(card,state);reviewCore.persist();});card.querySelector('.note').onchange=event=>{state.note=event.target.value;state.updated_at=new Date().toISOString();reviewCore.persist();};sync(card,state);root.appendChild(card);}
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
            title="CueQC v13 · 全量 false-drop 人工 gate",
            intro_html=intro_html,
            body_html='<div id="list"></div>',
            adapter_css=adapter_css,
            adapter_js=adapter_js,
        )
    )


def build(
    *,
    false_drop_manifest: Path,
    output_dir: Path,
    update_latest: bool = True,
    review_prompt: ResolvedAuditPrompt | None = None,
) -> dict[str, Any]:
    resolved_prompt = review_prompt or resolve_audit_prompt(
        default_prompt=DEFAULT_REVIEW_PROMPT,
    )
    source_rows = _rows(false_drop_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in source_rows:
        row_id = str(source.get("row_id") or "")
        if not row_id or row_id in seen:
            raise ValueError("false-drop row_id values must be non-empty and unique")
        seen.add(row_id)
        audio = Path(str(source.get("audio") or ""))
        if not audio.is_file():
            raise FileNotFoundError(f"false-drop source audio not found: {audio}")
        start = float(source.get("start_s") or 0.0)
        end = float(source.get("end_s") or 0.0)
        if end <= start:
            raise ValueError(f"false-drop interval must be positive: {row_id}")
        clip = media_dir / f"{_safe_name(row_id)}.mp3"
        slice_audio_clip(
            source_audio=audio,
            row={"start": start, "end": end, "duration_s": end - start},
            output_path=clip,
            fmt="mp3",
            bitrate="64k",
            sample_rate=16000,
            force=False,
        )
        rows.append(
            {
                **source,
                "audio_src": clip.relative_to(output_dir).as_posix(),
                "clip_duration_s": end - start,
            }
        )
    page = output_dir / "index.html"
    page.write_text(
        _page(rows, review_prompt=resolved_prompt.text),
        encoding="utf-8",
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "item_count": len(rows),
        "playback_contract": "exact_whole_subisland_v1",
        "false_drop_manifest": str(false_drop_manifest),
        "manual_verdicts": str(output_dir / "manual_verdicts.jsonl"),
        "manual_verdict_schema": MANUAL_VERDICT_SCHEMA,
        "manual_gate_status": "pending",
        "training_manifest_allowed": False,
        "review_prompt_source": resolved_prompt.source,
        "review_prompt_sha256": resolved_prompt.sha256,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if update_latest:
        update_audit_entrypoints(
            latest_html=page, title="CueQC v13 False-drop Audit"
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--false-drop-manifest", required=True)
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
    print(
        json.dumps(
            build(
                false_drop_manifest=Path(args.false_drop_manifest),
                output_dir=Path(args.output_dir),
                update_latest=not args.no_update_latest,
                review_prompt=prompt,
            ),
            ensure_ascii=False,
        )
    )
