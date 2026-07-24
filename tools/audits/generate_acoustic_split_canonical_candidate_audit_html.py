#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any
import wave


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
from tools.boundary.ja.acoustic_split_teacher_contracts import (  # noqa: E402
    ACOUSTIC_SPLIT_AUDIT_SUMMARY_SCHEMA,
    ACOUSTIC_SPLIT_MANUAL_VERDICT_SCHEMA,
)


DEFAULT_REVIEW_PROMPT = """本页复核 Split canonical teacher 对单个真实 candidate query 的二分类标签。先播放完整 query，再分别播放 candidate 左侧与右侧。若 candidate 两侧属于应独立送往下游的不同目标事件，并且在此处切开不会截断任何一侧，标 cut；若 candidate 位于同一目标事件内部，即使包含短停顿、呼吸、呻吟或动作声，也标 continue；无法可靠判断时标 unsure，训练映射为 ignore=-100。"""
SPLIT_CANONICAL_AXES = (
    AuditOptionAxis(field="verdict", options=("cut", "continue", "unsure")),
)
SPLIT_CANONICAL_RESULTS = {
    ("cut",): "cut",
    ("continue",): "continue",
    ("unsure",): "unsure",
}
validate_audit_option_contract(
    axes=SPLIT_CANONICAL_AXES,
    combination_results=SPLIT_CANONICAL_RESULTS,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _candidate_id(row: dict[str, Any]) -> str:
    return f"{row['window_id']}#f{int(row['feature_index']):05d}"


def _priority(row: dict[str, Any]) -> tuple[int, float, str]:
    if row.get("expected_gate_label"):
        rank = 0
    elif row.get("current_label") != row.get("label"):
        rank = 1
    elif row.get("legacy_label") and row.get("legacy_label") != row.get("label"):
        rank = 2
    elif row.get("label") == "cut":
        rank = 3
    elif float(row.get("confidence") or 0.0) < 0.9:
        rank = 4
    else:
        rank = 5
    return rank, float(row.get("confidence") or 0.0), _candidate_id(row)


def _requested_clip_bounds(
    *,
    row: dict[str, Any],
    center_s: float,
    duration_s: float,
    context_s: float,
) -> tuple[float, float]:
    """Use the exact teacher request when valid, otherwise a centered fallback."""

    try:
        start = float(row["request_clip_start_s"])
        end = float(row["request_clip_end_s"])
    except (KeyError, TypeError, ValueError):
        start = math.nan
        end = math.nan
    if (
        math.isfinite(start)
        and math.isfinite(end)
        and 0.0 <= start < center_s < end <= duration_s
    ):
        return start, end
    return max(0.0, center_s - context_s), min(duration_s, center_s + context_s)


def _slice_context(
    *,
    source: Path,
    output: Path,
    center_s: float,
    duration_s: float,
    context_s: float,
    row: dict[str, Any],
) -> tuple[float, float, float]:
    start, end = _requested_clip_bounds(
        row=row,
        center_s=center_s,
        duration_s=duration_s,
        context_s=context_s,
    )
    if end <= start:
        raise ValueError(f"invalid audit clip bounds: {start=} {end=}")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-i",
            str(source),
            "-t",
            f"{end - start:.6f}",
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output),
        ],
        check=True,
    )
    with wave.open(str(output), "rb") as handle:
        sample_rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        frame_count = int(handle.getnframes())
    if sample_rate != 16000 or channels != 1 or frame_count <= 0:
        raise ValueError(f"invalid materialized audit WAV: {output}")
    materialized_end = start + frame_count / sample_rate
    if center_s > materialized_end + 1e-6:
        raise ValueError("materialized audit clip no longer contains candidate query")
    return start, materialized_end, center_s - start


def _page(rows: list[dict[str, Any]], *, review_prompt: str) -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    prompt_html = html.escape(review_prompt).replace("\n", "<br>")
    intro_html = f"""<section class="contract"><h1>审计结构与选项意义</h1><div class="prompt"><b>本页审计提示</b><p>{prompt_html}</p></div><p>审计对象是一个固定的 Proposal candidate query，不负责新增 candidate，也不按句子、时长或概率阈值切分。页面播放的 WAV、标尺和 candidate 红线均来自 manifest 中同一组实际物化坐标。</p><table><thead><tr><th>选项</th><th>完整含义</th><th>训练映射</th></tr></thead><tbody><tr><td><code>cut</code></td><td>candidate 两侧是不同目标事件，且当前切点安全。</td><td><code>cut</code></td></tr><tr><td><code>continue</code></td><td>candidate 位于同一目标事件内部，不应在此切开。</td><td><code>continue</code></td></tr><tr><td><code>unsure</code></td><td>事件关系或边界安全性无法可靠判断。</td><td><code>ignore=-100</code></td></tr></tbody></table><p>三项覆盖单个 candidate query 的全部人工结果。只有已选择其中一项的条目会写入 <code>manual_verdicts.jsonl</code>。</p><div class="filters"><label>显示 <select id="filter"><option value="all">全部</option><option value="todo">未审</option><option value="disagree">teacher 与 current 不一致</option></select></label></div></section>"""
    adapter_css = """
.contract,article{background:#fff;border:1px solid #ccd6df;border-radius:10px;padding:14px;margin-bottom:14px}.prompt{background:#eef6ff;border-left:5px solid #315f9d;padding:10px 12px;margin:10px 0}.contract table{width:100%;border-collapse:collapse}.contract th,.contract td{border:1px solid #c9d3dc;padding:7px;text-align:left}.contract th{background:#edf1f5}.filters{margin-top:12px}article.done{border-left:6px solid #258b57}.meta,.labels,.play-controls,.choices{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.meta{color:#607080}.labels{margin:8px 0}.pill{background:#edf1f5;border-radius:4px;padding:3px 7px}.candidate-ruler{height:22px;position:relative;background:linear-gradient(90deg,#dce8f2 0 50%,#f2e2dc 50% 100%);border:1px solid #b9c0c9;border-radius:4px;overflow:hidden}.candidate-marker{position:absolute;top:0;bottom:0;width:3px;background:#b42318;transform:translateX(-1px)}.candidate-label{font-size:12px;color:#59616c;margin:4px 0 9px}.play-controls button,.choice{border:1px solid #8d99a5;border-radius:5px;background:#fff;padding:6px 9px;cursor:pointer}.choice.active{outline:3px solid #18212b;outline-offset:-2px}.choice[data-value="cut"].active{background:#f4b8b4}.choice[data-value="continue"].active{background:#bfe5cc}.choice[data-value="unsure"].active{background:#f3d49d}.note{width:100%;min-height:48px;margin-top:8px;box-sizing:border-box}pre{white-space:pre-wrap;background:#f6f7f9;padding:8px;border-radius:4px;font-size:12px}
"""
    adapter_js = r"""
const rows=__ROWS__,verdictSchema=__VERDICT_SCHEMA__,boundaryContract=__BOUNDARY_CONTRACT__;
const allowedVerdicts=new Set(['cut','continue','unsure']);
const reviewCore=createAuditReviewCore({storageKey:'acoustic-split-canonical-candidate-audit-v2:'+location.pathname,entries:rows,entryId:row=>row.candidate_id,defaultState:()=>({verdict:'',note:''}),isComplete:state=>allowedVerdicts.has(state.verdict),shouldSerialize:state=>allowedVerdicts.has(state.verdict),statusLabel:'Split canonical 裁决',filename:'manual_verdicts.jsonl',serialize:(row,state)=>({schema:verdictSchema,boundary_serialization_contract_id:boundaryContract,candidate_id:row.candidate_id,window_id:row.window_id,feature_index:row.feature_index,verdict:state.verdict,note:state.note||'',model_label:row.label,current_label:row.current_label,updated_at:state.updated_at||new Date().toISOString()})});
function sync(card,state){card.classList.toggle('done',allowedVerdicts.has(state.verdict));card.querySelectorAll('[data-value]').forEach(button=>button.classList.toggle('active',button.dataset.value===state.verdict));}
function rangeButton(label,start,end,className=''){return `<button type="button" class="${className}" data-play-start="${start}" data-play-end="${end}">${label} ${formatAuditSpan(start,end)}</button>`;}
function render(){stop();const filter=document.getElementById('filter').value,root=document.getElementById('list');root.innerHTML='';for(const row of rows){const state=reviewCore.ensure(row),disagree=row.current_label!==row.label;if(filter==='todo'&&allowedVerdicts.has(state.verdict))continue;if(filter==='disagree'&&!disagree)continue;const duration=Number(row.clip_duration_s),point=Number(row.candidate_offset_s),right=Math.max(0,duration-point),marker=Math.max(0,Math.min(100,100*point/duration)),card=document.createElement('article');card.innerHTML=`<h2>${escapeAuditHtml(row.candidate_id)}</h2><div class="meta"><span>${escapeAuditHtml(row.partition||'-')}</span><span>source t=${formatAuditTimestamp(row.time_s)}</span><span>实际 clip ${formatAuditSpan(row.clip_start_s,row.clip_end_s)}</span><span>左 ${formatAuditTimestamp(point)} / 右 ${formatAuditTimestamp(right)}</span><span>confidence=${Number(row.confidence||0).toFixed(2)}</span><span>${escapeAuditHtml((row.hard_case_categories||[]).join(', '))}</span></div><div class="labels"><span class="pill">current: ${escapeAuditHtml(row.current_label)}</span><span class="pill">legacy: ${escapeAuditHtml(row.legacy_label||'-')}</span><span class="pill">teacher: ${escapeAuditHtml(row.label)}</span><span class="pill">raw: ${escapeAuditHtml(row.omni_label)}</span></div><audio controls preload="metadata" src="${escapeAuditHtml(row.audio)}"></audio><div class="candidate-ruler"><span class="candidate-marker" style="left:${marker}%"></span></div><div class="candidate-label">红线 = 唯一 candidate query（物化音频内 ${formatAuditTimestamp(point)}）</div><div class="play-controls">${rangeButton('播放完整 candidate query',0,duration,'play-query')}${rangeButton('只听 candidate 左侧',0,point)}${rangeButton('只听 candidate 右侧',point,duration)}</div><div class="choices"><button type="button" class="choice" data-value="cut">cut：不同目标事件</button><button type="button" class="choice" data-value="continue">continue：同一目标事件</button><button type="button" class="choice" data-value="unsure">unsure</button></div><textarea class="note" placeholder="可选：记录事件关系、边界风险或 unsure 原因">${escapeAuditHtml(state.note||'')}</textarea><pre>${escapeAuditHtml(JSON.stringify({left_complete:row.left_complete,right_complete:row.right_complete,merged_better:row.merged_better,flags:row.flags,expected_gate_label:row.expected_gate_label,reason:row.reason},null,2))}</pre>`;const audio=card.querySelector('audio');audio.onplay=()=>{if(activeAudio&&activeAudio!==audio)stop();activeAudio=audio;};card.querySelectorAll('[data-play-start]').forEach(button=>button.onclick=()=>play(audio,button,Number(button.dataset.playStart),Number(button.dataset.playEnd)));card.querySelectorAll('[data-value]').forEach(button=>button.onclick=()=>{state.verdict=button.dataset.value;state.updated_at=new Date().toISOString();sync(card,state);reviewCore.persist();});card.querySelector('.note').onchange=event=>{state.note=event.target.value;state.updated_at=new Date().toISOString();reviewCore.persist();};sync(card,state);root.appendChild(card);}reviewCore.updateStatus();}
document.getElementById('filter').onchange=render;document.getElementById('stop').onclick=()=>{stop();reviewCore.updateStatus('已停止');};document.getElementById('save').onclick=()=>reviewCore.save();render();
"""
    adapter_js = (
        adapter_js.replace("__ROWS__", payload)
        .replace("__VERDICT_SCHEMA__", json.dumps(ACOUSTIC_SPLIT_MANUAL_VERDICT_SCHEMA))
        .replace(
            "__BOUNDARY_CONTRACT__",
            json.dumps(ACOUSTIC_BINARY_V12_CONTRACT.contract_id),
        )
    )
    return render_audit_review_page(
        AuditReviewPageSpec(
            title="Acoustic Split Canonical Candidate Audit",
            intro_html=intro_html,
            body_html='<div id="list"></div>',
            adapter_css=adapter_css,
            adapter_js=adapter_js,
        )
    )


def build_audit(
    *,
    selected_windows: Path,
    labels: Path,
    output_dir: Path,
    limit: int,
    context_s: float,
    update_nav: bool,
    review_prompt: ResolvedAuditPrompt | None = None,
) -> dict[str, Any]:
    resolved_prompt = review_prompt or resolve_audit_prompt(
        default_prompt=DEFAULT_REVIEW_PROMPT,
    )
    if not math.isfinite(context_s) or context_s <= 0.0:
        raise ValueError("context_s must be a positive finite value")
    windows = {str(row["window_id"]): row for row in _read_jsonl(selected_windows)}
    rows = sorted(_read_jsonl(labels), key=_priority)
    if limit > 0:
        rows = rows[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(rows, start=1):
        window = windows[str(row["window_id"])]
        source = Path(window["audio_path"])
        center = float(row["time_s"])
        duration = float(window["duration_s"])
        if not source.is_file():
            raise FileNotFoundError(source)
        if (
            not math.isfinite(center)
            or not math.isfinite(duration)
            or duration <= 0.0
            or not 0.0 <= center <= duration
        ):
            raise ValueError(
                f"invalid candidate/source timing: {center=} {duration=}"
            )
        clip = (
            output_dir
            / "audio"
            / f"{index:04d}_{row['window_id']}_f{int(row['feature_index']):05d}.wav"
        )
        clip_start, clip_end, candidate_offset = _slice_context(
            source=source,
            output=clip,
            center_s=center,
            duration_s=duration,
            context_s=context_s,
            row=row,
        )
        manifest.append(
            {
                **row,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "candidate_id": _candidate_id(row),
                "audio": clip.relative_to(output_dir).as_posix(),
                "clip_start_s": clip_start,
                "clip_end_s": clip_end,
                "clip_duration_s": clip_end - clip_start,
                "candidate_offset_s": candidate_offset,
            }
        )
    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )
    index_path = output_dir / "index.html"
    index_path.write_text(
        _page(manifest, review_prompt=resolved_prompt.text),
        encoding="utf-8",
    )
    summary = {
        "schema": ACOUSTIC_SPLIT_AUDIT_SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "title": "Acoustic Split Canonical Candidate Audit",
        "item_count": len(manifest),
        "label_counts": dict(Counter(str(row["label"]) for row in manifest)),
        "current_disagreement_count": sum(
            row["current_label"] != row["label"] for row in manifest
        ),
        "manual_verdicts": str(output_dir / "manual_verdicts.jsonl"),
        "manual_verdict_schema": ACOUSTIC_SPLIT_MANUAL_VERDICT_SCHEMA,
        "review_prompt_source": resolved_prompt.source,
        "review_prompt_sha256": resolved_prompt.sha256,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if update_nav:
        update_audit_entrypoints(latest_html=index_path, title=summary["title"])
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-windows", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--context-s", type=float, default=5.0)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--no-update-nav", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    prompt = resolve_audit_prompt(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        default_prompt=DEFAULT_REVIEW_PROMPT,
    )
    print(
        json.dumps(
            build_audit(
                selected_windows=Path(args.selected_windows),
                labels=Path(args.labels),
                output_dir=Path(args.output_dir),
                limit=args.limit,
                context_s=args.context_s,
                update_nav=not args.no_update_nav,
                review_prompt=prompt,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
