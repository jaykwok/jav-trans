#!/usr/bin/env python3
"""Generate one-page A/B review for broad-v3 and adaptive-partition Teachers."""
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

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.audits.review_page_core import (  # noqa: E402
    AuditOptionAxis,
    AuditReviewPageSpec,
    render_audit_review_page,
    validate_audit_option_contract,
)
from tools.boundary.ja.label_vocal_envelope_scorer_v12_compact_ab import (  # noqa: E402
    PREAUDIT_SCHEMA as COMPACT_PREAUDIT_SCHEMA,
    PROMPT_VERSION as COMPACT_PROMPT_VERSION,
)
from tools.boundary.ja.vocal_envelope_scorer_v12_teacher_contract import (  # noqa: E402
    SCORER_V12_FRAME_HOP_S,
    SCORER_V12_LOCAL_TIMESTAMP_STEP_S,
    SCORER_V12_TIME_GRID_CONTRACT_ID,
)


SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_prompt_ab_audit_summary_v3"
AUDIT_ITEM_SCHEMA = "vocal_envelope_scorer_v12_prompt_ab_audit_item_v3"
MANUAL_VERDICT_SCHEMA = "vocal_envelope_scorer_v12_prompt_ab_manual_verdict_v3"
PREFERENCE_OPTIONS = (
    "a_broad_v3_better",
    "b_adaptive_partition_better",
    "both_acceptable",
    "both_unacceptable",
    "complementary_tradeoffs",
    "unsure",
)


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


def _index(rows: Sequence[Mapping[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in output:
            raise ValueError(f"{name} requires unique non-empty source_id")
        output[source_id] = dict(row)
    if not output:
        raise ValueError(f"{name} must be non-empty")
    return output


def _labels(row: Mapping[str, Any], *, frame_count: int, name: str) -> list[str]:
    labels = [""] * frame_count
    for key, label in (
        ("vocal_spans", "vocal_candidate"),
        ("non_vocal_spans", "non_vocal_candidate"),
        ("unsure_spans", "unsure"),
    ):
        for span in row.get(key) or ():
            start, end = int(span["start_frame"]), int(span["end_frame"])
            if start < 0 or end <= start or end > frame_count:
                raise ValueError(f"{name} has invalid frame span")
            for frame in range(start, end):
                if labels[frame]:
                    raise ValueError(f"{name} has overlapping frame spans")
                labels[frame] = label
    if any(not label for label in labels):
        raise ValueError(f"{name} does not cover the complete frame timeline")
    return labels


def _difference_spans(a_labels: list[str], b_labels: list[str]) -> list[dict[str, Any]]:
    if len(a_labels) != len(b_labels):
        raise ValueError("A/B frame counts differ")
    output: list[dict[str, Any]] = []
    start = 0
    for end in range(1, len(a_labels) + 1):
        same_pair = end < len(a_labels) and (
            a_labels[end], b_labels[end]
        ) == (a_labels[start], b_labels[start])
        if same_pair:
            continue
        if a_labels[start] != b_labels[start]:
            output.append(
                {
                    "a_label": a_labels[start],
                    "b_label": b_labels[start],
                    "start_frame": start,
                    "end_frame": end,
                    "start_s": round(start * 0.02, 6),
                    "end_s": round(end * 0.02, 6),
                }
            )
        start = end
    return output


def _script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _page(payload: list[dict[str, Any]], *, a_sha: str, b_sha: str, audit_sha: str) -> str:
    validate_audit_option_contract(
        axes=(AuditOptionAxis(field="preference", options=PREFERENCE_OPTIONS),),
        combination_results={
            (option,): ("unsure" if option == "unsure" else "reviewed")
            for option in PREFERENCE_OPTIONS
        },
    )
    intro = """
<section class="audit-help">
  <h2>Scorer v12 broad-v3 / adaptive complete-partition Prompt A/B</h2>
  <p><b>A：</b>昨天的 broad-v3 完整时间轴三态 Teacher，使用 <code>MM:SS.mmm</code>，不含固定时长切分规则。</p>
  <p><b>B：</b>本次完整分区 Prompt，直接输出首尾相接的 vocal/non-vocal/unsure；每次最多听 20 秒，优先在约 15 秒附近的可靠 non-vocal 内提交并从切点重新计时，找不到安全落点时保留 5 秒前瞻。Teacher 坐标为 10ms 网格，本地使用同一个 vocal-safe 共享切点量化到 Scorer 的 20ms 帧。</p>
  <p>两臂的正类一致：对白、耳语、呻吟、喘息、吸呼气、亲吻/口腔声、咳嗽、歌唱及背景人声都属于 vocal；纯机械、肉体撞击/拍打、衣物/床体、水声、纯音乐、静音和环境噪声属于 non-vocal。</p>
  <p>重点比较两件事：是否把任何人类发声放进黄色；同一发声事件是否被不必要切碎。差异轨只显示 A/B 标签不同的区间，点击任意颜色条只播放该精确区间。</p>
</section>
<div id="cards"></div>
"""
    css = """
.audit-help,.ab-card{background:#fff;border:1px solid #d7dde4;border-radius:8px;padding:14px;margin:0 0 14px}.ab-card h3{margin:0 0 4px}.meta{color:#53606c;margin-bottom:8px}.full{display:flex;gap:8px;align-items:center;margin:8px 0}.full audio{flex:1}.ab-lanes{display:grid;gap:5px}.vocal-a{background:#2d9c68;color:#fff}.vocal-b{background:#147d56;color:#fff}.nonvocal-a{background:#f2ca52;color:#3c3000}.nonvocal-b{background:#d9a91d;color:#302300}.unsure-a,.unsure-b{background:#98a4af;color:#15202a}.diff-a-vocal{background:#dc5252;color:#fff}.diff-b-vocal{background:#3f79d8;color:#fff}.diff-other{background:#8c67b5;color:#fff}.verdict{display:grid;grid-template-columns:minmax(260px,1fr) 130px 2fr;gap:10px;align-items:end;margin-top:12px}.verdict label{display:grid;gap:4px}.verdict select,.verdict input[type=text]{padding:7px}.reviewed{display:flex!important;grid-auto-flow:column;justify-content:start;align-items:center}.audit-span{font-size:10px;overflow:hidden;white-space:nowrap}.audit-span.playing{outline:3px solid #111;z-index:2}@media(max-width:760px){.verdict{grid-template-columns:1fr}.audit-lane{grid-template-columns:92px 1fr}.audit-span{font-size:0}}
"""
    js = f"""
const entries={_script_json(payload)};
const aSha={_script_json(a_sha)},bSha={_script_json(b_sha)},auditSha={_script_json(audit_sha)};
const preferenceOptions=[
  ['', '请选择'],
  ['a_broad_v3_better','A broad-v3 更好'],
  ['b_adaptive_partition_better','B adaptive-partition 更好'],
  ['both_acceptable','两者均可接受'],
  ['both_unacceptable','两者均不可接受'],
  ['complementary_tradeoffs','各有优劣'],
  ['unsure','不确定']
];
const core=createAuditReviewCore({{
  entries,storageKey:'scorer-v12-prompt-ab:'+location.pathname+'|'+aSha+'|'+bSha,
  statusLabel:'已审计',entryId:entry=>entry.source_id,
  defaultState:()=>({{preference:'',reviewed_full_source:false,notes:'',updated_at:''}}),
  isComplete:state=>Boolean(state.preference&&state.reviewed_full_source),
  serialize:async(entry,state)=>({{
    schema:{_script_json(MANUAL_VERDICT_SCHEMA)},source_id:entry.source_id,
    video_id:entry.video_id,partition:entry.partition,audio_sha256:entry.audio_sha256,
    frame_count:entry.frame_count,duration_s:entry.duration_s,
    a_audit_manifest_sha256:aSha,b_preaudit_sha256:bSha,audit_manifest_sha256:auditSha,
    preference:state.preference,reviewed_full_source:state.reviewed_full_source,
    notes:state.notes,updated_at:state.updated_at
  }})
}});
const root=document.getElementById('cards');
function lane(container,audio,entry,label,spans,className){{appendAuditSpanLane({{container,audio,label,spans,durationS:entry.duration_s,className,title:(span,start,end)=>`${{label}} ${{formatAuditSpan(start,end)}}`}});}}
function render(){{
  stop();root.innerHTML='';
  for(const [index,entry] of entries.entries()){{
    const state=core.ensure(entry),card=document.createElement('article');card.className='ab-card';
    card.innerHTML=`<h3>${{index+1}}. ${{escapeAuditHtml(entry.source_id)}}</h3><div class="meta">${{escapeAuditHtml(entry.partition)}} · ${{entry.duration_s.toFixed(3)}}s · 差异 ${{entry.difference_spans.length}} 段</div><div class="full"><b>完整 source</b><audio controls preload="metadata" src="${{escapeAuditHtml(entry.audio)}}"></audio></div><div class="ab-lanes"></div><div class="verdict"><label>比较结论<select data-field="preference">${{preferenceOptions.map(([v,t])=>`<option value="${{v}}" ${{state.preference===v?'selected':''}}>${{t}}</option>`).join('')}}</select></label><label class="reviewed"><input type="checkbox" data-field="reviewed_full_source" ${{state.reviewed_full_source?'checked':''}}>已完整听完</label><label>备注<input type="text" data-field="notes" value="${{escapeAuditHtml(state.notes)}}"></label></div>`;
    const audio=card.querySelector('audio'),lanes=card.querySelector('.ab-lanes');
    lane(lanes,audio,entry,'A vocal',entry.a_vocal_spans,'vocal-a');
    lane(lanes,audio,entry,'A non-vocal',entry.a_non_vocal_spans,'nonvocal-a');
    lane(lanes,audio,entry,'A unsure',entry.a_unsure_spans,'unsure-a');
    lane(lanes,audio,entry,'B vocal',entry.b_vocal_spans,'vocal-b');
    lane(lanes,audio,entry,'B non-vocal',entry.b_non_vocal_spans,'nonvocal-b');
    lane(lanes,audio,entry,'B unsure',entry.b_unsure_spans,'unsure-b');
    appendAuditSpanLane({{container:lanes,audio,label:'A/B 差异',spans:entry.difference_spans,durationS:entry.duration_s,className:span=>span.a_label==='vocal_candidate'?'diff-a-vocal':span.b_label==='vocal_candidate'?'diff-b-vocal':'diff-other',title:(span,start,end)=>`${{span.a_label}} → ${{span.b_label}} · ${{formatAuditSpan(start,end)}}`}});
    for(const element of card.querySelectorAll('[data-field]')){{element.onchange=()=>{{const field=element.dataset.field;state[field]=element.type==='checkbox'?element.checked:element.value;state.updated_at=new Date().toISOString();core.persist();}};}}
    root.appendChild(card);
  }}core.updateStatus();
}}
document.getElementById('stop').onclick=stop;document.getElementById('save').onclick=async()=>{{const pending=entries.find(entry=>!core.isComplete(core.ensure(entry),entry));if(pending){{core.updateStatus('尚未完成 '+pending.source_id);return;}}await core.save();}};render();
"""
    return render_audit_review_page(
        AuditReviewPageSpec(
            title="Scorer v12 Teacher Prompt A/B review",
            intro_html=intro,
            body_html="",
            adapter_css=css,
            adapter_js=js,
        )
    )


def build(*, a_audit_manifest: Path, b_preaudit: Path, output_dir: Path) -> dict[str, Any]:
    a_audit_manifest = a_audit_manifest.resolve()
    b_preaudit = b_preaudit.resolve()
    a_rows = _index(_rows(a_audit_manifest), name="A audit manifest")
    b_rows = _index(_rows(b_preaudit), name="B compact preaudit")
    if set(a_rows) != set(b_rows):
        raise ValueError("A/B source IDs must match exactly")
    a_sha, b_sha = _sha256(a_audit_manifest), _sha256(b_preaudit)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    for index, source_id in enumerate(a_rows):
        a, b = a_rows[source_id], b_rows[source_id]
        if b.get("schema") != COMPACT_PREAUDIT_SCHEMA or b.get("prompt_version") != COMPACT_PROMPT_VERSION:
            raise ValueError(f"wrong B compact contract: {source_id}")
        for field, expected in (
            ("time_grid_contract_id", SCORER_V12_TIME_GRID_CONTRACT_ID),
            ("teacher_timestamp_step_s", SCORER_V12_LOCAL_TIMESTAMP_STEP_S),
            ("scorer_frame_hop_s", SCORER_V12_FRAME_HOP_S),
        ):
            actual = b.get(field)
            if field == "scorer_frame_hop_s" and actual is None:
                actual = b.get("frame_hop_s")
            if actual != expected:
                raise ValueError(f"wrong B {field}: {source_id}")
        for field in ("video_id", "partition", "audio_sha256", "frame_count"):
            if a.get(field) != b.get(field):
                raise ValueError(f"A/B {field} mismatch: {source_id}")
        duration_s, frame_count = float(a["duration_s"]), int(a["frame_count"])
        if abs(duration_s - float(b["duration_s"])) > 1e-9:
            raise ValueError(f"A/B duration mismatch: {source_id}")
        a_labels = _labels(a, frame_count=frame_count, name=f"A {source_id}")
        b_labels = _labels(b, frame_count=frame_count, name=f"B {source_id}")
        source_audio = (a_audit_manifest.parent / str(a["audio"])).resolve()
        if not source_audio.is_file() or _sha256(source_audio) != str(a["audio_sha256"]):
            raise ValueError(f"A audio identity mismatch: {source_id}")
        target = audio_dir / f"source-{index:03d}.wav"
        shutil.copy2(source_audio, target)
        payload.append(
            {
                "schema": AUDIT_ITEM_SCHEMA,
                "source_id": source_id,
                "video_id": str(a["video_id"]),
                "partition": str(a["partition"]),
                "audio": target.relative_to(output_dir).as_posix(),
                "audio_sha256": str(a["audio_sha256"]),
                "duration_s": duration_s,
                "frame_count": frame_count,
                "a_vocal_spans": list(a.get("vocal_spans") or ()),
                "a_non_vocal_spans": list(a.get("non_vocal_spans") or ()),
                "a_unsure_spans": list(a.get("unsure_spans") or ()),
                "b_vocal_spans": list(b.get("vocal_spans") or ()),
                "b_non_vocal_spans": list(b.get("non_vocal_spans") or ()),
                "b_unsure_spans": list(b.get("unsure_spans") or ()),
                "difference_spans": _difference_spans(a_labels, b_labels),
            }
        )
    audit_manifest = output_dir / "audit_manifest.jsonl"
    audit_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in payload),
        encoding="utf-8",
    )
    audit_sha = _sha256(audit_manifest)
    index_path = output_dir / "index.html"
    index_path.write_text(_page(payload, a_sha=a_sha, b_sha=b_sha, audit_sha=audit_sha), encoding="utf-8")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "source_count": len(payload),
        "a_label": "broad-v3",
        "a_audit_manifest": str(a_audit_manifest),
        "a_audit_manifest_sha256": a_sha,
        "b_label": "adaptive-complete-partition-v3-10ms-wire-20ms-frame",
        "b_preaudit": str(b_preaudit),
        "b_preaudit_sha256": b_sha,
        "audit_manifest": str(audit_manifest),
        "audit_manifest_sha256": audit_sha,
        "a_vocal_run_count": sum(len(row["a_vocal_spans"]) for row in payload),
        "a_nonvocal_run_count": sum(
            len(row["a_non_vocal_spans"]) for row in payload
        ),
        "b_vocal_run_count": sum(len(row["b_vocal_spans"]) for row in payload),
        "b_nonvocal_run_count": sum(
            len(row["b_non_vocal_spans"]) for row in payload
        ),
        "difference_run_count": sum(
            len(row["difference_spans"]) for row in payload
        ),
        "difference_frame_count": sum(
            int(span["end_frame"]) - int(span["start_frame"])
            for row in payload
            for span in row["difference_spans"]
        ),
        "time_grid_contract_id": SCORER_V12_TIME_GRID_CONTRACT_ID,
        "teacher_timestamp_step_s": SCORER_V12_LOCAL_TIMESTAMP_STEP_S,
        "scorer_frame_hop_s": SCORER_V12_FRAME_HOP_S,
        "manual_verdict_schema": MANUAL_VERDICT_SCHEMA,
        "manual_gate_status": "pending",
        "training_manifest_allowed": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_audit_entrypoints(latest_html=index_path, title="Scorer v12 Teacher Prompt A/B review")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-audit-manifest", required=True)
    parser.add_argument("--b-preaudit", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build(a_audit_manifest=Path(args.a_audit_manifest), b_preaudit=Path(args.b_preaudit), output_dir=Path(args.output_dir)), ensure_ascii=False))
