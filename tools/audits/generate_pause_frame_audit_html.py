#!/usr/bin/env python3
"""The labelling page: partition real audio into word / wordless voice / silence.

**This page shows no model output at all.** Not the blank runs, not the cut
points, not a waveform derived from the head. The whole value of the labels is
that they were produced without seeing the reading they will be used to judge -
an auditor shown a blank run next to the question would agree with it, and the
resulting agreement would be an artifact of the page rather than a measurement.
The comparison lives in `generate_pause_frame_review_html.py`, which runs
afterwards on frozen labels.

**Frame-aligned by construction.** Every boundary the editor can produce snaps to
one output frame (38.5 ms, the head's own resolution). Clicking a position finds
the frame, never a raw second, so a label can be compared with a posterior
without either side being resampled - the reason `drop_span_words_v1` cannot
answer this question is precisely that its spans do not line up with anything.

**Completeness is enforced at save, not while editing.** A partition has to cover
the window with no gaps and no overlaps, but demanding that after every click
would make the editor unusable; so gaps are allowed as an intermediate state and
`validateAuditPartition` rejects them on the way out. Every window starts as one
`unsure` span covering everything, which is the honest prior: nothing has been
decided yet, and a default of `silence` would let an unreviewed window serialize
as a real answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.audits.audit_nav import (  # noqa: E402
    audit_generated_at,
    update_audit_entrypoints,
)
from tools.audits.audit_prompt import resolve_audit_prompt  # noqa: E402
from tools.audits.pause_frame_audit import (  # noqa: E402
    FRAME_HOP_S,
    LABEL_COLOR,
    LABEL_TEXT,
    LABEL_UNSURE,
    MANUAL_LABEL_FILENAME,
    MANUAL_LABEL_SCHEMA,
    PAGE_SUMMARY_SCHEMA,
    PARTITION_EDITOR_CSS,
    PARTITION_LABELS,
    adapter_constants_js,
    label_legend_html,
    read_jsonl,
)
from tools.audits.review_page_core import (  # noqa: E402
    AuditReviewPageSpec,
    render_audit_review_page,
)

TITLE = "真实音频 · 逐帧安全切点标注"

DEFAULT_REVIEW_PROMPT = """把每一段音频切成连续的区间，每一帧都要有归属。三类：

A 有词 —— 正在说一个词。台词、名字、语气词里带实义的部分都算。
B 非语义发声 —— 喘息、呻吟、笑声、哭声。听得见有人在发声，但没有词。
C 静音 —— 没有人声。环境底噪、器械声、房间音都算静音。
? 不确定 —— 听不出来，或者喘息里像是埋着一个词。

判据是「这一刻在不在说词」，不是「响不响」。喘息可以很大声，仍然是 B；
很轻的耳语只要是词就是 A。**B 和 A 的分界是这次审计唯一真正要的东西**，
因为切点是在 B 上切还是在 A 上切，决定的是丢不丢内容。

拿不准就选 ?，不要猜。? 在统计里两边都不算，猜出来的 A/B 会被当成事实。"""

INTRO_HTML = (
    "<p>每段音频 8 秒，来自真实素材。把它切成连续区间并逐段标注；"
    "区间边界自动对齐到 38.5ms 一帧（对齐头的输出分辨率）。</p>"
    "<p>页面<b>不显示任何模型输出</b>——切点、blank 游程都不显示。"
    "先有不受影响的标注，比较放在另一个页面做。</p>"
)

EDITOR_JS = r"""
const state=createAuditReviewCore({
  entries:WINDOWS,
  storageKey:STORAGE_KEY,
  filename:LABEL_FILENAME,
  statusLabel:'已完成',
  entryId:entry=>entry.row_id,
  // One `unsure` span over the whole window: nothing has been decided, and a
  // default of `silence` would let an untouched window save as an answer.
  defaultState:entry=>({segments:[{id:'segment-0',label:'unsure',start_frame:0,end_frame:entry.frame_count}],note:''}),
  isComplete:(annotation,entry)=>{
    const result=validateAuditPartition(annotation.segments,entry.frame_count,PAUSE_LABELS);
    if(!result.ok)return false;
    // Complete means decided, so the initial all-unsure state does not count
    // even though it is a valid partition.
    return result.segments.some(segment=>segment.label!=='unsure');
  },
  shouldSerialize:(annotation,entry)=>state.isComplete(annotation,entry),
  serialize:async(entry,annotation)=>{
    const result=validateAuditPartition(annotation.segments,entry.frame_count,PAUSE_LABELS);
    if(!result.ok)throw new Error(`${entry.row_id}: ${result.error}`);
    return {
      schema:MANUAL_LABEL_SCHEMA,
      row_id:entry.row_id,
      frame_count:entry.frame_count,
      frame_hop_s:PAUSE_FRAME_HOP_S,
      // Signed so a later report can prove it read the labels it claims to.
      corrected_span_signature:await auditPartitionSha256(result.segments),
      segments:result.segments.map(segment=>Object.assign(
        {label:segment.label,start_frame:segment.start_frame,end_frame:segment.end_frame},
        auditPartitionSeconds(segment,PAUSE_FRAME_HOP_S)
      )),
      note:annotation.note||'',
      updated_at:new Date().toISOString()
    };
  }
});

let selected={};

function segmentsFor(entry){return state.ensure(entry).segments;}

function renderWindow(entry){
  const card=document.getElementById(`card-${entry.row_id}`);
  const strip=card.querySelector('.pause-strip');
  const error=card.querySelector('.pause-error');
  strip.innerHTML='';
  const segments=normalizeAuditPartition(segmentsFor(entry));
  state.ensure(entry).segments=segments;
  for(const segment of segments){
    const button=document.createElement('button');
    button.type='button';
    button.className='pause-seg'+(selected[entry.row_id]===segment.id?' selected':'');
    button.style.left=`${100*segment.start_frame/entry.frame_count}%`;
    button.style.width=`${Math.max(.4,100*(segment.end_frame-segment.start_frame)/entry.frame_count)}%`;
    button.style.background=PAUSE_LABEL_COLOR[segment.label]||'#888';
    const span=auditPartitionSeconds(segment,PAUSE_FRAME_HOP_S);
    button.textContent=PAUSE_LABEL_TEXT[segment.label]?PAUSE_LABEL_TEXT[segment.label][0]:'?';
    button.title=`${PAUSE_LABEL_TEXT[segment.label]||segment.label} ${formatAuditSpan(span.start_s,span.end_s)}`;
    button.onclick=event=>{
      event.stopPropagation();
      selected[entry.row_id]=segment.id;
      play(card.querySelector('audio'),button,span.start_s,span.end_s);
      renderWindow(entry);
    };
    strip.appendChild(button);
  }
  const result=validateAuditPartition(segments,entry.frame_count,PAUSE_LABELS);
  error.textContent=result.ok?'':result.error;
  state.persist();
}

// Splitting at a click, rather than dragging edges: a click maps to exactly one
// frame, so no boundary can land between frames however imprecise the pointer is.
function splitAt(entry,fraction){
  const frame=Math.min(entry.frame_count-1,Math.max(1,Math.round(fraction*entry.frame_count)));
  const segments=segmentsFor(entry);
  const target=segments.find(segment=>segment.start_frame<frame&&segment.end_frame>frame);
  if(!target)return;
  const tail={id:`segment-${Date.now()}`,label:target.label,start_frame:frame,end_frame:target.end_frame,category:'',reason:''};
  target.end_frame=frame;
  segments.push(tail);
  selected[entry.row_id]=tail.id;
  renderWindow(entry);
}

function assign(entry,label){
  const segments=segmentsFor(entry);
  const current=segments.find(segment=>segment.id===selected[entry.row_id]);
  if(!current){
    document.getElementById(`card-${entry.row_id}`).querySelector('.pause-error').textContent='先点一个区间再选标签';
    return;
  }
  current.label=label;
  renderWindow(entry);
}

function mergeSelected(entry){
  const segments=segmentsFor(entry).slice().sort((a,b)=>a.start_frame-b.start_frame);
  const position=segments.findIndex(segment=>segment.id===selected[entry.row_id]);
  if(position<0||position+1>=segments.length)return;
  const current=segments[position],next=segments[position+1];
  current.end_frame=next.end_frame;
  state.ensure(entry).segments=segments.filter(segment=>segment.id!==next.id);
  renderWindow(entry);
}

function resetWindow(entry){
  state.ensure(entry).segments=[{id:'segment-0',label:'unsure',start_frame:0,end_frame:entry.frame_count}];
  selected[entry.row_id]=null;
  renderWindow(entry);
}

for(const entry of WINDOWS){
  const card=document.getElementById(`card-${entry.row_id}`);
  const strip=card.querySelector('.pause-strip');
  strip.onclick=event=>{
    const box=strip.getBoundingClientRect();
    splitAt(entry,(event.clientX-box.left)/box.width);
  };
  card.querySelectorAll('[data-label]').forEach(button=>{
    button.onclick=()=>assign(entry,button.dataset.label);
  });
  card.querySelector('[data-action="merge"]').onclick=()=>mergeSelected(entry);
  card.querySelector('[data-action="reset"]').onclick=()=>resetWindow(entry);
  const note=card.querySelector('textarea');
  note.value=state.ensure(entry).note||'';
  note.oninput=()=>{state.ensure(entry).note=note.value;state.persist();};
  renderWindow(entry);
}
document.getElementById('stop').onclick=()=>stop();
document.getElementById('save').onclick=()=>state.save();
state.updateStatus();
"""


def _card_html(row: dict) -> str:
    row_id = str(row["row_id"])
    buttons = "".join(
        f'<button type="button" data-label="{label}" '
        f'style="background:{LABEL_COLOR[label]};color:#fff">{LABEL_TEXT[label]}</button>'
        for label in PARTITION_LABELS
    )
    return (
        f'<section class="pause-card" id="card-{row_id}">'
        f"<h3>{row_id}</h3>"
        f'<audio controls preload="none" src="{row["audio"]}"></audio>'
        f'<div class="pause-strip"></div>'
        f'<div class="pause-tools">{buttons}'
        f'<button type="button" data-action="merge">合并到右侧</button>'
        f'<button type="button" data-action="reset">重置</button></div>'
        f'<div class="pause-error"></div>'
        f'<div><small>点条带切分，点区间试听，选标签归类。</small></div>'
        f'<textarea rows="2" style="width:100%" placeholder="备注（可空）"></textarea>'
        f"</section>"
    )


def build(manifest_path: Path, output_dir: Path, prompt: str) -> dict:
    rows = read_jsonl(manifest_path)
    if not rows:
        raise SystemExit("manifest has no windows")
    windows = [
        {
            "row_id": str(row["row_id"]),
            "audio": str(row["audio"]),
            "frame_count": int(row["frame_count"]),
        }
        for row in rows
    ]
    body = (
        f'<section class="pause-card"><b>标注说明</b>'
        f'<pre style="white-space:pre-wrap">{prompt}</pre>{label_legend_html()}</section>'
        + "".join(_card_html(row) for row in rows)
    )
    adapter_js = (
        adapter_constants_js()
        + f"const WINDOWS={json.dumps(windows, ensure_ascii=False)};"
        + f"const STORAGE_KEY={json.dumps('pause-frame-audit-' + output_dir.name)};"
        + f"const LABEL_FILENAME={json.dumps(MANUAL_LABEL_FILENAME)};"
        + f"const MANUAL_LABEL_SCHEMA={json.dumps(MANUAL_LABEL_SCHEMA)};"
        + EDITOR_JS
    )
    html_text = render_audit_review_page(
        AuditReviewPageSpec(
            title=TITLE,
            intro_html=INTRO_HTML,
            body_html=body,
            adapter_css=PARTITION_EDITOR_CSS,
            adapter_js=adapter_js,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    summary = {
        "schema": PAGE_SUMMARY_SCHEMA,
        "generated_at": audit_generated_at(),
        "title": TITLE,
        "windows": len(rows),
        "frames_to_label": sum(int(row["frame_count"]) for row in rows),
        "frame_hop_s": FRAME_HOP_S,
        "labels": list(PARTITION_LABELS),
        "default_label": LABEL_UNSURE,
        # Stated in the artifact because it is the design property that makes the
        # labels usable as ground truth rather than as agreement.
        "shows_model_output": False,
        "manual_label_file": MANUAL_LABEL_FILENAME,
    }
    (output_dir / "page_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--no-nav", action="store_true")
    args = parser.parse_args()

    prompt = resolve_audit_prompt(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        default_prompt=DEFAULT_REVIEW_PROMPT,
    ).text
    output_dir = Path(args.output_dir)
    summary = build(Path(args.manifest), output_dir, prompt)
    if not args.no_nav:
        update_audit_entrypoints(latest_html=output_dir / "index.html", title=TITLE)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
