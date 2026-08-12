#!/usr/bin/env python3
"""The review page: human frame labels against the head's blank reading.

Generated only from labels that already exist. The labelling page shows no model
output, so the agreement measured here is not something the pages produced by
construction.

**What this settles.** The 07-31 falsification established that the blank reading
loses real words, and left two candidate explanations that its evidence could not
tell apart: a shared blank posterior cannot serve two conflicting tasks, or the
head is simply mistrained for this domain. The number that separates them is the
margin between the blank rate on wordless voice and on words, at frame
resolution:

  * a wide margin means the signal IS there and the discarding gate merely read
    it with a bad threshold - a pause branch is then worth building;
  * a margin near zero means no threshold on this posterior can work, and a pause
    branch trained on the same clean corpus would inherit the same blindness.

Either way this page reports the margin rather than a verdict, because the second
reading is only actionable together with real-domain training data that does not
exist yet.

**Per-frame, not per-span.** Both sides are expanded to a frame vector before
anything is counted. Span arithmetic was how the previous attempt at this
comparison went wrong: a 17-second span labelled `speech` was scored as having
lost fifteen seconds because the label's resolution and the gate's resolution
were different things being subtracted from each other.

**Frames the labeller left as `unsure` are counted in neither direction** and
reported on their own, for the same reason `evaluate_pregate_loss.py` does it:
folding them in would let an unresolved labelling question look like a result.
"""
from __future__ import annotations

import argparse
import hashlib
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
from tools.audits.pause_frame_audit import (  # noqa: E402
    FRAME_HOP_S,
    LABEL_COLOR,
    LABEL_NON_SEMANTIC,
    LABEL_SILENCE,
    LABEL_TEXT,
    LABEL_UNSURE,
    LABEL_WORD,
    MANUAL_LABEL_FILENAME,
    PARTITION_EDITOR_CSS,
    PARTITION_LABELS,
    RESULT_SCHEMA,
    adapter_constants_js,
    blank_frames_from_runs,
    confusion,
    expand_partition,
    label_legend_html,
    labelled_frame_totals,
    read_jsonl,
    separation_report,
)
from tools.audits.review_page_core import (  # noqa: E402
    AuditReviewPageSpec,
    render_audit_review_page,
)
from tools.audits.stats import wilson  # noqa: E402

TITLE = "真实音频 · 安全切点：人工帧标注 vs blank 读法"

INTRO_TEMPLATE = (
    "<p>逐帧比较：人工标注（本页<b>不可编辑</b>，来自不显示模型输出的标注页）"
    "与对齐头的 blank 读法。红色格子表示两者不一致。</p>"
    "<p>关键数字是<b>「非语义发声的 blank 率」减去「有词的 blank 率」</b>——"
    "这个余量决定 blank 到底是不是一个可用的停顿信号。"
    "07-31 实测该余量只有 7.4pp，本页在帧分辨率上重测。</p>"
    "<p>{provenance}</p>"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head_blank_runs(audio_path: Path, min_blank_s: float) -> tuple[list, int]:
    """Blank runs from the configured head, plus the frames it produced.

    Imported here rather than at module scope so the page can be regenerated
    from cached readings on a machine with no torch and no GPU.
    """
    import numpy as np
    import torch

    from asr.alignment import AlignmentHead, blank_runs
    from asr.encoder_features import EncoderFeatureConfig, Qwen3AsrEncoder
    from audio.loading import load_audio_16k_mono

    head = AlignmentHead.from_env()
    if head is None:
        raise SystemExit(
            "ASR_ALIGNMENT_HEAD_PATH is not set; the review page has nothing to compare"
        )
    extractor = Qwen3AsrEncoder(EncoderFeatureConfig(model_path="", device="cuda"))
    audio, rate = load_audio_16k_mono(str(audio_path))
    if rate != 16000:
        raise SystemExit(f"unexpected sample rate for {audio_path}: {rate}")
    features = extractor.encode_batch(
        [np.asarray(audio, dtype=np.float32)], sample_rate=16000
    )[0]
    log_probs = head.log_probs(features)
    runs = blank_runs(log_probs, upsample=head.upsample, min_seconds=min_blank_s)
    del torch
    return runs, int(log_probs.shape[0])


def build(
    *,
    manifest_path: Path,
    labels_path: Path,
    output_dir: Path,
    readings_path: Path | None,
    min_blank_s: float,
) -> dict:
    manifest = {str(row["row_id"]): row for row in read_jsonl(manifest_path)}
    labels = {str(row["row_id"]): row for row in read_jsonl(labels_path)}
    if not labels:
        raise SystemExit(f"no manual labels in {labels_path}")
    unknown = set(labels) - set(manifest)
    if unknown:
        raise SystemExit(f"labels reference unknown row_id: {sorted(unknown)}")

    cached = {}
    if readings_path and readings_path.exists():
        cached = {str(row["row_id"]): row for row in read_jsonl(readings_path)}

    rows: list[dict] = []
    readings: list[dict] = []
    pooled_labels: list[str] = []
    pooled_blank: list[bool] = []

    for row_id in sorted(labels):
        window = manifest[row_id]
        record = labels[row_id]
        frame_count = int(record.get("frame_count") or window["frame_count"])
        reading = cached.get(row_id)
        if reading is None:
            runs, _frames = _head_blank_runs(
                output_dir / str(window["audio"]), min_blank_s
            )
            reading = {
                "row_id": row_id,
                "min_blank_s": min_blank_s,
                "blank_runs": [[round(a, 6), round(b, 6)] for a, b in runs],
            }
        readings.append(reading)
        runs = [(float(a), float(b)) for a, b in reading["blank_runs"]]
        frame_labels = expand_partition(record.get("segments") or [], frame_count)
        blank = blank_frames_from_runs(runs, frame_count)
        pooled_labels.extend(frame_labels)
        pooled_blank.extend(blank)
        rows.append(
            {
                "row_id": row_id,
                "audio": str(window["audio"]),
                "frame_count": frame_count,
                "labels": frame_labels,
                "blank": blank,
                "note": str(record.get("note") or ""),
                "signature": str(record.get("corrected_span_signature") or ""),
            }
        )

    table = confusion(pooled_labels, pooled_blank)
    separation = separation_report(table)
    totals = labelled_frame_totals(pooled_labels)
    decisive = sum(
        totals.get(label, 0)
        for label in (LABEL_WORD, LABEL_NON_SEMANTIC, LABEL_SILENCE)
    )

    intervals = {}
    for label in (LABEL_WORD, LABEL_NON_SEMANTIC, LABEL_SILENCE):
        row = table.get(label) or {}
        total = int(row.get("blank", 0)) + int(row.get("non_blank", 0))
        interval = wilson(int(row.get("blank", 0)), total) if total > 0 else None
        if interval is not None:
            intervals[label] = [round(interval[0], 5), round(interval[1], 5)]

    result = {
        "schema": RESULT_SCHEMA,
        "generated_at": audit_generated_at(),
        "windows": len(rows),
        "frame_hop_s": FRAME_HOP_S,
        "min_blank_s": min_blank_s,
        # Provenance, for the same reason the onset audit now carries it: a
        # result that outlives its own inputs is worse than no result.
        "inputs": {
            "manifest": {"path": manifest_path.as_posix(), "sha256": _digest(manifest_path)},
            "labels": {
                "path": labels_path.as_posix(),
                "sha256": _digest(labels_path),
                "rows": len(labels),
            },
        },
        "labelled_frames": totals,
        "decisive_frames": decisive,
        "unsure_frames": totals.get(LABEL_UNSURE, 0),
        "unsure_share": (
            round(totals.get(LABEL_UNSURE, 0) / len(pooled_labels), 5)
            if pooled_labels
            else None
        ),
        "confusion_frames": table,
        "blank_rate_ci95": intervals,
        **separation,
    }

    body_rows = "".join(
        f'<section class="pause-card" id="card-{row["row_id"]}">'
        f'<h3>{row["row_id"]}</h3>'
        f'<audio controls preload="none" src="{row["audio"]}"></audio>'
        f'<div class="pause-compare"><small>人工标注</small><div class="pause-strip" data-lane="human"></div></div>'
        f'<div class="pause-compare"><small>blank 读法</small><div class="pause-strip" data-lane="head"></div></div>'
        f'<div class="pause-compare"><small>不一致</small><div class="pause-strip" data-lane="diff"></div></div>'
        + (f'<div><small>备注：{row["note"]}</small></div>' if row["note"] else "")
        + "</section>"
        for row in rows
    )

    margin = separation.get("margin_vs_non_semantic_pp")
    headline = (
        f'<section class="pause-card"><b>合计</b>'
        f'<p>可判定帧 {decisive}，unsure {result["unsure_frames"]}'
        f'（{(result["unsure_share"] or 0) * 100:.1f}%，两边都不算）。</p>'
        f'<p>blank 率：有词 {_pct(separation.get("blank_rate_word"))}，'
        f'非语义发声 {_pct(separation.get("blank_rate_non_semantic_vocal"))}，'
        f'静音 {_pct(separation.get("blank_rate_silence"))}。</p>'
        f'<p><b>余量（非语义发声 − 有词）：'
        f'{"n/a" if margin is None else f"{margin:+.2f}pp"}</b></p>'
        f"{label_legend_html()}</section>"
    )

    provenance = (
        f'标注文件 {labels_path.name}，{len(labels)} 个窗口，'
        f"sha256 {result['inputs']['labels']['sha256'][:16]}…"
    )
    adapter_js = (
        adapter_constants_js()
        + f"const ROWS={json.dumps(rows, ensure_ascii=False)};"
        + REVIEW_JS
    )
    html_text = render_audit_review_page(
        AuditReviewPageSpec(
            title=TITLE,
            intro_html=INTRO_TEMPLATE.format(provenance=provenance) + headline,
            body_html=body_rows,
            adapter_css=PARTITION_EDITOR_CSS + REVIEW_CSS,
            adapter_js=adapter_js,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "review.html").write_text(html_text, encoding="utf-8")
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if readings_path:
        with readings_path.open("w", encoding="utf-8") as handle:
            for reading in readings:
                handle.write(json.dumps(reading, ensure_ascii=False) + "\n")
    return result


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


REVIEW_CSS = """
.pause-strip{cursor:default}
.pause-cell{position:absolute;top:0;height:100%;border:0;padding:0}
"""

REVIEW_JS = r"""
// Painted per frame rather than per span: the whole reason this audit exists is
// that span-level agreement hid a frame-level disagreement.
function paint(lane,cells,audio,frameHop){
  lane.innerHTML='';
  const total=cells.length;
  for(let index=0;index<total;index+=1){
    const cell=cells[index];
    if(!cell)continue;
    const button=document.createElement('button');
    button.type='button';
    button.className='pause-cell';
    button.style.left=`${100*index/total}%`;
    button.style.width=`${100/total}%`;
    button.style.background=cell;
    const start=index*frameHop;
    button.title=formatAuditSpan(start,start+frameHop);
    button.onclick=()=>play(audio,button,start,Math.min(start+0.35,total*frameHop));
    lane.appendChild(button);
  }
}
for(const row of ROWS){
  const card=document.getElementById(`card-${row.row_id}`);
  const audio=card.querySelector('audio');
  paint(card.querySelector('[data-lane="human"]'),row.labels.map(label=>PAUSE_LABEL_COLOR[label]||'#888'),audio,PAUSE_FRAME_HOP_S);
  paint(card.querySelector('[data-lane="head"]'),row.blank.map(isBlank=>isBlank?'#c9d3dd':'#1f6f8b'),audio,PAUSE_FRAME_HOP_S);
  // Disagreement is defined only where the human was decisive: an `unsure`
  // frame is an unanswered question, not a point the head got wrong.
  paint(card.querySelector('[data-lane="diff"]'),row.labels.map((label,index)=>{
    if(label==='unsure')return null;
    const shouldBeBlank=label!=='word';
    return row.blank[index]===shouldBeBlank?null:'#d64545';
  }),audio,PAUSE_FRAME_HOP_S);
}
document.getElementById('stop').onclick=()=>stop();
// Nothing to save: this page reads frozen labels and must not be able to edit
// them, or the comparison would start moving with the reader's opinion.
const saveButton=document.getElementById('save');
saveButton.disabled=true;
saveButton.title='本页只读';
document.getElementById('status').textContent=`${ROWS.length} 个窗口 · 只读`;
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, help="the labelling page's dir")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--labels", default="")
    parser.add_argument(
        "--readings",
        default="",
        help="cache of head blank runs; written when absent, reused when present "
        "so the page can be rebuilt without a GPU",
    )
    parser.add_argument("--min-blank-s", type=float, default=0.6)
    parser.add_argument("--no-nav", action="store_true")
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    manifest_path = Path(args.manifest) if args.manifest else audit_dir / "manifest.jsonl"
    labels_path = (
        Path(args.labels) if args.labels else audit_dir / MANUAL_LABEL_FILENAME
    )
    readings_path = (
        Path(args.readings) if args.readings else audit_dir / "head_readings.jsonl"
    )
    result = build(
        manifest_path=manifest_path,
        labels_path=labels_path,
        output_dir=audit_dir,
        readings_path=readings_path,
        min_blank_s=args.min_blank_s,
    )
    if not args.no_nav:
        update_audit_entrypoints(latest_html=audit_dir / "review.html", title=TITLE)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
