#!/usr/bin/env python3
"""Compare Scorer v11 dual-evidence preaudit with existing human full truth."""
from __future__ import annotations

import argparse
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
from tools.audits.compare_candidate_island_preaudits import (  # noqa: E402
    AUDIO_SPAN_PLAYER_JS,
    _audio_url,
    _frame_bounds,
    _index,
    _label_runs,
    _labels,
    _rows,
    _sha256,
)


SUMMARY_SCHEMA = "candidate_island_dual_evidence_review_summary_v1"
DETAIL_SCHEMA = "candidate_island_dual_evidence_review_item_v1"


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


def _bridged_background_gaps(
    human_labels: list[str],
    protect: list[bool],
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
                "label": "bridge",
                "start_s": start * 0.02,
                "end_s": end * 0.02,
                "start_frame": start,
                "end_frame": end,
                "duration_s": (end - start) * 0.02,
                "protected_frames": protected_frames,
                "protected_ratio": protected_frames / max(end - start, 1),
                "fully_bridged": protected_frames == end - start,
            }
        )
    return result


def _page(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scorer v11 dual-evidence held-out review</title><style>
body{{margin:0;background:#f4f7fa;color:#18212b;font-family:Segoe UI,Microsoft YaHei,sans-serif}}header{{position:sticky;top:0;z-index:3;display:flex;gap:10px;align-items:center;background:#122233;color:#fff;padding:12px 18px}}header #status{{margin-left:auto}}main{{max-width:1500px;margin:auto;padding:16px}}article{{background:#fff;border:1px solid #ccd6df;border-radius:10px;padding:14px;margin-bottom:14px}}audio{{width:100%;margin:8px 0}}.lane{{display:grid;grid-template-columns:220px 1fr;gap:8px;align-items:center;margin:8px 0}}.lane-label{{display:flex;flex-direction:column;gap:2px}}.lane-label small{{color:#607080}}.track{{position:relative;height:40px;background:#e7ebef;border-radius:5px;overflow:hidden}}.span{{position:absolute;top:0;height:100%;border:0;min-width:2px;cursor:pointer;font-size:10px;overflow:hidden;white-space:nowrap}}.human-inside{{background:#315f9d;color:#fff}}.human-outside{{background:#8d98a5;color:#fff}}.human-unsure{{background:#725190;color:#fff}}.protect{{background:#27a2c2;color:#fff}}.remove{{background:#e5bb2c;color:#1d1d1d}}.final-inside{{background:#258b57;color:#fff}}.final-outside{{background:#f2cf45;color:#1d1d1d}}.final-unsure{{background:#d87800;color:#fff}}.conflict{{background:#8d3db7;color:#fff}}.bridge{{background:#6f8734;color:#fff}}.unsafe{{background:#d32626;color:#fff}}button.playing{{outline:3px solid #111;outline-offset:-3px}}.metrics{{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;font-size:12px}}.good{{color:#087443}}.bad{{color:#b3261e;font-weight:700}}.legend{{display:flex;gap:13px;flex-wrap:wrap}}.swatch{{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:-1px}}small{{color:#607080}}</style></head><body><header><b>Scorer v11 · Protect × Remove 双证据 held-out 对照</b><button id="stop" type="button">停止播放</button><span id="status"></span></header><main><section><p>人工蓝段是既有 Split 级语音锚点，不把锚点之间每段 background 自动视为 Scorer 反例。Protect 可以桥接同一连续对话包络中的 background；页面单独列出这些 gap 供听感判断，绝不按固定时长自动判错。硬失败仍是漏保护人工语音锚点，或 Remove-only 命中人工真语音。</p><p class="legend"><span><i class="swatch protect"></i>Protect evidence</span><span><i class="swatch remove"></i>Remove evidence</span><span><i class="swatch bridge"></i>被桥接 background gap</span><span><i class="swatch conflict"></i>冲突</span><span><i class="swatch unsafe"></i>真语音误删</span></p></section><div id="list"></div></main><script>
const rows={encoded};{AUDIO_SPAN_PLAYER_JS}
function esc(v){{return String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function lane(card,audio,row,label,spans,kind,metric=''){{const line=document.createElement('div');line.className='lane';line.innerHTML=`<div class="lane-label"><b>${{esc(label)}}</b><small>${{esc(metric)}}</small></div><div class="track"></div>`;const track=line.querySelector('.track');for(const span of spans){{const button=document.createElement('button'),start=Number(span.start_s),end=Number(span.end_s),suffix=span.label==='inside_candidate'?'inside':span.label==='outside_candidate'?'outside':'unsure';button.className=`span ${{kind==='human'?`human-${{suffix}}`:kind==='final'?`final-${{suffix}}`:kind}}`;button.style.left=`${{100*start/row.duration_s}}%`;button.style.width=`${{Math.max(.12,100*(end-start)/row.duration_s)}}%`;button.title=`${{label}} ${{span.label||kind}} ${{start.toFixed(2)}}–${{end.toFixed(2)}}s`;button.textContent=`${{start.toFixed(2)}}–${{end.toFixed(2)}}s`;button.onclick=()=>play(audio,button,start,end);track.appendChild(button);}}card.appendChild(line);}}
const root=document.getElementById('list');for(const row of rows){{const card=document.createElement('article');card.innerHTML=`<h2>${{esc(row.source_id)}}</h2><small>${{esc(row.partition)}} · ${{Number(row.duration_s).toFixed(2)}}s${{row.failed_closed?' · ⚠ teacher failed closed':''}}</small><audio controls preload="metadata" src="${{esc(row.audio)}}"></audio>`;const audio=card.querySelector('audio');lane(card,audio,row,'人工语音锚点 / BG',row.human_spans,'human',`inside ${{(100*row.human_inside_ratio).toFixed(1)}}%`);lane(card,audio,row,'Protect evidence',row.protect_spans,'protect',`anchor coverage ${{(100*row.protect_recall).toFixed(1)}}% · full spans ${{row.fully_protected_human_span_count}}/${{row.human_inside_span_count}}`);lane(card,audio,row,'Remove evidence',row.remove_spans,'remove',`anchor hits ${{row.remove_human_inside_frames}} frames`);lane(card,audio,row,'最终三态标签',row.final_spans,'final',`inside ${{(100*row.final_inside_ratio).toFixed(1)}}% · outside ${{(100*row.final_outside_ratio).toFixed(1)}}% · unsure ${{(100*row.final_unsure_ratio).toFixed(1)}}%`);lane(card,audio,row,'被 Protect 桥接的 BG gap',row.bridged_background_gaps,'bridge',`${{row.bridged_gap_count}} gaps · max ${{row.max_bridged_gap_s.toFixed(2)}}s`);lane(card,audio,row,'Protect / Remove 冲突',row.conflict_spans,'conflict',`${{row.conflict_frames}} frames`);lane(card,audio,row,'真语音被 outside 命中',row.unsafe_outside_spans,'unsafe',`${{row.unsafe_outside_frames}} frames / ${{row.unsafe_outside_s.toFixed(2)}}s`);const metrics=document.createElement('div');metrics.className='metrics';metrics.innerHTML=`<span>supervised ${{(100*row.supervised_ratio).toFixed(1)}}%</span><span class="${{row.protect_recall>=.95?'good':'bad'}}">anchor coverage ${{(100*row.protect_recall).toFixed(1)}}%</span><span>outside precision ${{(100*row.final_outside_precision).toFixed(1)}}%</span><span class="${{row.unsafe_outside_frames===0?'good':'bad'}}">true-speech outside=${{row.unsafe_outside_frames}}</span>`;card.appendChild(metrics);root.appendChild(card);}}
document.getElementById('stop').onclick=()=>{{stop();document.getElementById('status').textContent='已停止';}};document.getElementById('status').textContent=`${{rows.length}} sources · unsafe ${{rows.reduce((n,r)=>n+r.unsafe_outside_frames,0)}} frames`;
</script></body></html>"""


def generate(
    *,
    manifest: Path,
    human_verdicts: Path,
    candidate: Path,
    output_dir: Path,
    update_nav: bool = True,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    human_verdicts = human_verdicts.resolve()
    candidate = candidate.resolve()
    output_dir = output_dir.resolve()
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
        bridged_gaps = _bridged_background_gaps(human_labels, protect)
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
        "unsafe_outside_s": totals["unsafe_outside_frames"] * 0.02,
        "max_bridged_gap_s": totals["max_bridged_gap_frames"] * 0.02,
        "zero_true_speech_outside": totals["unsafe_outside_frames"] == 0,
        "training_manifest_allowed": False,
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
    index.write_text(_page(details), encoding="utf-8")
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
                manifest=Path(args.manifest),
                human_verdicts=Path(args.human_verdicts),
                candidate=Path(args.candidate),
                output_dir=Path(args.output_dir),
                update_nav=args.update_nav,
            ),
            ensure_ascii=False,
        )
    )
