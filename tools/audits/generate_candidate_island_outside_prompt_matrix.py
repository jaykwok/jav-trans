#!/usr/bin/env python3
"""Render multiple Scorer outside-teacher variants on one frozen-source page."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.audits.compare_candidate_island_preaudits import (  # noqa: E402
    AUDIO_SPAN_PLAYER_JS,
    _audio_url,
    _index,
    _label_runs,
    _labels,
    _sha256,
)


SUMMARY_SCHEMA = "candidate_island_outside_prompt_matrix_summary_v1"
DETAIL_SCHEMA = "candidate_island_outside_prompt_matrix_item_v1"


def _variant_specs(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip()
        raw_path = raw_path.strip()
        if separator != "=" or not name or not raw_path:
            raise ValueError(f"variant must use NAME=PATH: {value}")
        if name in names:
            raise ValueError(f"duplicate variant name: {name}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        names.add(name)
        result.append((name, path))
    if len(result) < 2:
        raise ValueError("outside prompt matrix requires at least two variants")
    return result


def _all_label_runs(labels: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label in ("inside_candidate", "unsure", "outside_candidate"):
        result.extend(_label_runs(labels, label=label))
    return sorted(
        result,
        key=lambda span: (span["start_s"], span["end_s"], span["label"]),
    )


def _page(payload: list[dict[str, Any]], *, variant_names: list[str]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    encoded_names = json.dumps(variant_names, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scorer outside prompt matrix</title><style>
body{{margin:0;background:#f4f7fa;color:#18212b;font-family:Segoe UI,Microsoft YaHei,sans-serif}}header{{position:sticky;top:0;z-index:3;display:flex;gap:10px;align-items:center;background:#122233;color:#fff;padding:12px 18px}}header #status{{margin-left:auto}}main{{max-width:1500px;margin:auto;padding:16px}}article{{background:#fff;border:1px solid #ccd6df;border-radius:10px;padding:14px;margin-bottom:14px}}audio{{width:100%;margin:8px 0}}.lane{{display:grid;grid-template-columns:230px 1fr;gap:8px;align-items:center;margin:9px 0}}.lane-label{{display:flex;flex-direction:column;gap:2px}}.lane-label small{{color:#617487}}.track{{position:relative;height:42px;background:#e6eaee;border-radius:5px;overflow:hidden}}.span{{position:absolute;top:0;height:100%;border:0;min-width:2px;cursor:pointer;font-size:10px;overflow:hidden;white-space:nowrap}}.outside{{background:#f2cf45;color:#1e1e1e}}.inside{{background:#258b57;color:#fff}}.unsure{{background:#d87800;color:#fff}}button.playing{{outline:3px solid #111;outline-offset:-3px}}.legend{{display:flex;gap:14px;flex-wrap:wrap}}.swatch{{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:-1px}}.warning{{color:#b3261e}}small{{color:#607080}}</style></head><body><header><b>Scorer Outside Prompt 四臂对照（同一冻结 {len(payload)} source）</b><button id="stop" type="button">停止播放</button><span id="status"></span></header><main><section><p>每个 source 只使用一个完整播放器；四条轨道共享同一音频坐标。点击任意色块只播放该精确区间，再次点击其他色块会取消尚未完成的旧加载。</p><p class="legend"><span><i class="swatch inside"></i>Inside / provisional keep</span><span><i class="swatch outside"></i>Outside candidate</span><span><i class="swatch unsure"></i>Unsure / ignore</span></p></section><div id="list"></div></main><script>
const rows={encoded};const variantNames={encoded_names};{AUDIO_SPAN_PLAYER_JS}
function esc(v){{return String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function lane(card,audio,row,name,variant){{const line=document.createElement('div');line.className='lane';line.innerHTML=`<div class="lane-label"><b>${{esc(name)}}</b><small>outside ${{(100*variant.outside_ratio).toFixed(1)}}% · keep ${{(100*variant.inside_ratio).toFixed(1)}}% · unsure ${{(100*variant.unsure_ratio).toFixed(1)}}%</small></div><div class="track"></div>`;const track=line.querySelector('.track');for(const span of variant.spans){{const button=document.createElement('button'),start=Number(span.start_s),end=Number(span.end_s),kind=span.label==='outside_candidate'?'outside':span.label==='unsure'?'unsure':'inside';button.className=`span ${{kind}}`;button.style.left=`${{100*start/row.duration_s}}%`;button.style.width=`${{Math.max(.12,100*(end-start)/row.duration_s)}}%`;button.title=`${{name}} ${{span.label}} ${{start.toFixed(2)}}–${{end.toFixed(2)}}s`;button.textContent=`${{start.toFixed(2)}}–${{end.toFixed(2)}}s`;button.onclick=()=>play(audio,button,start,end);track.appendChild(button);}}card.appendChild(line);if(variant.failed_closed){{const note=document.createElement('p');note.className='warning';note.textContent=`${{name}} fail-closed: ${{variant.reason||''}}`;card.appendChild(note);}}}}
const root=document.getElementById('list');for(const row of rows){{const card=document.createElement('article');card.innerHTML=`<h2>${{esc(row.source_id)}}</h2><small>${{Number(row.duration_s).toFixed(2)}}s</small><audio controls preload="metadata" src="${{esc(row.audio)}}"></audio>`;const audio=card.querySelector('audio');for(const name of variantNames)lane(card,audio,row,name,row.variants[name]);root.appendChild(card);}}
document.getElementById('stop').onclick=()=>{{stop();document.getElementById('status').textContent='已停止';}};document.getElementById('status').textContent=`${{rows.length}} sources · ${{variantNames.length}} variants`;</script></body></html>"""


def generate(
    *,
    manifest: Path,
    variants: list[tuple[str, Path]],
    output_dir: Path,
    limit: int = 0,
    update_nav: bool = True,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    output_dir = output_dir.resolve()
    sources = _index(manifest, name="source manifest")
    if limit > 0:
        sources = dict(list(sources.items())[:limit])
    indexes = {
        name: _index(path, name=f"variant {name}")
        for name, path in variants
    }
    for name, index in indexes.items():
        missing = set(sources) - set(index)
        if missing:
            raise ValueError(
                f"variant {name} is missing frozen sources: {sorted(missing)[:3]}"
            )

    totals = {
        name: {
            "frame_count": 0,
            "inside_frames": 0,
            "outside_frames": 0,
            "unsure_frames": 0,
            "outside_span_count": 0,
            "failed_closed_count": 0,
        }
        for name, _path in variants
    }
    details: list[dict[str, Any]] = []
    for source_id, source in sources.items():
        frame_count = int(source["frame_count"])
        item_variants: dict[str, Any] = {}
        for name, _path in variants:
            teacher = indexes[name][source_id]
            if int(teacher.get("frame_count") or 0) != frame_count:
                raise ValueError(f"variant {name} frame geometry mismatch: {source_id}")
            if str(teacher.get("audio_sha256") or "") != str(
                source.get("audio_sha256") or ""
            ):
                raise ValueError(f"variant {name} audio identity mismatch: {source_id}")
            labels = _labels(teacher, frame_count=frame_count)
            counts = {
                "inside_frames": labels.count("inside_candidate"),
                "outside_frames": labels.count("outside_candidate"),
                "unsure_frames": labels.count("unsure"),
            }
            spans = _all_label_runs(labels)
            outside_span_count = sum(
                span["label"] == "outside_candidate" for span in spans
            )
            aggregate = totals[name]
            aggregate["frame_count"] += frame_count
            for key, value in counts.items():
                aggregate[key] += value
            aggregate["outside_span_count"] += outside_span_count
            aggregate["failed_closed_count"] += int(
                bool(teacher.get("teacher_failed_closed"))
            )
            item_variants[name] = {
                "prompt_version": str(teacher.get("prompt_version") or ""),
                "reason": str(teacher.get("overall_reason") or ""),
                "failed_closed": bool(teacher.get("teacher_failed_closed")),
                "spans": spans,
                **counts,
                "outside_span_count": outside_span_count,
                "inside_ratio": counts["inside_frames"] / max(frame_count, 1),
                "outside_ratio": counts["outside_frames"] / max(frame_count, 1),
                "unsure_ratio": counts["unsure_frames"] / max(frame_count, 1),
            }
        details.append(
            {
                "schema": DETAIL_SCHEMA,
                "source_id": source_id,
                "duration_s": float(source["duration_s"]),
                "frame_count": frame_count,
                "audio": _audio_url(str(source["audio"]), manifest=manifest),
                "variants": item_variants,
            }
        )

    summary_variants: dict[str, Any] = {}
    for name, path in variants:
        aggregate = totals[name]
        frame_count = max(int(aggregate["frame_count"]), 1)
        summary_variants[name] = {
            "preaudit": str(path),
            "preaudit_sha256": _sha256(path),
            **aggregate,
            "inside_ratio": aggregate["inside_frames"] / frame_count,
            "outside_ratio": aggregate["outside_frames"] / frame_count,
            "unsure_ratio": aggregate["unsure_frames"] / frame_count,
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
    summary = {
        "schema": SUMMARY_SCHEMA,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "source_count": len(details),
        "variant_count": len(variants),
        "variant_order": [name for name, _path in variants],
        "variants": summary_variants,
        "limit": limit,
        "audit_navigation_updated": update_nav,
        "training_manifest_allowed": False,
        "per_source": str(detail_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    index = output_dir / "index.html"
    index.write_text(
        _page(details, variant_names=summary["variant_order"]),
        encoding="utf-8",
    )
    if update_nav:
        update_audit_entrypoints(
            latest_html=index,
            title="Scorer outside prompt four-arm matrix",
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Repeat NAME=PREAUDIT_JSONL in the desired display order.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
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
                variants=_variant_specs(args.variant),
                output_dir=Path(args.output_dir),
                limit=args.limit,
                update_nav=args.update_nav,
            ),
            ensure_ascii=False,
        )
    )
