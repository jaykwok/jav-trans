#!/usr/bin/env python3
"""Generate a blinded boundary-only A/B audit for two CTC checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import audit_generated_at, update_audit_entrypoints  # noqa: E402
from tools.audits.review_page_core import (  # noqa: E402
    AuditReviewPageSpec,
    render_audit_review_page,
)
from tools.omni.openai_compat import slice_audio_clip  # noqa: E402


SUMMARY_SCHEMA = "ctc_alignment_ab_audit_summary_v1"
ANSWER_SCHEMA = "ctc_alignment_ab_audit_answer_v1"
VERDICT_SCHEMA = "ctc_alignment_ab_manual_verdict_v1"
MINIMUM_DELTA_MS = 20.0
DEFAULT_CLIP_S = 2.5


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def galgame_candidates(
    *, details_a: list[dict[str, Any]], details_b: list[dict[str, Any]], composites: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_a = {(str(row["sample_id"]), int(row["core_index"])): row for row in details_a}
    by_b = {(str(row["sample_id"]), int(row["core_index"])): row for row in details_b}
    candidates: list[dict[str, Any]] = []
    for composite in composites:
        sample_id = str(composite["sample_id"])
        audio = Path(str(composite["audio"]))
        if not audio.is_absolute():
            audio = PROJECT_ROOT / audio
        for core_index, core in enumerate(composite.get("core_spans") or []):
            key = (sample_id, core_index)
            row_a, row_b = by_a.get(key), by_b.get(key)
            if row_a is None or row_b is None:
                continue
            core_start, core_end = float(core["start_s"]), float(core["end_s"])
            start_a = core_start + float(row_a["start_offset_edged_ms"]) / 1000.0
            start_b = core_start + float(row_b["start_offset_edged_ms"]) / 1000.0
            end_a = core_end - float(row_a["end_offset_edged_ms"]) / 1000.0
            end_b = core_end - float(row_b["end_offset_edged_ms"]) / 1000.0
            if not (0 <= start_a < end_a <= float(composite["duration_s"]) + 1e-6):
                continue
            if not (0 <= start_b < end_b <= float(composite["duration_s"]) + 1e-6):
                continue
            candidates.append(
                {
                    "candidate_id": f"galgame:{sample_id}:{core_index}",
                    "domain": "galgame",
                    "source_id": str(core.get("core_id") or key),
                    "audio": str(audio.resolve()),
                    "audio_duration_s": float(composite["duration_s"]),
                    "text": str(core["text"]),
                    "model_a_start_s": start_a,
                    "model_a_end_s": end_a,
                    "model_b_start_s": start_b,
                    "model_b_end_s": end_b,
                }
            )
    return candidates


def jav_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": f"jav:{row['row_id']}",
            "domain": "jav",
            "source_id": str(row.get("source_id") or row["row_id"]),
            "audio": str(Path(str(row["audio"])).resolve()),
            "audio_duration_s": float(row["audio_duration_s"]),
            "text": str(row["text"]),
            "model_a_start_s": float(row["model_a_start_s"]),
            "model_a_end_s": float(row["model_a_end_s"]),
            "model_b_start_s": float(row["model_b_start_s"]),
            "model_b_end_s": float(row["model_b_end_s"]),
        }
        for row in rows
    ]


def select_trials(
    candidates: list[dict[str, Any]],
    *,
    per_boundary: int,
    clip_s: float,
    seed: int,
    domains: tuple[str, ...] = ("galgame", "jav"),
    exclude_pairs: set[tuple[str, str]] | None = None,
    minimum_delta_ms: float = MINIMUM_DELTA_MS,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    excluded = exclude_pairs or set()
    for domain in domains:
        domain_rows = [row for row in candidates if row["domain"] == domain]
        for boundary in ("onset", "end"):
            field = "start" if boundary == "onset" else "end"
            eligible: list[dict[str, Any]] = []
            for row in domain_rows:
                if (str(row["candidate_id"]), boundary) in excluded:
                    continue
                a = float(row[f"model_a_{field}_s"])
                b = float(row[f"model_b_{field}_s"])
                duration = float(row["audio_duration_s"])
                if boundary == "onset" and max(a, b) + clip_s > duration:
                    continue
                if boundary == "end" and min(a, b) - clip_s < 0.0:
                    continue
                delta_ms = abs(a - b) * 1000.0
                row_minimum_delta_ms = max(
                    float(minimum_delta_ms),
                    float(row.get("minimum_delta_ms") or 0.0),
                )
                if delta_ms < row_minimum_delta_ms:
                    continue
                eligible.append({**row, "boundary": boundary, "delta_ms": delta_ms})
            eligible.sort(key=lambda row: (-float(row["delta_ms"]), row["candidate_id"]))
            take = min(per_boundary, len(eligible))
            top_count = min((take + 1) // 2, len(eligible))
            chosen = eligible[:top_count]
            remaining = eligible[top_count:]
            random_count = take - len(chosen)
            if random_count:
                positions = sorted(rng.choice(len(remaining), size=random_count, replace=False))
                chosen.extend(remaining[int(position)] for position in positions)
            selected.extend(chosen)
    rng.shuffle(selected)
    return selected


def _clip(
    *, audio: Path, start_s: float, end_s: float, output: Path
) -> None:
    slice_audio_clip(
        source_audio=audio,
        row={"start": start_s, "end": end_s, "duration_s": end_s - start_s},
        output_path=output,
        fmt="mp3",
        bitrate="64k",
        sample_rate=16000,
        force=False,
    )


def balanced_arm1_assignments(
    trials: list[dict[str, Any]], *, seed: int
) -> dict[int, str]:
    """Balance candidate/baseline arm position inside every audit stratum."""

    rng = np.random.default_rng(seed ^ 0xA5A5)
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, trial in enumerate(trials):
        grouped.setdefault((str(trial["domain"]), str(trial["boundary"])), []).append(index)
    assignments: dict[int, str] = {}
    for indices in grouped.values():
        flags = ["model_b"] * (len(indices) // 2) + ["model_a"] * (
            len(indices) - len(indices) // 2
        )
        rng.shuffle(flags)
        assignments.update(zip(indices, flags))
    return assignments


def materialize(
    trials: list[dict[str, Any]], *, output_dir: Path, clip_s: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arm1_by_index = balanced_arm1_assignments(trials, seed=seed)
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    page_rows: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    for index, trial in enumerate(trials):
        trial_id = f"boundary-ab-{index + 1:03d}"
        boundary = str(trial["boundary"])
        field = "start" if boundary == "onset" else "end"
        duration = float(trial["audio_duration_s"])
        a_time = float(trial[f"model_a_{field}_s"])
        b_time = float(trial[f"model_b_{field}_s"])
        arm_1 = arm1_by_index[index]
        arms = [arm_1, "model_a" if arm_1 == "model_b" else "model_b"]
        arm_times = {"model_a": a_time, "model_b": b_time}
        audio = Path(str(trial["audio"]))
        if not audio.is_file():
            raise FileNotFoundError(audio)
        if boundary == "onset":
            reference_start = max(0.0, min(a_time, b_time) - 1.0)
            reference_end = min(duration, max(a_time, b_time) + clip_s)
        else:
            reference_start = max(0.0, min(a_time, b_time) - clip_s)
            reference_end = min(duration, max(a_time, b_time) + 1.0)
        reference = media_dir / f"{trial_id}-reference.mp3"
        _clip(audio=audio, start_s=reference_start, end_s=reference_end, output=reference)
        arm_media: list[str] = []
        for arm_index, arm in enumerate(arms, start=1):
            moment = arm_times[arm]
            start = moment if boundary == "onset" else moment - clip_s
            end = moment + clip_s if boundary == "onset" else moment
            path = media_dir / f"{trial_id}-{arm_index}.mp3"
            _clip(audio=audio, start_s=start, end_s=end, output=path)
            arm_media.append(path.relative_to(output_dir).as_posix())
        page_rows.append(
            {
                "row_id": trial_id,
                "domain": "Held-out Galgame" if trial["domain"] == "galgame" else "真实 JAV",
                "boundary": "开头" if boundary == "onset" else "结尾",
                "text": str(trial["text"]),
                "reference_src": reference.relative_to(output_dir).as_posix(),
                "arm_1_src": arm_media[0],
                "arm_2_src": arm_media[1],
            }
        )
        answers.append(
            {
                "schema": ANSWER_SCHEMA,
                "row_id": trial_id,
                "candidate_id": trial["candidate_id"],
                "domain": trial["domain"],
                "boundary": boundary,
                "source_id": trial["source_id"],
                "arm_1": arms[0],
                "arm_2": arms[1],
                "model_a_time_s": round(a_time, 6),
                "model_b_time_s": round(b_time, 6),
                "delta_ms": round(abs(a_time - b_time) * 1000.0, 3),
            }
        )
    return page_rows, answers


CSS = """
.contract,article{background:#fff;border:1px solid #ccd6df;border-radius:10px;padding:14px;margin-bottom:14px}
.warn{background:#fff3e0;border-left:5px solid #c77700;padding:10px 12px;margin:10px 0}
article.done{border-left:6px solid #258b57}.meta{color:#607080}.text{font-size:16px;background:#f5f7f8;padding:10px;border-radius:6px}
.players{display:grid;grid-template-columns:1fr 1fr;gap:12px}.player{border:1px solid #d5dde4;border-radius:8px;padding:10px}.reference{margin:10px 0}
audio{width:100%}.controls{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.choice{padding:9px 13px;border:1px solid #8d99a5;border-radius:5px;background:#fff;cursor:pointer}
.choice.active{outline:3px solid #18212b;outline-offset:-2px;background:#cfe3d6}.note{width:100%;min-height:40px;box-sizing:border-box}
@media(max-width:760px){.players{grid-template-columns:1fr}}
"""


JS = r"""
const rows=__ROWS__;
const allowed=new Set(['arm_1_better','arm_2_better','equivalent_good','equivalent_bad','unsure']);
const reviewCore=createAuditReviewCore({storageKey:'ctc-alignment-ab-v1:'+location.pathname,entries:rows,entryId:r=>r.row_id,defaultState:()=>({verdict:'',note:''}),isComplete:s=>allowed.has(s.verdict),statusLabel:'A/B 边界裁决',filename:'manual_verdicts.jsonl',serialize:(row,state)=>({schema:'ctc_alignment_ab_manual_verdict_v1',boundary_serialization_contract_id:'ctc_alignment_ab_audit_v1',row_id:row.row_id,verdict:state.verdict||'unreviewed',note:state.note||'',updated_at:state.updated_at||new Date().toISOString()})});
function stopCard(card){card.querySelectorAll('audio').forEach(a=>a.pause());}
function sync(card,state){card.classList.toggle('done',allowed.has(state.verdict));card.querySelectorAll('[data-value]').forEach(b=>b.classList.toggle('active',b.dataset.value===state.verdict));}
const root=document.getElementById('list');
rows.forEach((row,index)=>{const state=reviewCore.ensure(row),card=document.createElement('article');
card.innerHTML=`<h2>${index+1} / ${rows.length}</h2><div class="meta">${escapeAuditHtml(row.domain)} · 判断${escapeAuditHtml(row.boundary)}边界</div><p class="text">${escapeAuditHtml(row.text)}</p><div class="reference"><b>共同上下文（只用于定位）</b><audio controls preload="metadata" src="${escapeAuditHtml(row.reference_src)}"></audio></div><div class="players"><div class="player"><b>片段 1</b><audio controls preload="metadata" src="${escapeAuditHtml(row.arm_1_src)}"></audio></div><div class="player"><b>片段 2</b><audio controls preload="metadata" src="${escapeAuditHtml(row.arm_2_src)}"></audio></div></div><div class="controls"><button class="choice" data-value="arm_1_better">1 更自然</button><button class="choice" data-value="arm_2_better">2 更自然</button><button class="choice" data-value="equivalent_good">都可接受</button><button class="choice" data-value="equivalent_bad">都有问题</button><button class="choice" data-value="unsure">不确定</button></div><textarea class="note" placeholder="可选：被切字、拖尾、过早或过晚">${escapeAuditHtml(state.note||'')}</textarea>`;
card.querySelectorAll('audio').forEach(audio=>audio.addEventListener('play',()=>card.querySelectorAll('audio').forEach(other=>{if(other!==audio)other.pause();})));
card.querySelectorAll('[data-value]').forEach(button=>button.onclick=()=>{stopCard(card);state.verdict=button.dataset.value;state.updated_at=new Date().toISOString();sync(card,state);reviewCore.persist();});
card.querySelector('.note').onchange=e=>{state.note=e.target.value;state.updated_at=new Date().toISOString();reviewCore.persist();};sync(card,state);root.appendChild(card);});
document.getElementById('stop').onclick=()=>{document.querySelectorAll('audio').forEach(a=>a.pause());reviewCore.updateStatus('已停止');};document.getElementById('save').onclick=()=>reviewCore.save();reviewCore.updateStatus();
"""


def render_page(rows: list[dict[str, Any]]) -> str:
    intro = """
    <section class="contract"><h2>盲听规则</h2>
    <div class="warn"><b>片段 1 / 2 的模型身份已随机化，页面内不含答案。</b></div>
    <p>共同上下文只帮助定位同一句话。判断开头时，选择既不切掉首音、也没有明显多余前导的片段；判断结尾时，选择尾音完整且没有明显多余拖尾的片段。</p>
    <p><b>都可接受</b>表示差异不可听或两边都自然；<b>都有问题</b>表示两边都切字、过早或过晚；听不清时选不确定，不要猜。</p></section>
    """
    adapter = JS.replace("__ROWS__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
    return render_audit_review_page(
        AuditReviewPageSpec(
            title="CTC 边界候选 · 匿名人耳 A/B",
            intro_html=intro,
            body_html='<div id="list"></div>',
            adapter_css=CSS,
            adapter_js=adapter,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-a-details")
    parser.add_argument("--geometry-b-details")
    parser.add_argument("--composites")
    parser.add_argument("--jav-predictions")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-boundary", type=int, default=12)
    parser.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_S)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--domain",
        dest="domains",
        action="append",
        choices=("galgame", "jav"),
        help="Domain to include; repeat for both. Defaults to both domains.",
    )
    parser.add_argument(
        "--exclude-answers",
        action="append",
        default=[],
        help="Prior answers.jsonl whose candidate/boundary pairs must not be reused.",
    )
    parser.add_argument("--no-update-latest", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    domains = tuple(dict.fromkeys(args.domains or ("galgame", "jav")))
    candidates: list[dict[str, Any]] = []
    if "galgame" in domains:
        galgame_inputs = (
            args.geometry_a_details,
            args.geometry_b_details,
            args.composites,
        )
        if not all(galgame_inputs):
            parser.error(
                "galgame requires --geometry-a-details, --geometry-b-details, and --composites"
            )
        candidates.extend(
            galgame_candidates(
                details_a=_rows(Path(args.geometry_a_details)),
                details_b=_rows(Path(args.geometry_b_details)),
                composites=_rows(Path(args.composites)),
            )
        )
    if "jav" in domains:
        if not args.jav_predictions:
            parser.error("jav requires --jav-predictions")
        candidates.extend(jav_candidates(_rows(Path(args.jav_predictions))))
    excluded = {
        (str(row["candidate_id"]), str(row["boundary"]))
        for path in args.exclude_answers
        for row in _rows(Path(path))
    }
    trials = select_trials(
        candidates,
        per_boundary=args.per_boundary,
        clip_s=float(args.clip_seconds),
        seed=int(args.seed),
        domains=domains,
        exclude_pairs=excluded,
    )
    page_rows, answers = materialize(
        trials, output_dir=output_dir, clip_s=float(args.clip_seconds), seed=int(args.seed)
    )
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in page_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for row in answers:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    page = output_dir / "index.html"
    rendered = render_page(page_rows)
    # Fail closed if a future page change accidentally embeds answer-bearing labels.
    for forbidden in ("model_a", "model_b", "ctc_aligner.pt", "delta_ms"):
        if forbidden in rendered:
            raise AssertionError(f"blind page leaked {forbidden}")
    page.write_text(rendered, encoding="utf-8")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "generated_at": audit_generated_at(),
        "page": str(page.resolve()),
        "review_items": len(page_rows),
        "counts": dict(
            Counter(f"{row['domain']}:{row['boundary']}" for row in answers)
        ),
        "domains": list(domains),
        "excluded_prior_pairs": len(excluded),
        "minimum_selected_delta_ms": min((row["delta_ms"] for row in answers), default=None),
        "maximum_selected_delta_ms": max((row["delta_ms"] for row in answers), default=None),
        "clip_seconds": float(args.clip_seconds),
        "blind": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.no_update_latest:
        update_audit_entrypoints(latest_html=page, title="CTC 边界候选 · 匿名人耳 A/B")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
