#!/usr/bin/env python3
"""Shared shell for blind two-answer clip audits.

Several questions in this project reduce to the same shape: play one clip, pick
A or B. Multi-class vocabularies were tried and are slow to listen through and
hard to pool, so the pages that matter are binary. What differs between them is
only the question, the two answer labels, and the framing - not the mechanics.

Three mechanics are enforced here rather than left to each adapter, because each
one is a way for an audit to quietly stop measuring anything:

  * the page is built only from the fields it needs, and a manifest carrying the
    answer is rejected outright rather than trusted not to be displayed;
  * `unsure` always exists and is never a third answer. It is the exit for an
    unjudgeable clip, because a forced guess becomes evidence;
  * clip order comes from the selector, which shuffles, so position leaks
    nothing about which stratum an item came from.

Choosing an answer also stops playback, so the clip being judged is never still
running while the next one is considered. Both `stop()` and `audio.pause()` are
called: the first unwinds the span player's timers and handlers, the second
covers playback the auditor started from the native `<audio controls>` element,
which the span player does not know about.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import re
from typing import Any

from tools.audits.review_page_core import (
    AuditOptionAxis,
    AuditReviewPageSpec,
    render_audit_review_page,
    validate_audit_option_contract,
)

UNSURE = "unsure"
# Fields a manifest may carry. Anything else is refused, because the reason an
# adapter would want extra fields is to display them.
ALLOWED_MANIFEST_FIELDS = {"schema", "row_id", "audio", "start_s", "end_s"}
# mp3 frame padding moves the decoded length by a few tens of ms; anything past
# this is a clip pointing at the wrong interval, not an encoder artefact.
CLIP_DURATION_TOLERANCE_S = 0.15

ADAPTER_CSS = """
.contract,article{background:#fff;border:1px solid #ccd6df;border-radius:10px;padding:14px;margin-bottom:14px}
.prompt{background:#eef6ff;border-left:5px solid #315f9d;padding:10px 12px;margin:10px 0}
.warn{background:#fff3e0;border-left:5px solid #c77700;padding:10px 12px;margin:10px 0}
.contract table{width:100%;border-collapse:collapse}
.contract th,.contract td{border:1px solid #c9d3dc;padding:7px;text-align:left;vertical-align:top}
.contract th{background:#edf1f5}
article.done{border-left:6px solid #258b57}
.meta{color:#607080;overflow-wrap:anywhere}
.controls{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin:8px 0}
.choice,.play-whole,.play-context{border:1px solid #8d99a5;border-radius:5px;background:#fff;padding:9px 14px;cursor:pointer;font-size:15px}
.play-whole{border-color:#315f9d;color:#1d3f6e;font-weight:600}
.play-context{border-style:dashed;color:#4a5a6a}
.play-whole.playing,.play-context.playing{background:#dfeaf7}
.choice.active{outline:3px solid #18212b;outline-offset:-2px}
.choice[data-role="a"].active{background:#bfe5cc}
.choice[data-role="b"].active{background:#cfd9e5}
.choice[data-role="unsure"].active{background:#f3d49d}
.note{width:100%;min-height:40px;box-sizing:border-box}
"""

ADAPTER_JS = r"""
const rows=__ROWS__,config=__CONFIG__;
const allowedVerdicts=new Set([config.optionA.value,config.optionB.value,'unsure'].concat(config.optionC?[config.optionC.value]:[]));
const reviewCore=createAuditReviewCore({storageKey:config.storageKey+':'+location.pathname,entries:rows,entryId:row=>row.row_id,defaultState:()=>({verdict:'',note:''}),isComplete:state=>allowedVerdicts.has(state.verdict),statusLabel:config.statusLabel,filename:'manual_verdicts.jsonl',serialize:(row,state)=>({schema:config.verdictSchema,boundary_serialization_contract_id:config.boundaryContract,row_id:row.row_id,start_s:Number(row.start_s),end_s:Number(row.end_s),verdict:state.verdict||'unreviewed',note:state.note||'',updated_at:state.updated_at||new Date().toISOString()})});
function sync(card,state){card.classList.toggle('done',allowedVerdicts.has(state.verdict));card.querySelectorAll('[data-value]').forEach(button=>button.classList.toggle('active',button.dataset.value===state.verdict));}
const root=document.getElementById('list');
rows.forEach((row,index)=>{const state=reviewCore.ensure(row),card=document.createElement('article');
card.innerHTML=`<h2>${index+1} / ${rows.length}</h2><div class="meta">${escapeAuditHtml(row.row_id)} · 片段 ${formatAuditTimestamp(row.clip_duration_s)}</div><audio controls preload="metadata" src="${escapeAuditHtml(row.audio_src)}"></audio><div class="controls"><button type="button" class="play-whole">${config.contextSeconds>0?'▶ 从切点':'重播'}</button>${config.contextSeconds>0?'<button type="button" class="play-context">▶ 带前文</button>':''}<button type="button" class="choice" data-role="a" data-value="${escapeAuditHtml(config.optionA.value)}">${escapeAuditHtml(config.optionA.label)}</button><button type="button" class="choice" data-role="b" data-value="${escapeAuditHtml(config.optionB.value)}">${escapeAuditHtml(config.optionB.label)}</button>${config.optionC?`<button type="button" class="choice" data-role="c" data-value="${escapeAuditHtml(config.optionC.value)}">${escapeAuditHtml(config.optionC.label)}</button>`:''}<button type="button" class="choice" data-role="unsure" data-value="unsure">不确定</button></div><textarea class="note" placeholder="${escapeAuditHtml(config.notePlaceholder)}">${escapeAuditHtml(state.note||'')}</textarea>`;
const audio=card.querySelector('audio'),playButton=card.querySelector('.play-whole');
const contextSeconds=Number(config.contextSeconds)||0;
playButton.onclick=()=>play(audio,playButton,contextSeconds,Number(row.clip_duration_s));
const contextButton=card.querySelector('.play-context');
if(contextButton){contextButton.onclick=()=>play(audio,contextButton,0,Number(row.clip_duration_s));}
card.querySelectorAll('[data-value]').forEach(button=>button.onclick=()=>{stop();audio.pause();state.verdict=button.dataset.value;state.updated_at=new Date().toISOString();sync(card,state);reviewCore.persist();});
card.querySelector('.note').onchange=event=>{state.note=event.target.value;state.updated_at=new Date().toISOString();reviewCore.persist();};
sync(card,state);root.appendChild(card);});
document.getElementById('stop').onclick=()=>{stop();reviewCore.updateStatus('已停止');};
document.getElementById('save').onclick=()=>reviewCore.save();
reviewCore.updateStatus();
"""


@dataclass(frozen=True)
class BinaryOption:
    value: str
    label: str


@dataclass(frozen=True)
class BinaryClipAuditSpec:
    title: str
    option_a: BinaryOption
    option_b: BinaryOption
    prompt: str
    intro_html: str
    verdict_schema: str
    storage_key: str
    status_label: str
    boundary_contract: str
    note_placeholder: str = "可选：听到了什么"
    # An optional third answer, for audits where a real category of clip fits
    # neither A nor B and would otherwise be forced into one of them or into
    # `unsure` - which would quietly mix "does not apply" with "cannot tell".
    # Default None keeps every existing two-option audit byte-identical.
    option_c: "BinaryOption | None" = None
    # Seconds of run-up carried at the head of every clip. When > 0 the card
    # gets a second play button that starts at 0 instead of at the cut, so the
    # auditor can hear what led into the moment being judged. The clip file is
    # the same either way, which keeps every clip the same length and stops
    # duration from encoding anything.
    context_seconds: float = 0.0


def option_contract(spec: BinaryClipAuditSpec) -> dict[tuple[str, ...], str]:
    """Every answer plus the exit, checked against the shared audit contract."""
    options = [spec.option_a, spec.option_b]
    if spec.option_c is not None:
        options.append(spec.option_c)
    values = tuple(option.value for option in options)
    if len(set(values)) != len(values) or UNSURE in values:
        raise ValueError(f"audit option values must be distinct and not {UNSURE!r}")
    axes = (AuditOptionAxis(field="verdict", options=(*values, UNSURE)),)
    results: dict[tuple[str, ...], str] = {
        (option.value,): option.value for option in options
    }
    results[(UNSURE,)] = UNSURE
    validate_audit_option_contract(axes=axes, combination_results=results)
    return results


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "item"


def check_manifest_row(row: dict[str, Any]) -> None:
    """Refuse anything that could tell the auditor the answer."""
    extra = set(row) - ALLOWED_MANIFEST_FIELDS
    if extra:
        raise ValueError(
            "manifest must not carry "
            f"{sorted(extra)}; keep answer-bearing fields in the key file"
        )


def render_page(spec: BinaryClipAuditSpec, rows: list[dict[str, Any]]) -> str:
    prompt_html = html.escape(spec.prompt).replace("\n", "<br>")
    intro = (
        '<section class="contract"><h2>这一页在问什么</h2>'
        f'<div class="prompt"><b>本页审计提示</b><p>{prompt_html}</p></div>'
        f"{spec.intro_html}</section>"
    )
    config = {
        "optionA": {"value": spec.option_a.value, "label": spec.option_a.label},
        "optionB": {"value": spec.option_b.value, "label": spec.option_b.label},
        "optionC": (
            {"value": spec.option_c.value, "label": spec.option_c.label}
            if spec.option_c is not None
            else None
        ),
        "contextSeconds": float(spec.context_seconds),
        "verdictSchema": spec.verdict_schema,
        "storageKey": spec.storage_key,
        "statusLabel": spec.status_label,
        "boundaryContract": spec.boundary_contract,
        "notePlaceholder": spec.note_placeholder,
    }
    adapter_js = ADAPTER_JS.replace(
        "__ROWS__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    ).replace("__CONFIG__", json.dumps(config, ensure_ascii=False))
    return render_audit_review_page(
        AuditReviewPageSpec(
            title=spec.title,
            intro_html=intro,
            body_html='<div id="list"></div>',
            adapter_css=ADAPTER_CSS,
            adapter_js=adapter_js,
        )
    )


def copy_clips(
    source_rows: list[dict[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    """Serve pre-cut clips unchanged, for auditing a decision made ON them.

    When the question is whether a teacher's call on a clip was right, re-cutting
    that clip from its source introduces a difference between what the teacher
    heard and what the human hears - and that difference lands entirely inside
    the disagreements, which is where it does the most damage. So the bytes are
    copied, not transcoded.

    The manifest interval must still describe the clip, and is checked against
    the file rather than trusted: a clip/label misalignment would otherwise show
    up as an unexplained false-drop rate.
    """
    import shutil

    from pipeline.audio import probe_video_duration_s

    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in source_rows:
        check_manifest_row(source)
        row_id = str(source.get("row_id") or "")
        if not row_id or row_id in seen:
            raise ValueError("audit row_id values must be non-empty and unique")
        seen.add(row_id)
        audio = Path(str(source.get("audio") or ""))
        if not audio.is_file():
            raise FileNotFoundError(f"audit source audio not found: {audio}")
        start = float(source.get("start_s") or 0.0)
        end = float(source.get("end_s") or 0.0)
        if end <= start:
            raise ValueError(f"audit interval must be positive: {row_id}")
        probed = probe_video_duration_s(str(audio))
        if probed is None:
            raise ValueError(f"could not probe clip duration: {audio}")
        if abs(probed - (end - start)) > CLIP_DURATION_TOLERANCE_S:
            raise ValueError(
                f"clip {audio.name} runs {probed:.3f}s but the label says "
                f"{end - start:.3f}s; clip and label are not the same interval"
            )
        clip = media_dir / f"{safe_name(row_id)}{audio.suffix}"
        shutil.copyfile(audio, clip)
        rows.append(
            {
                "row_id": row_id,
                "audio_src": clip.relative_to(output_dir).as_posix(),
                "start_s": 0.0,
                "end_s": round(probed, 6),
                "clip_duration_s": round(probed, 6),
            }
        )
    return rows


def cut_clips(
    source_rows: list[dict[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    """Slice each manifest interval to mp3, refusing answer-bearing manifests."""
    from tools.omni.openai_compat import slice_audio_clip

    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in source_rows:
        check_manifest_row(source)
        row_id = str(source.get("row_id") or "")
        if not row_id or row_id in seen:
            raise ValueError("audit row_id values must be non-empty and unique")
        seen.add(row_id)
        audio = Path(str(source.get("audio") or ""))
        if not audio.is_file():
            raise FileNotFoundError(f"audit source audio not found: {audio}")
        start = float(source.get("start_s") or 0.0)
        end = float(source.get("end_s") or 0.0)
        if end <= start:
            raise ValueError(f"audit interval must be positive: {row_id}")
        clip = media_dir / f"{safe_name(row_id)}.mp3"
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
                "row_id": row_id,
                "audio_src": clip.relative_to(output_dir).as_posix(),
                "start_s": start,
                "end_s": end,
                "clip_duration_s": end - start,
            }
        )
    return rows
