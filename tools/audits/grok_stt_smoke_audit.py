#!/usr/bin/env python3
"""Run a small Grok STT word-timestamp audit through OpenRouter.

The OpenRouter STT default response only contains ``text`` and ``usage``.  Grok
word timings are exposed when the request explicitly asks for ``verbose_json``
and ``timestamp_granularities=["word"]``.  This tool keeps the provider result
separate from training truth: it copies a fixed sample, stores the raw response,
and builds a page on which a human can judge transcript and timing quality.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import shutil
import sys
import time
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.audits.audit_nav import (  # noqa: E402
    audit_generated_at,
    update_audit_entrypoints,
)
from tools.audits.review_page_core import (  # noqa: E402
    AuditReviewPageSpec,
    render_audit_review_page,
)
from tools.omni.openai_compat import (  # noqa: E402
    load_env_file,
    normalize_openai_compat_base_url,
)


TITLE = "Grok STT 词时间边界 · 教师 Smoke 审计"
MODEL = "x-ai/grok-stt-1.0"
RESULT_SCHEMA = "grok_stt_smoke_result_v1"
SELECTION_SCHEMA = "grok_stt_smoke_selection_v1"
SUMMARY_SCHEMA = "grok_stt_smoke_summary_v1"
MANUAL_VERDICT_SCHEMA = "grok_stt_smoke_manual_verdict_v1"
MANUAL_VERDICT_FILENAME = "manual_verdicts.jsonl"
MIN_AUDITION_GAP_S = 0.2
FRAME_HOP_S = 1.0 / 26.0
MERGE_GAP_FRAMES = 4
MIN_LEXICAL_ISLAND_FRAMES = 2
BOUNDARY_IGNORE_FRAMES = 2
CUT_CANDIDATE_FRAMES = 8


TRANSCRIPT_OPTIONS = (
    ("correct", "转写正确"),
    ("minor_error", "少量错误"),
    ("major_error", "严重错误"),
    ("should_be_empty", "本应无词（幻听）"),
    ("unsure", "无法判断"),
)
TIMING_OPTIONS = (
    ("accurate", "边界准确"),
    ("usable", "基本可用"),
    ("bad", "明显不可用"),
    ("unsure", "无法判断"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(dict(value))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def build_request_payload(
    audio_bytes: bytes,
    *,
    model: str = MODEL,
    language: str = "ja",
    audio_format: str = "wav",
) -> dict[str, Any]:
    """The exact request that made OpenRouter preserve Grok's word timings."""

    return {
        "model": model,
        "language": language,
        "response_format": "verbose_json",
        "timestamp_granularities": ["word"],
        "provider": {
            "data_collection": "allow",
            "zdr": False,
            "require_parameters": True,
        },
        "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": audio_format,
        },
    }


def _union_seconds(words: Sequence[Mapping[str, Any]]) -> float:
    intervals = sorted(
        (float(word["start_s"]), float(word["end_s"])) for word in words
    )
    total = 0.0
    left: float | None = None
    right: float | None = None
    for start, end in intervals:
        if left is None:
            left, right = start, end
        elif start <= float(right):
            right = max(float(right), end)
        else:
            total += float(right) - left
            left, right = start, end
    if left is not None:
        total += float(right) - left
    return total


def word_gaps(
    words: Sequence[Mapping[str, Any]], duration_s: float
) -> list[dict[str, float]]:
    """Return uncovered intervals, including leading and trailing edges."""

    duration = max(0.0, float(duration_s))
    intervals = sorted(
        (
            max(0.0, min(duration, float(word["start_s"]))),
            max(0.0, min(duration, float(word["end_s"]))),
        )
        for word in words
    )
    gaps: list[dict[str, float]] = []
    cursor = 0.0
    for start, end in intervals:
        if start > cursor:
            gaps.append({"start_s": round(cursor, 6), "end_s": round(start, 6)})
        cursor = max(cursor, end)
    if duration > cursor:
        gaps.append({"start_s": round(cursor, 6), "end_s": round(duration, 6)})
    return gaps


def _punctuation_only(value: str) -> bool:
    characters = [character for character in str(value) if not character.isspace()]
    return bool(characters) and all(
        unicodedata.category(character).startswith("P") for character in characters
    )


def merge_lexical_islands(
    words: Sequence[Mapping[str, Any]],
    *,
    max_gap_s: float = MERGE_GAP_FRAMES * FRAME_HOP_S,
) -> list[dict[str, Any]]:
    """Close tokenizer-scale gaps without turning punctuation into speech."""

    lexical = [
        word
        for word in words
        if str(word.get("text") or "").strip()
        and not _punctuation_only(str(word.get("text") or ""))
    ]
    lexical.sort(
        key=lambda word: (float(word["start_s"]), float(word["end_s"]))
    )
    islands: list[dict[str, Any]] = []
    for word in lexical:
        start = float(word["start_s"])
        end = float(word["end_s"])
        text_value = str(word["text"])
        previous = islands[-1] if islands else None
        if previous is not None and start - float(previous["end_s"]) <= max_gap_s:
            previous["end_s"] = round(max(float(previous["end_s"]), end), 6)
            previous["text"] += text_value
            previous["unit_count"] += 1
        else:
            islands.append(
                {
                    "text": text_value,
                    "start_s": round(start, 6),
                    "end_s": round(end, 6),
                    "unit_count": 1,
                }
            )
    return islands


def _compress_frame_labels(
    labels: Sequence[str], *, duration_s: float
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    if not labels:
        return spans
    start = 0
    for index in range(1, len(labels) + 1):
        if index < len(labels) and labels[index] == labels[start]:
            continue
        spans.append(
            {
                "label": labels[start],
                "start_frame": start,
                "end_frame": index,
                "start_s": round(start * FRAME_HOP_S, 6),
                "end_s": round(min(float(duration_s), index * FRAME_HOP_S), 6),
            }
        )
        start = index
    return spans


def compile_frame_supervision(
    words: Sequence[Mapping[str, Any]], duration_s: float
) -> dict[str, Any]:
    """Compile Grok units into lexical islands and a complete frame partition.

    Word islands remain positive.  Only the two frames *outside* each teacher
    edge are ignored, so boundary jitter is not trained as a hard negative.  A
    cut candidate is a remaining non-word run of at least eight frames; it is a
    review aid, never an automatic training or runtime decision.
    """

    duration = max(0.0, float(duration_s))
    frame_count = int(duration / FRAME_HOP_S + 1e-9)
    merged_islands = merge_lexical_islands(words)
    minimum_island_s = MIN_LEXICAL_ISLAND_FRAMES * FRAME_HOP_S
    islands = [
        island
        for island in merged_islands
        if float(island["end_s"]) - float(island["start_s"]) >= minimum_island_s
    ]
    ignored_short_islands = [
        island
        for island in merged_islands
        if float(island["end_s"]) - float(island["start_s"]) < minimum_island_s
    ]
    labels = ["non_word"] * frame_count
    for island in islands:
        start_s = float(island["start_s"])
        end_s = float(island["end_s"])
        for frame in range(frame_count):
            center = (frame + 0.5) * FRAME_HOP_S
            if start_s <= center < end_s:
                labels[frame] = "word"
        ignore_start = max(0.0, start_s - BOUNDARY_IGNORE_FRAMES * FRAME_HOP_S)
        ignore_end = min(duration, end_s + BOUNDARY_IGNORE_FRAMES * FRAME_HOP_S)
        for frame in range(frame_count):
            if labels[frame] != "non_word":
                continue
            center = (frame + 0.5) * FRAME_HOP_S
            if ignore_start <= center < ignore_end:
                labels[frame] = "ignore"
    # A sub-two-frame island cannot be resolved by this head.  It may be a real
    # short interjection or a non-semantic vocalization misread as one; either
    # way, turning it into a hard positive or negative would invent certainty.
    for island in ignored_short_islands:
        ignore_start = max(
            0.0,
            float(island["start_s"]) - BOUNDARY_IGNORE_FRAMES * FRAME_HOP_S,
        )
        ignore_end = min(
            duration,
            float(island["end_s"]) + BOUNDARY_IGNORE_FRAMES * FRAME_HOP_S,
        )
        for frame in range(frame_count):
            center = (frame + 0.5) * FRAME_HOP_S
            if ignore_start <= center < ignore_end:
                labels[frame] = "ignore"
    supervision = _compress_frame_labels(labels, duration_s=duration)
    cut_candidates = [
        {
            "start_s": span["start_s"],
            "end_s": span["end_s"],
            "frame_count": int(span["end_frame"]) - int(span["start_frame"]),
        }
        for span in supervision
        if span["label"] == "non_word"
        and int(span["end_frame"]) - int(span["start_frame"])
        >= CUT_CANDIDATE_FRAMES
    ]
    return {
        "frame_hop_s": FRAME_HOP_S,
        "frame_count": frame_count,
        "merge_gap_frames": MERGE_GAP_FRAMES,
        "minimum_lexical_island_frames": MIN_LEXICAL_ISLAND_FRAMES,
        "boundary_ignore_frames": BOUNDARY_IGNORE_FRAMES,
        "cut_candidate_frames": CUT_CANDIDATE_FRAMES,
        "lexical_islands": islands,
        "ignored_short_islands": ignored_short_islands,
        "frame_supervision": supervision,
        "cut_candidates": cut_candidates,
    }


def normalize_response(
    response: Mapping[str, Any],
    *,
    fallback_duration_s: float,
) -> dict[str, Any]:
    """Normalize both xAI's ``text`` and OpenRouter's ``word`` token key."""

    duration = _number(response.get("duration"))
    if duration is None or duration <= 0:
        duration = float(fallback_duration_s)
    words: list[dict[str, Any]] = []
    invalid_words: list[dict[str, Any]] = []
    for index, raw in enumerate(response.get("words") or []):
        if not isinstance(raw, Mapping):
            invalid_words.append({"index": index, "reason": "not_an_object"})
            continue
        text_value = str(raw.get("word") or raw.get("text") or "")
        start = _number(raw.get("start"))
        end = _number(raw.get("end"))
        if not text_value or start is None or end is None or start < 0 or end <= start:
            invalid_words.append(
                {"index": index, "reason": "invalid_text_or_span", "raw": dict(raw)}
            )
            continue
        words.append(
            {
                "text": text_value,
                "start_s": round(start, 6),
                "end_s": round(end, 6),
            }
        )
    words.sort(key=lambda item: (item["start_s"], item["end_s"], item["text"]))
    overlaps = sum(
        1
        for previous, current in zip(words, words[1:])
        if float(current["start_s"]) < float(previous["end_s"])
    )
    out_of_range = sum(
        1
        for word in words
        if float(word["start_s"]) > duration or float(word["end_s"]) > duration + 0.05
    )
    gaps = word_gaps(words, duration)
    audible_gaps = [
        gap
        for gap in gaps
        if float(gap["end_s"]) - float(gap["start_s"]) >= MIN_AUDITION_GAP_S
    ]
    coverage = _union_seconds(words)
    return {
        "transcript": str(response.get("text") or ""),
        "language": str(response.get("language") or ""),
        "duration_s": round(duration, 6),
        "words": words,
        "gaps": audible_gaps,
        "segments": list(response.get("segments") or []),
        "usage": dict(response.get("usage") or {}),
        "diagnostics": {
            "word_count": len(words),
            "invalid_word_count": len(invalid_words),
            "overlap_count": overlaps,
            "out_of_range_count": out_of_range,
            "timed_coverage_s": round(coverage, 6),
            "timed_coverage_share": round(coverage / duration, 6) if duration else None,
            "audition_gap_count": len(audible_gaps),
            "longest_gap_s": round(
                max(
                    (
                        float(gap["end_s"]) - float(gap["start_s"])
                        for gap in gaps
                    ),
                    default=0.0,
                ),
                6,
            ),
        },
        "invalid_words": invalid_words,
    }


def _api_url(base_url: str) -> str:
    normalized = normalize_openai_compat_base_url(base_url)
    if not normalized:
        normalized = "https://openrouter.ai/api/v1"
    return normalized.rstrip("/") + "/audio/transcriptions"


def call_openrouter_stt(
    *,
    audio_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    language: str,
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    payload = build_request_payload(
        audio_path.read_bytes(),
        model=model,
        language=language,
        audio_format=audio_path.suffix.lstrip(".").lower() or "wav",
    )
    request = Request(
        _api_url(base_url),
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/local/jav-trans",
            "X-Title": "jav-trans Grok STT smoke audit",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=float(timeout_s)) as response:  # noqa: S310
            raw = json.loads(response.read().decode("utf-8"))
            headers = {
                "x-generation-id": str(response.headers.get("X-Generation-Id") or "")
            }
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter STT HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"OpenRouter STT transport error: {error.reason}") from error
    if not isinstance(raw, Mapping):
        raise RuntimeError("OpenRouter STT response must be a JSON object")
    return dict(raw), headers


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def _option_buttons(name: str, options: Sequence[tuple[str, str]]) -> str:
    return "".join(
        f'<button type="button" data-field="{html.escape(name)}" '
        f'data-value="{html.escape(value)}">{html.escape(label)}</button>'
        for value, label in options
    )


def _page_row(result: Mapping[str, Any]) -> dict[str, Any]:
    normalized = result.get("response") or {}
    diagnostics = normalized.get("diagnostics") or {}
    duration_s = float(normalized.get("duration_s") or result["source_duration_s"])
    compiled = compile_frame_supervision(
        list(normalized.get("words") or []), duration_s
    )
    return {
        "audit_id": str(result["audit_id"]),
        "audio": str(result["audio"]),
        "duration_s": duration_s,
        "transcript": str(normalized.get("transcript") or ""),
        "language": str(normalized.get("language") or ""),
        "words": list(normalized.get("words") or []),
        "gaps": list(normalized.get("gaps") or []),
        **compiled,
        "diagnostics": dict(diagnostics),
        "latency_s": float(result.get("latency_s") or 0.0),
        "cost": _number((normalized.get("usage") or {}).get("cost")),
    }


def build_page(results: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    page_rows = [_page_row(result) for result in results]
    cards: list[str] = []
    for index, row in enumerate(page_rows, start=1):
        diagnostics = row["diagnostics"]
        coverage = diagnostics.get("timed_coverage_share")
        coverage_text = "n/a" if coverage is None else f"{float(coverage) * 100:.1f}%"
        cost = row.get("cost")
        cost_text = "n/a" if cost is None else f"${float(cost):.6f}"
        transcript = html.escape(row["transcript"]) or "（空转写）"
        cards.append(
            f'<section class="teacher-card" id="card-{html.escape(row["audit_id"])}">'
            f'<div class="teacher-heading"><h2>样本 {index:02d}</h2>'
            f'<small>{float(row["duration_s"]):.2f}s · '
            f'{int(diagnostics.get("word_count") or 0)} 个时间单元 · '
            f'{len(row["lexical_islands"])} 个合并语音岛 · '
            f'{len(row["ignored_short_islands"])} 个短岛 ignore · '
            f'覆盖 {coverage_text} · 请求 {float(row["latency_s"]):.2f}s · {cost_text}</small></div>'
            f'<audio controls preload="none" src="{html.escape(row["audio"])}"></audio>'
            f'<div class="teacher-transcript"><b>Grok 转写</b><p>{transcript}</p></div>'
            f'<div class="teacher-timeline" data-timeline></div>'
            f'<div class="teacher-tokens" data-tokens></div>'
            f'<div class="teacher-options"><div><b>转写判断</b>'
            f'{_option_buttons("transcript", TRANSCRIPT_OPTIONS)}</div>'
            f'<div><b>时间判断</b>{_option_buttons("timing", TIMING_OPTIONS)}</div></div>'
            f'<textarea rows="2" placeholder="备注（可空）"></textarea>'
            f'<div class="teacher-warning" data-warning></div>'
            f'</section>'
        )

    total_cost = sum(float(row.get("cost") or 0.0) for row in page_rows)
    intro = (
        '<section class="teacher-intro"><p><b>怎么审：</b>先盲听整段，再看转写；'
        '点时间条或字符按钮可只听 Grok 标出的那一段，灰色按钮试听模型未覆盖的区间。'
        '最后分别判断转写和时间边界。</p>'
        '<p><b>灰色不等于静音或安全切点：</b>它只表示 Grok 没有返回词时间，'
        '里面仍可能是呼吸、呻吟、噪声或漏掉的台词；必须听过才能裁决。</p>'
        '<p><b>合并规则：</b>去掉纯标点，相邻间隔不超过 4 帧（约154ms）就合并；'
        '合并后不足 2 帧（约77ms）的孤立短岛降为 ignore；语音岛外侧各 2 帧也标为 ignore，'
        '连续 non-word 至少 8 帧（约308ms）才显示为潜在切点。'
        '颜色：<i class="legend raw"></i>原始单元 '
        '<i class="legend island"></i>合并语音岛 '
        '<i class="legend word"></i>word '
        '<i class="legend ignore"></i>ignore '
        '<i class="legend nonword"></i>non-word。</p>'
        '<p>样本来自三个既有抽样池、每池数量相同并使用匿名编号；'
        '<b>页面不显示原分类</b>。这里是教师候选审计，不会把结果自动写入训练标签。</p></section>'
    )
    adapter_js = (
        f"const ROWS={_json_for_script(page_rows)};"
        f"const STORAGE_KEY={_json_for_script('grok-stt-smoke-' + output_dir.name)};"
        f"const LABEL_FILENAME={_json_for_script(MANUAL_VERDICT_FILENAME)};"
        f"const VERDICT_SCHEMA={_json_for_script(MANUAL_VERDICT_SCHEMA)};"
        + PAGE_JS
    )
    html_text = render_audit_review_page(
        AuditReviewPageSpec(
            title=TITLE,
            intro_html=intro,
            body_html="".join(cards),
            adapter_css=PAGE_CSS,
            adapter_js=adapter_js,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    return {
        "schema": SUMMARY_SCHEMA,
        "generated_at": audit_generated_at(),
        "title": TITLE,
        "model": MODEL,
        "review_item_count": len(page_rows),
        "successful_results": len(page_rows),
        "word_units": sum(len(row["words"]) for row in page_rows),
        "total_audio_s": round(sum(float(row["duration_s"]) for row in page_rows), 6),
        "total_cost_usd": round(total_cost, 9),
        "manual_verdict_file": MANUAL_VERDICT_FILENAME,
        "shows_teacher_output": True,
        "shows_source_class": False,
        "training_manifest_allowed": False,
    }


PAGE_CSS = r"""
.teacher-intro,.teacher-card{background:#fff;border:1px solid #d7e0e8;border-radius:10px;padding:14px;margin:0 0 14px;box-shadow:0 2px 8px #10203012}
.teacher-intro p{margin:4px 0 8px}.teacher-heading{display:flex;gap:12px;align-items:baseline;justify-content:space-between}.teacher-heading h2{margin:0}
.teacher-transcript{border-left:4px solid #3d7ea6;background:#eef7fc;padding:9px 12px;margin:6px 0 10px}.teacher-transcript p{font-size:19px;margin:5px 0;line-height:1.6}
.teacher-tokens{display:flex;gap:5px;flex-wrap:wrap;margin:8px 0 12px}.teacher-token{border:1px solid #71a5c2;background:#eaf6fc;border-radius:5px;padding:5px 7px;cursor:pointer}.teacher-token small{display:block;font-size:9px}
.teacher-options{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:10px 0}.teacher-options b{display:block;margin-bottom:5px}.teacher-options button{margin:0 5px 5px 0;border:1px solid #9aa9b6;background:#f3f6f8;border-radius:5px;padding:6px 9px;cursor:pointer}.teacher-options button.selected{background:#195f82;color:#fff;border-color:#195f82}
.teacher-card textarea{width:100%;box-sizing:border-box}.teacher-warning{color:#a33131;margin-top:5px;min-height:1.2em}
.teacher-word-span{background:#4d9ac0;color:#07131a}.teacher-gap-span{background:#b8c0c7;color:#20262a}.teacher-gap-button{background:#d6dce1}.teacher-timeline .audit-span{font-size:0}
.teacher-island-span{background:#8256b5;color:#fff}.teacher-supervision-word{background:#2878a5;color:#fff}.teacher-supervision-ignore{background:#e0ad35;color:#241b08}.teacher-supervision-non-word{background:#aeb8c0;color:#172027}.teacher-cut-span{background:#4b9b66;color:#fff}
.teacher-short-island-span{background:#e0ad35;color:#241b08}
.legend{display:inline-block;width:13px;height:13px;border-radius:3px;vertical-align:-2px;margin:0 3px 0 8px}.legend.raw{background:#4d9ac0}.legend.island{background:#8256b5}.legend.word{background:#2878a5}.legend.ignore{background:#e0ad35}.legend.nonword{background:#aeb8c0}
@media(max-width:800px){.teacher-options{grid-template-columns:1fr}.teacher-heading{display:block}.audit-lane{grid-template-columns:1fr}}
"""


PAGE_JS = r"""
const state=createAuditReviewCore({
  entries:ROWS,
  storageKey:STORAGE_KEY,
  filename:LABEL_FILENAME,
  statusLabel:'已裁决',
  entryId:entry=>entry.audit_id,
  defaultState:_entry=>({transcript:'',timing:'',note:''}),
  isComplete:annotation=>Boolean(annotation.transcript&&annotation.timing),
  shouldSerialize:annotation=>Boolean(annotation.transcript||annotation.timing||annotation.note),
  serialize:(entry,annotation)=>({
    schema:VERDICT_SCHEMA,
    audit_id:entry.audit_id,
    transcript_verdict:annotation.transcript||'unreviewed',
    timing_verdict:annotation.timing||'unreviewed',
    note:annotation.note||'',
    updated_at:new Date().toISOString()
  })
});
function selectButton(card,field,value){
  card.querySelectorAll(`[data-field="${field}"]`).forEach(button=>button.classList.toggle('selected',button.dataset.value===value));
}
for(const row of ROWS){
  const card=document.getElementById(`card-${row.audit_id}`),audio=card.querySelector('audio');
  const timeline=card.querySelector('[data-timeline]');
  appendAuditSpanLane({container:timeline,audio,durationS:row.duration_s,label:'Grok 时间单元',metric:`${row.words.length} 个`,spans:row.words,className:'teacher-word-span',text:span=>span.text,title:span=>`${span.text} ${formatAuditSpan(span.start_s,span.end_s)}`});
  appendAuditSpanLane({container:timeline,audio,durationS:row.duration_s,label:'合并语音岛',metric:`${row.lexical_islands.length} 个`,spans:row.lexical_islands,className:'teacher-island-span',text:span=>span.text,title:span=>`${span.text} · ${span.unit_count} 单元 · ${formatAuditSpan(span.start_s,span.end_s)}`});
  appendAuditSpanLane({container:timeline,audio,durationS:row.duration_s,label:'短岛 ignore',metric:'< 2 帧',spans:row.ignored_short_islands,className:'teacher-short-island-span',text:span=>span.text,title:span=>`${span.text} · 降为 ignore · ${formatAuditSpan(span.start_s,span.end_s)}`});
  appendAuditSpanLane({container:timeline,audio,durationS:row.duration_s,label:'最终帧监督',metric:`hop ${(1000*row.frame_hop_s).toFixed(1)}ms`,spans:row.frame_supervision,className:span=>`teacher-supervision-${span.label}`,text:span=>span.label,title:span=>`${span.label} · frame ${span.start_frame}–${span.end_frame} · ${formatAuditSpan(span.start_s,span.end_s)}`});
  appendAuditSpanLane({container:timeline,audio,durationS:row.duration_s,label:'潜在切点区域',metric:'non-word ≥ 8 帧',spans:row.cut_candidates,className:'teacher-cut-span',text:()=>'',title:(_span,start,end)=>`候选区 ${formatAuditSpan(start,end)}`});
  appendAuditSpanLane({container:timeline,audio,durationS:row.duration_s,label:'未覆盖间隔',metric:'≥ 0.20s',spans:row.gaps,className:'teacher-gap-span',text:()=>'',title:(_span,start,end)=>`未覆盖 ${formatAuditSpan(start,end)}`});
  const tokens=card.querySelector('[data-tokens]');
  if(!row.words.length)tokens.innerHTML='<small>没有返回词时间单元</small>';
  for(const word of row.words){
    const button=document.createElement('button');
    button.type='button';button.className='teacher-token';
    button.innerHTML=`${escapeAuditHtml(word.text)}<small>${formatAuditSpan(word.start_s,word.end_s)}</small>`;
    button.onclick=()=>play(audio,button,Number(word.start_s),Number(word.end_s));
    tokens.appendChild(button);
  }
  appendAuditClipButtons({container:tokens,audio,label:'试听未覆盖间隔',spans:row.gaps,className:'teacher-gap-button'});
  const annotation=state.ensure(row);
  selectButton(card,'transcript',annotation.transcript);
  selectButton(card,'timing',annotation.timing);
  card.querySelectorAll('[data-field]').forEach(button=>button.onclick=()=>{
    annotation[button.dataset.field]=button.dataset.value;
    selectButton(card,button.dataset.field,button.dataset.value);
    state.persist();
  });
  const note=card.querySelector('textarea');note.value=annotation.note||'';
  note.oninput=()=>{annotation.note=note.value;state.persist();};
  const problems=[];
  if(Number(row.diagnostics.invalid_word_count)>0)problems.push(`无效时间单元 ${row.diagnostics.invalid_word_count}`);
  if(Number(row.diagnostics.overlap_count)>0)problems.push(`重叠 ${row.diagnostics.overlap_count}`);
  if(Number(row.diagnostics.out_of_range_count)>0)problems.push(`越界 ${row.diagnostics.out_of_range_count}`);
  card.querySelector('[data-warning]').textContent=problems.length?'结构告警：'+problems.join('，'):'';
}
document.getElementById('stop').onclick=()=>stop();
document.getElementById('save').onclick=()=>state.save();
state.updateStatus();
"""


def _first_config_value(config: Mapping[str, str], names: Sequence[str]) -> str:
    for name in names:
        value = str(config.get(name) or "").strip()
        if value:
            return value
    return ""


def _selection(
    source_dir: Path, row_ids: Sequence[str], output_dir: Path
) -> list[dict[str, Any]]:
    manifest = {str(row["row_id"]): row for row in read_jsonl(source_dir / "manifest.jsonl")}
    source_rows = {
        str(row["row_id"]): row for row in read_jsonl(source_dir / "selection.jsonl")
    }
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("--row-id values must be unique")
    unknown = [row_id for row_id in row_ids if row_id not in manifest or row_id not in source_rows]
    if unknown:
        raise ValueError(f"unknown source row_id(s): {unknown}")
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, row_id in enumerate(row_ids, start=1):
        audit_id = f"grok-{index:03d}"
        source_manifest = manifest[row_id]
        source_audio = source_dir / str(source_manifest["audio"])
        suffix = source_audio.suffix.lower() or ".wav"
        copied_audio = media_dir / f"{audit_id}{suffix}"
        shutil.copy2(source_audio, copied_audio)
        rows.append(
            {
                "schema": SELECTION_SCHEMA,
                "audit_id": audit_id,
                "source_row_id": row_id,
                "source_class": str(source_rows[row_id].get("source_class") or ""),
                "audio": f"media/{copied_audio.name}",
                "source_duration_s": float(source_manifest["duration_s"]),
                "audio_sha256": _sha256(copied_audio),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = _selection(source_dir, list(args.row_id), output_dir)
    write_jsonl(output_dir / "selection.jsonl", selection)

    config = load_env_file(Path(args.env_file).expanduser())
    api_key = _first_config_value(
        config,
        ("OPENROUTER_API_KEY", "OMNI_API_KEY", "OPENAI_API_KEY", "API_KEY"),
    )
    if not api_key:
        raise RuntimeError("OpenRouter API key not found in the configured env file")
    base_url = _first_config_value(config, ("OPENROUTER_BASE_URL", "OMNI_BASE_URL"))

    result_path = output_dir / "results.jsonl"
    # The first local smoke wrote the selection schema into otherwise complete
    # result rows because ``**selected`` followed the result schema literal.
    # Recognize that exact shape once, repair it in place, and then resume.  The
    # raw response is required so an actual selection row can never be mistaken
    # for a completed provider call.
    results: dict[str, dict[str, Any]] = {}
    repaired_schema = False
    for row in read_jsonl(result_path):
        is_result = row.get("schema") == RESULT_SCHEMA
        is_legacy_complete = (
            row.get("schema") == SELECTION_SCHEMA
            and isinstance(row.get("raw_response"), Mapping)
            and isinstance(row.get("response"), Mapping)
        )
        if row.get("model") != args.model or not (is_result or is_legacy_complete):
            continue
        repaired = dict(row)
        if is_legacy_complete:
            repaired["schema"] = RESULT_SCHEMA
            repaired_schema = True
        results[str(repaired["audit_id"])] = repaired
    if repaired_schema:
        write_jsonl(result_path, [results[key] for key in sorted(results)])
    for selected in selection:
        audit_id = str(selected["audit_id"])
        if audit_id in results and results[audit_id].get("audio_sha256") == selected["audio_sha256"]:
            print(f"grok_stt_resume audit_id={audit_id}", flush=True)
            continue
        audio_path = output_dir / str(selected["audio"])
        last_error: Exception | None = None
        for attempt in range(1, int(args.attempts) + 1):
            started = time.perf_counter()
            print(
                f"grok_stt_request audit_id={audit_id} attempt={attempt}/{args.attempts}",
                flush=True,
            )
            try:
                raw, response_headers = call_openrouter_stt(
                    audio_path=audio_path,
                    api_key=api_key,
                    base_url=base_url,
                    model=args.model,
                    language=args.language,
                    timeout_s=float(args.timeout_s),
                )
                latency = time.perf_counter() - started
                normalized = normalize_response(
                    raw,
                    fallback_duration_s=float(selected["source_duration_s"]),
                )
                result = {
                    **selected,
                    "schema": RESULT_SCHEMA,
                    "model": args.model,
                    "language_hint": args.language,
                    "request_contract": {
                        "response_format": "verbose_json",
                        "timestamp_granularities": ["word"],
                        "provider_require_parameters": True,
                    },
                    "latency_s": round(latency, 6),
                    "response": normalized,
                    "raw_response": raw,
                    "response_headers": response_headers,
                    "created_at": audit_generated_at(),
                }
                results[audit_id] = result
                write_jsonl(
                    result_path,
                    [results[key] for key in sorted(results)],
                )
                print(
                    f"grok_stt_result audit_id={audit_id} "
                    f"words={normalized['diagnostics']['word_count']} "
                    f"latency_s={latency:.2f}",
                    flush=True,
                )
                last_error = None
                break
            except Exception as error:  # noqa: BLE001
                last_error = error
                print(
                    f"grok_stt_error audit_id={audit_id} attempt={attempt} "
                    f"error={type(error).__name__}: {error}",
                    flush=True,
                )
                if attempt < int(args.attempts):
                    time.sleep(min(3.0, float(attempt)))
        if last_error is not None:
            raise RuntimeError(f"Grok STT failed for {audit_id}: {last_error}") from last_error

    ordered_results = [results[str(row["audit_id"])] for row in selection]
    summary = build_page(ordered_results, output_dir)
    source_counts: dict[str, int] = {}
    for row in selection:
        source_class = str(row["source_class"])
        source_counts[source_class] = source_counts.get(source_class, 0) + 1
    summary.update(
        {
            "source_audit": source_dir.as_posix(),
            "source_class_counts": source_counts,
            "selection": "selection.jsonl",
            "results": "results.jsonl",
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not args.no_nav:
        update_audit_entrypoints(latest_html=output_dir / "index.html", title=TITLE)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--row-id", action="append", required=True)
    parser.add_argument("--env-file", default="~/.config/omni/openrouter")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--no-nav", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
