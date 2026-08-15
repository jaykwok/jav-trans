#!/usr/bin/env python3
"""Generate a blinded per-cue A/B audit for two translation configurations.

Every translation change so far has been argued from prompt reasoning and spot
checks. This page asks the only question that settles it: on the *same* cue,
hearing the *same* audio, which of two Chinese lines is better - with the
configuration identity removed from the page.

The two arms are two ordinary runs of the same film. Nothing here re-translates
anything: point it at two `bilingual.json` files (or the `.zh.srt`/`.srt` pair)
produced by two runs that differ only in translation settings. The cheapest way
to make that pair is 重试 on a finished job after changing 翻译设置 - the retry
reuses the ASR artifacts, so the cue geometry is identical by construction.

That identity is checked, not assumed: cue count and every cue's start/end and
Japanese text must match across arms, and the tool stops if they do not. A
re-decoded run is *not* a valid arm - 2026-08-13 measured the same audio
re-transcribing 262/339 chunks differently - and comparing translations across
two different cue sets would be measuring the ASR, not the translator.

Only cues where the two arms actually produced different Chinese are eligible:
an identical line carries no preference information and would dilute the
sample. Arm order per card is balanced and randomized; the answer key lives in
`answers.jsonl`, never in the page, and the page is scanned for leaks before it
is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import audit_generated_at, update_audit_entrypoints  # noqa: E402
from tools.audits.review_page_core import (  # noqa: E402
    AuditReviewPageSpec,
    format_audit_timestamp,
    render_audit_review_page,
)
from tools.omni.openai_compat import slice_audio_clip  # noqa: E402


SUMMARY_SCHEMA = "translation_ab_audit_summary_v1"
ANSWER_SCHEMA = "translation_ab_audit_answer_v1"
VERDICT_SCHEMA = "translation_ab_manual_verdict_v1"
DEFAULT_CLIP_PAD_S = 0.35
VERDICTS = ("arm_1_better", "arm_2_better", "equivalent_good", "equivalent_bad", "unsure")
# Everything the page is allowed to know. The blind is enforced against this
# set rather than by searching the document for the arm names: an arm called
# `none` would match the core CSS, and an arm called `flash` could legitimately
# occur inside a subtitle line. Structure is checkable; substrings are not.
PAGE_ROW_KEYS = frozenset(
    {"row_id", "span", "ja", "clip_src", "arm_1_text", "arm_2_text"}
)


def assert_page_is_blind(page_rows: list[dict[str, Any]], rendered: str) -> None:
    for row in page_rows:
        extra = set(row) - PAGE_ROW_KEYS
        if extra:
            raise AssertionError(f"blind page row carries {sorted(extra)}")
    for forbidden in ('"arm_1":', '"arm_2":', '"cue_index":', "answers.jsonl"):
        if forbidden in rendered:
            raise AssertionError(f"blind page leaked {forbidden!r}")


def parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use --arm NAME=PATH")
    name, raw_path = value.split("=", 1)
    name, raw_path = name.strip(), raw_path.strip()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("use --arm NAME=PATH")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return name, path


def load_cues(path: Path) -> list[dict[str, Any]]:
    """Cues from a run's `bilingual.json`, in file order."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    blocks = payload["blocks"] if isinstance(payload, dict) and "blocks" in payload else payload
    cues: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        japanese = str(block.get("ja_text") or block.get("ja") or block.get("text") or "")
        chinese = str(block.get("zh_text") or block.get("zh") or "")
        cues.append(
            {
                "index": index,
                "start_s": float(block.get("start", 0.0)),
                "end_s": float(block.get("end", 0.0)),
                "ja": japanese,
                "zh": chinese,
            }
        )
    return cues


def require_same_cue_set(arms: dict[str, list[dict[str, Any]]]) -> None:
    """Both arms must be the same cues; otherwise this measures the ASR."""
    names = list(arms)
    reference_name, reference = names[0], arms[names[0]]
    for name in names[1:]:
        other = arms[name]
        if len(other) != len(reference):
            raise SystemExit(
                f"arms disagree on cue count: {reference_name}={len(reference)} "
                f"{name}={len(other)}. The arms must come from runs that share the "
                "ASR artifacts (retry the same job after changing 翻译设置); a "
                "re-decoded run is not a valid arm."
            )
        for left, right in zip(reference, other):
            same_geometry = (
                abs(left["start_s"] - right["start_s"]) <= 1e-6
                and abs(left["end_s"] - right["end_s"]) <= 1e-6
            )
            if not same_geometry or left["ja"] != right["ja"]:
                raise SystemExit(
                    f"arms disagree at cue {left['index']}: "
                    f"{reference_name}={left['start_s']:.3f}-{left['end_s']:.3f} "
                    f"{left['ja'][:24]!r} vs {name}={right['start_s']:.3f}-"
                    f"{right['end_s']:.3f} {right['ja'][:24]!r}. "
                    "Only the Chinese may differ between arms."
                )


def eligible_rows(
    arms: dict[str, list[dict[str, Any]]],
    *,
    min_ja_chars: int,
) -> list[dict[str, Any]]:
    first, second = list(arms)
    rows: list[dict[str, Any]] = []
    for left, right in zip(arms[first], arms[second]):
        if left["zh"].strip() == right["zh"].strip():
            continue
        if len(left["ja"].strip()) < min_ja_chars:
            continue
        if not left["zh"].strip() and not right["zh"].strip():
            continue
        rows.append(
            {
                "index": left["index"],
                "start_s": left["start_s"],
                "end_s": left["end_s"],
                "ja": left["ja"],
                first: left["zh"],
                second: right["zh"],
            }
        )
    return rows


def balanced_first_arm(count: int, arm_names: tuple[str, str], *, seed: int) -> list[str]:
    """Half the cards lead with each arm, in random order."""
    first, second = arm_names
    flags = [first] * (count // 2) + [second] * (count - count // 2)
    random.Random(seed ^ 0x5A5A).shuffle(flags)
    return flags


def materialize(
    rows: list[dict[str, Any]],
    *,
    arm_names: tuple[str, str],
    audio: Path,
    output_dir: Path,
    pad_s: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    leading = balanced_first_arm(len(rows), arm_names, seed=seed)
    page_rows: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        row_id = f"translation-ab-{position + 1:03d}"
        first = leading[position]
        second = arm_names[1] if first == arm_names[0] else arm_names[0]
        clip = media_dir / f"{row_id}.mp3"
        start = max(0.0, float(row["start_s"]) - pad_s)
        end = float(row["end_s"]) + pad_s
        slice_audio_clip(
            source_audio=audio,
            row={"start": start, "end": end, "duration_s": end - start},
            output_path=clip,
            fmt="mp3",
            bitrate="64k",
            sample_rate=16000,
            force=False,
        )
        page_rows.append(
            {
                "row_id": row_id,
                "span": f"{format_audit_timestamp(row['start_s'])}–"
                f"{format_audit_timestamp(row['end_s'])}",
                "ja": row["ja"],
                "clip_src": clip.relative_to(output_dir).as_posix(),
                "arm_1_text": row[first],
                "arm_2_text": row[second],
            }
        )
        answers.append(
            {
                "schema": ANSWER_SCHEMA,
                "row_id": row_id,
                "cue_index": int(row["index"]),
                "start_s": round(float(row["start_s"]), 6),
                "end_s": round(float(row["end_s"]), 6),
                "arm_1": first,
                "arm_2": second,
                "ja": row["ja"],
                "arm_1_zh": row[first],
                "arm_2_zh": row[second],
            }
        )
    return page_rows, answers


CSS = """
.contract,article{background:#fff;border:1px solid #ccd6df;border-radius:10px;padding:14px;margin-bottom:14px}
.warn{background:#fff3e0;border-left:5px solid #c77700;padding:10px 12px;margin:10px 0}
article.done{border-left:6px solid #258b57}
.meta{color:#607080;font-variant-numeric:tabular-nums}
.ja{font-size:17px;background:#f5f7f8;padding:10px;border-radius:6px;line-height:1.6}
.candidates{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:10px 0}
.candidate{border:1px solid #d5dde4;border-radius:8px;padding:10px}
.candidate b{color:#607080;font-size:12px}
.candidate p{font-size:16px;line-height:1.6;margin:6px 0 0;white-space:pre-wrap}
.controls{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}
.choice{padding:9px 13px;border:1px solid #8d99a5;border-radius:5px;background:#fff;cursor:pointer}
.choice.active{outline:3px solid #18212b;outline-offset:-2px;background:#cfe3d6}
.note{width:100%;min-height:40px;box-sizing:border-box}
@media(max-width:760px){.candidates{grid-template-columns:1fr}}
"""


JS = r"""
const rows=__ROWS__;
const allowed=new Set(['arm_1_better','arm_2_better','equivalent_good','equivalent_bad','unsure']);
const reviewCore=createAuditReviewCore({storageKey:'translation-ab-v1:'+location.pathname,entries:rows,entryId:r=>r.row_id,defaultState:()=>({verdict:'',note:''}),isComplete:s=>allowed.has(s.verdict),statusLabel:'译文裁决',filename:'manual_verdicts.jsonl',serialize:(row,state)=>({schema:'translation_ab_manual_verdict_v1',row_id:row.row_id,verdict:state.verdict||'unreviewed',note:state.note||'',updated_at:state.updated_at||new Date().toISOString()})});
function sync(card,state){card.classList.toggle('done',allowed.has(state.verdict));card.querySelectorAll('[data-value]').forEach(b=>b.classList.toggle('active',b.dataset.value===state.verdict));}
const root=document.getElementById('list');
rows.forEach((row,index)=>{const state=reviewCore.ensure(row),card=document.createElement('article');
card.innerHTML=`<h2>${index+1} / ${rows.length}</h2><div class="meta">${escapeAuditHtml(row.span)}</div><p class="ja">${escapeAuditHtml(row.ja)}</p><audio controls preload="metadata" src="${escapeAuditHtml(row.clip_src)}"></audio><div class="candidates"><div class="candidate"><b>甲</b><p>${escapeAuditHtml(row.arm_1_text)}</p></div><div class="candidate"><b>乙</b><p>${escapeAuditHtml(row.arm_2_text)}</p></div></div><div class="controls"><button class="choice" data-value="arm_1_better">甲更好</button><button class="choice" data-value="arm_2_better">乙更好</button><button class="choice" data-value="equivalent_good">都可用</button><button class="choice" data-value="equivalent_bad">都不可用</button><button class="choice" data-value="unsure">不确定</button></div><textarea class="note" placeholder="可选：错译、漏译、术语、语气、断句">${escapeAuditHtml(state.note||'')}</textarea>`;
card.querySelectorAll('[data-value]').forEach(button=>button.onclick=()=>{state.verdict=button.dataset.value;state.updated_at=new Date().toISOString();sync(card,state);reviewCore.persist();});
card.querySelector('.note').onchange=e=>{state.note=e.target.value;state.updated_at=new Date().toISOString();reviewCore.persist();};sync(card,state);root.appendChild(card);});
document.getElementById('stop').onclick=()=>{document.querySelectorAll('audio').forEach(a=>a.pause());reviewCore.updateStatus('已停止');};document.getElementById('save').onclick=()=>reviewCore.save();reviewCore.updateStatus();
"""


def render_page(rows: list[dict[str, Any]], *, title: str) -> str:
    intro = """
    <section class="contract"><h2>裁决规则</h2>
    <div class="warn"><b>甲 / 乙 的配置身份已随机化，页面内不含答案。</b>同一张卡片的左右两边不固定属于同一个配置。</div>
    <p>先听音频再读日文原文，然后判断哪一边的中文更符合这句话在这段音频里的意思与语气。顺序依次看：<b>有没有错译或漏译</b>、<b>术语与称呼是否一致</b>、<b>语气是否贴合</b>、<b>断句与长度是否好读</b>。</p>
    <p>只在两边确实有优劣时才选一边：差异不影响观感时选 <b>都可用</b>，两边都错到需要重写时选 <b>都不可用</b>，听不清或拿不准时选 <b>不确定</b>，不要猜。</p>
    <p>只有两边中文不同的 cue 会出现在这里；相同的没有信息量，已被排除。</p></section>
    """
    adapter = JS.replace("__ROWS__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
    return render_audit_review_page(
        AuditReviewPageSpec(
            title=title,
            intro_html=intro,
            body_html='<div id="list"></div>',
            adapter_css=CSS,
            adapter_js=adapter,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        type=parse_arm,
        metavar="NAME=BILINGUAL_JSON",
        help="exactly two, e.g. --arm none=... --arm medium=...",
    )
    parser.add_argument("--audio", required=True, help="the film audio both runs used")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="翻译配置 · 匿名人工 A/B")
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--min-ja-chars", type=int, default=4)
    parser.add_argument("--clip-pad-s", type=float, default=DEFAULT_CLIP_PAD_S)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--no-update-latest", action="store_true")
    args = parser.parse_args(argv)

    if len(args.arm) != 2:
        parser.error("exactly two --arm values are required")
    arm_names = tuple(name for name, _path in args.arm)
    if arm_names[0] == arm_names[1]:
        parser.error("the two arms need different names")
    arms = {name: load_cues(path) for name, path in args.arm}
    require_same_cue_set(arms)

    audio = Path(args.audio).expanduser()
    if not audio.is_absolute():
        audio = PROJECT_ROOT / audio
    if not audio.is_file():
        raise SystemExit(f"audio not found: {audio}")

    candidates = eligible_rows(arms, min_ja_chars=int(args.min_ja_chars))
    if not candidates:
        raise SystemExit(
            "the two arms produced identical Chinese on every cue; there is "
            "nothing to compare. Check that the arms really differ in settings."
        )
    rng = random.Random(int(args.seed))
    selected = sorted(
        rng.sample(candidates, min(int(args.sample), len(candidates))),
        key=lambda row: row["index"],
    )

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    page_rows, answers = materialize(
        selected,
        arm_names=arm_names,
        audio=audio,
        output_dir=output_dir,
        pad_s=float(args.clip_pad_s),
        seed=int(args.seed),
    )

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in page_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for row in answers:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    page = output_dir / "index.html"
    rendered = render_page(page_rows, title=str(args.title))
    assert_page_is_blind(page_rows, rendered)
    page.write_text(rendered, encoding="utf-8")

    summary = {
        "schema": SUMMARY_SCHEMA,
        "generated_at": audit_generated_at(),
        "page": str(page.resolve()),
        "arms": {name: str(path) for name, path in args.arm},
        "cues_total": len(next(iter(arms.values()))),
        "cues_differing": len(candidates),
        "review_items": len(page_rows),
        "leading_arm_counts": dict(Counter(row["arm_1"] for row in answers)),
        "clip_pad_s": float(args.clip_pad_s),
        "seed": int(args.seed),
        "blind": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.no_update_latest:
        update_audit_entrypoints(latest_html=page, title=str(args.title))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
