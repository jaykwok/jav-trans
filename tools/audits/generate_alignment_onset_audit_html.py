#!/usr/bin/env python3
"""One question: is the beginning of this clip cut off?

This is the first audit in the project about *timing* rather than about content,
so it cannot reuse the "does this contain words" wording. What is being tested
is whether the CTC alignment head puts a subtitle's start where the speech
actually starts, on real JAV audio - the one thing Phase 1 could not establish,
because its ground truth came from synthetic composites whose speech is clean
galgame.

An ear cannot report milliseconds, so the page does not ask for any. Every clip
is cut at some offset from a predicted line start and runs for the same fixed
duration, and the only question is whether the first sound is whole. Some clips
are cut deliberately late by a known amount; the rate at which those are heard
as chopped calibrates the ear, and the predicted onsets are then read against
that scale.

The page shows no labels and no timings. Clip length is constant across every
stratum on purpose: an earlier audit in this project was weakened when clip
duration leaked which arm a sample belonged to.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.audits.audit_nav import (  # noqa: E402
    audit_generated_at,
    update_audit_entrypoints,
)
from tools.audits.audit_prompt import (  # noqa: E402
    ResolvedAuditPrompt,
    resolve_audit_prompt,
)
from tools.audits.binary_clip_audit import (  # noqa: E402
    BinaryClipAuditSpec,
    BinaryOption,
    cut_clips,
    option_contract,
    render_page,
)

from tools.audits.select_alignment_onset_audit import (  # noqa: E402
    CONTEXT_SECONDS,
)

SUMMARY_SCHEMA = "alignment_onset_audit_page_summary_v1"
MANUAL_VERDICT_SCHEMA = "alignment_onset_manual_verdict_v1"

OPTION_INTACT = BinaryOption(value="intact", label="A 开头完整")
OPTION_CLIPPED = BinaryOption(value="clipped", label="B 开头被切掉")
OPTION_NON_SEMANTIC = BinaryOption(value="non_semantic", label="C 非语义发声")

DEFAULT_REVIEW_PROMPT = """只回答一个问题：从「切点」开始播放时，是不是从一个声音的正中间进去的？

两个播放按钮：
· **▶ 从切点**：这是被判断的那一下——模型认为字幕该从这里开始。
· **▶ 带前文**：同一段音频，往前多放 2 秒。用它听清切点前面发生了什么，再回头判断切点落在哪里。

A 开头完整：切点上第一个**说出来的词**是从它自己的起点开始的。切点后面先是静音、环境音、BGM 都不影响，只要那个词是完整地起来的，就选 A。
B 开头被切掉：切点已经落在一个词的中间了，起头被截掉，听上去像"半个音"或被硬切进去。
C 非语义发声：切点上那个声音不是词，是吮吸声、亲吻声、喘息、呻吟、哭笑、尖叫这一类。选了 C 就不用再判断它有没有被切掉。

只判断切点那一下，不要管结尾（所有片段都是硬切的）。整段完全没有声音、或者实在判断不了，选"不确定"。"""

INTRO_HTML = (
    '<div class="warn"><b>⚠ 这一页问的是「时间对不对」，不是「有没有词」</b>'
    "<p>之前几页都在问「这段有没有说出来的词」。这一页<b>完全不关心内容</b>——"
    "说了什么、听不听得懂、是不是呻吟，全都不影响答案。"
    "只判断一件事：<b>片段的第一个声音，是完整的，还是已经被切掉了开头</b>。</p></div>"
    "<p>项目到今天为止，字幕的段内时间轴一直是<b>编的</b>"
    "（<code>synthetic_proportional</code>：把文字按字数摊在整段窗口上）。"
    "现在有了一个 CTC 对齐头，第一次能真正测出每个字的时间。"
    "但它此前只在<b>合成语料</b>上验过，那里的语音是干净的 galgame 台词；"
    "<b>真实 JAV 音频上的对齐精度从没测过</b>，这一页就是那次测量。</p>"
    "<p><b>为什么这样问：</b>对齐误差是以毫秒计的，耳朵报不出毫秒。"
    "所以改成测「切在哪里能被听出来」——每个片段都从某个位置切开、长度完全一样，"
    "其中一部分是<b>故意</b>往后切了一点点的。"
    "故意切的那些有多少被你听出来，就标定出你的耳朵能分辨的尺度；"
    "再拿模型预测的那些去比，就能知道模型的误差落在哪一档。</p>"
    "<table><thead><tr><th>选项</th><th>完整含义</th></tr></thead><tbody>"
    "<tr><td><code>A 开头完整</code></td><td>开头第一个<b>说出来的词</b>是从它自己的"
    "起点开始的。开头有静音／环境音／BGM <b>都不影响</b>，只要那个词是完整起来的。</td></tr>"
    "<tr><td><code>B 开头被切掉</code></td><td>一进来就已经在一个词的中间了，"
    "听上去像半个音、像被硬切进去。</td></tr>"
    "<tr><td><code>C 非语义发声</code></td><td>开头那个声音<b>不是词</b>——吮吸、亲吻、"
    "喘息、呻吟、哭笑、尖叫等。选 C 就不用再判断它有没有被切掉。</td></tr>"
    "<tr><td><code>不确定</code></td><td>整段完全没有声音，或者实在判断不了。"
    "这不是答案，是出口——听不清时硬选会直接变成证据。</td></tr></tbody></table>"
    '<div class="warn"><b>关于 C 的取舍，先说清楚</b>'
    "<p>这个域里 ASR 转写出来的很多本来就是非语义发声（<code>はぁ、はぁ</code>、"
    "<code>あっ</code>、<code>イクッ</code>），所以 <b>C 可能会吃掉相当一部分片段</b>，"
    "而它们<b>不参与</b>时间轴统计（被切率只在 A、B 之间算）。"
    "如果最后 A+B 剩得太少，结论会因为样本量不足而读不出来——"
    "那种情况下会再补一批，不是把现有的硬凑。</p></div>"
    '<div class="warn"><b>两个播放按钮怎么用</b>'
    "<p><b>▶ 从切点</b>＝被判断的那一下，模型认为字幕应该从这里开始。"
    "<b>▶ 带前文</b>＝同一段音频往前多放 2 秒，用来听清切点前面发生了什么。</p>"
    "<p>建议先按<b>从切点</b>，拿不准再按<b>带前文</b>回头对一下：前文能让你听出"
    "「这个词是从切点起来的」还是「它在切点之前就已经开始了」。"
    "<b>前文对每一条都是一样的 2 秒</b>，所以它不会透露这条属于哪一组。</p></div>"
    "<p><b>只判断切点那一下</b>，结尾怎么样不用管（所有片段长度一样，"
    "结尾都是硬切的，那是设计如此，不是缺陷）。</p>"
    "<p>页面不显示任何标签，顺序已打乱。其中有一组片段是<b>往前多留了半秒</b>的，"
    "它们的开头不可能被切到——这组应该几乎全是 A。"
    "如果这组也出现不少 B，说明这次听音或者这个问法有问题，主结果就不能用。</p>"
)


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _spec(prompt: str) -> BinaryClipAuditSpec:
    return BinaryClipAuditSpec(
        title="开头完整 / 被切掉 · CTC 对齐头在真实音频上的起点精度",
        option_a=OPTION_INTACT,
        option_b=OPTION_CLIPPED,
        option_c=OPTION_NON_SEMANTIC,
        # Must match `select_alignment_onset_audit.py --context-seconds`, or the
        # "play from the cut" button would enter the clip at the wrong place and
        # every verdict would be about a moment nobody intended to test.
        context_seconds=CONTEXT_SECONDS,
        prompt=prompt,
        intro_html=INTRO_HTML,
        verdict_schema=MANUAL_VERDICT_SCHEMA,
        # Bumped to v2 when option C was added. A and B narrowed at the same
        # time - they used to cover any human sound, they now cover words only -
        # so v1 verdicts were made against a different question and must not
        # restore into this page. Two definitions averaged together is the exact
        # mistake that produced this project's 32% misattribution.
        storage_key="alignment-onset-audit-v2",
        status_label="开头完整/被切 裁决",
        boundary_contract=SUMMARY_SCHEMA,
        note_placeholder="可选：听到的第一个声音",
    )


def build(
    *,
    manifest: Path,
    output_dir: Path,
    update_latest: bool = True,
    review_prompt: ResolvedAuditPrompt | None = None,
) -> dict[str, Any]:
    resolved_prompt = review_prompt or resolve_audit_prompt(
        default_prompt=DEFAULT_REVIEW_PROMPT
    )
    spec = _spec(resolved_prompt.text)
    option_contract(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = cut_clips(_rows(manifest), output_dir)
    page = output_dir / "index.html"
    page.write_text(render_page(spec, rows), encoding="utf-8")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "generated_at": audit_generated_at(),
        "item_count": len(rows),
        "page": str(page),
        "prompt_source": resolved_prompt.source,
    }
    (output_dir / "page_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if update_latest:
        update_audit_entrypoints(latest_html=page, title=spec.title)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-update-latest", action="store_true")
    args = parser.parse_args()

    summary = build(
        manifest=Path(args.manifest),
        output_dir=Path(args.output_dir),
        update_latest=not args.no_update_latest,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
