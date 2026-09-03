#!/usr/bin/env python3
"""One question: does this clip contain a word you can make out?

The vocalisation filter has always decided this from text alone, by trying to
decompose the cue into kana that carry no lexical content. That rule is an
allow-list of noise and it has two known holes: it cannot reach a lone moan
between two lines of dialogue (nothing but context distinguishes a moan from a
gasp answering something said), and it cannot spell every onomatopoeia that
exists (`じゅぽっ`, `ごくんっ`, `ちゅぽっ` are in the corpus and were never in
the list).

A three-class frame head now supplies acoustic evidence, and combining the two
deletes about a third more cues than the text rule alone. Every statistic
available says that is an improvement and none of them can say it: the groups
those statistics are computed over are the text rule's own verdicts, so
"the text rule kept it and the acoustics dropped it" has no ground truth by
construction. Whether those cues contain words is a question about sound.

The page shows no text, no labels and no timings, and the order is shuffled.
Two of the four strata are references rather than measurements: one is what the
shipped filter already deletes, the other is cues carrying kanji, which certainly
contain words and which the new rule never touches. If the second group does not
come back as "has words", the question or the listening is wrong and the result
is void rather than negative.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
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

SUMMARY_SCHEMA = "vocalisation_verdict_audit_page_summary_v1"
MANUAL_VERDICT_SCHEMA = "vocalisation_verdict_manual_verdict_v1"

OPTION_HAS_WORDS = BinaryOption(value="has_words", label="A 有词")
OPTION_NO_WORDS = BinaryOption(value="no_words", label="B 只有非语义发声")

DEFAULT_REVIEW_PROMPT = """只回答一个问题：这一段里，有没有**至少一个你能听出意思的词**？

A 有词：听到了任何一个有意义的词就选 A。不需要听懂整句，不需要听清全部，
一个 `やめて`、`きもちいい`、`おにいちゃん`、`だめ`、`もっと` 就够了。
名词、动词、称呼、应答（`うん`/`はい`）都算。

B 只有非语义发声：整段从头到尾都是喘息、呻吟、吮吸声、亲吻声、咀嚼声、
哭腔、笑声、尖叫这一类，没有任何一个能听出意思的词。

不确定：整段几乎没有声音、只有 BGM/环境音、或者你实在分辨不出来。
这不是第三个答案，是出口——听不清时硬选会直接变成证据。

判断整段，不要只听开头。段落长度不一样是正常的。"""

INTRO_HTML = (
    '<div class="warn"><b>⚠ 这一页问的是「有没有词」，不是「时间对不对」</b>'
    "<p>上一次那一页问的是字幕的<b>起点</b>切得准不准。这一页<b>完全不关心时间</b>——"
    "开头切没切、结尾在哪里，都不影响答案。只判断一件事："
    "<b>这段音频里有没有至少一个能听出意思的词</b>。</p></div>"
    "<p>项目里删除呻吟字幕的过滤器，到今天为止<b>只看文本</b>："
    "把字幕拆成假名，如果整条都能被「无词义假名 + 拟声词表」吃掉，就删掉。"
    "这是一份<b>噪声白名单</b>，它有两个已知且补不完的洞：</p>"
    "<ul><li><b>孤立的呻吟删不掉。</b>规则要求连续 ≥2 条才删，因为夹在两句对白之间的"
    "一个 <code>あっ</code> 更可能是对刚才那句话的反应，而文本分不出这两者。"
    "一部片里就有 125 条因此留下。</li>"
    "<li><b>词表拼不全拟声词。</b><code>じゅぽっ</code>、<code>ごくんっ</code>、"
    "<code>ちゅぽっ</code> 这些都在语料里，从来不在表里。加词永远追不上。</li></ul>"
    "<p>现在对齐头多了一路输出，能逐帧判断「静音 / 非语义发声 / 言语」。"
    "把它和文本规则合起来用，八部片上<b>多删了约 1,158 条</b>（3,387 → 4,545，+34%）。"
    "问题是：<b>这多出来的一千多条，删对了吗？</b></p>"
    "<p><b>为什么必须靠耳朵：</b>所有能算的指标，分组依据都是<b>文本规则自己的判决</b>。"
    "「文本规则留下、声学要删」这一类，按定义就没有真值可比——"
    "再算下去也只是在问文本规则同不同意自己。这一页就是那个缺失的真值。</p>"
    "<table><thead><tr><th>选项</th><th>完整含义</th></tr></thead><tbody>"
    "<tr><td><code>A 有词</code></td><td>听到<b>任何一个</b>有意义的词。"
    "不需要听懂整句——一个 <code>だめ</code>、<code>もっと</code>、<code>うん</code> 就够。</td></tr>"
    "<tr><td><code>B 只有非语义发声</code></td><td>整段都是喘息、呻吟、吮吸、亲吻、"
    "咀嚼、哭腔、笑声、尖叫，没有任何一个能听出意思的词。</td></tr>"
    "<tr><td><code>不确定</code></td><td>几乎没有声音、只有 BGM／环境音、"
    "或实在分辨不出。这不是答案，是出口。</td></tr></tbody></table>"
    '<div class="warn"><b>这 60 条里混了两组「已知答案」的对照</b>'
    "<p>一组是<b>现役过滤器已经在删的</b>片段——它们应该基本都是 B。"
    "另一组是<b>带汉字的对白</b>，新规则从不碰它们——它们应该基本都是 A。</p>"
    "<p>这两组不参与结论，它们是用来<b>验证这次听音本身可不可信</b>的："
    "如果带汉字那组也听成 B，说明问法或者听音出了问题，"
    "主结果<b>作废</b>而不是算作负面结论。</p></div>"
    "<p>页面不显示文本、不显示时间、不显示分组，顺序已打乱。"
    "片段长度<b>已按统一区间（1.5–8 秒）抽样</b>，所以长短不透露它属于哪一组。</p>"
)


LABEL_SPAN_INTRO_HTML = (
    '<div class="warn"><b>⚠ 这一页听的是「训练标签」，不是字幕</b>'
    "<p>上一页听的是<b>成品字幕</b>该不该删。这一页听的是喂给模型的"
    "<b>标签本身对不对</b>——两回事，别用上一页的印象套。</p></div>"
    "<p>三类帧头（静音／非语义发声／言语）的标签里，有一路叫 <b>L1</b>："
    "把 NSFW 混合片的<b>完整脚本</b>（含呻吟）用通用对齐头强制对齐，"
    "按标点切块，逐块过共享的人声词表——判为人声的块，"
    "它占的那段音频就被标成 <code>vocalisation</code>。</p>"
    "<p><b>L1 是整套新增信息的唯一来源，也是唯一会往昂贵方向出错的地方。</b>"
    "把呻吟标成言语，只损失一点监督；把<b>言语</b>标成呻吟，"
    "就是在教模型「这个词是呻吟」——而下游的过滤器随后会把它删掉。</p>"
    "<p>已经跑过的两项检查都有同一个盲区：它们<b>读的是标签所来自的那份脚本</b>"
    "（汉字计数 0、块文本词频），所以脚本本身错了、或者对齐把语义块的音频"
    "拽进了人声块，这两项都看不出来。第三方才能裁决：本地 ASR"
    "<b>从没见过这些脚本</b>，让它独立解码这 60 段——"
    "它报出<b>零处</b>词义内容。这一页是同一批片段交给耳朵再判一次。</p>"
    "<p><b>为什么挑最长的 60 段：</b>跨度越长，标错代价越大、耳朵越容易判，"
    "而「对齐把语义块拽进人声块」产生的正是长跨度。"
    "这个抽样<b>偏向于发现问题</b>，对安全检查来说方向是对的。</p>"
    "<table><thead><tr><th>选项</th><th>完整含义</th></tr></thead><tbody>"
    "<tr><td><code>A 有词</code></td><td>听到<b>任何一个</b>有意义的词。"
    "这一段本不该被标成 <code>vocalisation</code>。</td></tr>"
    "<tr><td><code>B 只有非语义发声</code></td><td>整段是喘息、呻吟、吮吸、"
    "亲吻、咀嚼、哭腔、笑声、尖叫。标签是对的。</td></tr>"
    "<tr><td><code>不确定</code></td><td>几乎没有声音、只有 BGM／环境音、"
    "或实在分辨不出。</td></tr></tbody></table>"
    '<div class="warn"><b>这一页没有对照组</b>'
    "<p>上一页混了两组已知答案来验证听音本身。这一页<b>全部 60 条都是同一类</b>"
    "（被标为 <code>vocalisation</code> 的最长跨度），因为它问的是"
    "「这一类里有没有混进词」，不是在比较两组。"
    "判据是：<b>含可辨识词语的条数 ≤ 2</b>。</p></div>"
    "<p>页面不显示脚本文本、不显示时间、不显示 ASR 解码结果，顺序按跨度长度排列"
    "（同一类，长度不透露任何分组）。</p>"
)

SPLIT_FRAGMENT_INTRO_HTML = (
    '<div class="warn"><b>⚠ 这一页听的是「从一条字幕里切掉的那半段」</b>'
    "<p>前两页问的是<b>整条</b>字幕该不该删。这一页问的是"
    "<b>一条里只删掉一半</b>对不对——这是过滤器第一次动 cue 内部的文本。</p></div>"
    "<p>症状是这种条目：<code>れろれろ…あっ、エッチしてくださ</code>——"
    "前 3 秒是呻吟，后 1.4 秒是真话。它含真词，所以整条规则留；"
    "结果那 3 秒呻吟的文字照样上屏、照样进翻译。</p>"
    "<p><b>切的判据不是新的。</b>只有当「把这一段单独当成一条 cue 来判、"
    "并且用它自己那段音频重新测的帧后验」得出<b>删</b>时，这一段才会被切掉。"
    "所以它删不掉任何「作为独立 cue 会被保留」的东西。切点只落在标点分段的边界上，"
    "而且从含实词的那一段起就停下。</p>"
    "<p><b>但借来的判据不等于在新粒度上验过。</b>要判它对不对，只能听。</p>"
    "<table><thead><tr><th>选项</th><th>完整含义</th></tr></thead><tbody>"
    "<tr><td><code>A 有词</code></td><td>听到<b>任何一个</b>有意义的词，"
    "或者听到<b>半个被切断的词</b>。两者都说明这一刀切错了。</td></tr>"
    "<tr><td><code>B 只有非语义发声</code></td><td>整段是喘息、呻吟、吮吸、"
    "亲吻、咀嚼、哭腔、笑声、尖叫。</td></tr>"
    "<tr><td><code>不确定</code></td><td>几乎没有声音、只有 BGM／环境音、"
    "或实在分辨不出。</td></tr></tbody></table>"
    '<div class="warn"><b>混了两组「已知答案」的对照</b>'
    "<p>一组是<b>同一条 cue 里留下来的那半段</b>——它们应该基本都是 A。"
    "这组还有第二个作用：切点如果早了一点点，"
    "残缺会出现在<b>留下的这半</b>，在被删的那半里根本听不出来。</p>"
    "<p>另一组是<b>现役过滤器已经在整条删的</b>片段——它们应该基本都是 B。</p>"
    "<p>如果「留下的那半」也听成 B，说明问法或听音出了问题，"
    "主结果<b>作废</b>而不是算作负面结论。</p></div>"
    "<p>页面不显示文本、不显示时间、不显示分组，顺序已打乱。"
    "片段长度已按统一区间（1–8 秒）抽样。</p>"
)

FRAMINGS = {
    "joint_verdict": {
        "title": "有词 / 只有非语义发声 · 联合判决多删的那一千条对不对",
        "intro_html": INTRO_HTML,
        "storage_key": "vocalisation-verdict-audit-v1",
    },
    "label_spans": {
        "title": "有词 / 只有非语义发声 · L1 帧标签里被标为呻吟的最长跨度",
        "intro_html": LABEL_SPAN_INTRO_HTML,
        "storage_key": "frame-class-span-audit-v1",
    },
    "split_fragments": {
        "title": "有词 / 只有非语义发声 · 从一条字幕里切掉的那半段对不对",
        "intro_html": SPLIT_FRAGMENT_INTRO_HTML,
        "storage_key": "split-fragment-audit-v1",
    },
}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _spec(prompt: str, framing: str = "joint_verdict") -> BinaryClipAuditSpec:
    # The mechanics and the two answers are identical - the question really is
    # "does this contain a word" in both cases. Only the framing differs, and a
    # separate `storage_key` per framing keeps one page's saved verdicts from
    # restoring into the other, which would silently average two samples.
    chosen = FRAMINGS[framing]
    return BinaryClipAuditSpec(
        title=chosen["title"],
        option_a=OPTION_HAS_WORDS,
        option_b=OPTION_NO_WORDS,
        prompt=prompt,
        intro_html=chosen["intro_html"],
        verdict_schema=MANUAL_VERDICT_SCHEMA,
        storage_key=chosen["storage_key"],
        status_label="有词/无词 裁决",
        boundary_contract=SUMMARY_SCHEMA,
        note_placeholder="可选：听到的词",
    )


def build(
    *,
    manifest: Path,
    output_dir: Path,
    update_latest: bool = True,
    review_prompt: ResolvedAuditPrompt | None = None,
    framing: str = "joint_verdict",
) -> dict[str, Any]:
    resolved_prompt = review_prompt or resolve_audit_prompt(
        default_prompt=DEFAULT_REVIEW_PROMPT
    )
    spec = _spec(resolved_prompt.text, framing)
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
        update_audit_entrypoints(
            latest_html=page, title=spec.title, project_root=PROJECT_ROOT
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-update-latest", action="store_true")
    parser.add_argument(
        "--framing",
        default="joint_verdict",
        choices=sorted(FRAMINGS),
        help="which question this page frames: the finished cues the joint "
        "verdict would delete, or the L1 label spans themselves",
    )
    args = parser.parse_args()

    summary = build(
        manifest=Path(args.manifest),
        output_dir=Path(args.output_dir),
        update_latest=not args.no_update_latest,
        framing=args.framing,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
