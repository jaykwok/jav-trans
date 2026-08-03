"""Check `asr.decode_guard`'s two bounds against a real corpus.

The decode budget is `duration_s * TOKENS_PER_SECOND_CEILING`: an upper bound on
how much speech audio of that length can contain, so it cannot cut a
transcription short. The repetition guard's bar is a share of that budget, which
makes it a repetition *rate*. Both are measurements, so both need a way to be
re-measured - on a new domain, a new ASR checkpoint, a new tokenizer.

What this tool reports, from ASR result-cache entries alone (no GPU, no audio):

1. **The observed token rate.** `max` is the number the ceiling has to clear. Run
   the corpus with a deliberately generous budget first (`ASR_MAX_NEW_TOKENS`
   high enough that `decode_cap_truncations` is 0), or the top of the
   distribution is the budget measuring itself rather than the speech.

2. **Repetition, split three ways.** A chunk that emitted a stop token said
   everything it was going to say, so its repetition is real audio and must
   survive the guard. A chunk that ran out of budget, and a chunk whose tail
   already reaches its own bar, are both lower bounds - the second is the one
   the guard stopped, and lumping it in with the self-terminated group makes the
   tool argue in a circle ("the bar cut real content, so the bar is too low").

On the corpus this was built for the two repetition populations *overlap*, and
they overlap by construction: the guard's own bar pins the second group. That is
why the budget, not the repetition share, is what bounds the decode - the tool
says so out loud when it happens rather than offering a threshold anyway.

    $env:PYTHONIOENCODING = "utf-8"
    uv run python tools/asr/measure_repetition_budget.py tmp/asr_cache/<dir>

Add `--json <path>` to keep the raw per-chunk numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def longest_repeating_tail(ids: list[int], max_ngram: int) -> tuple[int, int, int]:
    """`(tokens, copies, ngram)` for the longest pure-repetition suffix.

    Matches what the guard looks at: the *tail*, because that is where a loop
    lives and because repetition followed by new content is not a loop.
    """
    best = (0, 0, 0)
    limit = min(max_ngram, len(ids) // 2)
    for ngram in range(1, limit + 1):
        unit = ids[len(ids) - ngram :]
        copies = 1
        while True:
            start = len(ids) - (copies + 1) * ngram
            if start < 0 or ids[start : start + ngram] != unit:
                break
            copies += 1
        if copies >= 2 and copies * ngram > best[0]:
            best = (copies * ngram, copies, ngram)
    return best


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def _describe(label: str, rows: list[dict], key: str, unit: str) -> None:
    values = [row[key] for row in rows]
    if not values:
        print(f"  {label}: 0 块")
        return
    print(
        f"  {label}: {len(values)} 块  "
        f"max={max(values):.2f}  p95={_quantile(values, 0.95):.2f}  "
        f"p50={statistics.median(values):.2f} {unit}"
    )
    for row in sorted(rows, key=lambda r: -r[key])[:5]:
        print(
            f"      {row[key]:5.2f}  {row['repeated_tokens']:3d} token "
            f"= {row['copies']}x{row['ngram']:<2d} / {row['duration']:5.1f}s  "
            f"{row['tail']!r}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path, help="an ASR result-cache directory")
    parser.add_argument(
        "--max-ngram",
        type=int,
        default=32,
        help="longest unit to look for while measuring (not the guard's limit)",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    from asr import decode_guard
    from asr.backends.qwen import active_qwen_asr_model_path, current_qwen_asr_backend
    from core.config import load_config
    from utils.model_paths import resolve_model_spec

    load_config()

    entries = [
        p for p in sorted(args.cache_dir.glob("*.json")) if p.name != "signature.json"
    ]
    if not entries:
        print(f"no cache entries under {args.cache_dir}", file=sys.stderr)
        return 1

    signature = args.cache_dir / "signature.json"
    generation = {}
    if signature.exists():
        generation = json.loads(signature.read_text(encoding="utf-8")).get(
            "generation", {}
        )
    guard = generation.get("asr_decode_loop_guard", "(unrecorded)")
    explicit = str(generation.get("asr_max_new_tokens") or "").strip()
    rate = str(generation.get("asr_decode_tokens_per_second") or "").strip()
    print(
        f"corpus: {len(entries)} 块  guard={guard}  "
        f"ASR_MAX_NEW_TOKENS={explicit or '(auto)'}  "
        f"tok/s ceiling={rate or decode_guard.TOKENS_PER_SECOND_CEILING}"
    )
    if str(guard) not in {"0", "false", "no", "off"}:
        print(
            "  [WARN] 这份语料是开着 guard 解出来的，自然收尾那一组是 guard 的输出"
            "而不是模型的；校准重复门槛请用 ASR_DECODE_LOOP_GUARD=0 重跑。"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_spec(
            active_qwen_asr_model_path() or None,
            generation.get("resolved_asr_model_id") or current_qwen_asr_backend(),
        ),
        trust_remote_code=True,
    )

    completed: list[dict] = []
    truncated: list[dict] = []
    guard_cut: list[dict] = []
    for path in entries:
        payload = json.loads(path.read_text(encoding="utf-8")).get("text_result") or {}
        text = payload.get("raw_text")
        duration = float(payload.get("duration") or 0.0)
        if not text or duration <= 0.0:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        budget = decode_guard.plausible_token_budget(duration)
        tokens, copies, ngram = longest_repeating_tail(ids, args.max_ngram)
        row = {
            "chunk": path.stem[:12],
            "duration": duration,
            "total_tokens": len(ids),
            "token_rate": len(ids) / duration,
            "chars_per_token": len(text) / max(1, len(ids)),
            "budget": budget,
            "repeated_tokens": tokens,
            "copies": copies,
            "ngram": ngram,
            "repeat_rate": tokens / duration,
            "share": tokens / budget if budget else 0.0,
            "speech_rate": (len(ids) - tokens) / duration,
            "tail": text[-40:],
        }
        bar = decode_guard.loop_guard_config(budget)[2]
        if bool((payload.get("asr_generation") or {}).get("truncated_at_cap")):
            truncated.append(row)
        elif tokens >= bar:
            # The guard ended this one, so its repetition is a lower bound too.
            guard_cut.append(row)
        else:
            completed.append(row)

    rows = completed + truncated + guard_cut
    if not rows:
        print("no usable entries (need raw_text and duration)", file=sys.stderr)
        return 1

    ceiling = decode_guard.tokens_per_second_ceiling()
    rates = [row["token_rate"] for row in completed] or [
        row["token_rate"] for row in rows
    ]
    speech = [row["speech_rate"] for row in completed] or [
        row["speech_rate"] for row in rows
    ]
    chars = [row["chars_per_token"] for row in rows]
    print("\n[1] 实测 token 速率（自然收尾的块，未被预算截断）:")
    print(
        f"  p50={statistics.median(rates):.2f}  p95={_quantile(rates, 0.95):.2f}  "
        f"max={max(rates):.2f} tok/s   现行上限 {ceiling:.1f} tok/s "
        f"→ 余量 {ceiling / max(rates, default=1.0):.1f}x"
    )
    print(
        f"  其中非重复部分 max={max(speech):.2f} tok/s"
        "（扣掉重复尾巴，这才是「说话」的速率）"
    )
    print(
        f"  chars/token p50={statistics.median(chars):.2f}"
        "  （≈1 表示 1 token≈1 音拍，速率上限可直接按发音速率读）"
    )
    if max(rates) >= ceiling:
        print(
            "  [WARN] 实测速率已达上限，说明预算正在截断真实语音——"
            "提高 ASR_DECODE_TOKENS_PER_SECOND 再量一遍。"
        )
    if truncated:
        print(
            f"  {len(truncated)}/{len(rows)} 块用完了自己的预算。"
            "预算按时长派生，所以这是失控计数，不是「上限太低」。"
        )

    print("\n[2] 尾部纯重复速率，按结束方式分组:")
    _describe("自然收尾（重复是真的，必须活下来）", completed, "repeat_rate", "tok/s")
    _describe("被重复门槛停（只是下界）", guard_cut, "repeat_rate", "tok/s")
    _describe("用完预算（只是下界）", truncated, "repeat_rate", "tok/s")

    max_ngram, min_repeats, min_tokens = decode_guard.loop_guard_config(
        decode_guard.plausible_token_budget(
            statistics.median([row["duration"] for row in rows])
        )
    )
    fraction = decode_guard.loop_budget_fraction()
    print(
        f"\n现行重复门槛：中位块 {min_tokens} token（= 预算的 {fraction:.0%}"
        f" = {fraction * ceiling:.1f} tok/s 纯重复）/ 单元最长 {max_ngram} token"
        f" / 至少 {min_repeats} 遍"
    )
    real_max = max((row["repeat_rate"] for row in completed), default=0.0)
    loop_min = min(
        (row["repeat_rate"] for row in truncated + guard_cut), default=float("inf")
    )
    if real_max:
        print(
            f"  真实重复最快 {real_max:.2f} tok/s，门槛余量 "
            f"{(fraction * ceiling) / real_max:.1f}x"
        )
    else:
        print("  这份语料没有可比的真实重复")
    if loop_min <= real_max:
        print(
            "  两组重叠：重复率分不开真实吟唱和解码失控（门槛自己就是分界线的来源）。"
            "别据此收紧——兜底是按时长派生的预算，重复门槛只买解码步数。"
        )
    if guard_cut:
        print(
            f"  {len(guard_cut)}/{len(rows)} 块被门槛停住。它们的重复率是被门槛钉出来的"
            "下界，不能拿来证明门槛的位置对不对。"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "tokens_per_second_ceiling": ceiling,
                    "loop_budget_fraction": fraction,
                    "completed": completed,
                    "guard_cut": guard_cut,
                    "truncated": truncated,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"\nper-chunk 数据 -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
