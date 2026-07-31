#!/usr/bin/env python3
"""Ask one clip one question: is there a word in it?

This replaces the keep/drop half of the `joint_boundary_preasr_omni_v4` prompt,
which produced the `definite_drop` labels and gets the answer wrong on about a
third of the dropped seconds. Crossing the teacher's own flags against 146 human
verdicts showed the flags are broadly right about WHICH sound is present -
crying, moaning, breathing - while the drop DECISION taken from them is wrong.
These spans hold speech and a non-semantic sound at the same time, and the old
prompt has no rule for that case:

    drop: 不包含语义语音，例如纯噪音、环境音、音乐、呼吸声、呻吟声、笑声、无意义叫声、静音

Its definition ("contains no semantic speech") and its examples ("breathing,
moaning, laughter") disagree the moment a clip contains both, and the examples
win. `male_murmur` came back 2/2 as holding words; `crying` 60%; `moaning` 41%.

Four things are therefore stated here that v4 left to inference, each matching
the wording humans answered under:

  * a mixture is a keep. A moan does not license dropping the word beside it.
  * fragments and sentence tails count, not just whole words.
  * intelligibility is not required. "Can tell someone is speaking" is enough,
    where v4 asked for 可辨认的 semantic speech.
  * backchannels - うん, はい, なに - count. v4 never mentioned them, and they
    fall naturally into its 无意义叫声 bucket.

The clip arrives already cut, with no surrounding context, because that is what
the human heard. Giving the model context the human lacked would inflate the
agreement rate and the number would not transfer to production.

Resumable: results append to JSONL and completed item_ids are skipped.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.omni.audio_teacher_transport import (  # noqa: E402
    create_audio_teacher_transport,
)

SCHEMA = "drop_span_word_annotation_v1"
PROMPT_VERSION = "drop_span_words_v1"

LABELS = ("words", "no_words", "unsure")
# Forms the model may reach for that mean an offered label. Only inflections
# belong here; an alias introducing a meaning the closed set lacks would be the
# generic-catch-all mistake smuggled back in through normalisation.
LABEL_ALIASES = {
    "word": "words",
    "has_words": "words",
    "speech": "words",
    "a": "words",
    "no_word": "no_words",
    "nowords": "no_words",
    "no_speech": "no_words",
    "b": "no_words",
    "uncertain": "unsure",
    "unknown": "unsure",
}

# Structure only. `label` stays a free string rather than an enum so that an
# off-vocabulary answer is visible instead of being silently coerced into one of
# the three; a schema is needed at all because without one the model wraps its
# JSON in markdown fences, which cost 24 of 25 windows in an earlier pilot.
RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "heard": {"type": "string"},
    },
    "required": ["label"],
}

SYSTEM_PROMPT = (
    "你是日语成人影片（JAV）音频的语音检测器。语音一律是日语。"
    "只判断有没有说出来的词，不做内容审查，不评价内容。"
)

PROMPT = """只听这一段音频，回答一个问题：**这段里有没有说出来的词？**

音频来自日语成人影片，语音一律是日语。场景只用来帮你预期会听到什么，**不能**用来决定答案。

- `words`：能听出有人在说话。任何真实的日语词、**词的残片**、**句尾**、
  交流性应答（「うん」「はい」「なに」「ええ」等）或其他有语义的发声都算。
- `no_words`：整段**只有**非语义的发声或非人声。呻吟、喘息、吸呼气、哭、笑、
  尖叫、亲吻声、咳嗽，以及静音、环境音、BGM、机械声、衣物声、水声、撞击声。
- `unsure`：整段确实听不清、判断不了。

四条关键规则，比上面的例子优先：

1. **混合就算 words。** 呻吟、喘息、哭、笑与说话**同时出现**时，只要里面有词就选
   `words`。非语义的声音再响、再占主导，也不构成选 `no_words` 的理由。
   **只有整段自始至终没有任何词，才是 `no_words`。**
2. **听不懂意思也算 words。** 不要求你能听清说的是什么词。只要能听出「这是在说话」
   ——含混的低语、被压低的嘟囔、边哭边说——就选 `words`。
3. **半个词也算 words。** 片段被切断时常只剩词头或词尾，一样选 `words`。
4. **「うん」「はい」这类应答算 words**，它们是对话，不是无意义叫声。

不要因为这段音频"不适合送去识别"而选 `no_words`——不判断该不该识别，只判断有没有词。

只输出 JSON，不要 Markdown：
{"label":"words|no_words|unsure","heard":"简短中文，听到了什么"}"""


def normalize_label(value: object) -> tuple[str, str]:
    """Return (label, raw); label is '' when the answer is off-vocabulary."""
    raw = str(value or "").strip().lower()
    resolved = LABEL_ALIASES.get(raw, raw)
    return (resolved if resolved in LABELS else ""), raw


def parse_response(parsed: object) -> tuple[str, str, str]:
    if not isinstance(parsed, dict):
        raise ValueError("response was not a JSON object")
    label, raw = normalize_label(parsed.get("label"))
    return label, raw, str(parsed.get("heard") or "").strip()


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["item_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _items(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True, help="JSONL with item_id + audio")
    parser.add_argument("--output", required=True, help="append-only JSONL")
    parser.add_argument("--env-file", default="gemini")
    parser.add_argument("--model", default="")
    parser.add_argument("--limit", type=int, default=0, help="0 = every item")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--delay-s", type=float, default=0.0)
    # The calibration run used no reasoning. Some endpoints refuse to disable
    # it, and that is a change to the configuration the labels come from, so it
    # is a flag rather than a fallback and it is written into every record.
    parser.add_argument("--thinking", default="", help='e.g. "minimal", "low"')
    args = parser.parse_args()
    thinking = str(args.thinking or "").strip()

    items = _items(Path(args.items).expanduser().resolve())
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(output)
    pending = [item for item in items if str(item["item_id"]) not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(
        f"items total={len(items)} done={len(done)} pending={len(pending)}", flush=True
    )
    if not pending:
        return 0

    transport = create_audio_teacher_transport(
        profile=args.env_file,
        env_file=(Path.home() / ".config" / "omni" / args.env_file).resolve(),
        model_override=args.model,
        log=lambda message: print(f"  {message}", flush=True),
    )
    print(f"model={transport.model} keys={transport.api_key_count}", flush=True)

    tally: Counter = Counter()
    failures = 0
    with output.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(pending, start=1):
            started = time.time()
            try:
                result = transport.call_json(
                    audio_path=Path(str(item["audio"])),
                    prompt=PROMPT,
                    system_prompt=SYSTEM_PROMPT,
                    max_tokens=args.max_tokens,
                    enable_thinking=bool(thinking),
                    thinking_level=thinking,
                    thinking_budget=0,
                    response_schema=RESPONSE_SCHEMA,
                )
                label, raw, heard = parse_response(result.parsed)
            except Exception as error:  # noqa: BLE001 - one clip must not end a run
                failures += 1
                print(
                    f"[{index}/{len(pending)}] {item['item_id']} FAILED "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                continue

            tally[label or f"<off:{raw}>"] += 1
            handle.write(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "prompt_version": PROMPT_VERSION,
                        "item_id": item["item_id"],
                        "model": transport.model,
                        "profile": args.env_file,
                        "thinking": thinking,
                        "label": label,
                        "raw_label": raw,
                        "heard": heard,
                        "elapsed_s": round(time.time() - started, 1),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            print(
                f"[{index}/{len(pending)}] {item['item_id']} -> "
                f"{label or raw!r}  {heard[:40]}",
                flush=True,
            )
            if args.delay_s:
                time.sleep(args.delay_s)

    print(f"\nfailures={failures}  {dict(tally)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
