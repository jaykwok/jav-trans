#!/usr/bin/env python3
"""Annotate the TYPE of already-segmented drop spans, by listening.

The type track was never annotated. It was recovered by stem-matching the
teacher's free-form `omni_flags`, and that recovery produced two systematic
errors in a row: `non_speech` and then bare `noise` both name the teacher's
DECISION ("this is not speech") rather than the sound, and each typed a large
block of spans `non_vocal` that a listening model calls human 75-85% of the
time. A third such word is not worth waiting for. This asks the audio directly.

What makes the question answerable now is that the segmentation is frozen. The
model is not asked whether a span should be dropped, nor where the boundaries
are - only what the sound in a known interval is. That is the same narrowing
that made scorer v12 work.

One field, not two. The A/B version of this prompt asked for a 4-way label AND
a category, and the model leaked category values into the label field 65 times.
Here it returns only the category, from a closed set, and the type is derived
from it by a table in this file - so the mapping is auditable code rather than
model output, and a category/type disagreement is unrepresentable.

The closed set deliberately has no generic "noise" member. That word is what
went wrong twice; a sound with no concrete name is `uncertain`, which costs a
span rather than mislabelling it.

Resumable: results append to JSONL and completed windows are skipped, because
the free tier is ~440 requests/day against ~770 windows.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.omni.audio_teacher_transport import (  # noqa: E402
    create_audio_teacher_transport,
)

SCHEMA = "typed_span_type_annotation_v1"

# Category -> type. Every emitted category maps to exactly one type, so the
# model never decides the taxonomy; it only names the sound.
CATEGORY_TYPES: dict[str, str] = {
    "dialogue": "speech",
    "whisper": "speech",
    "singing": "speech",
    "breath": "non_semantic_vocal",
    "moan": "non_semantic_vocal",
    "cry": "non_semantic_vocal",
    "laugh": "non_semantic_vocal",
    "scream": "non_semantic_vocal",
    "kiss": "non_semantic_vocal",
    "cough": "non_semantic_vocal",
    # Added after a 45-window pilot returned them as off-vocabulary answers.
    # They are real distinctions the annotator wanted and the set lacked.
    "sigh": "non_semantic_vocal",
    "gasp": "non_semantic_vocal",
    "music": "non_vocal",
    "impact": "non_vocal",
    "cloth": "non_vocal",
    "water": "non_vocal",
    "footsteps": "non_vocal",
    "machinery": "non_vocal",
    "silence": "non_vocal",
    "ambience": "non_vocal",
    "uncertain": "unsure",
}
CATEGORIES = tuple(CATEGORY_TYPES)
SPEECH_CATEGORIES = tuple(c for c, t in CATEGORY_TYPES.items() if t == "speech")

# Inflections the pilot produced for categories that were already offered.
# Only forms of an EXISTING category belong here: an alias that introduced a
# meaning the closed set does not have would be the generic-catch-all mistake
# smuggled back in through normalisation.
CATEGORY_ALIASES: dict[str, str] = {
    "screaming": "scream",
    "moaning": "moan",
    "breathing": "breath",
    "gasping": "gasp",
    "sighing": "sigh",
    "coughing": "cough",
    "crying": "cry",
    "laughing": "laugh",
    "laughter": "laugh",
    "kissing": "kiss",
    "footstep": "footsteps",
}

# Structure only. The category stays a free string rather than an enum on
# purpose: a schema that could not express an off-vocabulary answer would also
# hide whether the vocabulary fits, and this run is partly a test of that. A
# schema is needed at all because without one the model wraps its JSON in
# markdown fences; the earlier finding that schemas change DECISIONS and not
# just format is why this one constrains no value the model has to choose.
RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "span_categories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["id", "category"],
            },
        }
    },
    "required": ["span_categories"],
}


def resolve_type(category: str | None, source_label: str) -> tuple[str, bool]:
    """Type for a span, plus whether it contradicts the frozen segmentation.

    A `definite_drop` span was already judged to carry no semantic speech, so a
    `dialogue`/`whisper`/`singing` answer there cannot be written into the type
    track without asserting two incompatible things at once. The segmentation
    is the frozen input, so the type yields: the span becomes `unsure` rather
    than being forced either way, and the conflict is counted - a high rate
    would be evidence against the segmentation, which is worth knowing on its
    own and would be invisible if these were silently coerced.
    """
    resolved = CATEGORY_TYPES.get(category or "", "unsure")
    if resolved == "speech" and source_label != "definite_keep":
        return "unsure", True
    return resolved, False

SYSTEM_PROMPT = (
    "你是日语成人影片（JAV）音频的声音性质标注器。语音一律是日语。"
    "只描述听到的声音是什么，不做内容审查，不判断该不该送去识别。"
)


def build_prompt(duration_s: float, spans: list[dict]) -> str:
    listing = json.dumps(
        [
            {"id": s["id"], "start_s": round(s["start_s"], 3), "end_s": round(s["end_s"], 3)}
            for s in spans
        ],
        ensure_ascii=False,
    )
    categories = "|".join(CATEGORIES)
    return f"""只听这一个音频。音频来自日语成人影片（JAV），其中的语音一律是日语。
音频时间范围为 0.000 到 {duration_s:.3f} 秒；下列时间都相对此音频。

下面每个区间都已经确定不送去识别。你的唯一任务是说出**每个区间里占主导的声音是什么**。
场景是成人影片这件事只用来帮你预期会听到哪些声音，**不能**用来决定标签本身。

从以下闭集中选一个 category，不要自创，不要组合：{categories}

category 含义：
- dialogue / whisper / singing：能听出有词义的日语（对白、耳语、演唱）。
- breath / moan / cry / laugh / scream / kiss / cough：人的声道或口腔发出、但没有词义的声音。
- music：音乐或器乐。
- impact：肉体撞击、床体震动、物体碰撞、敲击。
- cloth：衣物摩擦、被褥摩擦、身体移动带出的布料声。
- water：水声、液体声、黏湿声。
- footsteps：脚步声。
- machinery：机械声、车辆声、电器声。
- silence：几乎无声。
- ambience：室内底噪、户外环境声、远处人群声等无法归入上面几类的背景声。
- uncertain：确实听不出来，或者不属于上面任何一类。

关键规则：
- **说不出具体是什么声音就选 uncertain，不要为了填满而猜。** 猜错比 uncertain 代价大得多。
- 不要用"不是说话"作为判断依据。喘息、呻吟、笑声都不是说话，但它们是人发出的。
  你要回答的是**这个声音由谁/由什么发出**，不是它有没有词义。
- 一个区间里有多种声音时，选**音量与时长上占主导**的那个。
- 纯音乐即使音色像人声也选 music；音乐上叠加真人演唱才选 singing。

必须对输入中的每个 id 恰好返回一次，不得发明 id，不写 reason。
spans={listing}

只输出 JSON，不要 Markdown：
{{"span_categories":[{{"id":"s000","category":"{CATEGORIES[0]}"}}]}}"""


def parse_response(parsed, spans: list[dict]) -> tuple[dict[str, str], list[str]]:
    """Return (id -> category) plus any categories outside the closed set."""
    if not isinstance(parsed, dict):
        raise ValueError("response was not a JSON object")
    rows = parsed.get("span_categories")
    if not isinstance(rows, list):
        raise ValueError("response has no span_categories list")
    wanted = {s["id"] for s in spans}
    categories: dict[str, str] = {}
    off_vocabulary: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        identifier = str(row.get("id", "")).strip()
        category = str(row.get("category", "")).strip().lower()
        category = CATEGORY_ALIASES.get(category, category)
        if identifier not in wanted:
            continue
        if category not in CATEGORY_TYPES:
            off_vocabulary.append(category)
            continue
        categories[identifier] = category
    return categories, off_vocabulary


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["window_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def window_spans(row: dict, source_labels: tuple[str, ...]) -> list[dict]:
    spans = [
        dict(span, id=f"s{index:03d}")
        for index, span in enumerate(
            [s for s in row["spans"] if s["source_label"] in source_labels]
        )
    ]
    return spans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="all.jsonl from the builder")
    parser.add_argument("--output", required=True, help="append-only JSONL")
    parser.add_argument("--env-file", default="gemini")
    parser.add_argument("--model", default="")
    parser.add_argument("--limit", type=int, default=0, help="0 = every window")
    parser.add_argument(
        "--source-labels",
        default="definite_drop",
        help="Comma-separated span source_labels to type.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--delay-s", type=float, default=0.0)
    parser.add_argument(
        "--order",
        choices=["video_coverage", "file"],
        default="video_coverage",
        help=(
            "video_coverage visits one window per video before revisiting any "
            "video. non_vocal is limited by how many DISTINCT videos carry it "
            "(66 of 93 in train), so a partial run should widen coverage "
            "rather than deepen a few videos."
        ),
    )
    args = parser.parse_args()

    source_labels = tuple(
        part.strip() for part in args.source_labels.split(",") if part.strip()
    )
    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    real = [
        row
        for row in rows
        if row["provenance"] == "real_omni_joint" and window_spans(row, source_labels)
    ]
    if args.order == "video_coverage":
        by_video: dict[str, list[dict]] = {}
        for row in real:
            by_video.setdefault(str(row.get("video_id")), []).append(row)
        for group in by_video.values():
            group.sort(key=lambda r: str(r["window_id"]))
        ordered: list[dict] = []
        depth = 0
        while any(len(group) > depth for group in by_video.values()):
            for video in sorted(by_video):
                group = by_video[video]
                if len(group) > depth:
                    ordered.append(group[depth])
            depth += 1
        real = ordered

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(output)
    pending = [row for row in real if row["window_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(
        f"windows total={len(real)} done={len(done)} pending={len(pending)}",
        flush=True,
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
    off_total: Counter = Counter()
    failures = 0
    conflicts = 0
    with output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(pending, start=1):
            spans = window_spans(row, source_labels)
            prompt = build_prompt(float(row["duration_s"]), spans)
            started = time.time()
            try:
                result = transport.call_json(
                    audio_path=Path(row["audio"]),
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    max_tokens=args.max_tokens,
                    enable_thinking=False,
                    thinking_level="",
                    thinking_budget=0,
                    response_schema=RESPONSE_SCHEMA,
                )
                categories, off_vocabulary = parse_response(result.parsed, spans)
            except Exception as error:  # noqa: BLE001 - one window must not end the run
                failures += 1
                print(
                    f"[{index}/{len(pending)}] {row['window_id']} FAILED "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                continue

            off_total.update(off_vocabulary)
            record = {
                "schema": SCHEMA,
                "window_id": row["window_id"],
                "video_id": row.get("video_id"),
                "dataset": row.get("dataset"),
                "model": transport.model,
                "elapsed_s": round(time.time() - started, 1),
                "off_vocabulary": off_vocabulary,
                "spans": [
                    {
                        "id": span["id"],
                        "start_s": span["start_s"],
                        "end_s": span["end_s"],
                        "source_label": span["source_label"],
                        "stem_type": span["type"],
                        "flags": span["flags"],
                        "category": categories.get(span["id"]),
                        **dict(
                            zip(
                                ("type", "segmentation_conflict"),
                                resolve_type(
                                    categories.get(span["id"]), span["source_label"]
                                ),
                                strict=True,
                            )
                        ),
                    }
                    for span in spans
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            tally.update(span["type"] for span in record["spans"])
            conflicts += sum(
                1 for span in record["spans"] if span["segmentation_conflict"]
            )
            missing = sum(1 for s in spans if s["id"] not in categories)
            print(
                f"[{index}/{len(pending)}] {row['window_id']} "
                f"spans={len(spans)} missing={missing} "
                f"{record['elapsed_s']:.0f}s  running={dict(tally)}",
                flush=True,
            )
            if args.delay_s:
                time.sleep(args.delay_s)

    print(f"\ntypes: {dict(tally)}")
    print(f"speech answers on non-keep spans (-> unsure): {conflicts}")
    print(f"off-vocabulary categories: {dict(off_total) or 'none'}")
    print(f"failed windows: {failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
