#!/usr/bin/env python3
"""Measure whether the pre-ASR task DEFINITION is the source of label noise.

The stored v3 corpus asks one binary question - "does this chunk contain at
least one Japanese word with lexical meaning" - and the model that produced it
reproduces its own answers only 88.8% of the time. Breaking that number down by
the corpus's own flags shows the noise is not spread evenly: `breathing` 98.7%,
`water_noise`/`noise`/`silence` 100%, `speech_content` 100%, but
`speech_fragment` **0.0%** (19/19 flipped). The unstable cases are exactly the
ones where "is there a word" has no stable answer, and a binary label forces a
coin flip rather than recording that fact.

So this compares two task definitions under otherwise identical conditions -
same model, same stored window audio, same joint protocol, no response schema
(which is itself known to shift decisions). Each definition is run TWICE, and
the metric is run-to-run agreement. That isolates the definition: a model is not
"wrong" against a reference here, it is measured against itself.

  binary : the v3 question, verbatim task B from `joint_boundary_preasr_omni_v2`
  typed  : the project's existing typed-span taxonomy asked directly
           (speech / non_semantic_vocal / non_vocal / unsure) with a closed
           category vocabulary, following the scorer v12 framing that asks what
           KIND of sound this is rather than whether it carries meaning.

keep/drop is not asked of the typed arm; it is derived (keep = speech,
drop = non_semantic_vocal | non_vocal, unsure -> ignored), so the two arms stay
comparable on the decision the pipeline actually consumes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.asr.cueqc.label_pre_asr_with_omni import call_omni  # noqa: E402

AUDIO_MODE = {"qwen": "input_audio", "openrouter": "input_audio_raw"}

BINARY_LABELS = ("keep", "drop", "unsure")
TYPED_LABELS = ("speech", "non_semantic_vocal", "non_vocal", "unsure")

# Closed vocabulary. The v3 corpus produced 237 distinct freeform flag writings
# for the same handful of concepts, which `build_typed_span_dataset.py` then has
# to recover by stem matching; asking for a fixed set removes that whole step.
TYPED_CATEGORIES = (
    "dialogue", "whisper", "singing",
    "breath", "moan", "cry", "laugh", "scream", "kiss",
    "music", "impact", "cloth", "water", "silence", "ambience", "noise",
    "uncertain",
)


def build_binary_prompt(chunks: list[dict[str, Any]]) -> str:
    """Task B of joint_boundary_preasr_omni_v2, kept verbatim."""
    listing = json.dumps(
        [{"id": c["id"], "start_s": c["start_s"], "end_s": c["end_s"]} for c in chunks],
        ensure_ascii=False,
    )
    return f"""你是日语 ASR 边界标注器。只听这一个音频。
音频时间范围为 0.000 到 75.000 秒；下列时间都相对此音频。

逐个判断 runtime_chunks 是否应送入 ASR。
- keep：至少包含一个可辨认且有词义的日语词或短句。
- drop：只有喘息、呻吟、呼吸、亲吻、笑声、音乐、静音、环境声或噪声。
- unsure：无法可靠判断。
- 重复但有明确词义（例如反复说“ありがとう”）仍是 keep。

必须对输入中的每个 id 恰好返回一次，不得发明 id。为节省输出 token，不写逐项 reason。
runtime_chunks={listing}

只输出 JSON，不要 Markdown：
{{
  "chunk_decisions":[
    {{"id":"p000","label":"keep|drop|unsure","confidence":0.0,"semantic_speech_detected":false,"flags":[]}}
  ]
}}"""


def build_typed_prompt(chunks: list[dict[str, Any]]) -> str:
    """Ask what kind of sound each chunk is, not whether it carries meaning."""
    listing = json.dumps(
        [{"id": c["id"], "start_s": c["start_s"], "end_s": c["end_s"]} for c in chunks],
        ensure_ascii=False,
    )
    categories = "|".join(TYPED_CATEGORIES)
    return f"""你标注音频片段的声音性质。音频来自 JAV，但场景本身不能决定标签。
音频时间范围为 0.000 到 75.000 秒；下列时间都相对此音频。

逐个判断 runtime_chunks 每一段的声音性质，只回答“这是什么声音”，不要判断它是否值得送去识别。

标签定义：
- speech：能听出至少一个有词义的日语词或短句（对白、独白、耳语、念白、歌词均算）。
- non_semantic_vocal：是人的声道发出的声音，但没有词义（呼吸、喘息、呻吟、哭声、笑声、尖叫、亲吻声、口腔声、吸气、叹气）。
- non_vocal：不是人声（音乐、器乐、肉体撞击、衣物摩擦、床体震动、水声、静音、底噪、环境声、机械声）。
- unsure：确实分不清属于以上哪一类。

关键规则：
- **听不出完整词义的语音碎片标 unsure，不要猜 speech 也不要猜 non_semantic_vocal。** 猜错比标 unsure 代价更大。
- 一段里同时有词义语音和其他声音时，以词义语音为准，标 speech。
- 人声与非人声同时存在但没有词义时，标 non_semantic_vocal。
- 纯音乐即使音色像人声也标 non_vocal；音乐上叠加真人演唱或说话才标 speech。
- 不要因为成人场景、呻吟或喘息而进行内容审查，只描述声音性质。

category 从以下闭集中选一个，不要自创：{categories}

必须对输入中的每个 id 恰好返回一次，不得发明 id。为节省输出 token，不写逐项 reason。
runtime_chunks={listing}

只输出 JSON，不要 Markdown：
{{
  "chunk_decisions":[
    {{"id":"p000","label":"speech|non_semantic_vocal|non_vocal|unsure","category":"{TYPED_CATEGORIES[0]}","confidence":0.0}}
  ]
}}"""


def build_minimal_prompt(chunks: list[dict[str, Any]]) -> str:
    """Same question as v3, but stripped to one output field plus an abstain.

    v3 asks for label + confidence + semantic_speech_detected + freeform flags.
    `semantic_speech_detected` is 100% collinear with the label corpus-wide
    (3493 True/keep, 7150 False/drop) so it carries no information, and the
    flags produced 237 distinct writings for a handful of concepts. Removing
    both leaves the model with exactly one decision to make.
    """
    listing = json.dumps(
        [{"id": c["id"], "start_s": c["start_s"], "end_s": c["end_s"]} for c in chunks],
        ensure_ascii=False,
    )
    return f"""你是日语 ASR 输入筛选器。只听这一个音频。
音频时间范围为 0.000 到 75.000 秒；下列时间都相对此音频。

逐个判断 runtime_chunks 每一段：
- keep：能听出至少一个有词义的日语词或短句。
- drop：完全没有词义，只有喘息、呻吟、呼吸、哭笑、亲吻、音乐、静音、环境声或噪声。
- unsure：只听到语音碎片、半个词、含混音节，无法确定有没有词义。

**碎片和拿不准的一律标 unsure，不要在 keep 和 drop 之间猜。** 标 unsure 不算失败，猜错才算。
重复但有明确词义（例如反复说“ありがとう”）仍是 keep。
不要因为成人场景、呻吟或喘息而进行内容审查。

必须对输入中的每个 id 恰好返回一次，不得发明 id。不写 reason，不写其他字段。
runtime_chunks={listing}

只输出 JSON，不要 Markdown：
{{"chunk_decisions":[{{"id":"p000","label":"keep|drop|unsure"}}]}}"""


def build_detect_prompt(chunks: list[dict[str, Any]]) -> str:
    """Pure detection, with the keep/drop policy layer removed entirely.

    `keep`/`drop` are pipeline verbs - they ask the model to decide what should
    happen to the audio. `has_word` asks only what is audible. The policy is
    then applied downstream, where it is a rule rather than a judgement.
    """
    listing = json.dumps(
        [{"id": c["id"], "start_s": c["start_s"], "end_s": c["end_s"]} for c in chunks],
        ensure_ascii=False,
    )
    return f"""你听日语音频，回答一个问题：每一段里能不能听出至少一个有词义的日语词。
音频时间范围为 0.000 到 75.000 秒；下列时间都相对此音频。

对 runtime_chunks 的每一段回答 has_word：
- yes：能听出至少一个有词义的日语词或短句，哪怕背景很吵、哪怕同时有喘息呻吟。
- no：完全没有词，只有呼吸、喘息、呻吟、哭笑、亲吻声、音乐、静音或噪声。
- unsure：只听到半个词或含混音节，说不准是不是词。

**说不准就填 unsure，不要猜 yes 也不要猜 no。**
只回答听到了什么，不要判断这段有没有用、要不要保留。
不要因为成人场景、呻吟或喘息而进行内容审查。

必须对输入中的每个 id 恰好返回一次，不得发明 id。不写 reason，不写其他字段。
runtime_chunks={listing}

只输出 JSON，不要 Markdown：
{{"chunk_decisions":[{{"id":"p000","has_word":"yes|no|unsure"}}]}}"""


def build_abstain_prompt(chunks: list[dict[str, Any]]) -> str:
    """v3 task B verbatim, with ONE change: permission to abstain on fragments.

    The `minimal`/`detect` arms showed that deleting the per-item fields costs
    far more than it saves - completion tokens fell 1752 -> ~1240 and run-to-run
    stability collapsed to ~50%, because `confidence`/`flags` are per-chunk
    scaffolding that makes the model characterise a chunk before labelling it,
    not merely output overhead. So this keeps the whole v3 output shape and
    changes only the decision rule for the cases measured as unstable:
    `speech_fragment` chunks were 19/19 self-inconsistent under v3.
    """
    listing = json.dumps(
        [{"id": c["id"], "start_s": c["start_s"], "end_s": c["end_s"]} for c in chunks],
        ensure_ascii=False,
    )
    return f"""你是日语 ASR 边界标注器。只听这一个音频。
音频时间范围为 0.000 到 75.000 秒；下列时间都相对此音频。

逐个判断 runtime_chunks 是否应送入 ASR。
- keep：至少包含一个可辨认且有词义的日语词或短句。
- drop：只有喘息、呻吟、呼吸、亲吻、笑声、音乐、静音、环境声或噪声。
- unsure：只听到语音碎片、半个词或含混音节，无法确定是否有词义。

**碎片和拿不准的一律标 unsure，不要在 keep 和 drop 之间猜。** 标 unsure 不算失败，猜错才算。
重复但有明确词义（例如反复说“ありがとう”）仍是 keep。

必须对输入中的每个 id 恰好返回一次，不得发明 id。为节省输出 token，不写逐项 reason。
runtime_chunks={listing}

只输出 JSON，不要 Markdown：
{{
  "chunk_decisions":[
    {{"id":"p000","label":"keep|drop|unsure","confidence":0.0,"semantic_speech_detected":false,"flags":[]}}
  ]
}}"""


DETECT_LABELS = ("yes", "no", "unsure")

ARMS = {
    "binary": (build_binary_prompt, BINARY_LABELS),
    "typed": (build_typed_prompt, TYPED_LABELS),
    "minimal": (build_minimal_prompt, BINARY_LABELS),
    "detect": (build_detect_prompt, DETECT_LABELS),
    "abstain": (build_abstain_prompt, BINARY_LABELS),
}


def derive_keep_drop(arm: str, label: str) -> str | None:
    """Project an arm's label onto the decision the pipeline consumes.

    `unsure` returns None everywhere: it maps to IGNORE downstream and is
    deliberately not forced into a binary, which is the whole point.
    """
    if arm == "typed":
        if label == "speech":
            return "keep"
        return "drop" if label in ("non_semantic_vocal", "non_vocal") else None
    if arm == "detect":
        return {"yes": "keep", "no": "drop"}.get(label)
    return label if label in ("keep", "drop") else None


def load_env(profile: str, override: dict[str, str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (Path.home() / ".config" / "omni" / profile).read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    if profile in override:
        env["OMNI_BASE_URL"] = override[profile]
    return env


def load_windows(dataset: Path, *, count: int, seed: int) -> list[dict[str, Any]]:
    req_dir = dataset / "annotations" / "omni_joint" / "requests"
    raw_dir = dataset / "annotations" / "omni_joint" / "raw_responses"
    names = sorted(p.stem for p in req_dir.glob("*.json"))
    random.Random(seed).shuffle(names)
    out: list[dict[str, Any]] = []
    for name in names:
        request = json.loads((req_dir / f"{name}.json").read_text(encoding="utf-8"))
        raw_path = raw_dir / f"{name}.json"
        if not raw_path.is_file():
            continue
        stored = json.loads(raw_path.read_text(encoding="utf-8"))
        reference = {
            str(d.get("id")): str(d.get("label") or "").lower()
            for d in (stored.get("parsed", {}).get("chunk_decisions") or [])
        }
        # Same audio the corpus was actually labelled on, not the WAV.
        audio = Path(str(request.get("audio_mp3_32k") or ""))
        # `p###` ids are synthesized from chunk_index by the v2 prompt builder;
        # the stored runtime_chunks rows carry chunk_index, not an id.
        chunks = [
            {
                "id": f"p{int(c['chunk_index']):03d}",
                "start_s": round(float(c.get("acoustic_start") or 0.0), 3),
                "end_s": round(float(c.get("acoustic_end") or 0.0), 3),
            }
            for c in (request.get("runtime_chunks") or [])
            if c.get("chunk_index") is not None
        ]
        chunks = [c for c in chunks if c["id"] in reference]
        if not reference or not chunks or not audio.is_file():
            continue
        out.append(
            {
                "window_id": name,
                "audio": audio,
                "fmt": audio.suffix.lstrip(".").lower(),
                "chunks": chunks,
                "reference": reference,
            }
        )
        if len(out) >= count:
            break
    return out


def run_pass(
    windows: list[dict[str, Any]],
    *,
    arm: str,
    profile: str,
    model: str,
    env: dict[str, str],
    thinking: bool,
    budget: int,
    workers: int,
    timeout_s: float,
    tag: str,
) -> dict[str, Any]:
    build, _ = ARMS[arm]
    out: dict[str, Any] = {}
    lock = threading.Lock()
    done = {"n": 0}

    def one(window: dict[str, Any]) -> None:
        try:
            parsed, raw = call_omni(
                audio_path=window["audio"],
                fmt=window["fmt"],
                audio_content_mode=AUDIO_MODE.get(profile, "input_audio"),
                model=model,
                api_key=env["OMNI_API_KEY"].split(",")[0].strip(),
                base_url=env["OMNI_BASE_URL"],
                timeout_s=timeout_s,
                store_stream_chunks=False,
                prompt=build(window["chunks"]),
                max_tokens=4096,
                enable_thinking=thinking,
                thinking_budget=budget,
                provider_profile=profile,
                response_format=None,
            )
        except Exception as error:  # noqa: BLE001
            with lock:
                out[window["window_id"]] = {"error": str(error)[:200]}
                done["n"] += 1
            return
        payload = parsed if isinstance(parsed, dict) else {}
        usage = (raw or {}).get("usage") or {}
        with lock:
            out[window["window_id"]] = {
                "decisions": {
                    # the detect arm answers `has_word` rather than `label`
                    str(d.get("id")): str(
                        d.get("label") or d.get("has_word") or ""
                    ).strip().lower()
                    for d in (payload.get("chunk_decisions") or [])
                },
                "categories": {
                    str(d.get("id")): str(d.get("category") or "").strip().lower()
                    for d in (payload.get("chunk_decisions") or [])
                },
                "completion_tokens": usage.get("completion_tokens"),
            }
            done["n"] += 1
            if done["n"] % 10 == 0:
                print(f"  [{tag}] {done['n']}/{len(windows)}", flush=True)

    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, windows))
    print(f"  [{tag}] done in {time.time() - started:.0f}s", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="datasets/train/omni-joint-boundary-preasr-v3"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--profile", default="qwen")
    parser.add_argument("--model", default="qwen3-omni-flash")
    parser.add_argument("--thinking", type=int, default=1)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--arms", default="binary,typed")
    parser.add_argument("--base-url", action="append", default=[])
    args = parser.parse_args(argv)

    override = {}
    for item in args.base_url:
        key, _, url = item.partition("=")
        override[key.strip()] = url.strip()

    dataset = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    windows = load_windows(dataset, count=args.windows, seed=args.seed)
    total = sum(len(w["reference"]) for w in windows)
    env = load_env(args.profile, override)
    print(f"{len(windows)} windows / {total} chunks | model={args.model} "
          f"thinking={bool(args.thinking)} budget={args.thinking_budget}")
    print(f"endpoint={env['OMNI_BASE_URL']} | {args.passes} passes per arm, no schema\n")

    results: dict[str, list[dict[str, Any]]] = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        print(f"=== arm={arm} ===")
        results[arm] = [
            run_pass(
                windows,
                arm=arm,
                profile=args.profile,
                model=args.model,
                env=env,
                thinking=bool(args.thinking),
                budget=args.thinking_budget,
                workers=args.workers,
                timeout_s=args.timeout_s,
                tag=f"{arm}#{i + 1}",
            )
            for i in range(args.passes)
        ]

    reference = {w["window_id"]: w["reference"] for w in windows}
    (output / "task_definition_ab.json").write_text(
        json.dumps(
            {
                "schema": "pre_asr_task_definition_ab_v1",
                "model": args.model,
                "thinking": bool(args.thinking),
                "windows": len(windows),
                "chunks": total,
                "reference": reference,
                "arms": results,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"\n{'arm':<10}{'win ok':>9}{'run-to-run':>13}{'committed r2r':>15}"
          f"{'abstain%':>10}{'coverage':>10}{'derived vs v3':>16}{'tok':>7}")
    for arm, passes in results.items():
        _, labels = ARMS[arm]
        both = same = 0
        unsure = graded = 0
        dsame = dtot = 0
        csame = ctot = 0
        okw = min(sum(1 for v in p.values() if "decisions" in v) for p in passes)
        toks = [v["completion_tokens"] for p in passes for v in p.values()
                if v.get("completion_tokens")]
        for w, ref in reference.items():
            ds = [p.get(w, {}).get("decisions") or {} for p in passes]
            for cid, want in ref.items():
                vals = [d.get(cid) for d in ds]
                if not all(v in labels for v in vals):
                    continue
                both += 1
                same += len(set(vals)) == 1
                unsure += sum(v == "unsure" for v in vals)
                graded += len(vals)
                # Stability restricted to chunks BOTH passes committed on: the
                # point of an abstain option is that what survives it is solid.
                derived = [derive_keep_drop(arm, v) for v in vals]
                if all(x in ("keep", "drop") for x in derived):
                    ctot += 1
                    csame += len(set(derived)) == 1
                # decision agreement vs the v3 keep/drop, first pass only
                got = derived[0]
                if got in ("keep", "drop"):
                    dtot += 1
                    dsame += got == want
        print(f"{arm:<10}{okw:>6}/{len(windows):<2}{100*same/max(both,1):>12.1f}%"
              f"{100*csame/max(ctot,1):>14.1f}%"
              f"{100*unsure/max(graded,1):>9.1f}%"
              f"{100*ctot/max(both,1):>9.1f}%"
              f"{100*dsame/max(dtot,1):>14.1f}% (n={dtot})"
              f"{sum(toks)/max(len(toks),1):>7.0f}")

    for arm, passes in results.items():
        dist = Counter(v for p in passes for r in p.values()
                       for v in (r.get("decisions") or {}).values())
        print(f"\n{arm} label distribution (both passes): {dict(dist.most_common())}")
        if arm == "typed":
            cats = Counter(v for p in passes for r in p.values()
                           for v in (r.get("categories") or {}).values() if v)
            off = {c: n for c, n in cats.items() if c not in TYPED_CATEGORIES}
            print(f"  categories: {dict(cats.most_common(10))}")
            print(f"  off-vocabulary categories: {off or 'none'}")

    for arm, passes in results.items():
        errs = [v["error"][:70] for p in passes for v in p.values() if "error" in v]
        if errs:
            print(f"\n{arm} failures ({len(errs)}):")
            for e, n in Counter(errs).most_common(3):
                print(f"   {n:>3}  {e}")
    print(f"\nwrote {output / 'task_definition_ab.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
