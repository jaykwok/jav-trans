#!/usr/bin/env python3
"""A/B the pre-ASR keep/drop teacher across providers on identical chunks.

The 13.7 h of real training data behind the typed-span target was labelled by
`qwen3.5-omni-flash` (per `label_source` in pre_asr/labels.jsonl). Before
spending quota to extend it, this asks whether a
different model labels the SAME chunks the same way - because a mixed-provider
training set only helps if the providers agree on the target.

Deliberately not routed through `label_joint_boundary_preasr_with_omni.py`: that
tool writes dataset-shaped artifacts and would need a provider switch threaded
through it for an experiment. Here the only thing shared with production is the
prompt builder, imported rather than copied, so the arms cannot silently drift
from the target they are supposed to be measuring.

The existing v3 label is a REFERENCE, not ground truth - it is one model's
opinion. Agreement against it measures reproducibility of the current corpus,
not correctness; only a human audit settles correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from tools.datasets.label_joint_boundary_preasr_with_omni import (  # noqa: E402
    _build_pre_asr_prompt,
)

LABELS = ("keep", "drop", "unsure")
# Stratify on the TRAINING label, which has the borderline class: `omni_label`
# only ever holds keep/drop, because a low-confidence decision is demoted to
# `ambiguous_ignore` downstream rather than being returned as "unsure".
STRATA = ("definite_keep", "definite_drop", "ambiguous_ignore")
# Which audio content encoding each provider's OpenAI-compatible surface wants.
AUDIO_MODE = {"qwen": "input_audio", "openrouter": "input_audio_raw"}


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def sample_chunks(dataset: Path, *, per_label: int, seed: int) -> list[dict[str, Any]]:
    """Stratified by the existing label so disagreement is visible per class."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for line in (dataset / "pre_asr" / "labels.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = str(row.get("label") or "")
        if label in STRATA and Path(row["audio"]).is_file():
            by_label[label].append(row)
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    for label in STRATA:
        pool = sorted(by_label.get(label, []), key=lambda r: str(r["candidate_id"]))
        rng.shuffle(pool)
        picked.extend(pool[:per_label])
    return picked


def run_arm(
    chunks: list[dict[str, Any]],
    *,
    profile: str,
    model: str,
    env: dict[str, str],
    thinking: bool,
    budget: int,
    workers: int,
    timeout_s: float,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    done = {"n": 0}

    def one(row: dict[str, Any]) -> None:
        prompt = _build_pre_asr_prompt(
            {"duration_s": float(row.get("duration_s") or 0.0)}, item_id="p000"
        )
        try:
            parsed, raw = call_omni(
                audio_path=Path(row["audio"]),
                fmt="wav",
                audio_content_mode=AUDIO_MODE.get(profile, "input_audio"),
                model=model,
                api_key=env["OMNI_API_KEY"].split(",")[0].strip(),
                base_url=env["OMNI_BASE_URL"],
                timeout_s=timeout_s,
                store_stream_chunks=False,
                prompt=prompt,
                max_tokens=1024,
                enable_thinking=thinking,
                thinking_budget=budget,
                provider_profile=profile,
            )
        except Exception as error:  # noqa: BLE001 - recorded per chunk
            with lock:
                results[row["candidate_id"]] = {"error": str(error)[:200]}
                done["n"] += 1
            return
        payload = parsed if isinstance(parsed, dict) else {}
        usage = (raw or {}).get("usage") or {}
        with lock:
            results[row["candidate_id"]] = {
                "label": str(payload.get("label") or "").strip().lower(),
                "confidence": payload.get("confidence"),
                "semantic_speech_detected": payload.get("semantic_speech_detected"),
                "flags": payload.get("flags") or [],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
            done["n"] += 1
            if done["n"] % 20 == 0:
                print(f"  [{model}] {done['n']}/{len(chunks)}", flush=True)

    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, chunks))
    print(f"  [{model}] done in {time.time() - started:.0f}s", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="datasets/train/omni-joint-boundary-preasr-v3"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-label", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help="profile:model[:thinking[:budget]], e.g. qwen:qwen3-omni-flash:1:512",
    )
    parser.add_argument(
        "--base-url",
        action="append",
        default=[],
        help=(
            "Override a profile endpoint as profile=URL. The private MaaS "
            "deployment caps thinking_budget at 0, so reasoning is impossible "
            "there; the public DashScope endpoint serves the same model with "
            "reasoning enabled."
        ),
    )
    args = parser.parse_args(argv)

    dataset = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    chunks = sample_chunks(dataset, per_label=args.per_label, seed=args.seed)
    print(f"sampled {len(chunks)} chunks: "
          f"{Counter(str(c.get('label')) for c in chunks)}")

    arms: dict[str, dict[str, Any]] = {}
    for spec in args.arm:
        parts = spec.split(":")
        profile, model = parts[0], parts[1]
        thinking = len(parts) > 2 and parts[2] not in ("0", "false", "")
        budget = int(parts[3]) if len(parts) > 3 else 0
        env = load_env(Path.home() / ".config" / "omni" / profile)
        for override in args.base_url:
            key, _, url = override.partition("=")
            if key.strip() == profile:
                env = dict(env, OMNI_BASE_URL=url.strip())
        print(
            f"\n=== arm {model} (profile={profile}, thinking={thinking}, "
            f"budget={budget}, endpoint={env['OMNI_BASE_URL']}) ==="
        )
        arms[model] = run_arm(
            chunks,
            profile=profile,
            model=model,
            env=env,
            thinking=thinking,
            budget=budget,
            workers=args.workers,
            timeout_s=args.timeout_s,
        )

    reference = {c["candidate_id"]: str(c.get("omni_label") or "") for c in chunks}
    strata = {c["candidate_id"]: str(c.get("label") or "") for c in chunks}
    payload = {
        "schema": "pre_asr_teacher_model_ab_v1",
        "dataset": str(dataset),
        "chunks": len(chunks),
        "reference_model": "qwen3.5-omni-flash (existing v3 labels)",
        "reference": reference,
        "strata": strata,
        "arms": arms,
    }
    (output / "ab_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"\n{'arm':<26}{'ok':>6}{'vs reference':>14}{'prompt tok':>12}{'compl tok':>11}")
    for model, res in arms.items():
        ok = [k for k, v in res.items() if v.get("label") in LABELS]
        same = sum(1 for k in ok if res[k]["label"] == reference[k])
        pt = [v["prompt_tokens"] for v in res.values() if v.get("prompt_tokens")]
        ct = [v["completion_tokens"] for v in res.values() if v.get("completion_tokens")]
        print(f"{model:<26}{len(ok):>4}/{len(res):<2}"
              f"{100 * same / max(len(ok), 1):>13.1f}%"
              f"{sum(pt) / max(len(pt), 1):>12.0f}{sum(ct) / max(len(ct), 1):>11.0f}")

    models = list(arms)
    if len(models) == 2:
        a, b = models
        both = [k for k in arms[a]
                if arms[a][k].get("label") in LABELS and arms[b].get(k, {}).get("label") in LABELS]
        agree = sum(1 for k in both if arms[a][k]["label"] == arms[b][k]["label"])
        print(f"\n{a} vs {b}: {100 * agree / max(len(both), 1):.1f}% agree on {len(both)} chunks")
        confusion: Counter = Counter()
        for k in both:
            confusion[(reference[k], arms[a][k]["label"], arms[b][k]["label"])] += 1
        print(f"\n{'reference':>10}{a[:14]:>16}{b[:14]:>16}{'n':>5}")
        for (ref, la, lb), n in confusion.most_common(12):
            print(f"{ref:>10}{la:>16}{lb:>16}{n:>5}")
    print(f"\nwrote {output / 'ab_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
