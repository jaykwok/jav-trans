#!/usr/bin/env python3
"""Replay the EXACT v3 joint request against other models, and diff the answers.

The 13.7 h corpus was labelled by `joint_boundary_preasr_omni_v2`: one call per
75 s window, whole-window audio, every chunk id and split candidate in a single
prompt. The current labeller (`v4_mmss_mmm`) instead issues isolated per-chunk
calls on chunk-only audio. Those are different tasks - a chunk judged with its
siblings audible is not the chunk judged alone - so a model comparison run under
v4 cannot tell you whether a model reproduces the v3 corpus.

This replays the stored v3 prompt byte-for-byte with the stored window audio, so
the only variable left is the model. That makes the result answerable in the way
that matters before spending quota: can provider X extend this corpus without
silently changing what the labels mean?

Reference answers come from the same stored artifact, so no re-derivation can
drift from what the corpus actually contains.
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

AUDIO_MODE = {"qwen": "input_audio", "openrouter": "input_audio_raw"}
CHUNK_LABELS = ("keep", "drop", "unsure")

# The v2 joint prompt asks for all three blocks; constrain the shape without
# changing the task, because unconstrained JSON was the single largest failure
# source in the per-chunk A/B (13/120 for one provider, 0/30 once schema'd).
JOINT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "joint_boundary_preasr",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "chunk_decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {"type": "string", "enum": list(CHUNK_LABELS)},
                            "confidence": {"type": "number"},
                            "semantic_speech_detected": {"type": "boolean"},
                            "flags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "label"],
                    },
                },
                "split_decisions": {"type": "array", "items": {"type": "object"}},
                "missed_boundaries": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["chunk_decisions"],
        },
    },
}


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


def load_windows(
    dataset: Path, *, count: int, seed: int, audio_field: str
) -> list[dict[str, Any]]:
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
        audio = Path(str(request.get(audio_field) or ""))
        if not reference or not audio.is_file():
            continue
        out.append(
            {
                "window_id": name,
                "prompt": request["prompt"],
                "audio": audio,
                "fmt": audio.suffix.lstrip(".").lower(),
                "reference": reference,
                "prompt_version": request.get("prompt_version"),
            }
        )
        if len(out) >= count:
            break
    return out


def run_arm(
    windows: list[dict[str, Any]],
    *,
    profile: str,
    model: str,
    env: dict[str, str],
    thinking: bool,
    budget: int,
    workers: int,
    timeout_s: float,
    use_schema: bool,
) -> dict[str, Any]:
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
                prompt=window["prompt"],
                max_tokens=4096,
                enable_thinking=thinking,
                thinking_budget=budget,
                provider_profile=profile,
                response_format=JOINT_SCHEMA if use_schema else None,
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
                    str(d.get("id")): str(d.get("label") or "").lower()
                    for d in (payload.get("chunk_decisions") or [])
                },
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
            done["n"] += 1
            if done["n"] % 5 == 0:
                print(f"  [{model}] {done['n']}/{len(windows)}", flush=True)

    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, windows))
    print(f"  [{model}] done in {time.time() - started:.0f}s", flush=True)
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
    parser.add_argument("--no-schema", action="store_true")
    parser.add_argument(
        "--audio-field",
        default="audio_mp3_32k",
        choices=("audio_mp3_32k", "training_audio_wav"),
        help=(
            "Which stored audio to replay. The v3 corpus was labelled on "
            "`audio_mp3_32k`; replaying the WAV changes the codec as well as "
            "the model, so agreement numbers stop being attributable."
        ),
    )
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--base-url", action="append", default=[])
    args = parser.parse_args(argv)

    override = {}
    for item in args.base_url:
        key, _, url = item.partition("=")
        override[key.strip()] = url.strip()

    dataset = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    windows = load_windows(
        dataset, count=args.windows, seed=args.seed, audio_field=args.audio_field
    )
    total_chunks = sum(len(w["reference"]) for w in windows)
    print(f"replaying {len(windows)} windows / {total_chunks} chunk decisions "
          f"(prompt_version={windows[0]['prompt_version']}, "
          f"audio={args.audio_field}, schema={not args.no_schema})")

    arms: dict[str, Any] = {}
    for spec in args.arm:
        parts = spec.split(":")
        profile, model = parts[0], parts[1]
        thinking = len(parts) > 2 and parts[2] not in ("0", "false", "")
        budget = int(parts[3]) if len(parts) > 3 else 0
        env = load_env(profile, override)
        print(f"\n=== {model} (profile={profile}, thinking={thinking}, "
              f"budget={budget}, endpoint={env['OMNI_BASE_URL']}) ===")
        arms[model] = run_arm(
            windows,
            profile=profile,
            model=model,
            env=env,
            thinking=thinking,
            budget=budget,
            workers=args.workers,
            timeout_s=args.timeout_s,
            use_schema=not args.no_schema,
        )

    reference = {w["window_id"]: w["reference"] for w in windows}
    (output / "replay_results.json").write_text(
        json.dumps(
            {
                "schema": "pre_asr_joint_replay_ab_v1",
                "dataset": str(dataset),
                "audio_field": args.audio_field,
                "windows": len(windows),
                "chunk_decisions": total_chunks,
                "reference": reference,
                "arms": arms,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"\n{'arm':<26}{'windows ok':>12}{'chunks':>8}{'coverage':>10}{'agree':>9}")
    per_arm_labels: dict[str, Counter] = {}
    for model, res in arms.items():
        ok = [w for w, v in res.items() if "decisions" in v]
        matched = agree = 0
        labels: Counter = Counter()
        for w in ok:
            got, ref = res[w]["decisions"], reference[w]
            for cid, want in ref.items():
                if cid in got and got[cid] in CHUNK_LABELS:
                    matched += 1
                    labels[got[cid]] += 1
                    agree += got[cid] == want
        per_arm_labels[model] = labels
        cov = matched / max(total_chunks, 1)
        print(f"{model:<26}{len(ok):>7}/{len(res):<4}{matched:>8}{100*cov:>9.1f}%"
              f"{100*agree/max(matched,1):>8.1f}%")

    print(f"\n{'arm':<26}{'keep':>7}{'drop':>7}{'unsure':>8}")
    for model, labels in per_arm_labels.items():
        print(f"{model:<26}{labels['keep']:>7}{labels['drop']:>7}{labels['unsure']:>8}")
    ref_labels = Counter(v for r in reference.values() for v in r.values())
    print(f"{'v3 reference':<26}{ref_labels['keep']:>7}{ref_labels['drop']:>7}"
          f"{ref_labels['unsure']:>8}")

    names = list(arms)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = agree = 0
            for w in reference:
                da = arms[a].get(w, {}).get("decisions") or {}
                db = arms[b].get(w, {}).get("decisions") or {}
                for cid in reference[w]:
                    if da.get(cid) in CHUNK_LABELS and db.get(cid) in CHUNK_LABELS:
                        both += 1
                        agree += da[cid] == db[cid]
            print(f"\n{a} vs {b}: {100*agree/max(both,1):.1f}% agree on {both} chunks")

    for model, res in arms.items():
        errs = [v["error"][:70] for v in res.values() if "error" in v]
        if errs:
            print(f"\n{model} failures ({len(errs)}):")
            for e, n in Counter(errs).most_common(3):
                print(f"   {n:>3}  {e}")
    print(f"\nwrote {output / 'replay_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
