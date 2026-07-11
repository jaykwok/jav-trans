#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.asr.cueqc.label_pre_asr_with_omni import (  # noqa: E402
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    append_jsonl,
    call_omni,
    first_env_value,
    load_env_file,
    slice_audio_clip,
)


PROMPT_VERSION = "semantic_split_omni_label_v1"
PROMPT = """你是 ASR chunk 切分质量标注器。音频中间位置 2.000 秒是候选切分点。
请判断这里是否适合把音频切成两个 ASR chunk。

标签定义：
- cut：候选点左右像两个独立语义片段，切开后各自适合 ASR/字幕。
- continue：候选点只是同一句话内部停顿、喘息、拖音、犹豫或短静音，不应切开。
- unsure：边界不确定，左右语义不完整，或音频太模糊。

不要仅因为静音、喘息、呻吟、呼吸声就判断为 cut。重点判断切开后左右两段是否都适合作为独立语义单元；如果合在一起更像完整一句话，应标 continue。

只输出 JSON，不要输出 Markdown：
{
  "label": "cut|continue|unsure",
  "confidence": 0.0-1.0,
  "left_complete": true|false,
  "right_complete": true|false,
  "merged_better": true|false,
  "flags": ["short_pause", "breath", "moan", "laughter", "same_sentence", "topic_shift", "speaker_change", "low_snr", "music"],
  "reason": "简短中文理由"
}
"""


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select(rows: list[dict], *, max_items: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    bins = ((0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.01))
    per_bin = max_items // len(bins)
    selected: list[dict] = []
    for low, high in bins:
        pool = [row for row in rows if low <= float(row["p_cut"]) < high]
        rng.shuffle(pool)
        selected.extend(pool[:per_bin])
    if len(selected) < max_items:
        used = {int(row["index"]) for row in selected}
        pool = [row for row in rows if int(row["index"]) not in used]
        rng.shuffle(pool)
        selected.extend(pool[: max_items - len(selected)])
    return sorted(selected, key=lambda row: int(row["index"]))


def _normalized_label(value: object) -> str:
    label = str(value or "").strip().lower()
    return label if label in {"cut", "continue", "unsure"} else "unsure"


def run(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    rows = _select(
        _read_jsonl(Path(args.candidates)),
        max_items=args.max_items,
        seed=args.seed,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "selected_candidates.jsonl"
    selected_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    labels_path = output / "omni_split_labels.jsonl"
    raw_path = output / "omni_raw_responses.jsonl"
    existing = {
        int(row["index"])
        for row in _read_jsonl(labels_path)
    } if labels_path.exists() else set()
    _model_name, model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or model or "qwen3.5-omni-flash"
    _key_name, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_name, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    audio = Path(args.audio)
    counts: Counter[str] = Counter()
    for position, row in enumerate(rows, start=1):
        index = int(row["index"])
        if index in existing and not args.prepare_only:
            continue
        center = float(row["time_s"])
        clip_row = {
            "start": max(0.0, center - 2.0),
            "end": center + 2.0,
            "duration_s": 4.0,
        }
        clip_path = output / "audio_clips" / f"split-{index:05d}-{center:.3f}.mp3"
        slice_audio_clip(
            source_audio=audio,
            row=clip_row,
            output_path=clip_path,
            fmt="mp3",
            bitrate=args.audio_bitrate,
            sample_rate=16000,
            force=False,
        )
        if args.prepare_only:
            continue
        try:
            parsed, raw = call_omni(
                audio_path=clip_path,
                fmt="mp3",
                audio_content_mode=args.audio_content_mode,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=args.timeout_s,
                store_stream_chunks=False,
                prompt=PROMPT,
            )
        except Exception as exc:  # noqa: BLE001
            parsed = {
                "label": "unsure",
                "confidence": 0.0,
                "left_complete": False,
                "right_complete": False,
                "merged_better": False,
                "flags": ["api_error"],
                "reason": f"Omni request failed: {exc}",
            }
            raw = {"error": repr(exc)}
        omni_label = _normalized_label(parsed.get("label"))
        confidence = min(1.0, max(0.0, float(parsed.get("confidence") or 0.0)))
        label = omni_label if confidence >= args.confidence else "unsure"
        if label == "cut" and (
            not bool(parsed.get("left_complete"))
            or not bool(parsed.get("right_complete"))
            or bool(parsed.get("merged_better"))
        ):
            label = "unsure"
        result = {
            "schema": "semantic_split_omni_label_v1",
            "index": index,
            "audio": str(audio),
            "time_s": center,
            "current_p_cut": float(row["p_cut"]),
            "current_label": str(row["label"]),
            "label": label,
            "omni_label": omni_label,
            "confidence": confidence,
            "left_complete": bool(parsed.get("left_complete")),
            "right_complete": bool(parsed.get("right_complete")),
            "merged_better": bool(parsed.get("merged_better")),
            "flags": list(parsed.get("flags") or []),
            "reason": str(parsed.get("reason") or ""),
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "audio_clip": str(clip_path),
        }
        append_jsonl(labels_path, result)
        append_jsonl(
            raw_path,
            {
                "schema": "semantic_split_omni_raw_v1",
                "index": index,
                "parsed": parsed,
                "response": raw,
            },
        )
        counts[label] += 1
        if args.sleep_s:
            time.sleep(args.sleep_s)
        if position % 25 == 0:
            print(
                f"processed={position}/{len(rows)} labels={dict(counts)}",
                flush=True,
            )
    summary = {
        "schema": "semantic_split_omni_run_v1",
        "selected": len(rows),
        "labels": dict(counts),
        "model": model,
        "confidence_threshold": args.confidence,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-items", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", default="")
    parser.add_argument("--audio-bitrate", default="32k")
    parser.add_argument("--audio-content-mode", default=os.getenv("OMNI_AUDIO_CONTENT_MODE", "input_audio"))
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
