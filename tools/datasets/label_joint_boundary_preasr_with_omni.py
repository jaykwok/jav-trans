#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.asr.cueqc.label_pre_asr_v10_with_omni import (  # noqa: E402
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    call_omni,
    first_env_value,
    load_env_file,
    normalize_omni_label,
    slice_audio_clip,
    training_label_from_omni,
)


PROMPT_VERSION = "joint_boundary_preasr_omni_v2"
JOINT_SCHEMA = "joint_boundary_preasr_omni_label_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _select_split_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: str,
) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return sorted(rows, key=lambda row: int(row["index"]))
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    accepted = [row for row in rows if bool(row.get("accepted"))]
    rng.shuffle(accepted)
    selected.extend(accepted[: max(1, limit // 3)])
    used = {int(row["index"]) for row in selected}
    bins = ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01))
    for low, high in bins:
        pool = [
            row
            for row in rows
            if int(row["index"]) not in used
            and low <= float(row.get("p_cut") or 0.0) < high
        ]
        rng.shuffle(pool)
        take = max(1, (limit - len(selected)) // max(1, len(bins)))
        for row in pool[:take]:
            selected.append(row)
            used.add(int(row["index"]))
    if len(selected) < limit:
        pool = [row for row in rows if int(row["index"]) not in used]
        rng.shuffle(pool)
        selected.extend(pool[: limit - len(selected)])
    return sorted(selected[:limit], key=lambda row: float(row["time_s"]))


def _select_chunk_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: str,
) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return sorted(rows, key=lambda row: int(row["chunk_index"]))
    rng = random.Random(seed)
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row.get("duration_s") or 0.0),
            int(row["chunk_index"]),
        ),
    )
    selected = ranked[: max(1, limit // 3)]
    used = {int(row["chunk_index"]) for row in selected}
    pool = [row for row in rows if int(row["chunk_index"]) not in used]
    rng.shuffle(pool)
    selected.extend(pool[: limit - len(selected)])
    return sorted(selected, key=lambda row: int(row["chunk_index"]))


def _build_prompt(
    split_items: list[dict[str, Any]],
    chunk_items: list[dict[str, Any]],
    *,
    duration_s: float,
) -> str:
    split_payload = [
        {
            "id": f"s{position:03d}",
            "time_s": round(float(row["time_s"]), 3),
            "current": str(row.get("label") or ""),
            "p_cut": round(float(row.get("p_cut") or 0.0), 4),
        }
        for position, row in enumerate(split_items)
    ]
    chunk_payload = [
        {
            "id": f"p{position:03d}",
            "start_s": round(float(row["start"]), 3),
            "end_s": round(float(row["end"]), 3),
        }
        for position, row in enumerate(chunk_items)
    ]
    return f"""你是日语 ASR 边界与语义声音联合标注器。只听这一个音频，一次完成三类标注。
音频时间范围为 0.000 到 {duration_s:.3f} 秒；下列时间都相对此音频。

任务 A：逐个判断 split_candidates 的候选切点。
- cut：左右是可独立送入 ASR/字幕的语义单元，且左右都完整。
- continue：同一句内部停顿、喘息、拖音、重复、呻吟或短静音；合并更自然。
- unsure：无法可靠判断。
- 不要仅因静音、喘息或说话人变化就切。cut 必须同时满足 left_complete=true、right_complete=true、merged_better=false。

任务 A2：报告漏掉的句界。如果音频中存在明显的语义句界，但 split_candidates 里 ±0.30 秒内没有任何候选，
把它的时间（三位小数）加入 missed_boundaries 并给出 confidence；没有漏掉的句界就返回空数组。

任务 B：逐个判断 runtime_chunks 是否应送入 ASR。
- keep：至少包含一个可辨认且有词义的日语词或短句。
- drop：只有喘息、呻吟、呼吸、亲吻、笑声、音乐、静音、环境声或噪声。
- unsure：无法可靠判断。
- 重复但有明确词义（例如反复说“ありがとう”）仍是 keep。

必须对输入中的每个 id 恰好返回一次，不得发明 id。为节省输出 token，不写逐项 reason。
split_candidates={json.dumps(split_payload, ensure_ascii=False, separators=(",", ":"))}
runtime_chunks={json.dumps(chunk_payload, ensure_ascii=False, separators=(",", ":"))}

只输出 JSON，不要 Markdown：
{{
  "split_decisions":[
    {{"id":"s000","label":"cut|continue|unsure","confidence":0.0,"left_complete":false,"right_complete":false,"merged_better":true,"flags":[]}}
  ],
  "missed_boundaries":[
    {{"time_s":0.000,"confidence":0.0}}
  ],
  "chunk_decisions":[
    {{"id":"p000","label":"keep|drop|unsure","confidence":0.0,"semantic_speech_detected":false,"flags":[]}}
  ]
}}"""


def _response_by_id(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        item_id = str(item.get("id") or "")
        if item_id and item_id not in result:
            result[item_id] = dict(item)
    return result


def _confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, parsed))


def _normalize_split_decision(
    item: Mapping[str, Any] | None,
    *,
    confidence_threshold: float,
) -> dict[str, Any]:
    payload = dict(item or {})
    label = str(payload.get("label") or "unsure").strip().lower()
    if label not in {"cut", "continue", "unsure"}:
        label = "unsure"
    confidence = _confidence(payload.get("confidence"))
    if confidence < confidence_threshold:
        label = "unsure"
    left_complete = bool(payload.get("left_complete"))
    right_complete = bool(payload.get("right_complete"))
    merged_better = bool(payload.get("merged_better"))
    if label == "cut" and (
        not left_complete or not right_complete or merged_better
    ):
        label = "unsure"
    return {
        "label": label,
        "confidence": confidence,
        "left_complete": left_complete,
        "right_complete": right_complete,
        "merged_better": merged_better,
        "flags": [str(value) for value in (payload.get("flags") or [])],
    }


def _normalize_chunk_decision(
    item: Mapping[str, Any] | None,
    *,
    keep_confidence: float,
    drop_confidence: float,
) -> dict[str, Any]:
    payload = dict(item or {})
    omni_label = normalize_omni_label(payload.get("label"))
    confidence = _confidence(payload.get("confidence"))
    label = training_label_from_omni(
        label=omni_label,
        confidence=confidence,
        keep_confidence=keep_confidence,
        drop_confidence=drop_confidence,
    )
    return {
        "label": label,
        "omni_label": omni_label,
        "confidence": confidence,
        "semantic_speech_detected": bool(payload.get("semantic_speech_detected")),
        "flags": [str(value) for value in (payload.get("flags") or [])],
    }


def _normalize_missed_boundaries(
    value: Any,
    *,
    duration_s: float,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    """Keep in-window missed-boundary reports; drop junk and low confidence."""

    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in value:
        if not isinstance(item, Mapping):
            continue
        try:
            time_s = float(item.get("time_s"))
        except (TypeError, ValueError):
            continue
        if not 0.0 < time_s < duration_s:
            continue
        confidence = _confidence(item.get("confidence"))
        if confidence < confidence_threshold:
            continue
        key = int(round(time_s * 1000.0))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "time_s": round(time_s, 3),
                "confidence": confidence,
                "flags": [str(flag) for flag in (item.get("flags") or [])],
            }
        )
    return sorted(rows, key=lambda row: row["time_s"])


def _call_with_retry(
    *,
    audio_path: Path,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    audio_content_mode: str,
    timeout_s: float,
    max_tokens: int,
    audio_fmt: str = "wav",
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_error: BaseException | None = None
    for attempt in range(6):
        try:
            return call_omni(
                audio_path=audio_path,
                fmt=audio_fmt,
                audio_content_mode=audio_content_mode,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=timeout_s,
                store_stream_chunks=False,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            message = str(exc).lower()
            if (
                "429" in message
                or "limit_requests" in message
                or "rate limit" in message
            ):
                time.sleep(min(30.0, 2.0 * (attempt + 1)))
                continue
            if attempt < 2:
                time.sleep(1.0 + attempt)
                continue
            break
    assert last_error is not None
    raise last_error


def _process_window(
    row: dict[str, Any],
    *,
    output: Path,
    segments_root: Path,
    split_limit: int,
    chunk_limit: int,
    split_confidence: float,
    keep_confidence: float,
    drop_confidence: float,
    model: str,
    api_key: str,
    base_url: str,
    audio_content_mode: str,
    timeout_s: float,
    max_tokens: int,
    prepare_only: bool,
) -> dict[str, Any]:
    window_id = str(row["window_id"])
    split_rows = _select_split_rows(
        _read_jsonl(Path(row["semantic_split_metadata"])),
        limit=split_limit,
        seed=f"{window_id}:split",
    )
    chunk_rows = _select_chunk_rows(
        _read_jsonl(Path(row["pre_asr_candidates"])),
        limit=chunk_limit,
        seed=f"{window_id}:chunk",
    )
    prompt = _build_prompt(
        split_rows,
        chunk_rows,
        duration_s=float(row["duration_s"]),
    )
    request_path = output / "requests" / f"{window_id}.json"
    _write_json(
        request_path,
        {
            "schema": "joint_boundary_preasr_omni_request_v1",
            "window_id": window_id,
            "prompt_version": PROMPT_VERSION,
            "audio_mp3_32k": row["omni_mp3_32k"],
            "training_audio_wav": row["audio_wav"],
            "split_candidates": split_rows,
            "runtime_chunks": chunk_rows,
            "prompt": prompt,
        },
    )
    if prepare_only:
        return {
            "window_id": window_id,
            "prepared": True,
            "split_count": len(split_rows),
            "chunk_count": len(chunk_rows),
        }
    try:
        # Upload the training-grade WAV directly: Qwen3.5-Omni encodes audio at
        # a fixed token rate (independent of bitrate), so the 32k MP3 resample
        # cost no tokens and only low-passed/quantized the breathy/whisper
        # textures that matter for keep/drop. WAV also keeps the Omni label and
        # the training feature on the exact same audio.
        omni_audio_path = Path(row["audio_wav"])
        parsed, raw = _call_with_retry(
            audio_path=omni_audio_path,
            prompt=prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            audio_content_mode=audio_content_mode,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            audio_fmt="wav",
        )
    except Exception as exc:  # noqa: BLE001
        _write_json(
            output / "errors" / f"{window_id}.json",
            {
                "schema": "joint_boundary_preasr_omni_error_v1",
                "window_id": window_id,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return {
            "window_id": window_id,
            "prepared": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    split_by_id = _response_by_id(parsed.get("split_decisions"))
    chunk_by_id = _response_by_id(parsed.get("chunk_decisions"))
    missed_boundaries = [
        {
            "schema": "joint_split_missed_boundary_v1",
            "window_id": window_id,
            "prompt_version": PROMPT_VERSION,
            "model": model,
            **missed,
        }
        for missed in _normalize_missed_boundaries(
            parsed.get("missed_boundaries"),
            duration_s=float(row["duration_s"]),
            confidence_threshold=split_confidence,
        )
    ]
    split_labels: list[dict[str, Any]] = []
    for position, candidate in enumerate(split_rows):
        item_id = f"s{position:03d}"
        decision = _normalize_split_decision(
            split_by_id.get(item_id),
            confidence_threshold=split_confidence,
        )
        split_labels.append(
            {
                "schema": "joint_semantic_split_omni_label_v1",
                "window_id": window_id,
                "feature_index": int(candidate["index"]),
                "time_s": float(candidate["time_s"]),
                "current_label": str(candidate.get("label") or ""),
                "current_p_cut": float(candidate.get("p_cut") or 0.0),
                "prompt_id": item_id,
                "prompt_version": PROMPT_VERSION,
                "model": model,
                **decision,
            }
        )
    pre_asr_labels: list[dict[str, Any]] = []
    for position, candidate in enumerate(chunk_rows):
        item_id = f"p{position:03d}"
        decision = _normalize_chunk_decision(
            chunk_by_id.get(item_id),
            keep_confidence=keep_confidence,
            drop_confidence=drop_confidence,
        )
        segment_label = str(decision["label"])
        segment_path = (
            segments_root
            / segment_label
            / f"{window_id}-chunk{int(candidate['chunk_index']):05d}.wav"
        )
        slice_audio_clip(
            source_audio=Path(row["audio_wav"]),
            row=candidate,
            output_path=segment_path,
            fmt="wav",
            bitrate="",
            sample_rate=16000,
            force=False,
        )
        pre_asr_labels.append(
            {
                "schema": "pre_asr_omni_label_v1",
                "sample_id": str(candidate["sample_id"]),
                "candidate_id": str(candidate["candidate_id"]),
                "audio_id": str(candidate["audio_id"]),
                "video_id": str(candidate["video_id"]),
                "window_id": window_id,
                "chunk_index": int(candidate["chunk_index"]),
                "start": float(candidate["start"]),
                "end": float(candidate["end"]),
                "duration_s": float(candidate["duration_s"]),
                "label": segment_label,
                "display_decision": (
                    "keep"
                    if segment_label == "definite_keep"
                    else "drop"
                    if segment_label == "definite_drop"
                    else "ambiguous_ignore"
                ),
                "training_label_included": segment_label
                in {"definite_keep", "definite_drop"},
                "label_source": f"omni:{model}",
                "omni_label": decision["omni_label"],
                "omni_confidence": decision["confidence"],
                "omni_semantic_speech_detected": decision[
                    "semantic_speech_detected"
                ],
                "omni_flags": decision["flags"],
                "prompt_id": item_id,
                "prompt_version": PROMPT_VERSION,
                "audio": str(segment_path.resolve()),
                "audio_format": "wav",
                "omni_request_audio": str(omni_audio_path.resolve()),
                "omni_request_audio_format": "wav",
                "feature_schema": str(candidate.get("feature_schema") or ""),
                "runtime_adapter": str(candidate.get("runtime_adapter") or ""),
            }
        )
    result = {
        "schema": JOINT_SCHEMA,
        "window_id": window_id,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "request": str(request_path.resolve()),
        "audio_mp3_32k": str(Path(row["omni_mp3_32k"]).resolve()),
        "training_audio_wav": str(Path(row["audio_wav"]).resolve()),
        "split_labels": split_labels,
        "pre_asr_labels": pre_asr_labels,
        "missed_boundaries": missed_boundaries,
        "response_usage": raw.get("usage"),
        "response_finish_reasons": raw.get("finish_reasons"),
    }
    _write_json(output / "joint_labels" / f"{window_id}.json", result)
    _write_json(
        output / "raw_responses" / f"{window_id}.json",
        {
            "schema": "joint_boundary_preasr_omni_raw_v1",
            "window_id": window_id,
            "parsed": parsed,
            "response": raw,
        },
    )
    return {
        "window_id": window_id,
        "prepared": False,
        "split_count": len(split_labels),
        "chunk_count": len(pre_asr_labels),
    }


def _collect_completed(
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    split_rows: list[dict[str, Any]] = []
    pre_asr_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    for path in sorted((output / "joint_labels").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        split_rows.extend(payload.get("split_labels") or [])
        pre_asr_rows.extend(payload.get("pre_asr_labels") or [])
        missed_rows.extend(payload.get("missed_boundaries") or [])
    return split_rows, pre_asr_rows, missed_rows


def run(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    dataset = Path(args.dataset_dir)
    output = Path(args.output_dir or dataset / "annotations" / "omni_joint")
    segments_root = dataset / "pre_asr" / "audio_wav"
    output.mkdir(parents=True, exist_ok=True)
    windows = _read_jsonl(dataset / "source_windows.jsonl")
    completed = {
        path.stem for path in (output / "joint_labels").glob("*.json")
    }
    selected = [
        row for row in windows if str(row["window_id"]) not in completed
    ]
    if args.max_windows is not None:
        selected = selected[: args.max_windows]
    _model_name, configured_model = first_env_value(
        ("OMNI_MODEL", "QWEN_OMNI_MODEL")
    )
    model = args.model or configured_model or "qwen3.5-omni-flash"
    _key_name, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_name, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    if not args.prepare_only and not api_key:
        raise RuntimeError("Omni API key is missing")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _process_window,
                row,
                output=output,
                segments_root=segments_root,
                split_limit=args.max_split_candidates,
                chunk_limit=args.max_runtime_chunks,
                split_confidence=args.split_confidence,
                keep_confidence=args.keep_confidence,
                drop_confidence=args.drop_confidence,
                model=model,
                api_key=api_key,
                base_url=base_url,
                audio_content_mode=args.audio_content_mode,
                timeout_s=args.timeout_s,
                max_tokens=args.max_tokens,
                prepare_only=args.prepare_only,
            ): str(row["window_id"])
            for row in selected
        }
        for position, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"completed={position}/{len(selected)} "
                f"window={result['window_id']} "
                f"error={result.get('error', '')}",
                flush=True,
            )
    split_rows, pre_asr_rows, missed_rows = _collect_completed(output)
    _write_jsonl(output / "split_labels.jsonl", split_rows)
    _write_jsonl(output / "pre_asr_labels.jsonl", pre_asr_rows)
    _write_jsonl(output / "missed_boundaries.jsonl", missed_rows)
    _write_jsonl(dataset / "semantic_split" / "labels.jsonl", split_rows)
    _write_jsonl(dataset / "semantic_split" / "missed_boundaries.jsonl", missed_rows)
    _write_jsonl(dataset / "pre_asr" / "labels.jsonl", pre_asr_rows)
    split_counts = Counter(str(row["label"]) for row in split_rows)
    pre_asr_counts = Counter(str(row["label"]) for row in pre_asr_rows)
    errors = [row for row in results if row.get("error")]
    _write_json(
        output / "summary.json",
        {
            "schema": "joint_boundary_preasr_omni_summary_v1",
            "dataset_dir": str(dataset.resolve()),
            "window_count": len(windows),
            "completed_window_count": len(
                list((output / "joint_labels").glob("*.json"))
            ),
            "attempted_this_run": len(selected),
            "errors_this_run": len(errors),
            "split_label_count": len(split_rows),
            "split_labels": dict(split_counts),
            "missed_boundary_count": len(missed_rows),
            "missed_boundary_window_count": len(
                {str(item.get("window_id")) for item in missed_rows}
            ),
            "pre_asr_label_count": len(pre_asr_rows),
            "pre_asr_labels": dict(pre_asr_counts),
            "omni_audio_format": "wav",
            "training_audio_format": "wav",
            "sample_rate": 16000,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "single_request_per_window": True,
            "semantic_split_dataset": str(
                (dataset / "semantic_split").resolve()
            ),
            "pre_asr_dataset": str((dataset / "pre_asr").resolve()),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use one Omni request per MP3 window to label both semantic split "
            "candidates and post-boundary Pre-ASR chunks, then slice WAV training audio."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets/train/omni-joint-boundary-preasr-v1",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--max-split-candidates", type=int, default=32)
    parser.add_argument("--max-runtime-chunks", type=int, default=48)
    parser.add_argument("--split-confidence", type=float, default=0.80)
    parser.add_argument("--keep-confidence", type=float, default=0.80)
    parser.add_argument("--drop-confidence", type=float, default=0.90)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--audio-content-mode",
        default=os.getenv("OMNI_AUDIO_CONTENT_MODE", "input_audio"),
    )
    args = parser.parse_args()
    if args.max_split_candidates <= 0 or args.max_runtime_chunks <= 0:
        parser.error("candidate limits must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
