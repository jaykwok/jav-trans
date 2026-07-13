#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.asr.cueqc.label_pre_asr_with_omni import (  # noqa: E402
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    audio_content_part,
    call_omni,
    extract_json_object,
    first_env_value,
    load_env_file,
    slice_audio_clip,
)


SCHEMA = "inner_edge_safe_zone_teacher_v1"
SELECTION_SCHEMA = "inner_edge_safe_zone_selection_v1"
PROMPT_VERSION = "inner_edge_safe_zone_omni_plus_monotonic_v2"
DEFAULT_MODEL = "qwen3.5-omni-plus"
LABELS = {"left_clipped", "safe", "right_clipped", "unsure"}
DEFAULT_BOUNDARIES = (
    "MKMP-636.H265-e5cfe4fcd6-w03#outer003#b000",
    "787HNAMH-002-a80a2b0215-w00#outer002#b007",
    "787HNAMH-002-a80a2b0215-w00#outer002#b008",
    "KRND-017-239dac3a1f-w05#outer003#b009",
    "KRND-017-239dac3a1f-w05#outer003#b010",
)

SYSTEM_PROMPT = """你是日语语音内部安全区标注器。每次请求只处理一个已经确认语义上应该切开的 boundary。不要重新判断语义是否可切，不要转录，不要输出时间戳。

同一请求中的候选全部属于同一个 target semantic boundary，并严格按时间从早到晚排列。必须联合比较整组候选，只追踪这一个 target boundary；即使更早处本身也是静音，只要它位于目标左语义单元完成之前，仍是 left_clipped，不能误判为另一个 safe boundary。

每个候选音频都由同一原始 speech island 在一个 proposer candidate 处分成“左段 + 1 秒静音 + 右段”，然后按原顺序拼接。只判断插入静音的位置是否完整保留 target boundary 两侧的语音：
- left_clipped：候选太早，目标左语义单元尚未完整留在左段（可能只缺最后 mora，也可能整个目标左语义单元仍在右段）。
- safe：切点位于明确内部非语音区，左右语音都完整。
- right_clipped：候选太晚，目标右语义单元的开头已被留在左段，右段缺失句首。
- unsure：连续语音、重叠语音、音质不足或无法可靠判断。

除 unsure 外，整组标签必须随候选从早到晚保持单调：零个或多个 left_clipped，随后零个或多个连续 safe，最后零个或多个 right_clipped。绝不能从 safe/right_clipped 回到 left_clipped，也不能出现两个分离的 safe 区。

例：若 c00/c01 还没保留完整左句，c02/c03 位于同一安全停顿，c04 已吃掉右句开头，则输出 left_clipped,left_clipped,safe,safe,right_clipped。

日语句尾 mora 零容忍，尤其注意「ね、よ、な、か、の、わ、です、ます」的弱读、气声和拖长。不要因为 1 秒静音听感更自然就判 safe；只依据静音插入点两侧是否保留完整语音。

只输出 JSON：
{"boundary_id":"...","candidates":[{"candidate_id":"c00","label":"left_clipped|safe|right_clipped|unsure","confidence":0.0,"reason":"简短中文理由"}]}
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_boundaries(
    islands_path: Path,
    semantic_labels_path: Path,
    boundary_ids: list[str],
    candidate_count: int = 7,
) -> list[dict[str, Any]]:
    if not 5 <= candidate_count <= 9:
        raise ValueError("candidate_count must be between 5 and 9")
    islands = {str(row["island_id"]): row for row in _read_jsonl(islands_path)}
    labels = {str(row["island_id"]): row for row in _read_jsonl(semantic_labels_path)}
    selected = []
    for boundary_id in boundary_ids:
        island_id, marker = boundary_id.rsplit("#b", 1)
        boundary_index = int(marker)
        island = islands[island_id]
        cuts = sorted(labels[island_id].get("cuts") or [], key=lambda row: float(row["time_s"]))
        event_time = float(cuts[boundary_index]["time_s"])
        previous = float(cuts[boundary_index - 1]["time_s"]) if boundary_index else 0.0
        following = float(cuts[boundary_index + 1]["time_s"]) if boundary_index + 1 < len(cuts) else float(island["duration_s"])
        region_start = (previous + event_time) / 2.0
        region_end = (event_time + following) / 2.0
        candidates = [
            row for row in sorted(island.get("candidates") or [], key=lambda row: float(row["relative_time_s"]))
            if region_start <= float(row["relative_time_s"]) <= region_end
        ]
        if len(candidates) < 5:
            raise ValueError(f"{boundary_id} has only {len(candidates)} proposer candidates in its adaptive region")
        anchor = min(range(len(candidates)), key=lambda index: abs(float(candidates[index]["relative_time_s"]) - event_time))
        start = max(0, min(anchor - candidate_count // 2, len(candidates) - candidate_count))
        chosen = candidates[start : start + min(candidate_count, len(candidates))]
        selected.append(
            {
                "schema": SELECTION_SCHEMA,
                "boundary_id": boundary_id,
                "island_id": island_id,
                "window_id": island["window_id"],
                "duration_s": float(island["duration_s"]),
                "span_start_s": float(island["span_start_s"]),
                "span_end_s": float(island["span_end_s"]),
                "source_audio": str(island["source_audio"]),
                "candidates": [
                    {
                        "candidate_id": f"c{index:02d}",
                        "feature_index": int(row["feature_index"]),
                        "relative_time_s": float(row["relative_time_s"]),
                        "p_cut": float(row.get("p_cut") or 0.0),
                    }
                    for index, row in enumerate(chosen)
                ],
            }
        )
    return selected


def _build_candidate_audio(source: Path, output: Path, duration_s: float, cut_s: float) -> None:
    import subprocess

    output.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        f"[0:a]atrim=start=0:end={cut_s:.6f},asetpts=PTS-STARTPTS[left];"
        "anullsrc=r=16000:cl=mono:d=1[silence];"
        f"[0:a]atrim=start={cut_s:.6f}:end={duration_s:.6f},asetpts=PTS-STARTPTS[right];"
        "[left][silence][right]concat=n=3:v=0:a=1[out]"
    )
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-filter_complex", filters, "-map", "[out]", "-ac", "1", "-ar", "16000", str(output)], check=True)


def _validate(parsed: dict[str, Any], boundary: dict[str, Any]) -> list[dict[str, Any]]:
    if str(parsed.get("boundary_id") or "") != boundary["boundary_id"]:
        raise ValueError("boundary_id mismatch")
    expected = [row["candidate_id"] for row in boundary["candidates"]]
    rows = parsed.get("candidates")
    if not isinstance(rows, list) or [str(row.get("candidate_id")) for row in rows] != expected:
        raise ValueError("candidate ids must exactly match request order")
    for row in rows:
        if str(row.get("label")) not in LABELS:
            raise ValueError("invalid safe-zone label")
        confidence = float(row.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
    ranks = {"left_clipped": 0, "safe": 1, "right_clipped": 2}
    definite = [ranks[str(row["label"])] for row in rows if str(row["label"]) != "unsure"]
    if definite != sorted(definite):
        raise ValueError("definite labels must be monotonic left_clipped -> safe -> right_clipped")
    return rows


def _call_multi(*, audio_paths: list[Path], boundary_id: str, model: str, api_key: str, base_url: str, timeout_s: float, thinking_budget: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_s}
    if base_url:
        client_kwargs["base_url"] = base_url
    content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps({"boundary_id": boundary_id, "candidate_ids": [f"c{i:02d}" for i in range(len(audio_paths))]}, ensure_ascii=False)}]
    for index, path in enumerate(audio_paths):
        content.append({"type": "text", "text": f"candidate_id=c{index:02d}"})
        content.append(audio_content_part(path, fmt="wav", mode="input_audio"))
    kwargs: dict[str, Any] = {}
    if thinking_budget > 0:
        kwargs["extra_body"] = {"enable_thinking": True, "thinking_budget": thinking_budget}
    stream = OpenAI(**client_kwargs).chat.completions.create(model=model, temperature=0, max_tokens=768, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}], modalities=["text"], stream=True, stream_options={"include_usage": True}, **kwargs)
    parts = []
    chunks = []
    for chunk in stream:
        payload = chunk.model_dump(mode="json")
        chunks.append(payload)
        choices = getattr(chunk, "choices", None) or []
        if choices:
            parts.append(getattr(choices[0].delta, "content", None) or "")
    text = "".join(parts)
    return extract_json_object(text), {"content": text, "chunks": chunks, "multi_audio": True}


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "selected_boundaries.jsonl"
    if selected_path.exists() and args.resume_selection:
        selected = _read_jsonl(selected_path)
    else:
        selected = select_boundaries(Path(args.selected_islands), Path(args.semantic_labels), list(args.boundary_id or DEFAULT_BOUNDARIES), int(args.candidate_count))
        selected_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    for boundary in selected:
        island_audio = output / "request_audio" / f"{boundary['island_id'].replace('#', '__')}.wav"
        slice_audio_clip(source_audio=Path(boundary["source_audio"]), row={"start": boundary["span_start_s"], "end": boundary["span_end_s"]}, output_path=island_audio, fmt="wav", bitrate="256k", sample_rate=16000, force=False)
        boundary["original_audio"] = str(island_audio)
        for candidate in boundary["candidates"]:
            plan_audio = output / "request_audio" / boundary["boundary_id"].replace("#", "__") / f"{candidate['candidate_id']}.wav"
            if not plan_audio.exists():
                _build_candidate_audio(island_audio, plan_audio, float(boundary["duration_s"]), float(candidate["relative_time_s"]))
            candidate["audio"] = str(plan_audio)
    selected_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    if args.prepare_only:
        return {"selected_boundaries": len(selected), "candidate_count": sum(len(row["candidates"]) for row in selected)}

    _model_env, configured_model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or configured_model or DEFAULT_MODEL
    _key_env, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_env, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    labels_path = output / "safe_zone_labels.jsonl"
    raw_path = output / "omni_raw_responses.jsonl"
    existing = {str(row["boundary_id"]) for row in _read_jsonl(labels_path)}
    fallback_count = 0
    for boundary in selected:
        if boundary["boundary_id"] in existing:
            continue
        audio_paths = [Path(row["audio"]) for row in boundary["candidates"]]
        parsed: dict[str, Any]
        raw: dict[str, Any]
        try:
            if args.request_mode == "single":
                raise RuntimeError("single request mode selected")
            parsed = {}
            raw = {}
            validation_error: Exception | None = None
            for _attempt in range(1, int(args.max_attempts) + 1):
                parsed, raw = _call_multi(audio_paths=audio_paths, boundary_id=boundary["boundary_id"], model=model, api_key=api_key, base_url=base_url, timeout_s=float(args.timeout_s), thinking_budget=int(args.thinking_budget))
                try:
                    rows = _validate(parsed, boundary)
                    validation_error = None
                    break
                except ValueError as error:
                    validation_error = error
            if validation_error is not None:
                raise validation_error
        except ValueError:
            raise
        except Exception as multi_error:
            fallback_count += 1
            rows = []
            raw = {"multi_audio_error": repr(multi_error), "multi_audio": False, "responses": []}
            for candidate, audio_path in zip(boundary["candidates"], audio_paths):
                prompt = json.dumps({"boundary_id": boundary["boundary_id"], "candidate_ids": [candidate["candidate_id"]]}, ensure_ascii=False)
                one, one_raw = call_omni(audio_path=audio_path, fmt="wav", audio_content_mode="input_audio", model=model, api_key=api_key, base_url=base_url, timeout_s=float(args.timeout_s), store_stream_chunks=False, prompt=prompt, system_prompt=SYSTEM_PROMPT, max_tokens=256, enable_thinking=True, thinking_budget=int(args.thinking_budget))
                one_boundary = {**boundary, "candidates": [candidate]}
                rows.extend(_validate(one, one_boundary))
                raw["responses"].append(one_raw)
                if float(args.request_interval_s) > 0:
                    time.sleep(float(args.request_interval_s))
        label = {"schema": SCHEMA, "prompt_version": PROMPT_VERSION, "model": model, "boundary_id": boundary["boundary_id"], "island_id": boundary["island_id"], "candidates": rows}
        with labels_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(label, ensure_ascii=False) + "\n")
        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"boundary_id": boundary["boundary_id"], "response": raw}, ensure_ascii=False) + "\n")
    summary = {"schema": "inner_edge_safe_zone_teacher_summary_v1", "selected_boundaries": len(selected), "labeled_boundaries": len(_read_jsonl(labels_path)), "fallback_boundaries": fallback_count, "labels": str(labels_path)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-islands", required=True)
    parser.add_argument("--semantic-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boundary-id", action="append", default=[])
    parser.add_argument("--candidate-count", type=int, default=7)
    parser.add_argument("--model", default="")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--request-interval-s", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-mode", choices=("multi", "single"), default="multi")
    parser.add_argument("--resume-selection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
