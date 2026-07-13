#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID, qwen_asr_repo_tag  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.features import load_cached_feature  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    LEGACY_SPEECH_ISLAND_SCORER_SCHEMA,
    SpeechIslandScorerBundle,
    load_speech_island_scorer_checkpoint,
)
from tools.asr.cueqc.label_pre_asr_with_omni import (  # noqa: E402
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    audio_content_part,
    call_omni,
    extract_json_object,
    first_env_value,
    load_env_file,
)


SCHEMA = "semantic_source_candidate_teacher_v2"
SELECTION_SCHEMA = "semantic_source_learned_candidate_selection_v1"
PROMPT_VERSION = "semantic_source_hierarchical_candidate_marker_v2"
DEFAULT_MODEL = "qwen3.5-omni-plus"
LABELS = ("discardable", "semantic_target", "unsure")
SOURCE_LABELS = ("discardable", "contains_semantic", "unsure")
DEFAULT_SCORER = (
    PROJECT_ROOT
    / "src"
    / "checkpoints"
    / qwen_asr_repo_tag(QWEN_ASR_17B_REPO_ID)
    / f"speech_island_scorer_v8.{qwen_asr_repo_tag(QWEN_ASR_17B_REPO_ID)}.pt"
)

SYSTEM_PROMPT = """你是日语隔离 source utterance 的语义候选分类器。每次请求只处理一个短 source utterance 中的 5–9 个已给定声学候选；不要输出时间戳，不要重新划分区间。

用户会提供该 source utterance 的可信数据集参考文本。参考文本只用于区分有语言语义的词句与喘息、呻吟、亲吻声、笑声、非词叫声；不得根据文本猜测标记位置。候选位置只能由音频中的 1 秒静音标记确定。

每个候选音频只包含该候选的自适应声学邻域，并在精确候选帧插入 1 秒静音。只判断静音标记紧邻位置的声学状态，不要因为参考文本或邻域较远处存在词语就把标记处判为 semantic_target：

- semantic_target：静音标记切入或紧贴清楚可辨、具有语言语义、值得进入字幕的前景人声。短词、句尾 mora 和助词只要可辨也属于 semantic_target。
- discardable：静音标记处只有纯背景音乐、环境/机械噪声、纯喘息、呻吟、亲吻声、笑声、无意义叫声或短促非词 vocalization，或远处/嘈杂/不可辨且无字幕价值的背景人声。
- unsure：标记正处于类别转换、疑似包含词语但听不清、与 semantic_target 重叠而无法可靠分离，或无法确定是否有语义内容。

高召回不等于把纯 BGM 或非语言 vocalization 标成语音。只要可能有真实词语但证据不足，必须 unsure。不要转录、引用或改写具体台词；reason 只能描述声学类别与可辨识性证据。每个候选独立分类，不要求标签平滑或单调。

只输出 JSON：
{"sample_id":"...","candidates":[{"candidate_id":"c00","label":"discardable|semantic_target|unsure","confidence":0.0,"reason":"简短声学理由"}]}
"""

SOURCE_GATE_PROMPT = """你是日语隔离 source utterance 的语义存在性分类器。用户提供一条完整短音频和可信数据集参考文本。只判断整条 source 是否包含至少一个清楚可辨、具有语言语义、值得字幕化的词或短句；不要输出时间戳，不要划分区间。

- contains_semantic：至少包含一个清楚可辨的词、助词、应答词或短句；即使同时有喘息、呻吟或亲吻声也选此类。
- discardable：整条只有喘息、呻吟、亲吻声、笑声、无意义叫声、短促拟声或其他非词 vocalization，没有可辨词语。
- unsure：参考文本或音频疑似有词但无法可靠确认。

参考文本用于避免把非词 vocalization 幻听成台词，但最终必须同时尊重音频。不要转录，不要输出 Markdown。

只输出 JSON：
{"sample_id":"...","label":"discardable|contains_semantic|unsure","confidence":0.0,"reason":"简短理由"}
"""


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def learned_frame_embeddings(
    bundle: SpeechIslandScorerBundle,
    *,
    ptm: np.ndarray,
    mfcc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if bundle.schema != LEGACY_SPEECH_ISLAND_SCORER_SCHEMA:
        raise ValueError("candidate proposal requires the promoted legacy binary scorer")
    frame_total = min(int(ptm.shape[0]), int(mfcc.shape[0]))
    features = np.concatenate(
        [
            np.asarray(ptm[:frame_total, : bundle.ptm_dim], dtype=np.float32),
            np.asarray(mfcc[:frame_total], dtype=np.float32),
        ],
        axis=1,
    )
    mean = np.asarray(bundle.normalization["feature_mean"], dtype=np.float32)
    std = np.maximum(
        np.asarray(bundle.normalization["feature_std"], dtype=np.float32), 1e-6
    )
    normalized = np.ascontiguousarray((features - mean) / std, dtype=np.float32)
    with torch.inference_mode():
        values = torch.from_numpy(normalized).unsqueeze(0).to(bundle.device)
        mask = torch.ones((1, frame_total), dtype=torch.long, device=bundle.device)
        hidden = bundle.model.backbone(
            bundle.model.proj(values), attention_mask=mask
        )
        hidden = bundle.model.norm(hidden)[0]
        probabilities = torch.sigmoid(bundle.model.head(hidden)[:, 0])
    return (
        np.ascontiguousarray(hidden.detach().cpu().numpy(), dtype=np.float32),
        np.ascontiguousarray(probabilities.detach().cpu().numpy(), dtype=np.float32),
    )


def select_candidate_frames(embeddings: np.ndarray, candidate_count: int) -> list[int]:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < candidate_count:
        raise ValueError("not enough learned frame embeddings for candidate selection")
    if not 5 <= candidate_count <= 9:
        raise ValueError("candidate_count must be between 5 and 9")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    semantic = values / np.maximum(norms, 1e-6)
    timeline = (
        ((np.arange(values.shape[0], dtype=np.float32) + 0.5) / values.shape[0])
        * 2.0
        - 1.0
    )[:, None]
    descriptors = np.concatenate([semantic, timeline], axis=1)
    center = descriptors.mean(axis=0, keepdims=True)
    first = int(np.argmin(np.sum((descriptors - center) ** 2, axis=1)))
    selected = [first]
    min_distance = np.sum((descriptors - descriptors[first]) ** 2, axis=1)
    while len(selected) < candidate_count:
        index = int(np.argmax(min_distance))
        selected.append(index)
        distance = np.sum((descriptors - descriptors[index]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -1.0
    return sorted(selected)


def candidate_cells(
    frame_indexes: list[int],
    *,
    frame_hop_s: float,
    duration_s: float,
) -> list[dict[str, Any]]:
    markers = [
        min(float(duration_s), (int(index) + 0.5) * float(frame_hop_s))
        for index in frame_indexes
    ]
    cells: list[dict[str, Any]] = []
    for position, (frame_index, marker_s) in enumerate(
        zip(frame_indexes, markers, strict=True)
    ):
        start_s = 0.0 if position == 0 else (markers[position - 1] + marker_s) / 2.0
        end_s = (
            float(duration_s)
            if position + 1 == len(markers)
            else (marker_s + markers[position + 1]) / 2.0
        )
        cells.append(
            {
                "candidate_id": f"c{position:02d}",
                "feature_index": int(frame_index),
                "marker_s": float(marker_s),
                "context_start_s": float(start_s),
                "context_end_s": float(end_s),
            }
        )
    return cells


def select_candidates(
    *,
    samples: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    scorer: SpeechIslandScorerBundle,
    candidate_count: int,
) -> list[dict[str, Any]]:
    by_audio = {str(row["audio_id"]): row for row in feature_rows}
    selected: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        feature_row = by_audio[sample_id]
        ptm, mfcc = load_cached_feature(Path(str(feature_row["feature_path"])))
        embeddings, old_probabilities = learned_frame_embeddings(
            scorer, ptm=ptm, mfcc=mfcc
        )
        frame_hop_s = float(feature_row.get("frame_hop_s") or 0.02)
        frame_indexes = select_candidate_frames(embeddings, candidate_count)
        cells = candidate_cells(
            frame_indexes,
            frame_hop_s=frame_hop_s,
            duration_s=float(sample["duration_s"]),
        )
        for cell in cells:
            cell["proposal_speech_probability"] = float(
                old_probabilities[cell["feature_index"]]
            )
        selected.append(
            {
                "schema": SELECTION_SCHEMA,
                "sample_id": sample_id,
                "audio": str(sample["audio"]),
                "duration_s": float(sample["duration_s"]),
                "source": str(sample.get("source") or ""),
                "audit_focus": str(sample.get("audit_focus") or ""),
                "reference_text": str(sample["reference_text"]),
                "feature_path": str(feature_row["feature_path"]),
                "frame_hop_s": frame_hop_s,
                "proposal_scorer_sha256": scorer.sha256,
                "selection_mode": "learned_hidden_farthest_medoid_v1",
                "candidates": cells,
            }
        )
    return selected


def _build_candidate_audio(
    *,
    source: Path,
    original_output: Path,
    marked_output: Path,
    start_s: float,
    marker_s: float,
    end_s: float,
) -> None:
    original_output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-af",
            f"atrim=start={start_s:.6f}:end={end_s:.6f},asetpts=PTS-STARTPTS",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(original_output),
        ],
        check=True,
    )
    filters = (
        f"[0:a]atrim=start={start_s:.6f}:end={marker_s:.6f},asetpts=PTS-STARTPTS[left];"
        "anullsrc=r=16000:cl=mono:d=1[silence];"
        f"[0:a]atrim=start={marker_s:.6f}:end={end_s:.6f},asetpts=PTS-STARTPTS[right];"
        "[left][silence][right]concat=n=3:v=0:a=1[out]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(marked_output),
        ],
        check=True,
    )


def materialize_candidate_audio(selected: list[dict[str, Any]], output: Path) -> None:
    for sample in selected:
        source = Path(sample["audio"])
        sample_dir = output / "request_audio" / str(sample["sample_id"])
        for candidate in sample["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            original = sample_dir / f"{candidate_id}__original.wav"
            marked = sample_dir / f"{candidate_id}__marked.wav"
            if not original.exists() or not marked.exists():
                _build_candidate_audio(
                    source=source,
                    original_output=original,
                    marked_output=marked,
                    start_s=float(candidate["context_start_s"]),
                    marker_s=float(candidate["marker_s"]),
                    end_s=float(candidate["context_end_s"]),
                )
            candidate["original_audio"] = str(original)
            candidate["marked_audio"] = str(marked)


def _validate(parsed: dict[str, Any], sample: dict[str, Any]) -> list[dict[str, Any]]:
    if str(parsed.get("sample_id") or "") != str(sample["sample_id"]):
        raise ValueError("sample_id mismatch")
    expected = [str(row["candidate_id"]) for row in sample["candidates"]]
    rows = parsed.get("candidates")
    if not isinstance(rows, list) or [str(row.get("candidate_id")) for row in rows] != expected:
        raise ValueError("candidate ids must exactly match request order")
    validated: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("label") or "")
        confidence = float(row.get("confidence"))
        reason = str(row.get("reason") or "")
        if label not in LABELS:
            raise ValueError("invalid semantic speech candidate label")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        validated.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "label": label,
                "confidence": confidence,
                "reason": reason,
            }
        )
    return validated


def _validate_source_gate(
    parsed: dict[str, Any], sample: dict[str, Any]
) -> dict[str, Any]:
    if str(parsed.get("sample_id") or "") != str(sample["sample_id"]):
        raise ValueError("source gate sample_id mismatch")
    label = str(parsed.get("label") or "")
    confidence = float(parsed.get("confidence"))
    if label not in SOURCE_LABELS:
        raise ValueError("invalid semantic source gate label")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("source gate confidence must be in [0, 1]")
    return {
        "sample_id": str(sample["sample_id"]),
        "label": label,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or ""),
    }


def _call_source_gate(
    *,
    sample: dict[str, Any],
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: float,
    thinking_budget: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return call_omni(
        audio_path=Path(sample["audio"]),
        fmt=Path(sample["audio"]).suffix.lstrip(".") or "wav",
        audio_content_mode="input_audio",
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout_s=timeout_s,
        store_stream_chunks=False,
        prompt=json.dumps(
            {
                "sample_id": sample["sample_id"],
                "reference_text": sample["reference_text"],
            },
            ensure_ascii=False,
        ),
        system_prompt=SOURCE_GATE_PROMPT,
        max_tokens=256,
        enable_thinking=True,
        thinking_budget=thinking_budget,
    )


def _call_multi(
    *,
    sample: dict[str, Any],
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: float,
    thinking_budget: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_s}
    if base_url:
        client_kwargs["base_url"] = base_url
    candidate_ids = [str(row["candidate_id"]) for row in sample["candidates"]]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "sample_id": sample["sample_id"],
                    "candidate_ids": candidate_ids,
                    "reference_text": sample["reference_text"],
                },
                ensure_ascii=False,
            ),
        }
    ]
    for candidate in sample["candidates"]:
        content.append(
            {"type": "text", "text": f"candidate_id={candidate['candidate_id']}"}
        )
        content.append(
            audio_content_part(
                Path(candidate["marked_audio"]), fmt="wav", mode="input_audio"
            )
        )
    kwargs: dict[str, Any] = {}
    if thinking_budget > 0:
        kwargs["extra_body"] = {
            "enable_thinking": True,
            "thinking_budget": thinking_budget,
        }
    stream = OpenAI(**client_kwargs).chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
        **kwargs,
    )
    parts: list[str] = []
    chunks: list[dict[str, Any]] = []
    for chunk in stream:
        chunks.append(chunk.model_dump(mode="json"))
        choices = getattr(chunk, "choices", None) or []
        if choices:
            parts.append(getattr(choices[0].delta, "content", None) or "")
    response_text = "".join(parts)
    return extract_json_object(response_text), {
        "content": response_text,
        "chunks": chunks,
        "multi_audio": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "selected_candidates.jsonl"
    if selected_path.exists() and args.resume_selection:
        selected = _rows(selected_path)
    else:
        apply_vram_safety_cap(0.95)
        scorer = load_speech_island_scorer_checkpoint(
            args.proposal_scorer, device=args.proposal_device
        )
        selected = select_candidates(
            samples=_rows(Path(args.samples)),
            feature_rows=_rows(Path(args.feature_manifest)),
            scorer=scorer,
            candidate_count=int(args.candidate_count),
        )
    materialize_candidate_audio(selected, output)
    selected_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    if args.prepare_only:
        return {
            "selected_samples": len(selected),
            "candidate_count": sum(len(row["candidates"]) for row in selected),
            "selection": str(selected_path),
        }

    _model_env, configured_model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or configured_model or DEFAULT_MODEL
    _key_env, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_env, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    if not api_key:
        raise RuntimeError("Omni API key is required")
    source_gate_path = output / "source_gate_labels.jsonl"
    source_gate_raw_path = output / "source_gate_raw_responses.jsonl"
    source_gates = {
        str(row["sample_id"]): row for row in _rows(source_gate_path)
    }
    for sample in selected:
        sample_id = str(sample["sample_id"])
        if sample_id in source_gates:
            continue
        validation_error: Exception | None = None
        for attempt in range(1, int(args.max_attempts) + 1):
            parsed, raw = _call_source_gate(
                sample=sample,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=float(args.timeout_s),
                thinking_budget=int(args.thinking_budget),
            )
            try:
                gate = _validate_source_gate(parsed, sample)
                validation_error = None
                break
            except ValueError as error:
                validation_error = error
            finally:
                with source_gate_raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "sample_id": sample_id,
                                "attempt": attempt,
                                "response": raw,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        if validation_error is not None:
            raise validation_error
        with source_gate_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(gate, ensure_ascii=False) + "\n")
        source_gates[sample_id] = gate
    labels_path = output / "candidate_labels.jsonl"
    raw_path = output / "omni_raw_responses.jsonl"
    existing = {str(row["sample_id"]) for row in _rows(labels_path)}
    fallback_samples = 0
    for sample in selected:
        sample_id = str(sample["sample_id"])
        if sample_id in existing:
            continue
        source_gate = source_gates[sample_id]
        if source_gate["label"] != "contains_semantic":
            propagated_label = (
                "discardable"
                if source_gate["label"] == "discardable"
                else "unsure"
            )
            classified = [
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "label": propagated_label,
                    "confidence": float(source_gate["confidence"]),
                    "reason": "source gate: " + str(source_gate["reason"]),
                    "label_source": "source_gate_propagated",
                }
                for candidate in sample["candidates"]
            ]
            raw = {
                "multi_audio": False,
                "candidate_request_skipped": True,
                "source_gate": source_gate,
            }
        else:
            try:
                if args.request_mode == "single":
                    raise RuntimeError("single request mode selected")
                validation_error = None
                for _attempt in range(int(args.max_attempts)):
                    parsed, raw = _call_multi(
                        sample=sample,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                        timeout_s=float(args.timeout_s),
                        thinking_budget=int(args.thinking_budget),
                    )
                    try:
                        classified = _validate(parsed, sample)
                        validation_error = None
                        break
                    except ValueError as error:
                        validation_error = error
                if validation_error is not None:
                    raise validation_error
            except ValueError:
                raise
            except Exception as multi_error:
                fallback_samples += 1
                classified = []
                raw = {
                    "multi_audio": False,
                    "multi_audio_error": repr(multi_error),
                    "responses": [],
                }
                for candidate in sample["candidates"]:
                    prompt = json.dumps(
                        {
                            "sample_id": sample["sample_id"],
                            "candidate_ids": [candidate["candidate_id"]],
                            "reference_text": sample["reference_text"],
                        },
                        ensure_ascii=False,
                    )
                    parsed, one_raw = call_omni(
                        audio_path=Path(candidate["marked_audio"]),
                        fmt="wav",
                        audio_content_mode="input_audio",
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                        timeout_s=float(args.timeout_s),
                        store_stream_chunks=False,
                        prompt=prompt,
                        system_prompt=SYSTEM_PROMPT,
                        max_tokens=256,
                        enable_thinking=True,
                        thinking_budget=int(args.thinking_budget),
                    )
                    classified.extend(
                        _validate(parsed, {**sample, "candidates": [candidate]})
                    )
                    raw["responses"].append(one_raw)
                    if float(args.request_interval_s) > 0:
                        time.sleep(float(args.request_interval_s))
            classified = [
                {**row, "label_source": "candidate_marker"} for row in classified
            ]
        by_id = {str(row["candidate_id"]): row for row in classified}
        label = {
            "schema": SCHEMA,
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "sample_id": sample["sample_id"],
            "audio": sample["audio"],
            "duration_s": sample["duration_s"],
            "source": sample["source"],
            "audit_focus": sample["audit_focus"],
            "reference_text": sample["reference_text"],
            "source_gate": source_gate,
            "selection_mode": sample["selection_mode"],
            "proposal_scorer_sha256": sample["proposal_scorer_sha256"],
            "candidates": [
                {**candidate, **by_id[str(candidate["candidate_id"])]}
                for candidate in sample["candidates"]
            ],
        }
        with labels_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(label, ensure_ascii=False) + "\n")
        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"sample_id": sample["sample_id"], "response": raw},
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "schema": "semantic_source_candidate_teacher_summary_v2",
        "selected_samples": len(selected),
        "source_gate_counts": {
            label: sum(1 for row in source_gates.values() if row["label"] == label)
            for label in SOURCE_LABELS
        },
        "labeled_samples": len(_rows(labels_path)),
        "candidate_count": sum(len(row["candidates"]) for row in selected),
        "fallback_samples": fallback_samples,
        "labels": str(labels_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify isolated semantic source candidates with Omni."
    )
    parser.add_argument("--samples", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--proposal-scorer", default=str(DEFAULT_SCORER))
    parser.add_argument("--proposal-device", default="cuda")
    parser.add_argument("--candidate-count", type=int, default=9)
    parser.add_argument("--model", default="")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--request-interval-s", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-mode", choices=("multi", "single"), default="multi")
    parser.add_argument(
        "--resume-selection", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
