#!/usr/bin/env python3
"""Create Scorer v12 single-pass tri-state vocal-envelope preaudit labels.

One Gemini call labels vocal-envelope, definite non-vocal, and unsure regions
together.  OpenRouter and Google AI Studio use isolated transports.  The result
remains review-only evidence until a separate canonical compiler binds a
complete human audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.ja.vocal_envelope_v12 import (  # noqa: E402
    VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
    VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
)
from tools.omni.timestamp_contract import (  # noqa: E402
    TIMESTAMP_CONTRACT_ID,
    TIMESTAMP_PROMPT_CONTRACT_ZH,
    parse_mmss_span,
    parse_mmss_timestamp,
    timestamp_request_contract,
)
from tools.omni.audio_teacher_transport import (  # noqa: E402
    OPENROUTER_GEMINI_EXECUTION_CONTRACT,
    AudioTeacherTransport,
    create_audio_teacher_transport,
)
from tools.omni.audio_teacher_batch import (  # noqa: E402
    iter_completed_audio_teacher_items,
    resolve_worker_count,
)
from tools.omni.gemini_native import (  # noqa: E402
    GEMINI_NATIVE_EXECUTION_CONTRACT,
    GEMINI_NATIVE_MODEL,
)
from tools.boundary.ja.vocal_envelope_scorer_v12_calibration import (  # noqa: E402
    CALIBRATION_ARTIFACT_SHA256,
    evidence_span_signature,
    load_approved_calibration,
)


FRAME_HOP_S = 0.02
CONTRACT_ID = "boundary_acoustic_binary_v12"
SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_single_pass_tristate_summary_v2"
PROGRESS_SCHEMA = "vocal_envelope_scorer_v12_single_pass_tristate_progress_v2"
PROMPT_PROFILE = "vocal-envelope-single-pass-tristate-v3"
PROMPT_VERSION = "vocal-envelope-single-pass-tristate-v3-scorer-duty-gemini36-medium-mmss"
EXPECTED_REASONING = "medium"
EXPECTED_MAX_TOKENS = 8192
PROVIDER_CONTRACTS: dict[str, dict[str, str]] = {
    "openrouter": {
        "model": "google/gemini-3.6-flash",
        "execution_contract": OPENROUTER_GEMINI_EXECUTION_CONTRACT,
        "transport": "openai_compatible_input_audio",
    },
    "gemini": {
        "model": GEMINI_NATIVE_MODEL,
        "execution_contract": GEMINI_NATIVE_EXECUTION_CONTRACT,
        "transport": "google_ai_interactions_inline_audio",
    },
}


TRISTATE_SYSTEM_PROMPT = """你是 1.7B Scorer v12 的 Human Vocal Event Envelope 单次三态预审 teacher。
音频主要来自日语 JAV、Galgame 或类似场景，但场景、亲密互动、声音强度和声音来源本身都不能直接决定标签。

【Scorer 在真实工作流中的职责】
Scorer 是任何下游模型运行前的第一层，只负责生成高召回、连续的人类发声事件候选包络。它不判断语义、不判断字幕价值、不转写、不切句、不区分说话人，也不负责删除呻吟或喘息。后续 Proposal/Split 负责隔离独立事件，CueQC 负责语义 keep/drop，Inner 只裁最终 keep 岛首尾。因此不要把 Split、CueQC 或 Inner 的职责提前强加给 Scorer。

你必须对当前完整 source 输出一个按时间排序、无重叠、无缺口并覆盖完整音频的 segments 数组。每段只能使用以下三种标签之一：

1. vocal_candidate：存在任何由人类声道、口腔或呼吸系统产生的发声证据，或非发声声音与疑似人声重叠。包括清晰或含混对白、耳语、气声、呻吟、喘息、吸气、呼气、叹气、哭、笑、咳嗽、喷嚏、清嗓、抽鼻、亲吻/唾液/口腔声、歌唱、远处或背景人声。像「あ、ん、はぁ」无论是词语、应答还是纯呻吟都属于 vocal。ASR 是否能识别、是否有翻译价值均无关。

2. non_vocal_candidate：有较高把握确认完全没有任何人类发声的纯非发声区间。包括纯机械、肉体撞击或拍打、动作声、衣物摩擦、床体震动、水声、纯器乐、静音、底噪、风扇空调、电流、交通和环境噪声。肉体撞击由人体动作产生也不等于人声，不能以“来自人体”为理由标成 vocal；但只要撞击、床体、水声或音乐下叠有任何疑似呻吟、呼吸或对白，重叠区就不能标 non_vocal。

3. unsure：无法可靠确认是否叠有人声、边界无法安全定位，或 vocal 与 non-vocal 证据不可分离。不要猜测；unsure 在训练中会被忽略。

【发声事件包络规则】
正类单位是连续的人类发声事件包络，不是逐音节 VAD。一次连续发声事件内部的短停顿、吸气、释气、弱尾音和非语义过渡应随包络保留；不要按字、音节、每次喘息脉冲或极短能量谷切碎。纯撞击声本身不是 vocal 证据：若它形成声学上独立且可安全分离的纯非发声区间，标 non_vocal；若它与人声重叠，标 vocal；若只是短暂嵌入且无法安全独立切开，标 unsure。不要跨越声学上独立的长纯非发声区域合并两个事件，也不得使用固定时长阈值。

【边界与完整覆盖】
- 第一段必须从 00:00.000 开始，最后一段必须精确结束于请求给出的 duration_ts。
- 相邻段必须首尾严格相接；不得重叠或留空白；相邻同标签必须合并。
- 覆盖完整词头、气声、衰减和尾音。混合区优先保护 vocal，真正无法判断才用 unsure。
- 不要产生短到无法对应一个 20ms Scorer 帧的区间；这是坐标分辨率要求，不是声音类别的时长规则。

只输出 JSON，不要输出 Markdown、解释或额外文字：
{"source_id":"...","segments":[{"start_ts":"00:00.000","end_ts":"00:01.000","label":"vocal_candidate|non_vocal_candidate|unsure","category":"vocal|mixed_vocal|mechanical|impact|cloth|bed|water|music|silence|ambience|other|uncertain","reason":"简短声学理由"}],"overall_reason":"..."}
""" + "\n" + TIMESTAMP_PROMPT_CONTRACT_ZH

NONVOCAL_CATEGORIES = frozenset({"mechanical", "impact", "cloth", "bed", "water", "music", "silence", "ambience", "other"})
SEGMENT_LABELS = frozenset(
    {"vocal_candidate", "non_vocal_candidate", "unsure"}
)

TRISTATE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "source_id": {"type": "string"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_ts": {"type": "string"},
                    "end_ts": {"type": "string"},
                    "label": {
                        "type": "string",
                        "enum": [
                            "vocal_candidate",
                            "non_vocal_candidate",
                            "unsure",
                        ],
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "vocal",
                            "mixed_vocal",
                            "mechanical",
                            "impact",
                            "cloth",
                            "bed",
                            "water",
                            "music",
                            "silence",
                            "ambience",
                            "other",
                            "uncertain",
                        ],
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "start_ts",
                    "end_ts",
                    "label",
                    "category",
                    "reason",
                ],
            },
        },
        "overall_reason": {"type": "string"},
    },
    "required": ["source_id", "segments", "overall_reason"],
}


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json(path, payload)


def _resolve_audio(row: Mapping[str, Any], *, manifest: Path) -> Path:
    value = str(row.get("audio") or row.get("audio_path") or "")
    if not value:
        raise ValueError(f"source {row.get('source_id')} has no audio path")
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [manifest.parent / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(candidates[0])


def _frame_count(row: Mapping[str, Any], duration_s: float) -> int:
    frame_count = int(row.get("frame_count") or round(duration_s / FRAME_HOP_S))
    if frame_count <= 0:
        raise ValueError(f"invalid frame_count for {row.get('source_id')}")
    if abs(frame_count * FRAME_HOP_S - duration_s) > FRAME_HOP_S:
        raise ValueError(f"duration/frame geometry mismatch for {row.get('source_id')}")
    return frame_count


def _boundary_frame(
    *, left_label: str, right_label: str, boundary_s: float, frame_count: int
) -> int:
    scaled = boundary_s / FRAME_HOP_S
    if left_label == "vocal_candidate":
        value = math.ceil(scaled - 1e-9)
    elif right_label == "vocal_candidate":
        value = math.floor(scaled + 1e-9)
    elif left_label == "non_vocal_candidate":
        value = math.floor(scaled + 1e-9)
    elif right_label == "non_vocal_candidate":
        value = math.ceil(scaled - 1e-9)
    else:
        value = round(scaled)
    return max(0, min(frame_count, int(value)))


def _normalize_segments(
    parsed: Mapping[str, Any], *, duration_s: float, frame_count: int
) -> dict[str, list[dict[str, Any]]]:
    raw = parsed.get("segments")
    if not isinstance(raw, list) or not raw:
        raise ValueError("teacher response segments must be a non-empty array")
    advertised = timestamp_request_contract(duration_s)["duration_ts"]
    advertised_end = parse_mmss_timestamp(
        advertised,
        field="duration_ts",
        duration_s=duration_s,
    )
    parsed_segments: list[dict[str, Any]] = []
    previous_end = 0.0
    previous_label = ""
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"segments[{index}] must be an object")
        label = str(item.get("label") or "")
        if label not in SEGMENT_LABELS:
            raise ValueError(f"segments[{index}] has unsupported label: {label!r}")
        start, end = parse_mmss_span(
            item,
            field=f"segments[{index}]",
            duration_s=duration_s,
        )
        assert start is not None and end is not None
        if index == 0 and abs(start) > 0.0005:
            raise ValueError("single-pass segments must start at 00:00.000")
        if index > 0 and abs(start - previous_end) > 0.0005:
            raise ValueError("single-pass segments must be contiguous without gaps")
        if label == previous_label:
            raise ValueError("adjacent single-pass segments with the same label must merge")
        category = str(item.get("category") or "other")
        if label == "non_vocal_candidate" and category not in NONVOCAL_CATEGORIES:
            raise ValueError(f"unsupported non-vocal category: {category}")
        parsed_segments.append(
            {
                "label": label,
                "start_wire_s": start,
                "end_wire_s": end,
                "category": category,
                "reason": str(item.get("reason") or ""),
            }
        )
        previous_end = end
        previous_label = label
    if abs(previous_end - advertised_end) > 0.0005:
        raise ValueError(
            "single-pass segments must end exactly at advertised duration_ts"
        )

    boundaries = [0]
    for index in range(1, len(parsed_segments)):
        boundaries.append(
            _boundary_frame(
                left_label=str(parsed_segments[index - 1]["label"]),
                right_label=str(parsed_segments[index]["label"]),
                boundary_s=float(parsed_segments[index]["start_wire_s"]),
                frame_count=frame_count,
            )
        )
    boundaries.append(frame_count)
    output: dict[str, list[dict[str, Any]]] = {
        "vocal_spans": [],
        "non_vocal_spans": [],
        "unsure_spans": [],
        "conflict_spans": [],
    }
    output_key = {
        "vocal_candidate": "vocal_spans",
        "non_vocal_candidate": "non_vocal_spans",
        "unsure": "unsure_spans",
    }
    for index, item in enumerate(parsed_segments):
        start_frame, end_frame = boundaries[index], boundaries[index + 1]
        if end_frame <= start_frame:
            raise ValueError(
                f"segments[{index}] is shorter than the 20ms frame resolution"
            )
        span = {
            "label": item["label"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_s": round(start_frame * FRAME_HOP_S, 6),
            "end_s": round(end_frame * FRAME_HOP_S, 6),
            "reason": item["reason"],
        }
        if item["label"] == "non_vocal_candidate":
            span["category"] = item["category"]
        output[output_key[str(item["label"])]].append(span)
    return output


def _request_prompt(row: Mapping[str, Any], *, feedback: str = "") -> str:
    payload = {
        "source_id": str(row["source_id"]),
        **timestamp_request_contract(float(row["duration_s"])),
        "task": "vocal_event_envelope_single_pass_tristate",
        "labels": ["vocal_candidate", "non_vocal_candidate", "unsure"],
        "coverage": "complete_contiguous_source_timeline",
    }
    if feedback:
        payload["previous_validation_error"] = feedback
    return json.dumps(payload, ensure_ascii=False)


def _validate_manifest(rows: list[dict[str, Any]], *, manifest: Path) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("v12 source manifest is empty")
    source_ids: set[str] = set()
    core_ids: set[str] = set()
    partitions: dict[str, str] = {}
    video_partitions: dict[str, str] = {}
    result: list[dict[str, Any]] = []
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in source_ids:
            raise ValueError(f"duplicate/empty v12 source_id: {source_id!r}")
        partition = str(row.get("partition") or "")
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"v12 source has invalid partition: {source_id}")
        video_id = str(row.get("video_id") or "")
        if not video_id:
            raise ValueError(f"v12 source has no frozen video_id: {source_id}")
        previous_video_partition = video_partitions.setdefault(video_id, partition)
        if previous_video_partition != partition:
            raise ValueError(f"v12 video crosses partitions: {video_id}")
        core_values = row.get("core_ids") or row.get("core_id") or []
        if isinstance(core_values, str):
            core_values = [core_values]
        cores = [str(value) for value in core_values if str(value)]
        if len(cores) != 1 or cores[0] in core_ids:
            raise ValueError(f"v12 core must be present once and unique: {source_id}")
        audio = _resolve_audio(row, manifest=manifest)
        duration = float(row.get("duration_s") or 0.0)
        if duration <= 0:
            raise ValueError(f"v12 source duration is invalid: {source_id}")
        if row.get("audio_sha256") and _sha256(audio) != str(row["audio_sha256"]):
            raise ValueError(f"v12 source audio SHA mismatch: {source_id}")
        copied = dict(row)
        copied.update({"source_id": source_id, "video_id": video_id, "partition": partition, "core_ids": cores, "audio": str(audio), "audio_sha256": _sha256(audio), "duration_s": duration, "frame_count": _frame_count(row, duration)})
        source_ids.add(source_id)
        core_ids.add(cores[0])
        partitions[source_id] = partition
        result.append(copied)
    if not {"train", "val", "test"}.issubset(set(partitions.values())):
        raise ValueError("v12 frozen manifest must contain train/val/test partitions")
    return result


def _call_teacher(
    *,
    row: Mapping[str, Any],
    transport: AudioTeacherTransport,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    audio = Path(str(row["audio"]))
    last_error: Exception | None = None
    for attempt in range(1, int(args.max_attempts) + 1):
        try:
            prompt = _request_prompt(
                row,
                feedback="" if last_error is None else str(last_error),
            )
            response = transport.call_json(
                audio_path=audio,
                prompt=prompt,
                system_prompt=TRISTATE_SYSTEM_PROMPT,
                max_tokens=EXPECTED_MAX_TOKENS,
                enable_thinking=True,
                thinking_level=EXPECTED_REASONING,
                thinking_budget=0,
                response_schema=TRISTATE_RESPONSE_SCHEMA,
                store_stream_chunks=False,
                require_provider_parameters=True,
            )
            parsed, raw = response.parsed, response.raw
            if str(parsed.get("source_id") or "") != str(row["source_id"]):
                raise ValueError("teacher source_id mismatch")
            normalized = _normalize_segments(
                parsed,
                duration_s=float(row["duration_s"]),
                frame_count=int(row["frame_count"]),
            )
            return dict(parsed), raw, normalized
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < int(args.max_attempts):
                time.sleep(min(8.0, float(attempt)))
    raise RuntimeError(
        f"v12 single-pass teacher failed for {row['source_id']}: {last_error}"
    ) from last_error


def run(args: argparse.Namespace) -> dict[str, Any]:
    profile = str(args.env_file)
    if profile not in PROVIDER_CONTRACTS:
        raise ValueError("Scorer v12 teacher only supports openrouter or gemini")
    calibration_paths = (
        args.calibration_manifest,
        args.calibration_preaudit,
        args.calibration_verdicts,
    )
    if any(calibration_paths) and not all(calibration_paths):
        raise ValueError(
            "v12 calibration seed requires manifest, preaudit and verdicts together"
        )
    if all(calibration_paths) and profile != "gemini":
        raise ValueError("the fixed v12 pilot calibration uses the gemini profile")
    provider_contract = PROVIDER_CONTRACTS[profile]
    manifest = Path(args.manifest).expanduser().resolve()
    manifest_sha = _sha256(manifest)
    rows = _validate_manifest(_rows(manifest), manifest=manifest)
    full_row_index = {str(row["source_id"]): row for row in rows}
    calibration: dict[str, Any] | None = None
    if all(calibration_paths):
        calibration = load_approved_calibration(
            manifest=Path(args.calibration_manifest),
            preaudit=Path(args.calibration_preaudit),
            verdicts=Path(args.calibration_verdicts),
            expected_hashes=CALIBRATION_ARTIFACT_SHA256,
        )
        missing_calibration_sources = set(calibration["sources"]) - set(full_row_index)
        if missing_calibration_sources:
            raise ValueError(
                "v12 target manifest omits calibrated pilot sources: "
                f"{sorted(missing_calibration_sources)}"
            )
    if args.source_id:
        wanted = list(dict.fromkeys(args.source_id))
        index = {row["source_id"]: row for row in rows}
        missing = [value for value in wanted if value not in index]
        if missing:
            raise ValueError(f"manifest missing source ids: {missing}")
        rows = [index[value] for value in wanted]
    if args.limit > 0:
        rows = rows[: int(args.limit)]
    transport = create_audio_teacher_transport(
        profile=profile,
        env_file=(Path.home() / ".config" / "omni" / profile).resolve(),
        model_override=str(args.model or ""),
        timeout_s=float(args.timeout_s),
        log=lambda message: print(message, flush=True),
    )
    model = transport.model
    expected_model = provider_contract["model"]
    if model != expected_model:
        raise ValueError(
            f"Scorer v12 {profile} profile requires {expected_model}, got {model}"
        )
    execution_contract = provider_contract["execution_contract"]
    if transport.execution_contract != execution_contract:
        raise ValueError("Scorer v12 provider execution contract mismatch")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / "preaudit.jsonl"
    raw_path = output / "raw_responses.jsonl"
    progress_path = output / "progress.json"
    existing: dict[str, dict[str, Any]] = {}
    if labels_path.is_file():
        selected_ids = {str(row["source_id"]) for row in rows}
        selected_index = {str(row["source_id"]): row for row in rows}
        for saved in _rows(labels_path):
            source_id = str(saved.get("source_id") or "")
            if not source_id or source_id in existing or source_id not in selected_ids:
                raise ValueError(f"invalid v12 resume preaudit source: {source_id!r}")
            current = selected_index[source_id]
            checks = {
                "schema": VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "model": model,
                "provider_profile": profile,
                "env_file_name": profile,
                "reasoning_effort": EXPECTED_REASONING,
                "max_tokens": EXPECTED_MAX_TOKENS,
                "prompt_version": PROMPT_VERSION,
                "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
                "teacher_execution_contract_id": execution_contract,
                "source_manifest_sha256": manifest_sha,
                "partition": current["partition"],
                "audio_sha256": current["audio_sha256"],
                "frame_count": current["frame_count"],
                "sample_count": current.get("sample_count"),
            }
            for field, expected in checks.items():
                if saved.get(field) != expected:
                    raise ValueError(f"v12 resume preaudit {field} mismatch: {source_id}")
            if list(saved.get("core_ids") or ()) != list(current["core_ids"]):
                raise ValueError(f"v12 resume preaudit core mismatch: {source_id}")
            existing[source_id] = saved
    calibration_seed_ids: set[str] = set()
    if calibration is not None:
        selected_index = {str(row["source_id"]): row for row in rows}
        calibration_seed_ids = set(selected_index) & set(calibration["sources"])
        for source_id in sorted(calibration_seed_ids):
            current = selected_index[source_id]
            calibration_source = calibration["sources"][source_id]
            for field in (
                "video_id",
                "partition",
                "audio_sha256",
                "duration_s",
                "frame_count",
                "sample_rate",
                "sample_count",
            ):
                if current.get(field) != calibration_source.get(field):
                    raise ValueError(
                        f"v12 calibration seed source {field} drift: {source_id}"
                    )
            if list(current.get("core_ids") or ()) != list(
                calibration_source.get("core_ids") or ()
            ):
                raise ValueError(f"v12 calibration seed core drift: {source_id}")
            if source_id in existing:
                saved = existing[source_id]
                if saved.get("calibration_id") != calibration["calibration_id"]:
                    raise ValueError(
                        f"v12 calibration resume row lacks seed provenance: {source_id}"
                    )
                for field, expected in (
                    (
                        "calibration_manifest_sha256",
                        calibration["hashes"]["manifest"],
                    ),
                    (
                        "calibration_preaudit_sha256",
                        calibration["hashes"]["preaudit"],
                    ),
                    (
                        "calibration_verdicts_sha256",
                        calibration["hashes"]["verdicts"],
                    ),
                ):
                    if saved.get(field) != expected:
                        raise ValueError(
                            f"v12 calibration resume {field} mismatch: {source_id}"
                        )
                if evidence_span_signature(
                    saved,
                    frame_count=int(current["frame_count"]),
                    source_id=source_id,
                ) != calibration["signatures"][source_id]:
                    raise ValueError(
                        f"v12 calibration resume span drift: {source_id}"
                    )
                continue
            seeded = dict(calibration["evidence"][source_id])
            seeded.update(
                {
                    "video_id": current["video_id"],
                    "partition": current["partition"],
                    "core_ids": list(current["core_ids"]),
                    "audio": current["audio"],
                    "audio_sha256": current["audio_sha256"],
                    "duration_s": current["duration_s"],
                    "frame_count": current["frame_count"],
                    "sample_rate": current.get("sample_rate"),
                    "sample_count": current.get("sample_count"),
                    "source_manifest": str(manifest),
                    "source_manifest_sha256": manifest_sha,
                    "calibration_id": calibration["calibration_id"],
                    "calibration_manifest_sha256": calibration["hashes"]["manifest"],
                    "calibration_preaudit_sha256": calibration["hashes"]["preaudit"],
                    "calibration_verdicts_sha256": calibration["hashes"]["verdicts"],
                    "preaudit_provenance": (
                        "fixed_human_approved_pilot25_rebound_to_full_manifest_v1"
                    ),
                }
            )
            with labels_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(seeded, ensure_ascii=False, sort_keys=True) + "\n"
                )
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "source_id": source_id,
                            "calibration_seed": True,
                            "calibration_id": calibration["calibration_id"],
                            "calibration_preaudit_sha256": calibration["hashes"][
                                "preaudit"
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            existing[source_id] = seeded
    pending = [row for row in rows if row["source_id"] not in existing]
    worker_count = (
        resolve_worker_count(
            requested=int(args.workers),
            provider_limit=int(transport.max_concurrency),
            item_count=len(pending),
        )
        if pending
        else 0
    )
    started = time.perf_counter()
    _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running", "model": model, "provider_profile": profile, "reasoning_effort": "medium", "max_tokens": EXPECTED_MAX_TOKENS, "worker_count": worker_count, "completed": len(existing), "total": len(rows), "pending": len(pending)})

    def execute_row(row: dict[str, Any]) -> dict[str, Any]:
        print(
            f"v12_teacher_dispatch source={row['source_id']} "
            f"pass=single_tristate workers={worker_count}",
            flush=True,
        )
        request_started = time.perf_counter()
        try:
            response, response_raw, normalized = _call_teacher(
                row=row,
                transport=transport,
                args=args,
            )
            return {
                "ok": True,
                "response": response,
                "response_raw": response_raw,
                "normalized": normalized,
                "request_s": time.perf_counter() - request_started,
            }
        except Exception as error:  # noqa: BLE001
            return {
                "ok": False,
                "error": error,
                "request_s": time.perf_counter() - request_started,
            }

    completed_items = iter_completed_audio_teacher_items(
        items=pending,
        worker=execute_row,
        max_workers=max(1, worker_count),
        sequential_interval_s=(
            float(args.request_interval_s) if worker_count == 1 else 0.0
        ),
    )
    failures: list[tuple[str, Exception]] = []
    for completed in completed_items:
        row = completed.item
        outcome = completed.result
        request_elapsed = float(outcome["request_s"])
        if not outcome["ok"]:
            error = outcome["error"]
            failures.append((str(row["source_id"]), error))
            print(
                f"v12_teacher_error source={row['source_id']} "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
            _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running_with_failures", "model": model, "provider_profile": profile, "worker_count": worker_count, "completed": len(existing), "failed": len(failures), "total": len(rows), "pending": len(rows) - len(existing) - len(failures), "last_source_id": row["source_id"], "last_error": repr(error)})
            continue
        response = outcome["response"]
        response_raw = outcome["response_raw"]
        normalized = outcome["normalized"]
        label = {
            "schema": VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
            "boundary_serialization_contract_id": CONTRACT_ID,
            "source_id": row["source_id"], "video_id": str(row.get("video_id") or ""),
            "partition": row["partition"], "core_ids": row["core_ids"],
            "audio": row["audio"], "audio_sha256": row.get("audio_sha256") or _sha256(Path(row["audio"])),
            "duration_s": row["duration_s"], "frame_count": row["frame_count"], "frame_hop_s": FRAME_HOP_S,
            "sample_rate": row.get("sample_rate"), "sample_count": row.get("sample_count"),
            "model": model, "provider_profile": profile,
            "env_file_name": profile,
            "transport": transport.transport_name,
            "api_key_count": transport.api_key_count,
            "reasoning_effort": "medium", "max_tokens": EXPECTED_MAX_TOKENS,
            "temperature": None, "top_p": None, "top_k": None,
            "prompt_profile": PROMPT_PROFILE, "prompt_version": PROMPT_VERSION,
            "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
            "teacher_execution_contract_id": execution_contract,
            "source_manifest": str(manifest), "source_manifest_sha256": manifest_sha,
            "single_pass_response": response,
            "vocal_spans": normalized["vocal_spans"],
            "non_vocal_spans": normalized["non_vocal_spans"],
            "unsure_spans": normalized["unsure_spans"],
            "conflict_spans": [],
            "teacher_failed_closed": False, "training_manifest_allowed": False,
            "unsure_training_label": VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
            "preaudit_provenance": f"{profile}:{model}:single_pass_tristate_evidence",
        }
        with labels_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"source_id": row["source_id"], "single_pass": response_raw},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        existing[row["source_id"]] = label
        elapsed = time.perf_counter() - started
        rate = len(existing) / max(elapsed, 1e-9)
        eta = (len(rows) - len(existing)) / max(rate, 1e-9)
        print(
            f"v12_teacher_result={len(existing)}/{len(rows)} "
            f"source={row['source_id']} request_s={request_elapsed:.1f} "
            f"eta_s={eta:.0f}",
            flush=True,
        )
        _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running_with_failures" if failures else "running", "model": model, "provider_profile": profile, "reasoning_effort": "medium", "max_tokens": EXPECTED_MAX_TOKENS, "worker_count": worker_count, "completed": len(existing), "failed": len(failures), "total": len(rows), "pending": len(rows) - len(existing) - len(failures), "last_source_id": row["source_id"], "last_request_s": round(request_elapsed, 3), "elapsed_s": round(elapsed, 3), "eta_s": round(eta, 3)})
    if failures:
        source_id, error = failures[0]
        _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "failed", "model": model, "provider_profile": profile, "worker_count": worker_count, "completed": len(existing), "failed": len(failures), "total": len(rows), "pending": len(rows) - len(existing), "last_source_id": source_id, "last_error": repr(error)})
        raise RuntimeError(
            f"v12 Teacher failed for {len(failures)} source(s); "
            f"first={source_id}: {error}"
        ) from error
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "model": model,
        "provider_profile": profile,
        "env_file_name": profile,
        "transport": transport.transport_name,
        "api_key_count": transport.api_key_count,
        "worker_count": worker_count,
        "reasoning_effort": "medium",
        "max_tokens": EXPECTED_MAX_TOKENS,
        "request_count": len(existing) - len(calibration_seed_ids),
        "calls_per_new_source": 1,
        "calibration_seed_count": len(calibration_seed_ids),
        "calibration_id": calibration["calibration_id"] if calibration else None,
        "calibration_manifest_sha256": (
            calibration["hashes"]["manifest"] if calibration else None
        ),
        "calibration_preaudit_sha256": (
            calibration["hashes"]["preaudit"] if calibration else None
        ),
        "calibration_verdicts_sha256": (
            calibration["hashes"]["verdicts"] if calibration else None
        ),
        "omitted_sampling_parameters": ["temperature", "top_p", "top_k"],
        "prompt_profile": PROMPT_PROFILE,
        "prompt_version": PROMPT_VERSION,
        "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
        "teacher_execution_contract_id": execution_contract,
        "source_manifest": str(manifest),
        "source_manifest_sha256": manifest_sha,
        "source_count": len(rows),
        "result_count": len(existing),
        "results": str(labels_path),
        "raw_responses": str(raw_path),
        "training_manifest_allowed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "completed", "model": model, "provider_profile": profile, "reasoning_effort": "medium", "max_tokens": EXPECTED_MAX_TOKENS, "completed": len(existing), "total": len(rows), "pending": 0, "elapsed_s": round(time.perf_counter() - started, 3), "summary": str(output / "summary.json")})
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--env-file",
        choices=("openrouter", "gemini"),
        default="gemini",
        help="openrouter uses its OpenAI-compatible API; gemini uses Google AI Studio Interactions.",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-interval-s", type=float, default=0.5)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 uses one worker per native Gemini key; compatible providers remain single-worker.",
    )
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--calibration-preaudit")
    parser.add_argument("--calibration-verdicts")
    args = parser.parse_args(argv)
    if (
        args.limit < 0
        or args.max_attempts <= 0
        or args.request_interval_s < 0
        or args.workers < 0
    ):
        parser.error("limit/attempt/interval values are invalid")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
