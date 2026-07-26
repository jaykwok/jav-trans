#!/usr/bin/env python3
"""Run the adaptive complete-partition Scorer v12 prompt as an A/B arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Mapping
import wave

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.ja.vocal_envelope_v12 import (  # noqa: E402
    VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
    VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
)
from tools.boundary.ja.label_vocal_envelope_scorer_v12_with_omni import (  # noqa: E402
    FRAME_HOP_S,
    PROVIDER_CONTRACTS,
    _sha256,
    _validate_manifest,
)
from tools.boundary.ja.vocal_envelope_scorer_v12_teacher_contract import (  # noqa: E402
    SCORER_V12_FRAME_HOP_S,
    SCORER_V12_LOCAL_TIMESTAMP_STEP_S,
    SCORER_V12_TIME_GRID_CONTRACT_ID,
    quantize_vocal_partition_boundary_frame,
    require_scorer_v12_local_timestamp,
)
from tools.omni.audio_teacher_batch import (  # noqa: E402
    iter_completed_audio_teacher_items,
    resolve_worker_count,
)
from tools.omni.audio_teacher_transport import (  # noqa: E402
    AudioTeacherTransport,
    create_audio_teacher_transport,
)


PREAUDIT_SCHEMA = "vocal_envelope_scorer_v12_adaptive_partition_preaudit_v3"
SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_adaptive_partition_summary_v3"
PROGRESS_SCHEMA = "vocal_envelope_scorer_v12_adaptive_partition_progress_v3"
PROMPT_PROFILE = "vocal-envelope-adaptive-complete-partition-v3"
PROMPT_VERSION = "vocal-envelope-adaptive-complete-partition-v3-gemini36-medium-10ms-wire-20ms-frame"
EXPECTED_REASONING = "medium"
EXPECTED_MAX_TOKENS = 8192
LABEL_VALUES = frozenset({"vocal", "non_vocal", "unsure"})
DEFAULT_WINDOW_FRAMES = 1000
DEFAULT_TARGET_COMMIT_FRAMES = 750
DEFAULT_MIN_COMMIT_FRAMES = 500
DEFAULT_MAX_COMMIT_FRAMES = 900
DEFAULT_NONVOCAL_SEAM_FRAMES = 20
WINDOW_CHECKPOINT_SCHEMA = "vocal_envelope_adaptive_partition_window_checkpoint_v2"

COMPACT_SYSTEM_PROMPT = """你是人类发声事件包络检测器。输入一段音频窗口，把整个窗口切成
连续、无缝、无重叠的区间，每个区间贴一个标签。

【输出契约】
严格 JSON 数组，无其他文字。时间为相对窗口起点的秒数，量化到 0.01 秒，
最多两位小数；不要输出毫秒三位小数或比 0.01 秒更细的伪精度。
[{"s":0.00,"e":3.20,"t":"vocal"},{"s":3.20,"e":4.10,"t":"non_vocal"}, ...]
t: vocal | non_vocal | unsure
第一个区间必须从 0.00 开始，最后一个必须到窗口结束。
相邻区间首尾相接：前一段的 e 等于后一段的 s。不允许留空隙。
这些 0.01 秒坐标会由本地编译器用同一个共享切点量化到 Scorer 的 0.02 秒帧；
不要为追求更细的时间数值而拆分区间。

【仲裁规则，优先于一切】
只要能听到任何人类发声证据 —— 哪怕微弱、含混、被撞击声或音乐掩盖 ——
标 vocal。
non_vocal 只用于你能确信整段完全没有任何人类发声的区间。
判断不了就标 vocal，不要标 non_vocal，也不要标 unsure。
unsure 仅用于你确信有声音但完全无法归类的极少数情况。

【切分方式】
一个 vocal 区间 = 一次连续的发声事件，也就是你会作为一整句字幕
呈现的单位，或一段没有明确停顿的连续呻吟/说话。

只在你能清楚听到发声完全停止、并且停止持续到足以让人感觉"这一句
结束了"的位置断开。
句中换气、词首送气、弱尾音、音量起伏、浊音与气声之间的切换
都留在同一个区间内，不要在这些位置断开。

若一个 vocal 区间超过 15 秒，在其中最明显的一次停止处断开，
不要为了缩短而在无停顿处硬切。

【vocal】
对白、耳语、气声、呻吟、
喘息、吸气、呼气、叹气、哭、笑、咳嗽、清嗓、亲吻/唾液声、歌唱、
背景人声。「あ、ん、はぁ」无论是词语、应答还是纯呻吟都算。
ASR 能否识别、有无翻译价值均无关。
人声与其他声音重叠时，重叠区标 vocal。

【non_vocal】
纯机械声、肉体撞击/拍打、动作声、衣物摩擦、床体震动、水声、纯器乐、
静音、底噪、空调、电流、环境噪声。肉体撞击虽由人体产生但不是人声。
"""

COMPACT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "s": {"type": "number"},
            "e": {"type": "number"},
            "t": {"type": "string", "enum": sorted(LABEL_VALUES)},
        },
        "required": ["s", "e", "t"],
    },
}


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _content_sha256(value: Any) -> str:
    encoded = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runs(labels: list[int]) -> list[dict[str, Any]]:
    if not labels:
        return []
    names = {0: "non_vocal_candidate", 1: "vocal_candidate", -100: "unsure"}
    output: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index < len(labels) and labels[index] == labels[start]:
            continue
        output.append(
            {
                "label": names[labels[start]],
                "start_frame": start,
                "end_frame": index,
                "start_s": round(start * FRAME_HOP_S, 6),
                "end_s": round(index * FRAME_HOP_S, 6),
            }
        )
        start = index
    return output


def normalize_compact_response(
    parsed: Any, *, duration_s: float, frame_count: int
) -> dict[str, Any]:
    if not isinstance(parsed, list):
        raise ValueError("partition Teacher response root must be a JSON array")
    if not parsed:
        raise ValueError("partition Teacher response must cover the complete timeline")
    wire_duration = round(float(duration_s), 2)
    normalized: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, item in enumerate(parsed):
        if not isinstance(item, Mapping):
            raise ValueError(f"span[{index}] must be an object")
        if set(item) != {"s", "e", "t"}:
            raise ValueError(f"span[{index}] has missing or extra fields")
        start = require_scorer_v12_local_timestamp(
            item["s"], field=f"span[{index}].s"
        )
        end = require_scorer_v12_local_timestamp(
            item["e"], field=f"span[{index}].e"
        )
        label = str(item["t"])
        if label not in LABEL_VALUES:
            raise ValueError(f"span[{index}] has an unsupported enum value")
        if start < 0 or end <= start or end > wire_duration + 1e-9:
            raise ValueError(f"span[{index}] has invalid or out-of-range bounds")
        if abs(start - previous_end) > 1e-9:
            raise ValueError("partition Teacher spans must be contiguous without gaps")
        normalized.append({"s": start, "e": end, "t": label})
        previous_end = end
    if abs(previous_end - wire_duration) > 1e-9:
        raise ValueError("partition Teacher response must end at the advertised duration")

    label_values = {
        "vocal": 1,
        "non_vocal": 0,
        "unsure": VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
    }
    boundaries = [0]
    for index in range(1, len(normalized)):
        boundaries.append(
            quantize_vocal_partition_boundary_frame(
                left_label=str(normalized[index - 1]["t"]),
                right_label=str(normalized[index]["t"]),
                boundary_s=float(normalized[index]["s"]),
                frame_count=frame_count,
                frame_hop_s=FRAME_HOP_S,
            )
        )
    boundaries.append(frame_count)
    frame_labels: list[int] = []
    for index, item in enumerate(normalized):
        start_frame, end_frame = boundaries[index], boundaries[index + 1]
        if end_frame <= start_frame:
            raise ValueError(
                f"span[{index}] collapses below the 20ms Scorer frame resolution"
            )
        frame_labels.extend(
            [label_values[str(item["t"])]] * (end_frame - start_frame)
        )
    if len(frame_labels) != frame_count:
        raise RuntimeError("quantized partition does not cover every Scorer frame")
    compiled = _runs(frame_labels)
    return {
        "time_grid_contract_id": SCORER_V12_TIME_GRID_CONTRACT_ID,
        "teacher_timestamp_step_s": SCORER_V12_LOCAL_TIMESTAMP_STEP_S,
        "scorer_frame_hop_s": SCORER_V12_FRAME_HOP_S,
        "quantized_boundary_frames": boundaries,
        "original_spans": normalized,
        "vocal_spans": [span for span in compiled if span["label"] == "vocal_candidate"],
        "non_vocal_spans": [span for span in compiled if span["label"] == "non_vocal_candidate"],
        "unsure_spans": [span for span in compiled if span["label"] == "unsure"],
    }


def _frame_labels(normalized: Mapping[str, Any], *, frame_count: int) -> list[int]:
    labels = [VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX] * frame_count
    values = {
        "vocal_spans": 1,
        "non_vocal_spans": 0,
        "unsure_spans": VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
    }
    assigned = [False] * frame_count
    for key, value in values.items():
        for span in normalized.get(key) or ():
            start, end = int(span["start_frame"]), int(span["end_frame"])
            if not (0 <= start < end <= frame_count):
                raise ValueError("normalized partition span is out of bounds")
            for frame in range(start, end):
                if assigned[frame]:
                    raise ValueError("normalized partition spans overlap")
                labels[frame] = value
                assigned[frame] = True
    if not all(assigned):
        raise ValueError("normalized partition does not cover every frame")
    return labels


def choose_adaptive_commit_frame(
    normalized: Mapping[str, Any],
    *,
    window_frame_count: int,
    target_commit_frames: int = DEFAULT_TARGET_COMMIT_FRAMES,
    min_commit_frames: int = DEFAULT_MIN_COMMIT_FRAMES,
    max_commit_frames: int = DEFAULT_MAX_COMMIT_FRAMES,
    nonvocal_seam_frames: int = DEFAULT_NONVOCAL_SEAM_FRAMES,
) -> tuple[int, str]:
    if window_frame_count <= target_commit_frames:
        return window_frame_count, "source_tail"
    lower = max(1, min(min_commit_frames, window_frame_count - 1))
    upper = max(lower, min(max_commit_frames, window_frame_count - 1))
    candidates: list[int] = []
    margin = max(1, nonvocal_seam_frames // 2)
    for span in normalized.get("non_vocal_spans") or ():
        start, end = int(span["start_frame"]), int(span["end_frame"])
        if end - start < nonvocal_seam_frames:
            continue
        safe_start, safe_end = start + margin, end - margin
        if safe_start > safe_end:
            continue
        candidates.append(max(safe_start, min(target_commit_frames, safe_end)))
    bounded = [value for value in candidates if lower <= value <= upper]
    if bounded:
        return min(bounded, key=lambda value: (abs(value - target_commit_frames), value)), "definite_nonvocal_seam"
    return min(target_commit_frames, window_frame_count - 1), "target_fallback"


def _safe_source_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "source"
    return f"{cleaned[:96]}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"


def _slice_pcm_wav(
    source: Path,
    target: Path,
    *,
    start_s: float,
    end_s: float,
) -> float:
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        if params.comptype != "NONE":
            raise ValueError(f"rolling Teacher requires PCM WAV: {source}")
        start_sample = max(0, min(params.nframes, round(start_s * params.framerate)))
        end_sample = max(start_sample + 1, min(params.nframes, round(end_s * params.framerate)))
        reader.setpos(start_sample)
        payload = reader.readframes(end_sample - start_sample)
    with wave.open(str(target), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(payload)
    return (end_sample - start_sample) / params.framerate


def _call_window(
    row: Mapping[str, Any],
    *,
    audio_path: Path,
    duration_s: float,
    frame_count: int,
    window_index: int,
    source_start_frame: int,
    transport: AudioTeacherTransport,
    max_attempts: int,
) -> tuple[Any, dict[str, Any], dict[str, Any], float]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            prompt = json.dumps(
                {
                    "source_id": str(row["source_id"]),
                    "window_index": window_index,
                    "source_start_s": round(source_start_frame * FRAME_HOP_S, 2),
                    "duration_s": round(duration_s, 2),
                    "timestamp_step_s": SCORER_V12_LOCAL_TIMESTAMP_STEP_S,
                    "scorer_frame_hop_s": SCORER_V12_FRAME_HOP_S,
                    "time_grid_contract_id": SCORER_V12_TIME_GRID_CONTRACT_ID,
                    "coordinate_origin": "window_start_0.00s",
                    "coverage": "complete_contiguous_window_timeline",
                },
                ensure_ascii=False,
            )
            result = transport.call_json(
                audio_path=audio_path,
                prompt=prompt,
                system_prompt=COMPACT_SYSTEM_PROMPT,
                max_tokens=EXPECTED_MAX_TOKENS,
                enable_thinking=True,
                thinking_level=EXPECTED_REASONING,
                thinking_budget=0,
                response_schema=COMPACT_RESPONSE_SCHEMA,
                require_object=False,
                require_provider_parameters=True,
            )
            normalized = normalize_compact_response(
                result.parsed,
                duration_s=duration_s,
                frame_count=frame_count,
            )
            return result.parsed, result.raw, normalized, time.perf_counter() - started
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < max_attempts:
                time.sleep(min(8.0, float(attempt)))
    raise RuntimeError(
        f"partition Teacher failed for {row['source_id']} window {window_index}: {last_error}"
    ) from last_error


def _call(
    row: Mapping[str, Any],
    *,
    transport: AudioTeacherTransport,
    max_attempts: int,
    window_dir: Path,
    checkpoint_dir: Path,
    source_manifest_sha256: str,
    selection_manifest_sha256: str,
    window_frames: int,
    target_commit_frames: int,
    min_commit_frames: int,
    max_commit_frames: int,
    nonvocal_seam_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], float]:
    source_id = str(row["source_id"])
    source_audio = Path(str(row["audio"]))
    source_frame_count = int(row["frame_count"])
    if source_frame_count <= 0:
        raise ValueError(f"invalid source frame count: {source_id}")
    cursor = 0
    committed_labels: list[int] = []
    windows: list[dict[str, Any]] = []
    raw_windows: list[dict[str, Any]] = []
    total_request_s = 0.0
    safe_id = _safe_source_id(source_id)
    while cursor < source_frame_count:
        window_index = len(windows)
        window_end = min(source_frame_count, cursor + window_frames)
        requested_start_s = cursor * FRAME_HOP_S
        requested_end_s = min(float(row["duration_s"]), window_end * FRAME_HOP_S)
        audio_path = window_dir / safe_id / f"window-{window_index:03d}.wav"
        actual_duration_s = _slice_pcm_wav(
            source_audio,
            audio_path,
            start_s=requested_start_s,
            end_s=requested_end_s,
        )
        local_frame_count = window_end - cursor
        audio_sha256 = _sha256(audio_path)
        checkpoint_path = checkpoint_dir / safe_id / f"window-{window_index:03d}.json"
        checkpoint_contract = {
            "schema": WINDOW_CHECKPOINT_SCHEMA,
            "source_id": source_id,
            "source_audio_sha256": str(row["audio_sha256"]),
            "source_manifest_sha256": source_manifest_sha256,
            "selection_manifest_sha256": selection_manifest_sha256,
            "prompt_version": PROMPT_VERSION,
            "time_grid_contract_id": SCORER_V12_TIME_GRID_CONTRACT_ID,
            "system_prompt_sha256": _content_sha256(COMPACT_SYSTEM_PROMPT),
            "response_schema_sha256": _content_sha256(COMPACT_RESPONSE_SCHEMA),
            "window_index": window_index,
            "window_audio_sha256": audio_sha256,
            "source_start_frame": cursor,
            "source_end_frame": window_end,
            "local_frame_count": local_frame_count,
            "actual_audio_duration_s": actual_duration_s,
        }
        if checkpoint_path.is_file():
            checkpoint = _read_json(checkpoint_path)
            for field, expected in checkpoint_contract.items():
                if checkpoint.get(field) != expected:
                    raise ValueError(
                        f"adaptive window checkpoint {field} mismatch: "
                        f"{source_id} window {window_index}"
                    )
            parsed = checkpoint["parsed_response"]
            raw = checkpoint["raw_response"]
            normalized = checkpoint["normalized_partition"]
            request_s = 0.0
            window_provider_profile = str(checkpoint["provider_profile"])
            window_model = str(checkpoint["model"])
            window_transport = str(checkpoint["transport"])
            window_execution_contract = str(checkpoint["teacher_execution_contract_id"])
            print(
                f"partition_window_resume source={source_id} window={window_index + 1}",
                flush=True,
            )
        else:
            parsed, raw, normalized, request_s = _call_window(
                row,
                audio_path=audio_path,
                duration_s=actual_duration_s,
                frame_count=local_frame_count,
                window_index=window_index,
                source_start_frame=cursor,
                transport=transport,
                max_attempts=max_attempts,
            )
            window_provider_profile = transport.profile
            window_model = transport.model
            window_transport = transport.transport_name
            window_execution_contract = transport.execution_contract
            _atomic_json(
                checkpoint_path,
                {
                    **checkpoint_contract,
                    "provider_profile": window_provider_profile,
                    "model": window_model,
                    "transport": window_transport,
                    "teacher_execution_contract_id": window_execution_contract,
                    "parsed_response": parsed,
                    "raw_response": raw,
                    "normalized_partition": normalized,
                    "request_s": request_s,
                },
            )
        commit_frames, cut_kind = choose_adaptive_commit_frame(
            normalized,
            window_frame_count=local_frame_count,
            target_commit_frames=target_commit_frames,
            min_commit_frames=min_commit_frames,
            max_commit_frames=max_commit_frames,
            nonvocal_seam_frames=nonvocal_seam_frames,
        )
        if not (0 < commit_frames <= local_frame_count):
            raise RuntimeError(f"adaptive rolling made no progress: {source_id}")
        local_labels = _frame_labels(normalized, frame_count=local_frame_count)
        committed_labels.extend(local_labels[:commit_frames])
        committed_end = cursor + commit_frames
        windows.append(
            {
                "window_index": window_index,
                "audio": str(audio_path),
                "audio_sha256": audio_sha256,
                "provider_profile": window_provider_profile,
                "model": window_model,
                "transport": window_transport,
                "teacher_execution_contract_id": window_execution_contract,
                "source_start_frame": cursor,
                "source_end_frame": window_end,
                "source_start_s": round(cursor * FRAME_HOP_S, 6),
                "source_end_s": round(window_end * FRAME_HOP_S, 6),
                "actual_audio_duration_s": actual_duration_s,
                "committed_end_frame": committed_end,
                "committed_end_s": round(committed_end * FRAME_HOP_S, 6),
                "commit_frame_count": commit_frames,
                "cut_kind": cut_kind,
                "checkpoint": str(checkpoint_path),
                "response": parsed,
            }
        )
        raw_windows.append(
            {
                "window_index": window_index,
                "source_start_frame": cursor,
                "source_end_frame": window_end,
                "committed_end_frame": committed_end,
                "cut_kind": cut_kind,
                "provider_profile": window_provider_profile,
                "model": window_model,
                "response": raw,
            }
        )
        total_request_s += request_s
        print(
            f"partition_window_result source={source_id} "
            f"window={window_index + 1} input={requested_start_s:.2f}-{requested_end_s:.2f}s "
            f"commit={committed_end * FRAME_HOP_S:.2f}s cut={cut_kind} "
            f"request_s={request_s:.1f}",
            flush=True,
        )
        cursor = committed_end
    if len(committed_labels) != source_frame_count:
        raise RuntimeError(f"adaptive rolling did not cover complete source: {source_id}")
    compiled = _runs(committed_labels)
    normalized_source = {
        "time_grid_contract_id": SCORER_V12_TIME_GRID_CONTRACT_ID,
        "teacher_timestamp_step_s": SCORER_V12_LOCAL_TIMESTAMP_STEP_S,
        "scorer_frame_hop_s": SCORER_V12_FRAME_HOP_S,
        "time_grid_quantization": "shared_boundary_vocal_safe_to_20ms_frames",
        "rolling_windows": windows,
        "vocal_spans": [span for span in compiled if span["label"] == "vocal_candidate"],
        "non_vocal_spans": [span for span in compiled if span["label"] == "non_vocal_candidate"],
        "unsure_spans": [span for span in compiled if span["label"] == "unsure"],
    }
    return windows, raw_windows, normalized_source, total_request_s


def run(args: argparse.Namespace) -> dict[str, Any]:
    window_frames = round(float(args.window_s) / FRAME_HOP_S)
    target_commit_frames = round(float(args.target_commit_s) / FRAME_HOP_S)
    min_commit_frames = round(float(args.min_commit_s) / FRAME_HOP_S)
    max_commit_frames = round(float(args.max_commit_s) / FRAME_HOP_S)
    nonvocal_seam_frames = round(float(args.nonvocal_seam_s) / FRAME_HOP_S)
    if not (
        0 < min_commit_frames <= target_commit_frames <= max_commit_frames < window_frames
        and nonvocal_seam_frames > 0
    ):
        raise ValueError("adaptive rolling frame geometry is invalid")
    source_manifest = Path(args.manifest).resolve()
    source_manifest_sha = _sha256(source_manifest)
    rows = _validate_manifest(_rows(source_manifest), manifest=source_manifest)
    row_index = {str(row["source_id"]): row for row in rows}
    selection_manifest = Path(args.selection_manifest).resolve()
    selection_sha = _sha256(selection_manifest)
    selected_ids = [str(row.get("source_id") or "") for row in _rows(selection_manifest)]
    if not selected_ids or any(not value for value in selected_ids) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selection manifest requires unique non-empty source_id values")
    missing = [source_id for source_id in selected_ids if source_id not in row_index]
    if missing:
        raise ValueError(f"selection manifest IDs are absent from source manifest: {missing}")
    selected = [row_index[source_id] for source_id in selected_ids]

    profile = str(args.env_file)
    provider = PROVIDER_CONTRACTS[profile]
    transport = create_audio_teacher_transport(
        profile=profile,
        env_file=(Path.home() / ".config" / "omni" / profile).resolve(),
        model_override=str(args.model or ""),
        timeout_s=float(args.timeout_s),
        log=lambda message: print(message, flush=True),
    )
    if transport.model != provider["model"] or transport.execution_contract != provider["execution_contract"]:
        raise ValueError("partition Teacher provider contract mismatch")
    workers = resolve_worker_count(
        requested=int(args.workers),
        provider_limit=int(transport.max_concurrency),
        item_count=len(selected),
    )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    window_dir = output / "request_windows"
    checkpoint_dir = output / "window_checkpoints"
    preaudit_path = output / "preaudit.jsonl"
    raw_path = output / "raw_responses.jsonl"
    progress_path = output / "progress.json"
    existing: dict[str, dict[str, Any]] = {}
    if preaudit_path.is_file():
        for saved in _rows(preaudit_path):
            source_id = str(saved.get("source_id") or "")
            if source_id not in selected_ids or source_id in existing:
                raise ValueError(f"invalid partition Teacher resume row: {source_id!r}")
            for field, expected in (
                ("schema", PREAUDIT_SCHEMA),
                ("source_manifest_sha256", source_manifest_sha),
                ("selection_manifest_sha256", selection_sha),
                ("prompt_version", PROMPT_VERSION),
                ("time_grid_contract_id", SCORER_V12_TIME_GRID_CONTRACT_ID),
                ("rolling_window_frames", window_frames),
                ("rolling_target_commit_frames", target_commit_frames),
                ("rolling_min_commit_frames", min_commit_frames),
                ("rolling_max_commit_frames", max_commit_frames),
                ("rolling_nonvocal_seam_frames", nonvocal_seam_frames),
            ):
                if saved.get(field) != expected:
                    raise ValueError(f"partition Teacher resume {field} mismatch: {source_id}")
            existing[source_id] = saved
    pending = [row for row in selected if str(row["source_id"]) not in existing]
    started = time.perf_counter()
    _atomic_json(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running", "completed": len(existing), "total": len(selected), "pending": len(pending), "worker_count": workers})

    def worker(row: Mapping[str, Any]):
        return _call(
            row,
            transport=transport,
            max_attempts=int(args.max_attempts),
            window_dir=window_dir,
            checkpoint_dir=checkpoint_dir,
            source_manifest_sha256=source_manifest_sha,
            selection_manifest_sha256=selection_sha,
            window_frames=window_frames,
            target_commit_frames=target_commit_frames,
            min_commit_frames=min_commit_frames,
            max_commit_frames=max_commit_frames,
            nonvocal_seam_frames=nonvocal_seam_frames,
        )

    for completed in iter_completed_audio_teacher_items(
        items=pending,
        worker=worker,
        max_workers=workers,
        sequential_interval_s=float(args.request_interval_s),
    ):
        row = completed.item
        parsed, raw, normalized, request_s = completed.result
        source_id = str(row["source_id"])
        source_provider_profiles = sorted(
            {str(window["provider_profile"]) for window in parsed}
        )
        source_models = sorted({str(window["model"]) for window in parsed})
        source_execution_contracts = sorted(
            {str(window["teacher_execution_contract_id"]) for window in parsed}
        )
        artifact = {
            "schema": PREAUDIT_SCHEMA,
            "boundary_serialization_contract_id": "boundary_acoustic_binary_v12",
            "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
            "source_id": source_id,
            "video_id": str(row["video_id"]),
            "partition": str(row["partition"]),
            "core_ids": list(row["core_ids"]),
            "audio": str(row["audio"]),
            "audio_sha256": str(row["audio_sha256"]),
            "duration_s": float(row["duration_s"]),
            "frame_count": int(row["frame_count"]),
            "frame_hop_s": FRAME_HOP_S,
            "teacher_timestamp_step_s": SCORER_V12_LOCAL_TIMESTAMP_STEP_S,
            "scorer_frame_hop_s": SCORER_V12_FRAME_HOP_S,
            "time_grid_contract_id": SCORER_V12_TIME_GRID_CONTRACT_ID,
            "time_grid_quantization": "shared_boundary_vocal_safe_to_20ms_frames",
            "provider_profile": source_provider_profiles[0] if len(source_provider_profiles) == 1 else "mixed",
            "provider_profiles": source_provider_profiles,
            "model": source_models[0] if len(source_models) == 1 else "mixed",
            "models": source_models,
            "env_file_name": profile,
            "transport": "mixed" if len(source_execution_contracts) > 1 else str(parsed[0]["transport"]),
            "teacher_execution_contract_id": source_execution_contracts[0] if len(source_execution_contracts) == 1 else "mixed",
            "teacher_execution_contract_ids": source_execution_contracts,
            "reasoning_effort": EXPECTED_REASONING,
            "max_tokens": EXPECTED_MAX_TOKENS,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "prompt_profile": PROMPT_PROFILE,
            "prompt_version": PROMPT_VERSION,
            "system_prompt_sha256": _content_sha256(COMPACT_SYSTEM_PROMPT),
            "response_schema_sha256": _content_sha256(COMPACT_RESPONSE_SCHEMA),
            "source_manifest_sha256": source_manifest_sha,
            "selection_manifest_sha256": selection_sha,
            "rolling_window_frames": window_frames,
            "rolling_target_commit_frames": target_commit_frames,
            "rolling_min_commit_frames": min_commit_frames,
            "rolling_max_commit_frames": max_commit_frames,
            "rolling_nonvocal_seam_frames": nonvocal_seam_frames,
            "raw_partition_windows": parsed,
            **normalized,
            "training_manifest_allowed": False,
            "unsure_training_label": VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
        }
        with preaudit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(artifact, ensure_ascii=False, sort_keys=True) + "\n")
        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"source_id": source_id, "response": raw}, ensure_ascii=False, sort_keys=True) + "\n")
        existing[source_id] = artifact
        elapsed = time.perf_counter() - started
        print(f"partition_teacher_result={len(existing)}/{len(selected)} source={source_id} windows={len(parsed)} request_s={request_s:.1f}", flush=True)
        _atomic_json(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running", "completed": len(existing), "total": len(selected), "pending": len(selected) - len(existing), "worker_count": workers, "last_source_id": source_id, "last_request_s": round(request_s, 3), "elapsed_s": round(elapsed, 3)})

    provider_profiles = sorted(
        {str(row.get("provider_profile") or "") for row in existing.values()}
    )
    models = sorted({str(row.get("model") or "") for row in existing.values()})
    summary = {
        "schema": SUMMARY_SCHEMA,
        "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_manifest_sha,
        "selection_manifest": str(selection_manifest),
        "selection_manifest_sha256": selection_sha,
        "source_count": len(selected),
        "result_count": len(existing),
        "provider_profile": provider_profiles[0] if len(provider_profiles) == 1 else "mixed",
        "provider_profiles": provider_profiles,
        "model": models[0] if len(models) == 1 else "mixed",
        "models": models,
        "reasoning_effort": EXPECTED_REASONING,
        "max_tokens": EXPECTED_MAX_TOKENS,
        "omitted_sampling_parameters": ["temperature", "top_p", "top_k"],
        "prompt_profile": PROMPT_PROFILE,
        "prompt_version": PROMPT_VERSION,
        "teacher_timestamp_step_s": SCORER_V12_LOCAL_TIMESTAMP_STEP_S,
        "scorer_frame_hop_s": SCORER_V12_FRAME_HOP_S,
        "time_grid_contract_id": SCORER_V12_TIME_GRID_CONTRACT_ID,
        "time_grid_quantization": "shared_boundary_vocal_safe_to_20ms_frames",
        "system_prompt_sha256": _content_sha256(COMPACT_SYSTEM_PROMPT),
        "response_schema_sha256": _content_sha256(COMPACT_RESPONSE_SCHEMA),
        "rolling_window_frames": window_frames,
        "rolling_window_s": round(window_frames * FRAME_HOP_S, 6),
        "rolling_target_commit_frames": target_commit_frames,
        "rolling_target_commit_s": round(target_commit_frames * FRAME_HOP_S, 6),
        "rolling_min_commit_frames": min_commit_frames,
        "rolling_max_commit_frames": max_commit_frames,
        "rolling_nonvocal_seam_frames": nonvocal_seam_frames,
        "request_count": sum(len(row.get("rolling_windows") or ()) for row in existing.values()),
        "preaudit": str(preaudit_path),
        "preaudit_sha256": _sha256(preaudit_path),
        "raw_responses": str(raw_path),
        "training_manifest_allowed": False,
    }
    _atomic_json(output / "summary.json", summary)
    _atomic_json(progress_path, {"schema": PROGRESS_SCHEMA, "status": "completed", "completed": len(existing), "total": len(selected), "pending": 0, "worker_count": workers, "elapsed_s": round(time.perf_counter() - started, 3), "summary": str(output / "summary.json")})
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-file", choices=("gemini", "openrouter"), default="gemini")
    parser.add_argument("--model", default="")
    # A source performs several dependent rolling-window requests.  Auto-scaling
    # by API-key count creates a provider-wide burst even when each key has its
    # own local RPM ledger, so adaptive labeling is deliberately serial by default.
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-interval-s", type=float, default=0.5)
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--target-commit-s", type=float, default=15.0)
    parser.add_argument("--min-commit-s", type=float, default=10.0)
    parser.add_argument("--max-commit-s", type=float, default=18.0)
    parser.add_argument("--nonvocal-seam-s", type=float, default=0.4)
    args = parser.parse_args(argv)
    if (
        args.workers < 0
        or args.timeout_s <= 0
        or args.max_attempts <= 0
        or args.request_interval_s < 0
        or min(
            args.window_s,
            args.target_commit_s,
            args.min_commit_s,
            args.max_commit_s,
            args.nonvocal_seam_s,
        )
        <= 0
    ):
        parser.error("worker/timeout/attempt/interval values are invalid")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
