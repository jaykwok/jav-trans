#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import wave
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.config import load_config  # noqa: E402

load_config()

from tools.audits.generate_boundary_preference_audit_html import generate_audit  # noqa: E402
from tools.boundary.boundary_preference import (  # noqa: E402
    AXES,
    CANDIDATE_SCHEMA,
    OFFSETS_MS,
    aligned_offset_frames,
    build_blind_items,
    candidate_id,
    candidate_observables,
    is_nonlexical_text,
    perturbation_category,
    read_jsonl,
    select_balanced_candidates,
    stable_hash,
    write_jsonl,
)


@dataclass(frozen=True)
class CaseSpec:
    video_id: str
    video_label: str
    media_path: Path


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path) -> str:
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: Sequence[str]) -> None:
    print("run=" + " ".join(str(value) for value in command), flush=True)
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def prepare_audio(case: CaseSpec, *, output_dir: Path, ffmpeg_bin: str) -> Path:
    audio_path = output_dir / "audio" / f"{case.video_id}.16k-mono.wav"
    if audio_path.exists() and audio_path.stat().st_size > 44:
        return audio_path
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(case.media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]
    )
    return audio_path


def _evenly_spaced(values: Sequence[dict[str, Any]], count: int, *, salt: str) -> list[dict[str, Any]]:
    if len(values) < count:
        return []
    ordered = sorted(
        values,
        key=lambda row: (
            float(row.get("boundary_time_s") or 0.0),
            stable_hash(salt, row.get("boundary_index")),
        ),
    )
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for slot in range(count):
        target = int(round(((slot + 0.5) * len(ordered) / count) - 0.5))
        target = max(0, min(len(ordered) - 1, target))
        radius = 0
        while target in used and radius < len(ordered):
            radius += 1
            left = target - radius
            right = target + radius
            if left >= 0 and left not in used:
                target = left
                break
            if right < len(ordered) and right not in used:
                target = right
                break
        if target in used:
            return []
        used.add(target)
        selected.append(ordered[target])
    return selected


def _prepool_for_case(
    boundary_rows: Sequence[dict[str, Any]],
    *,
    case: CaseSpec,
    frame_hop_s: float,
    per_category: int,
) -> list[dict[str, Any]]:
    remaining = list(boundary_rows)
    selected: list[dict[str, Any]] = []
    used_boundaries: set[int] = set()
    categories = [
        (axis, offset_ms, aligned_offset_frames(offset_ms, frame_hop_s))
        for axis in AXES
        for offset_ms in OFFSETS_MS
    ]
    for axis, offset_ms, offset_frames in categories:
        valid: list[dict[str, Any]] = []
        offset_s = offset_frames * frame_hop_s
        for row in remaining:
            boundary_index = int(row["boundary_index"])
            if boundary_index in used_boundaries:
                continue
            left_interval = dict(row["left_interval"])
            right_interval = dict(row["right_interval"])
            if axis == "right.start":
                challenger_interval = {
                    "start": float(right_interval["start"]) + offset_s,
                    "end": float(right_interval["end"]),
                }
            else:
                challenger_interval = {
                    "start": float(left_interval["start"]),
                    "end": float(left_interval["end"]) + offset_s,
                }
            if (
                challenger_interval["start"] < 0.0
                or challenger_interval["end"] > float(row["duration_s"]) + 1e-6
                or challenger_interval["end"] - challenger_interval["start"] < 0.4
            ):
                continue
            valid.append(
                {
                    **row,
                    "axis": axis,
                    "offset_ms": int(round(offset_s * 1000.0)),
                    "offset_frames": offset_frames,
                    "offset_s": round(offset_s, 6),
                    "challenger_interval": {
                        "start": round(challenger_interval["start"], 6),
                        "end": round(challenger_interval["end"], 6),
                    },
                }
            )
        picked = _evenly_spaced(
            valid,
            per_category,
            salt=f"{case.video_id}:{axis}:{offset_frames}",
        )
        if len(picked) < per_category:
            raise ValueError(
                f"not enough valid boundaries for {case.video_id} {axis} {offset_ms}ms: "
                f"need={per_category} available={len(valid)}"
            )
        for row in picked:
            boundary_index = int(row["boundary_index"])
            used_boundaries.add(boundary_index)
            baseline_interval = (
                dict(row["right_interval"])
                if axis == "right.start"
                else dict(row["left_interval"])
            )
            challenger_interval = dict(row["challenger_interval"])
            baseline_left_end = float(row["left_interval"]["end"])
            baseline_right_start = float(row["right_interval"]["start"])
            challenger_left_end = (
                float(challenger_interval["end"])
                if axis == "left.end"
                else baseline_left_end
            )
            challenger_right_start = (
                float(challenger_interval["start"])
                if axis == "right.start"
                else baseline_right_start
            )
            gap_consumed_s = (
                max(0.0, baseline_right_start - challenger_right_start)
                if axis == "right.start"
                else max(0.0, challenger_left_end - baseline_left_end)
            )
            crossing = gap_consumed_s > 0.0
            row.update(
                {
                    "schema": CANDIDATE_SCHEMA,
                    "candidate_id": candidate_id(
                        video_id=case.video_id,
                        boundary_index=boundary_index,
                        axis=axis,
                        offset_frames=offset_frames,
                    ),
                    "video_id": case.video_id,
                    "video_label": case.video_label,
                    "media_path": project_rel(case.media_path),
                    "audio_path": str(row["audio_path"]),
                    "baseline_interval": baseline_interval,
                    "challenger_interval": challenger_interval,
                    "baseline_boundary": {
                        "left_end_s": round(baseline_left_end, 6),
                        "right_start_s": round(baseline_right_start, 6),
                    },
                    "challenger_boundary": {
                        "left_end_s": round(challenger_left_end, 6),
                        "right_start_s": round(challenger_right_start, 6),
                    },
                    "gap_crossing": crossing,
                    "gap_consumed_s": round(gap_consumed_s, 6),
                    "gap_overlap_s": round(
                        max(0.0, challenger_left_end - challenger_right_start),
                        6,
                    ),
                    "perturbation_category": perturbation_category(axis, offset_frames),
                }
            )
            selected.append(row)
    return selected


def build_boundary_candidates(
    cases: Sequence[CaseSpec],
    *,
    output_dir: Path,
    ffmpeg_bin: str,
    device: str,
    per_category: int,
) -> list[dict[str, Any]]:
    from audio.chunk_packer import PackingLayoutConfig, _materialize_packed_chunks
    from boundary.features import make_feature_bundle
    from boundary.ja.backend import SpeechBoundaryJaBackend, SpeechBoundaryJaConfig
    from boundary.planner import BoundaryPlannerConfig, plan_boundary_chunks
    from boundary.refiner import load_frame_sequence_refiner_checkpoint
    from boundary.sequence_features import (
        FRAME_SEQUENCE_FEATURE_SCHEMA,
        FrameSequenceFeatureConfig,
        FrameSequenceFeatureProvider,
        feature_extraction_signature,
    )

    checkpoint_path = project_path(
        os.getenv("BOUNDARY_REFINER_MODEL_PATH", "").strip()
        or "src/boundary/checkpoints/boundary_refiner.pt"
    )
    refiner = load_frame_sequence_refiner_checkpoint(checkpoint_path, device=device)
    env_config = SpeechBoundaryJaConfig.from_env()
    backend = SpeechBoundaryJaBackend(
        replace(
            env_config,
            device=device,
            export_sequence_features=True,
        )
    )
    planner_config = BoundaryPlannerConfig(
        frame_hop_s=float(os.getenv("BOUNDARY_FEATURE_FRAME_HOP_S", "0.02")),
        max_core_chunk_s=float(os.getenv("BOUNDARY_PLANNER_MAX_CORE_CHUNK_S", "5.0")),
        target_chunk_s=float(os.getenv("BOUNDARY_PLANNER_TARGET_CHUNK_S", "3.0")),
        min_chunk_s=float(os.getenv("BOUNDARY_PLANNER_MIN_CHUNK_S", "0.4")),
        max_splits_per_segment=int(
            os.getenv("BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT", "16")
        ),
        sequence_batch_size=int(
            os.getenv("BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE", "256")
        ),
    )
    all_candidates: list[dict[str, Any]] = []
    try:
        for case_index, case in enumerate(cases, start=1):
            audio_path = prepare_audio(case, output_dir=output_dir, ffmpeg_bin=ffmpeg_bin)
            print(
                f"boundary_case={case_index}/{len(cases)} video={case.video_id} audio={audio_path}",
                flush=True,
            )
            result = backend.segment(str(audio_path))
            frames = result.parameters.get("sequence_feature_frames")
            if not isinstance(frames, Mapping):
                raise ValueError("SpeechBoundary-JA did not export sequence feature frames")
            feature_config = FrameSequenceFeatureConfig(
                target_chunk_s=planner_config.target_chunk_s
            )
            provider = FrameSequenceFeatureProvider(
                duration_s=float(result.audio_duration_sec),
                frame_hop_s=float(frames.get("frame_hop_s") or env_config.frame_hop_s),
                ptm=frames.get("ptm") or [],
                mfcc=frames.get("mfcc") or [],
                config=feature_config,
            )
            provider.validate_for_checkpoint(
                refiner.feature_names,
                refiner.feature_schema_hash,
            )
            frame_scores = result.parameters.get("frame_scores")
            cut_scores = result.parameters.get("cut_frame_scores")
            score_hop = float(result.parameters.get("frame_hop_s") or env_config.frame_hop_s)
            planned = plan_boundary_chunks(
                result.segments,
                features=make_feature_bundle(
                    frame_hop_s=score_hop,
                    speech_scores=frame_scores,
                    cut_scores=cut_scores,
                ),
                config=planner_config,
                sequence_refiner=refiner,
                sequence_feature_provider=provider,
            )
            packed = _materialize_packed_chunks(
                planned,
                layout=PackingLayoutConfig(),
            )
            feature_names = provider.feature_names()
            feature_signature = feature_extraction_signature(
                config=feature_config,
                feature_names=feature_names,
            )
            boundary_rows: list[dict[str, Any]] = []
            for boundary_index, (left, right) in enumerate(zip(planned, planned[1:])):
                decision = left.boundary_decision
                if decision is None:
                    continue
                raw_left = left.islands[-1]
                raw_right = right.islands[0]
                left_chunk = packed[boundary_index]
                right_chunk = packed[boundary_index + 1]
                feature = provider.features_for_boundary(
                    left_start_s=raw_left.start,
                    left_end_s=raw_left.end,
                    right_start_s=raw_right.start,
                    right_end_s=raw_right.end,
                )
                boundary_rows.append(
                    {
                        "boundary_index": boundary_index,
                        "audio_path": project_rel(audio_path),
                        "duration_s": round(float(result.audio_duration_sec), 6),
                        "frame_hop_s": provider.frame_hop_s,
                        "raw_left_end_s": round(float(raw_left.end), 6),
                        "raw_right_start_s": round(float(raw_right.start), 6),
                        "left_interval": {
                            "start": round(float(left_chunk.start), 6),
                            "end": round(float(left_chunk.end), 6),
                        },
                        "right_interval": {
                            "start": round(float(right_chunk.start), 6),
                            "end": round(float(right_chunk.end), 6),
                        },
                        "boundary_time_s": round(
                            (float(left_chunk.end) + float(right_chunk.start)) / 2.0,
                            6,
                        ),
                        "context_start": round(
                            max(0.0, min(float(left_chunk.start), float(right_chunk.start)) - 1.2),
                            6,
                        ),
                        "context_end": round(
                            min(
                                float(result.audio_duration_sec),
                                max(float(left_chunk.end), float(right_chunk.end)) + 1.2,
                            ),
                            6,
                        ),
                        "feature_schema": FRAME_SEQUENCE_FEATURE_SCHEMA,
                        "feature_schema_hash": refiner.feature_schema_hash,
                        "feature_signature": feature_signature,
                        "feature_names": feature_names,
                        "sequence_feature": feature,
                    }
                )
            case_candidates = _prepool_for_case(
                boundary_rows,
                case=case,
                frame_hop_s=provider.frame_hop_s,
                per_category=per_category,
            )
            all_candidates.extend(case_candidates)
            write_json(
                output_dir / "boundary" / f"{case.video_id}.summary.json",
                {
                    "video_id": case.video_id,
                    "media_path": project_rel(case.media_path),
                    "audio_path": project_rel(audio_path),
                    "duration_s": result.audio_duration_sec,
                    "speech_segments": len(result.segments),
                    "planned_chunks": len(planned),
                    "frame_hop_s": provider.frame_hop_s,
                    "prepool_candidates": len(case_candidates),
                    "category_counts": dict(
                        Counter(row["perturbation_category"] for row in case_candidates)
                    ),
                    "checkpoint": project_rel(checkpoint_path),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "checkpoint_signature": refiner.signature(),
                    "planner_config": planner_config.signature(),
                    "boundary_config": asdict(env_config),
                },
            )
            del frames
            del provider
            del result
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
        del refiner
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return all_candidates


def materialize_candidate_clips(
    candidates: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
) -> dict[tuple[str, str], Path]:
    by_video: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        by_video.setdefault(str(candidate["video_id"]), []).append(candidate)
    paths: dict[tuple[str, str], Path] = {}
    for video_id, rows in by_video.items():
        source_audio = project_path(str(rows[0]["audio_path"]))
        clip_dir = output_dir / "clips" / video_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        with wave.open(str(source_audio), "rb") as source:
            sample_rate = source.getframerate()
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or sample_rate != 16000
            ):
                raise ValueError(f"prepared audio must be 16k mono PCM16: {source_audio}")
            for candidate in rows:
                for identity in ("baseline", "challenger"):
                    interval = candidate[f"{identity}_interval"]
                    start_frame = max(0, int(round(float(interval["start"]) * sample_rate)))
                    end_frame = min(
                        source.getnframes(),
                        int(round(float(interval["end"]) * sample_rate)),
                    )
                    clip_path = clip_dir / f"{candidate['candidate_id']}.{identity}.wav"
                    if not clip_path.exists() or clip_path.stat().st_size <= 44:
                        source.setpos(start_frame)
                        frames = source.readframes(max(0, end_frame - start_frame))
                        with wave.open(str(clip_path), "wb") as target:
                            target.setnchannels(1)
                            target.setsampwidth(2)
                            target.setframerate(sample_rate)
                            target.writeframes(frames)
                    paths[(str(candidate["candidate_id"]), identity)] = clip_path
    return paths


def run_candidate_asr(
    candidates: Sequence[Mapping[str, Any]],
    *,
    clip_paths: Mapping[tuple[str, str], Path],
    device: str,
) -> list[dict[str, Any]]:
    os.environ["ASR_SLIDING_CONTEXT_SEGS"] = "0"
    from asr.backends.registry import _resolve_asr_backend, get_backend_label
    from asr.pipeline import (
        _align_TRANSCRIPTION_results,
        _alignment_outcome_for_chunk,
        _finalize_aligned_chunk_without_asr_retry,
        _qc_items_by_chunk,
        _run_TRANSCRIPTION_qc,
        _transcribe_asr_chunks_text_only,
        _with_alignment_fallback_window,
        collect_adaptive_precision_review,
    )
    from asr.prealign import prepare_text_for_alignment, strip_alignment_punctuation
    from tools.asr.diagnostics.diagnose_asr_alignment import diagnostic_qc_metrics

    chunks: list[dict[str, Any]] = []
    job_meta: list[tuple[str, str]] = []
    for candidate in candidates:
        for identity in ("baseline", "challenger"):
            interval = candidate[f"{identity}_interval"]
            index = len(chunks)
            chunks.append(
                {
                    "index": index,
                    "path": str(clip_paths[(str(candidate["candidate_id"]), identity)]),
                    "start": float(interval["start"]),
                    "end": float(interval["end"]),
                    "duration": float(interval["end"]) - float(interval["start"]),
                    "core_start": float(interval["start"]),
                    "core_end": float(interval["end"]),
                    "speech_segments": [
                        {
                            "start": float(interval["start"]),
                            "end": float(interval["end"]),
                            "score": 1.0,
                        }
                    ],
                }
            )
            job_meta.append((str(candidate["candidate_id"]), identity))

    backend = _resolve_asr_backend(device)
    log: list[str] = []

    def on_stage(message: str) -> None:
        print(f"stage={message}", flush=True)

    try:
        backend.load(on_stage=on_stage)
        text_results, text_timings = _transcribe_asr_chunks_text_only(
            backend,
            chunks,
            "Boundary preference ASR",
            on_stage=on_stage,
        )
        qc_report, qc_timings = _run_TRANSCRIPTION_qc(
            chunks,
            text_results,
            log,
            on_stage=on_stage,
        )
        text_results, qc_report, _ = collect_adaptive_precision_review(
            chunks,
            text_results,
            qc_report,
        )
        text_results = [
            _with_alignment_fallback_window(chunk, result)
            for chunk, result in zip(chunks, text_results, strict=True)
        ]
        qc_items, review_items = _qc_items_by_chunk(qc_report)
        backend.unload_model(on_stage=on_stage)
        prepared, align_timings = _align_TRANSCRIPTION_results(
            backend,
            text_results,
            on_stage=on_stage,
        )
        output: list[dict[str, Any]] = []
        for chunk, text_result, prepared_result, meta in zip(
            chunks,
            text_results,
            prepared,
            job_meta,
            strict=True,
        ):
            chunk_result, chunk_log = prepared_result
            chunk_words, chunk_log = _finalize_aligned_chunk_without_asr_retry(
                chunk,
                chunk_result,
                list(chunk_log),
            )
            chunk_index = int(chunk["index"])
            qc_item = qc_items.get(chunk_index)
            review_item = review_items.get(chunk_index)
            outcome = _alignment_outcome_for_chunk(
                chunk=chunk,
                chunk_result=chunk_result,
                chunk_words=chunk_words,
                chunk_log=chunk_log,
                qc_item=qc_item,
                review_item=review_item,
            )
            text = str(text_result.get("text") or text_result.get("raw_text") or "")
            raw_text = str(text_result.get("raw_text") or text)
            prealign = prepare_text_for_alignment(text)
            qc_metrics, merged_qc_item = diagnostic_qc_metrics(
                chunk=chunk,
                analysis_text=text,
                raw_text=raw_text,
                qc_item=qc_item,
            )
            repeat_profile = (
                dict(qc_metrics.get("max_repeat") or {})
                if isinstance(qc_metrics.get("max_repeat"), Mapping)
                else {}
            )
            repetition_repair = (
                dict(qc_metrics.get("repetition_repair") or {})
                if isinstance(qc_metrics.get("repetition_repair"), Mapping)
                else {}
            )
            if not repeat_profile and repetition_repair.get("changed"):
                repeat_profile = {
                    "unit": repetition_repair.get("unit") or "",
                    "run": repetition_repair.get("run") or 0,
                    "ratio": repetition_repair.get("ratio") or 0.0,
                }
            compact_chars = len(strip_alignment_punctuation(prealign.display_text))
            duration = max(0.001, float(chunk["duration"]))
            candidate_value, identity = meta
            output.append(
                {
                    "candidate_id": candidate_value,
                    "identity": identity,
                    "text": text,
                    "raw_text": raw_text,
                    "display_text": prealign.display_text,
                    "asr_qc_severity": str(
                        (merged_qc_item or {}).get("severity")
                        or outcome.get("asr_qc_severity")
                        or ""
                    ),
                    "asr_qc_reasons": list(
                        (merged_qc_item or {}).get("reasons")
                        or outcome.get("asr_qc_reasons")
                        or []
                    ),
                    "alignment_quality": outcome.get("alignment_quality"),
                    "fallback_type": outcome.get("fallback_type"),
                    "fallback_subtype": outcome.get("fallback_subtype"),
                    "sentinel": bool(outcome.get("sentinel_lines")),
                    "sentinel_lines": list(outcome.get("sentinel_lines") or []),
                    "nonlexical_text": is_nonlexical_text(text),
                    "repeat_profile": repeat_profile,
                    "repetition_repair": repetition_repair,
                    "text_density": dict(qc_metrics.get("text_density") or {}),
                    "compact_chars": compact_chars,
                    "chars_per_sec": round(compact_chars / duration, 6),
                    "cue_density_cps": round(compact_chars / duration, 6),
                    "word_count": len(chunk_words),
                }
            )
        print(
            "asr_backend={backend} text_s={text_s:.3f} qc_s={qc_s:.3f} align_s={align_s:.3f}".format(
                backend=get_backend_label(),
                text_s=float(text_timings.get("text_transcribe_s") or 0.0),
                qc_s=float(qc_timings.get("asr_qc_s") or 0.0),
                align_s=float(align_timings.get("alignment_s") or 0.0),
            ),
            flush=True,
        )
        return output
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


def attach_results(
    candidates: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (str(row.get("candidate_id") or ""), str(row.get("identity") or "")): dict(row)
        for row in result_rows
    }
    output: list[dict[str, Any]] = []
    for raw in candidates:
        row = dict(raw)
        candidate_value = str(row["candidate_id"])
        baseline = indexed.get((candidate_value, "baseline"))
        challenger = indexed.get((candidate_value, "challenger"))
        if baseline is None or challenger is None:
            raise ValueError(f"missing ASR result for {candidate_value}")
        row["baseline_result"] = baseline
        row["challenger_result"] = challenger
        row.update(candidate_observables(row, baseline, challenger))
        output.append(row)
    return output


def build_summary(
    *,
    cases: Sequence[CaseSpec],
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    blind_rows: Sequence[Mapping[str, Any]],
    answer_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    audit_dir: Path,
) -> dict[str, Any]:
    return {
        "schema": "boundary_preference_pilot_build_v1",
        "dataset_id": "true-v5-boundary-preference-pilot-20260611",
        "cases": [
            {
                "video_id": case.video_id,
                "video_label": case.video_label,
                "media_path": project_rel(case.media_path),
            }
            for case in cases
        ],
        "candidate_pool_count": len(candidates),
        "unique_selected_count": len(selected),
        "blind_review_count": len(blind_rows),
        "hidden_duplicate_count": sum(
            1 for row in answer_rows if bool(row.get("is_hidden_duplicate"))
        ),
        "video_counts": dict(Counter(str(row["video_id"]) for row in selected)),
        "axis_counts": dict(Counter(str(row["axis"]) for row in selected)),
        "offset_ms_counts": dict(Counter(str(row["offset_ms"]) for row in selected)),
        "category_counts": dict(
            Counter(str(row["perturbation_category"]) for row in selected)
        ),
        "risk_bucket_counts": dict(Counter(str(row["risk_bucket"]) for row in selected)),
        "signals_used_for_selection": [
            "asr_text_stability",
            "asr_qc",
            "repeat_nonlexical",
            "forced_fallback_sentinel",
            "cue_density",
            "gap_crossing",
        ],
        "signals_explicitly_not_used": [
            "local_cer",
            "avg_logprob",
            "no_speech_prob",
            "token_confidence",
        ],
        "outputs": {
            "candidates": project_rel(output_dir / "selected_candidates.jsonl"),
            "blind_manifest": project_rel(output_dir / "blind_items.jsonl"),
            "answer_key": project_rel(output_dir / "answer_key.jsonl"),
            "empty_labels": project_rel(output_dir / "manual_boundary_preference_labels.jsonl"),
            "audit_html": project_rel(audit_dir / "index.html"),
        },
        "gate": {
            "hidden_duplicate_consistency": ">=0.80",
            "usable_labels": ">=90",
            "decisive_ratio": ">0.50",
            "challenger_wins": ">25",
            "challenger_category_coverage": ">3",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the first true-v5 boundary perturbation preference pilot "
            "(108 unique + 12 hidden duplicates)."
        )
    )
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        required=True,
        metavar=("VIDEO_ID", "VIDEO_LABEL", "MEDIA"),
    )
    parser.add_argument(
        "--output-dir",
        default="agents/temp/speech-boundary-ja/true-v5-boundary-preference-pilot",
    )
    parser.add_argument(
        "--audit-dir",
        default="agents/audits/true-v5-boundary-preference-pilot",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asr-backend", default="")
    parser.add_argument("--asr-batch-size", default="")
    parser.add_argument("--aligner-batch-size", type=int)
    parser.add_argument("--prepool-per-category", type=int, default=16)
    parser.add_argument("--unique-per-video", type=int, default=36)
    parser.add_argument("--hidden-duplicates", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--force-boundary", action="store_true")
    parser.add_argument("--force-asr", action="store_true")
    parser.add_argument("--update-entrypoints", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.prepool_per_category < 5:
        parser.error("--prepool-per-category must be at least 5")
    if args.unique_per_video != 36:
        parser.error("the first-stage pilot requires --unique-per-video 36")
    if args.hidden_duplicates != 12:
        parser.error("the first-stage pilot requires --hidden-duplicates 12")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.asr_backend:
        os.environ["ASR_BACKEND"] = args.asr_backend
    if args.asr_batch_size:
        os.environ["ASR_BATCH_SIZE"] = args.asr_batch_size
    if args.aligner_batch_size:
        os.environ["ALIGNER_BATCH_SIZE"] = str(args.aligner_batch_size)
        os.environ["ALIGN_LONG_CHUNK_BATCH_SIZE"] = str(args.aligner_batch_size)
    os.environ["ASR_WORKER_MODE"] = "subprocess"
    os.environ["ASR_SLIDING_CONTEXT_SEGS"] = "0"
    os.environ["SPEECH_BOUNDARY_JA_EXPORT_SEQUENCE_FEATURES"] = "1"
    force_asr = bool(args.force_asr or args.force_boundary)
    if force_asr:
        os.environ["ASR_CHECKPOINT_ENABLED"] = "0"

    cases = [
        CaseSpec(
            video_id=values[0],
            video_label=values[1],
            media_path=project_path(values[2]),
        )
        for values in args.case
    ]
    if len(cases) != 3:
        raise SystemExit("the first-stage pilot requires exactly three --case values")
    for case in cases:
        if not case.media_path.exists():
            raise FileNotFoundError(case.media_path)

    output_dir = project_path(args.output_dir)
    audit_dir = project_path(args.audit_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ASR_CHUNK_ROOT"] = str(output_dir / "clips")
    boundary_pool_path = output_dir / "candidate_prepool.jsonl"
    asr_results_path = output_dir / "candidate_asr_results.jsonl"

    if boundary_pool_path.exists() and not args.force_boundary:
        candidate_pool = read_jsonl(boundary_pool_path)
        print(f"reuse_boundary_candidates={len(candidate_pool)}", flush=True)
    else:
        candidate_pool = build_boundary_candidates(
            cases,
            output_dir=output_dir,
            ffmpeg_bin=args.ffmpeg_bin,
            device=args.device,
            per_category=args.prepool_per_category,
        )
        write_jsonl(boundary_pool_path, candidate_pool)

    clip_paths = materialize_candidate_clips(candidate_pool, output_dir=output_dir)
    if asr_results_path.exists() and not force_asr:
        asr_results = read_jsonl(asr_results_path)
        print(f"reuse_asr_results={len(asr_results)}", flush=True)
    else:
        asr_results = run_candidate_asr(
            candidate_pool,
            clip_paths=clip_paths,
            device=args.device,
        )
        write_jsonl(asr_results_path, asr_results)

    scored = attach_results(candidate_pool, asr_results)
    write_jsonl(output_dir / "scored_candidate_pool.jsonl", scored)
    selected = select_balanced_candidates(
        scored,
        video_ids=[case.video_id for case in cases],
        per_video=args.unique_per_video,
    )
    selected_path = output_dir / "selected_candidates.jsonl"
    write_jsonl(selected_path, selected)
    blind_rows, answer_rows = build_blind_items(
        selected,
        hidden_duplicate_count=args.hidden_duplicates,
        seed=args.seed,
    )
    blind_path = output_dir / "blind_items.jsonl"
    answer_path = output_dir / "answer_key.jsonl"
    write_jsonl(blind_path, blind_rows)
    write_jsonl(answer_path, answer_rows)
    empty_labels_path = output_dir / "manual_boundary_preference_labels.jsonl"
    if not empty_labels_path.exists():
        empty_labels_path.write_text("", encoding="utf-8")

    generate_audit(
        blind_manifest=blind_path,
        output_dir=audit_dir,
        title="True v5 边界偏好盲测（120 条）",
        dataset_id="true-v5-boundary-preference-pilot-20260611",
        update_entrypoints=args.update_entrypoints,
    )
    summary = build_summary(
        cases=cases,
        candidates=scored,
        selected=selected,
        blind_rows=blind_rows,
        answer_rows=answer_rows,
        output_dir=output_dir,
        audit_dir=audit_dir,
    )
    write_json(output_dir / "summary.json", summary)
    print(f"candidate_pool={len(scored)}")
    print(f"unique_selected={len(selected)}")
    print(f"blind_review_items={len(blind_rows)}")
    print(f"audit_html={audit_dir / 'index.html'}")
    print(f"answer_key={answer_path}")
    print(f"manual_labels={empty_labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
