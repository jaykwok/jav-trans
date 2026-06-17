#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402
from boundary.ja import (  # noqa: E402
    LabelRecord,
    build_negative_record,
    build_training_examples,
    write_jsonl as write_label_jsonl,
)


CANDIDATE_SCHEMA = "boundary_hardcase_candidate_from_cueqc_v1"
ENRICHED_SCHEMA = "cueqc_drop_v51_source_candidate_v1"
PREFERENCE_SEED_SCHEMA = "boundary_preference_seed_from_cueqc_drop_v1"
SUMMARY_SCHEMA = "cueqc_drop_v51_source_prep_summary_v1"

FRAME_NEGATIVE_ROUTE = "speech_boundary_frame_negative_candidate"
PREFERENCE_ROUTE = "boundary_preference_candidate"


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_display_path(path: str | Path | None) -> str:
    if not path:
        return ""
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(PROJECT_ROOT)).replace("/", "\\")
    except ValueError:
        return str(raw)


def local_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(dict(row))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def discover_latest_candidates() -> Path:
    paths = sorted(
        (PROJECT_ROOT / "agents" / "temp").glob(
            "*_boundary-hardcase-candidates-from-cueqc/cueqc_confirmed_drop_candidates.jsonl"
        )
    )
    if not paths:
        raise FileNotFoundError(
            "no CueQC hard-case candidate manifest found under agents/temp/*_boundary-hardcase-candidates-from-cueqc"
        )
    return paths[-1]


def audit_item_paths_from_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[Path]:
    roots: set[Path] = set()
    for candidate in candidates:
        paths = list(candidate.get("source_label_paths") or [])
        for evidence in candidate.get("source_evidence") or []:
            if isinstance(evidence, Mapping) and evidence.get("source_label_path"):
                paths.append(str(evidence["source_label_path"]))
        for value in paths:
            label_path = project_path(str(value))
            roots.add(label_path.parent / "cueqc_prediction_audit_items.jsonl")
    return sorted(roots)


def load_audit_items(paths: Sequence[Path]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for path in paths:
        if not path.exists():
            missing.append(repo_display_path(path))
            continue
        for row in read_jsonl(path):
            item = dict(row)
            item["_audit_items_path"] = repo_display_path(path)
            for key in (str(item.get("sample_id") or ""), str(item.get("audit_id") or "")):
                if key:
                    by_key.setdefault(key, []).append(item)
    return by_key, missing


def pick_audit_item(candidate: Mapping[str, Any], audit_items: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    keys = [str(candidate.get("sample_id") or ""), str(candidate.get("audit_id") or "")]
    preferred_dirs = {
        str(Path(project_path(path)).parent)
        for path in candidate.get("source_label_paths") or []
    }
    for key in keys:
        if not key:
            continue
        items = audit_items.get(key) or []
        if not items:
            continue
        for item in items:
            audit_path = str(project_path(str(item.get("_audit_items_path") or "")).parent)
            if audit_path in preferred_dirs:
                return item
        return items[-1]
    return None


def safe_stem(value: Any) -> str:
    raw = str(value or "sample")
    clean = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in raw)
    return clean.strip("._") or "sample"


def row_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return default


def source_audio_path(candidate: Mapping[str, Any], audit_item: Mapping[str, Any] | None) -> Path | None:
    media = audit_item.get("media") if isinstance(audit_item, Mapping) else None
    if isinstance(media, Mapping):
        for key in ("audio_path", "source_audio_path"):
            value = media.get(key)
            if value:
                return project_path(str(value))
    for key in ("source_audio_path", "audio", "audio_path"):
        value = candidate.get(key)
        if value:
            return project_path(str(value))
    return None


def enrich_candidate(candidate: Mapping[str, Any], audit_item: Mapping[str, Any] | None) -> dict[str, Any]:
    media = audit_item.get("media") if isinstance(audit_item, Mapping) else {}
    audio_path = source_audio_path(candidate, audit_item)
    enriched = {
        **dict(candidate),
        "schema": ENRICHED_SCHEMA,
        "source_candidate_schema": str(candidate.get("schema") or ""),
        "source_audio_path": repo_display_path(audio_path) if audio_path else "",
        "audit_item_found": audit_item is not None,
        "audit_items_path": str((audit_item or {}).get("_audit_items_path") or ""),
        "audit_media": dict(media or {}) if isinstance(media, Mapping) else {},
        "context_start": row_float(audit_item or {}, "context_start", row_float(candidate, "start")),
        "context_end": row_float(audit_item or {}, "context_end", row_float(candidate, "end")),
        "chunk_subtitle_cues": list((audit_item or {}).get("chunk_subtitle_cues") or []),
        "context_subtitle_cues": list((audit_item or {}).get("context_subtitle_cues") or []),
        "aligned_segments": list((audit_item or {}).get("aligned_segments") or []),
        "v51_source_status": "source_prepared_candidate",
    }
    return enriched


def slice_audio(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    start_sample = max(0, min(len(audio), int(round(start_s * sample_rate))))
    end_sample = max(0, min(len(audio), int(round(end_s * sample_rate))))
    if end_sample <= start_sample:
        return np.zeros(max(1, int(round(0.02 * sample_rate))), dtype=np.float32)
    return np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)


def with_boundary_metadata(record: LabelRecord, metadata: Mapping[str, Any]) -> LabelRecord:
    return LabelRecord(
        audio_id=record.audio_id,
        source=record.source,
        duration_s=record.duration_s,
        text=record.text,
        teacher_segments=record.teacher_segments,
        frame_hop_s=record.frame_hop_s,
        speech_frames=record.speech_frames,
        label_quality=record.label_quality,
        frame_weights=record.frame_weights,
        boundary_metadata=dict(metadata),
    )


def materialize_negative_sources(
    candidates: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    frame_hop_s: float,
    min_duration_s: float,
) -> tuple[list[LabelRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    audio_dir = output_dir / "speech_boundary_negative_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records: list[LabelRecord] = []
    manifest_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    current_path: Path | None = None
    current_audio: np.ndarray | None = None
    current_sample_rate = 0
    ordered = sorted(
        candidates,
        key=lambda row: (
            str(row.get("source_audio_path") or ""),
            float(row.get("start") or 0.0),
            str(row.get("sample_id") or ""),
        ),
    )
    for row in ordered:
        sample_id = str(row.get("sample_id") or row.get("candidate_id") or "")
        audio_value = str(row.get("source_audio_path") or "")
        if not audio_value:
            skipped.append({"sample_id": sample_id, "reason": "missing_source_audio_path"})
            continue
        source_audio = project_path(audio_value)
        if not source_audio.exists():
            skipped.append(
                {
                    "sample_id": sample_id,
                    "source_audio_path": repo_display_path(source_audio),
                    "reason": "source_audio_not_found",
                }
            )
            continue
        start = max(0.0, row_float(row, "start"))
        end = max(start, row_float(row, "end", start))
        if end <= start:
            skipped.append({"sample_id": sample_id, "reason": "non_positive_span", "start": start, "end": end})
            continue
        if source_audio != current_path:
            current_audio, current_sample_rate = load_audio_16k_mono(str(source_audio))
            current_path = source_audio
        assert current_audio is not None
        source_duration_s = len(current_audio) / current_sample_rate if current_sample_rate > 0 else 0.0
        clip_start = min(start, source_duration_s)
        clip_end = min(end, source_duration_s)
        duration_s = max(0.0, clip_end - clip_start)
        if duration_s < min_duration_s:
            skipped.append(
                {
                    "sample_id": sample_id,
                    "source_audio_path": repo_display_path(source_audio),
                    "reason": "too_short_after_clamp",
                    "duration_s": round(duration_s, 6),
                }
            )
            continue
        clip = slice_audio(current_audio, current_sample_rate, clip_start, clip_end)
        audio_id = f"cueqc_drop_neg_{safe_stem(sample_id)}"
        output_audio = audio_dir / f"{audio_id}.wav"
        sf.write(str(output_audio), clip, current_sample_rate)
        actual_duration_s = len(clip) / current_sample_rate if current_sample_rate > 0 else duration_s
        record = build_negative_record(
            audio_id=audio_id,
            source="cueqc_drop_frame_negative",
            duration_s=actual_duration_s,
            text=str(row.get("text") or ""),
            frame_hop_s=frame_hop_s,
        )
        metadata = {
            "source": "cueqc_drop_hardcase_candidate",
            "candidate_id": row.get("candidate_id"),
            "sample_id": sample_id,
            "video_id": row.get("video_id"),
            "chunk_index": row.get("chunk_index"),
            "source_audio_path": repo_display_path(source_audio),
            "source_start_s": round(clip_start, 6),
            "source_end_s": round(clip_end, 6),
            "reason_tags": list(row.get("reason_tags") or []),
            "display_prob_drop_mean": row.get("display_prob_drop_mean"),
            "route_reason": row.get("route_reason"),
        }
        records.append(with_boundary_metadata(record, metadata))
        manifest_rows.append(
            {
                "audio_id": audio_id,
                "audio": repo_display_path(output_audio),
                "duration_s": round(actual_duration_s, 6),
                "source": "cueqc_drop_frame_negative",
                "source_audio_path": repo_display_path(source_audio),
                "source_start_s": round(clip_start, 6),
                "source_end_s": round(clip_end, 6),
                "sample_id": sample_id,
                "candidate_id": str(row.get("candidate_id") or ""),
                "video_id": str(row.get("video_id") or ""),
                "chunk_index": int(row.get("chunk_index") or 0),
                "text": str(row.get("text") or ""),
                "label_quality": "negative",
                "speech_frame_count": 0,
                "frame_count": len(record.speech_frames),
            }
        )
    return records, manifest_rows, skipped


def build_preference_seed(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": PREFERENCE_SEED_SCHEMA,
        "seed_id": f"cueqc-pref-seed-{safe_stem(row.get('sample_id'))}",
        "candidate_id": str(row.get("candidate_id") or ""),
        "sample_id": str(row.get("sample_id") or ""),
        "video_id": str(row.get("video_id") or ""),
        "video_label": str(row.get("video_label") or ""),
        "chunk_index": int(row.get("chunk_index") or 0),
        "source_audio_path": str(row.get("source_audio_path") or ""),
        "start": row_float(row, "start"),
        "end": row_float(row, "end"),
        "duration_s": row_float(row, "duration_s"),
        "context_start": row_float(row, "context_start", row_float(row, "start")),
        "context_end": row_float(row, "context_end", row_float(row, "end")),
        "text": str(row.get("text") or ""),
        "text_bucket": str(row.get("text_bucket") or ""),
        "reason_tags": list(row.get("reason_tags") or []),
        "route_reason": str(row.get("route_reason") or ""),
        "display_prob_drop_mean": row.get("display_prob_drop_mean"),
        "chunk_subtitle_cues": list(row.get("chunk_subtitle_cues") or []),
        "context_subtitle_cues": list(row.get("context_subtitle_cues") or []),
        "aligned_segments": list(row.get("aligned_segments") or []),
        "next_step": "build neighbor-context boundary A/B or explicit start/end delta target before v5.1 training",
        "direct_v51_training": False,
    }


def prepare_cueqc_drop_v51_sources(
    *,
    candidates_path: Path,
    output_dir: Path,
    audit_item_paths: Sequence[Path] | None = None,
    frame_hop_s: float = 0.02,
    min_negative_duration_s: float = 0.02,
) -> dict[str, Any]:
    candidates_path = candidates_path.resolve()
    output_dir = output_dir.resolve()
    candidates = read_jsonl(candidates_path)
    for row in candidates:
        schema = str(row.get("schema") or "")
        if schema and schema != CANDIDATE_SCHEMA:
            raise ValueError(f"unsupported candidate schema {schema!r}")
    paths = list(audit_item_paths or audit_item_paths_from_candidates(candidates))
    audit_items, missing_audit_item_paths = load_audit_items(paths)

    enriched: list[dict[str, Any]] = []
    missing_item_samples: list[str] = []
    for candidate in candidates:
        item = pick_audit_item(candidate, audit_items)
        if item is None:
            missing_item_samples.append(str(candidate.get("sample_id") or ""))
        enriched.append(enrich_candidate(candidate, item))

    frame_negative_rows = [
        row for row in enriched if str(row.get("candidate_route") or "") == FRAME_NEGATIVE_ROUTE
    ]
    preference_rows = [
        row for row in enriched if str(row.get("candidate_route") or "") == PREFERENCE_ROUTE
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    negative_records, negative_manifest, negative_skipped = materialize_negative_sources(
        frame_negative_rows,
        output_dir=output_dir,
        frame_hop_s=frame_hop_s,
        min_duration_s=min_negative_duration_s,
    )
    preference_seed_rows = [build_preference_seed(row) for row in preference_rows]

    enriched_path = output_dir / "enriched_confirmed_drop_candidates.jsonl"
    preference_seed_path = output_dir / "boundary_preference_seed_candidates.jsonl"
    labels_path = output_dir / "speech_boundary_negative_labels.jsonl"
    manifest_path = output_dir / "speech_boundary_negative_manifest.json"
    skipped_path = output_dir / "speech_boundary_negative_skipped.json"
    training_manifest_path = output_dir / "speech_boundary_training_manifest.jsonl"
    training_skipped_path = output_dir / "speech_boundary_training_manifest_skipped.json"

    write_jsonl(enriched_path, enriched)
    write_jsonl(preference_seed_path, preference_seed_rows)
    write_label_jsonl(labels_path, negative_records)
    write_json(manifest_path, negative_manifest)
    write_json(skipped_path, negative_skipped)
    examples, training_skipped = build_training_examples(
        negative_records,
        manifest_audio_map={row["audio_id"]: row["audio"] for row in negative_manifest},
        audio_root=None,
        trainable_only=True,
    )
    write_jsonl(training_manifest_path, (example.__dict__ for example in examples))
    write_json(training_skipped_path, training_skipped)

    route_counts = Counter(str(row.get("candidate_route") or "") for row in enriched)
    video_counts = Counter(str(row.get("video_id") or "") for row in enriched)
    negative_video_counts = Counter(str(row.get("video_id") or "") for row in negative_manifest)
    preference_video_counts = Counter(str(row.get("video_id") or "") for row in preference_seed_rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidates_path": repo_display_path(candidates_path),
        "output_dir": repo_display_path(output_dir),
        "audit_item_paths": [repo_display_path(path) for path in paths],
        "missing_audit_item_paths": missing_audit_item_paths,
        "outputs": {
            "enriched_candidates": repo_display_path(enriched_path),
            "speech_boundary_negative_labels": repo_display_path(labels_path),
            "speech_boundary_negative_manifest": repo_display_path(manifest_path),
            "speech_boundary_negative_skipped": repo_display_path(skipped_path),
            "speech_boundary_training_manifest": repo_display_path(training_manifest_path),
            "speech_boundary_training_manifest_skipped": repo_display_path(training_skipped_path),
            "boundary_preference_seed_candidates": repo_display_path(preference_seed_path),
            "summary_json": repo_display_path(output_dir / "summary.json"),
            "summary_md": repo_display_path(output_dir / "summary.md"),
        },
        "counts": {
            "input_candidates": len(candidates),
            "enriched_candidates": len(enriched),
            "missing_audit_item_samples": len(missing_item_samples),
            "frame_negative_candidates": len(frame_negative_rows),
            "frame_negative_materialized": len(negative_records),
            "frame_negative_skipped": len(negative_skipped),
            "speech_boundary_training_examples": len(examples),
            "speech_boundary_training_skipped": len(training_skipped),
            "boundary_preference_seed_candidates": len(preference_seed_rows),
        },
        "candidate_route_counts": dict(route_counts),
        "candidate_video_counts": dict(video_counts),
        "frame_negative_video_counts": dict(negative_video_counts),
        "boundary_preference_video_counts": dict(preference_video_counts),
        "missing_audit_item_sample_ids": missing_item_samples[:50],
        "v51_boundary": {
            "direct_boundary_refiner_dataset_emitted": False,
            "speech_boundary_negative_labels_emitted": True,
            "boundary_preference_seed_emitted": True,
            "next_steps": [
                "use speech_boundary_negative_labels/manifest as a hard-negative source in a guarded SpeechBoundary-JA finetune",
                "turn boundary_preference_seed_candidates into neighbor-context A/B or explicit start/end delta labels before Boundary Refiner v5.1 training",
            ],
        },
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# CueQC Drop v5.1 Source Prep",
        "",
        f"- Candidates: `{summary['candidates_path']}`",
        f"- Output: `{summary['output_dir']}`",
        "",
        "## Counts",
        "",
        f"- Input candidates: `{counts['input_candidates']}`",
        f"- Frame-negative candidates/materialized/skipped: `{counts['frame_negative_candidates']}` / `{counts['frame_negative_materialized']}` / `{counts['frame_negative_skipped']}`",
        f"- SpeechBoundary training examples/skipped: `{counts['speech_boundary_training_examples']}` / `{counts['speech_boundary_training_skipped']}`",
        f"- Boundary preference seed candidates: `{counts['boundary_preference_seed_candidates']}`",
        f"- Missing audit item samples: `{counts['missing_audit_item_samples']}`",
        "",
        "## Outputs",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in summary["outputs"].items())
    lines.extend(
        [
            "",
            "## Boundary v5.1 Boundary",
            "",
            "- This prep emits SpeechBoundary-JA negative labels and Boundary preference seeds.",
            "- It does not emit `boundary_refiner_frame_sequence_dataset_v5`.",
            "- Preference seeds still need A/B or explicit delta labels before Boundary Refiner v5.1 training.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare trainable source artifacts from CueQC drop hard-case candidates: "
            "SpeechBoundary-JA negative labels plus Boundary preference seeds."
        )
    )
    parser.add_argument(
        "--candidates",
        default="",
        help="cueqc_confirmed_drop_candidates.jsonl. Defaults to the latest agents/temp/*_boundary-hardcase-candidates-from-cueqc manifest.",
    )
    parser.add_argument("--audit-items", nargs="*", default=None, help="Optional cueqc_prediction_audit_items.jsonl paths.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to agents/temp/YYYYMMDD_HHMMSS_boundary-v5.1-sources-from-cueqc.",
    )
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--min-negative-duration-s", type=float, default=0.02)
    args = parser.parse_args(argv)
    if args.frame_hop_s <= 0.0:
        parser.error("--frame-hop-s must be positive")
    if args.min_negative_duration_s <= 0.0:
        parser.error("--min-negative-duration-s must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidates_path = project_path(args.candidates) if args.candidates else discover_latest_candidates()
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "agents" / "temp" / f"{local_timestamp()}_boundary-v5.1-sources-from-cueqc"
    )
    summary = prepare_cueqc_drop_v51_sources(
        candidates_path=candidates_path,
        output_dir=output_dir,
        audit_item_paths=[project_path(path) for path in args.audit_items] if args.audit_items else None,
        frame_hop_s=args.frame_hop_s,
        min_negative_duration_s=args.min_negative_duration_s,
    )
    print(f"output_dir={summary['output_dir']}")
    print(f"negative_labels={summary['outputs']['speech_boundary_negative_labels']}")
    print(f"negative_manifest={summary['outputs']['speech_boundary_negative_manifest']}")
    print(f"preference_seed={summary['outputs']['boundary_preference_seed_candidates']}")
    print(
        "frame_negative={materialized}/{total} preference_seed={pref} training_examples={examples}".format(
            materialized=summary["counts"]["frame_negative_materialized"],
            total=summary["counts"]["frame_negative_candidates"],
            pref=summary["counts"]["boundary_preference_seed_candidates"],
            examples=summary["counts"]["speech_boundary_training_examples"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
