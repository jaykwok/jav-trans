#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.ja import (  # noqa: E402
    LabelRecord,
    TrainingExample,
    build_supervised_record,
    write_jsonl as write_label_jsonl,
    write_training_manifest,
)


SUMMARY_SCHEMA = "speech_boundary_positive_anchor_replay_v1"
DEFAULT_SOURCE_SPECS = (
    "anime_nsfw=55=datasets/train/boundary-sources/japanese-anime-speech-v2-nsfw-60k/hf_audio_manifest.json",
    "anime_sfw=20=datasets/train/boundary-sources/japanese-anime-speech-v2-sfw-40k/hf_audio_manifest.json",
    "galgame=25=datasets/train/boundary-sources/galgame-asr-100k-ogg/manifest.jsonl",
)


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


def safe_stem(value: Any) -> str:
    raw = str(value or "sample")
    clean = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in raw)
    return clean.strip("._") or "sample"


def read_manifest(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"manifest must be a JSON list or JSONL: {path}")
        rows = payload
    else:
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"manifest JSONL row must be an object: {path}:{line_number}")
            rows.append(row)
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def row_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return default


def parse_source_spec(spec: str) -> tuple[str, float, Path]:
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise ValueError("source spec must be name=weight=manifest_path")
    name = parts[0].strip()
    if not name:
        raise ValueError("source spec name must not be empty")
    weight = float(parts[1])
    if weight <= 0.0:
        raise ValueError("source spec weight must be positive")
    path = project_path(parts[2])
    if not path.exists():
        raise FileNotFoundError(path)
    return name, weight, path


def candidate_rows(
    *,
    rows: Iterable[Mapping[str, Any]],
    source_name: str,
    source_manifest: Path,
    min_duration_s: float,
    max_duration_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("error"):
            skipped.append({"index": index, "source_group": source_name, "reason": "source_error"})
            continue
        audio_value = row.get("audio")
        if not audio_value:
            skipped.append({"index": index, "source_group": source_name, "reason": "missing_audio"})
            continue
        audio_path = project_path(str(audio_value))
        duration_s = row_float(row, "duration_s", 0.0)
        if duration_s < min_duration_s:
            skipped.append(
                {
                    "index": index,
                    "source_group": source_name,
                    "reason": "too_short",
                    "duration_s": duration_s,
                }
            )
            continue
        if max_duration_s > 0.0 and duration_s > max_duration_s:
            skipped.append(
                {
                    "index": index,
                    "source_group": source_name,
                    "reason": "too_long",
                    "duration_s": duration_s,
                }
            )
            continue
        item = dict(row)
        item["_source_group"] = source_name
        item["_source_manifest"] = repo_display_path(source_manifest)
        item["_audio_path"] = repo_display_path(audio_path)
        item["_duration_s"] = duration_s
        candidates.append(item)
    return candidates, skipped


def allocate_counts(groups: Sequence[tuple[str, float, list[dict[str, Any]], Path]], count: int) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    total_weight = sum(weight for _, weight, _, _ in groups)
    raw_counts = [(count * weight / total_weight) for _, weight, _, _ in groups]
    counts = [int(value) for value in raw_counts]
    remaining = count - sum(counts)
    order = sorted(
        range(len(raw_counts)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def with_anchor_metadata(record: LabelRecord, metadata: Mapping[str, Any]) -> LabelRecord:
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


def select_existing_rows(
    *,
    rows: Sequence[dict[str, Any]],
    source_name: str,
    target_count: int,
    rng: random.Random,
    sample_with_replacement: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if target_count <= 0:
        return selected, skipped
    if sample_with_replacement:
        max_attempts = max(target_count * 20, target_count + 100)
        attempts = 0
        while len(selected) < target_count and attempts < max_attempts:
            attempts += 1
            row = dict(rng.choice(list(rows)))
            audio_path = project_path(str(row["_audio_path"]))
            if audio_path.exists():
                selected.append(row)
            else:
                skipped.append(
                    {
                        "source_group": source_name,
                        "reason": "selected_audio_not_found",
                        "audio": repo_display_path(audio_path),
                    }
                )
        if len(selected) < target_count:
            raise ValueError(
                f"source group {source_name} only resolved {len(selected)}/{target_count} selected audio rows"
            )
        return selected, skipped

    order = list(range(len(rows)))
    rng.shuffle(order)
    for index in order:
        row = dict(rows[index])
        audio_path = project_path(str(row["_audio_path"]))
        if audio_path.exists():
            selected.append(row)
            if len(selected) >= target_count:
                break
        else:
            skipped.append(
                {
                    "source_group": source_name,
                    "reason": "selected_audio_not_found",
                    "audio": repo_display_path(audio_path),
                    "candidate_index": index,
                }
            )
    if len(selected) < target_count:
        raise ValueError(
            f"source group {source_name} only resolved {len(selected)}/{target_count} selected audio rows"
        )
    return selected, skipped


def build_positive_anchor_replay(
    *,
    source_specs: Sequence[str],
    output_dir: Path,
    count: int = 1500,
    seed: int = 250617,
    min_duration_s: float = 0.2,
    max_duration_s: float = 12.0,
    frame_hop_s: float = 0.02,
    source_prefix: str = "speech_boundary_anchor_positive",
    sample_with_replacement: bool = False,
) -> dict[str, Any]:
    rng = random.Random(seed)
    parsed_sources: list[tuple[str, float, list[dict[str, Any]], Path]] = []
    skipped_rows: list[dict[str, Any]] = []
    source_input_counts: dict[str, int] = {}
    source_candidate_counts: dict[str, int] = {}
    for spec in source_specs:
        name, weight, path = parse_source_spec(spec)
        raw_rows = read_manifest(path)
        rows, skipped = candidate_rows(
            rows=raw_rows,
            source_name=name,
            source_manifest=path,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
        )
        source_input_counts[name] = len(raw_rows)
        source_candidate_counts[name] = len(rows)
        skipped_rows.extend(skipped)
        if not rows:
            raise ValueError(f"no valid source rows for {name}: {repo_display_path(path)}")
        parsed_sources.append((name, weight, rows, path))

    target_counts = allocate_counts(parsed_sources, count)
    records: list[LabelRecord] = []
    manifest_rows: list[dict[str, Any]] = []
    source_group_counts: Counter[str] = Counter()
    source_name_counts: Counter[str] = Counter()
    output_index = 0
    for (name, _, rows, path), target_count in zip(parsed_sources, target_counts, strict=True):
        if target_count > len(rows) and not sample_with_replacement:
            raise ValueError(
                f"source group {name} needs {target_count} rows but only {len(rows)} are available; "
                "use --sample-with-replacement or lower --count"
            )
        selected, selected_skipped = select_existing_rows(
            rows=rows,
            source_name=name,
            target_count=target_count,
            rng=rng,
            sample_with_replacement=sample_with_replacement,
        )
        skipped_rows.extend(selected_skipped)
        for local_index, row in enumerate(selected):
            source_audio_id = str(row.get("audio_id") or Path(str(row.get("audio") or "")).stem)
            audio_id = f"anchor_{safe_stem(name)}_{output_index:06d}_{safe_stem(source_audio_id)[:48]}"
            duration_s = float(row["_duration_s"])
            source = f"{source_prefix}:{name}"
            text = str(row.get("text") or "")
            record = build_supervised_record(
                audio_id=audio_id,
                source=source,
                duration_s=duration_s,
                text=text,
                speech_segments=[{"start": 0.0, "end": duration_s}],
                frame_hop_s=frame_hop_s,
            )
            metadata = {
                "source": "positive_anchor_replay",
                "source_group": name,
                "source_manifest": repo_display_path(path),
                "source_audio_id": source_audio_id,
                "source_audio_path": str(row["_audio_path"]),
                "source_input": str(row.get("input") or ""),
                "source_sample_index": local_index,
                "source_weight": next(weight for group_name, weight, _, _ in parsed_sources if group_name == name),
            }
            records.append(with_anchor_metadata(record, metadata))
            manifest_rows.append(
                {
                    "audio_id": audio_id,
                    "audio": str(row["_audio_path"]),
                    "duration_s": round(duration_s, 6),
                    "source": source,
                    "source_group": name,
                    "source_manifest": repo_display_path(path),
                    "source_audio_id": source_audio_id,
                    "source_input": str(row.get("input") or ""),
                    "text": text,
                    "label_quality": record.label_quality,
                    "speech_frame_count": sum(int(value) for value in record.speech_frames),
                    "frame_count": len(record.speech_frames),
                }
            )
            source_group_counts[name] += 1
            source_name_counts[str(row.get("source") or "")] += 1
            output_index += 1

    # Keep deterministic source allocation but shuffle final row order for training replay.
    order = list(range(len(records)))
    rng.shuffle(order)
    records = [records[index] for index in order]
    manifest_rows = [manifest_rows[index] for index in order]

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "positive_anchor_labels.jsonl"
    manifest_path = output_dir / "positive_anchor_manifest.json"
    training_manifest_path = output_dir / "positive_anchor_training_manifest.jsonl"
    skipped_path = output_dir / "positive_anchor_skipped.json"
    training_skipped_path = output_dir / "positive_anchor_training_manifest_skipped.json"
    summary_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"

    write_label_jsonl(labels_path, records)
    write_json(manifest_path, manifest_rows)
    write_json(skipped_path, skipped_rows)
    examples = [
        TrainingExample(
            audio_id=record.audio_id,
            source=record.source,
            label_quality=record.label_quality,
            duration_s=record.duration_s,
            frame_hop_s=record.frame_hop_s,
            audio_path=str(manifest_row["audio"]),
            label_index=index,
            speech_frame_count=sum(int(value) for value in record.speech_frames),
            frame_count=len(record.speech_frames),
        )
        for index, (record, manifest_row) in enumerate(zip(records, manifest_rows, strict=True))
    ]
    training_skipped: list[dict[str, Any]] = []
    write_training_manifest(path=training_manifest_path, examples=examples)
    write_json(training_skipped_path, training_skipped)
    total_frames = sum(len(record.speech_frames) for record in records)
    speech_frames = sum(sum(int(value) for value in record.speech_frames) for record in records)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": repo_display_path(output_dir),
        "source_specs": list(source_specs),
        "source_input_counts": source_input_counts,
        "source_candidate_counts": source_candidate_counts,
        "count": len(records),
        "seed": seed,
        "source_group_counts": dict(source_group_counts),
        "source_counts": dict(source_name_counts),
        "duration_s_total": round(sum(float(row["duration_s"]) for row in manifest_rows), 6),
        "frame_count": total_frames,
        "speech_frame_count": speech_frames,
        "speech_frame_ratio": speech_frames / total_frames if total_frames else 0.0,
        "training_examples": len(examples),
        "training_skipped": len(training_skipped),
        "source_rows_skipped": len(skipped_rows),
        "filters": {
            "min_duration_s": min_duration_s,
            "max_duration_s": max_duration_s,
            "frame_hop_s": frame_hop_s,
        },
        "outputs": {
            "positive_anchor_labels": repo_display_path(labels_path),
            "positive_anchor_manifest": repo_display_path(manifest_path),
            "positive_anchor_training_manifest": repo_display_path(training_manifest_path),
            "positive_anchor_skipped": repo_display_path(skipped_path),
            "positive_anchor_training_manifest_skipped": repo_display_path(training_skipped_path),
            "summary_json": repo_display_path(summary_path),
            "summary_md": repo_display_path(summary_md_path),
        },
    }
    write_json(summary_path, summary)
    summary_md_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# SpeechBoundary Positive Anchor Replay",
        "",
        f"- Output: `{summary['output_dir']}`",
        f"- Anchors: `{summary['count']}`",
        f"- Training examples: `{summary['training_examples']}`",
        f"- Source groups: `{summary['source_group_counts']}`",
        f"- Speech frame ratio: `{summary['speech_frame_ratio']}`",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build SpeechBoundary-JA positive/synthetic anchor replay labels from anime and galgame "
            "source manifests. Defaults use boosted NSFW weighting: nsfw=55, sfw=20, galgame=25."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="name=weight=manifest_path. Repeatable. Defaults to anime_nsfw=55, anime_sfw=20, galgame=25.",
    )
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=250617)
    parser.add_argument("--min-duration-s", type=float, default=0.2)
    parser.add_argument("--max-duration-s", type=float, default=12.0)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--source-prefix", default="speech_boundary_anchor_positive")
    parser.add_argument("--sample-with-replacement", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to agents/temp/YYYYMMDD_HHMMSS_speech-boundary-positive-anchor-replay.",
    )
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.min_duration_s <= 0.0:
        parser.error("--min-duration-s must be positive")
    if args.max_duration_s < 0.0:
        parser.error("--max-duration-s must be non-negative")
    if args.max_duration_s and args.max_duration_s < args.min_duration_s:
        parser.error("--max-duration-s must be >= --min-duration-s")
    if args.frame_hop_s <= 0.0:
        parser.error("--frame-hop-s must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "agents" / "temp" / f"{local_timestamp()}_speech-boundary-positive-anchor-replay"
    )
    summary = build_positive_anchor_replay(
        source_specs=args.source or DEFAULT_SOURCE_SPECS,
        output_dir=output_dir,
        count=args.count,
        seed=args.seed,
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
        frame_hop_s=args.frame_hop_s,
        source_prefix=args.source_prefix,
        sample_with_replacement=args.sample_with_replacement,
    )
    print(f"output_dir={summary['output_dir']}")
    print(f"labels={summary['outputs']['positive_anchor_labels']}")
    print(f"manifest={summary['outputs']['positive_anchor_manifest']}")
    print(f"training_manifest={summary['outputs']['positive_anchor_training_manifest']}")
    print(
        "anchors={count} training_examples={examples} groups={groups}".format(
            count=summary["count"],
            examples=summary["training_examples"],
            groups=json.dumps(summary["source_group_counts"], ensure_ascii=False, sort_keys=True),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
