#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CLUSTER_LABEL_SCHEMA = "cueqc_cluster_label_v1"
CANDIDATE_SCHEMA = "speech_boundary_hard_negative_candidate_from_cueqc_v1"
SUMMARY_SCHEMA = "cueqc_cluster_seed_drop_hardcase_export_summary_v1"

FRAME_NEGATIVE_ROUTE = "speech_boundary_frame_negative_candidate"
DISPLAY_DROP = "drop"
SEED_USE = "use_seed"


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


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be an object: {path}")
    return dict(payload)


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


def discover_latest_cluster_labels() -> Path:
    paths = sorted((PROJECT_ROOT / "agents" / "audits").glob("*/cueqc_cluster_labels.jsonl"))
    if not paths:
        raise FileNotFoundError("no cueqc_cluster_labels.jsonl found under agents/audits/*")
    return paths[-1]


def discover_clusters_from_audit_summary(cluster_labels_path: Path) -> Path:
    summary_path = cluster_labels_path.parent / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"--clusters was not provided and audit summary is missing: {repo_display_path(summary_path)}"
        )
    summary = read_json(summary_path)
    value = summary.get("source_clusters")
    if not value:
        raise ValueError(f"audit summary has no source_clusters: {repo_display_path(summary_path)}")
    path = project_path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"source_clusters not found: {repo_display_path(path)}")
    return path


def row_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def text_observation_bucket(row: Mapping[str, Any]) -> str:
    text = str(row.get("text") or "").strip()
    features = row.get("text_features")
    char_count = 0
    if isinstance(features, Mapping):
        char_count = int(row_float(features, "char_count", 0.0))
    if not text or char_count <= 0:
        return "empty_or_punctuation"
    if char_count <= 4:
        return "short_text"
    if char_count <= 16:
        return "medium_text"
    return "long_text"


def source_audio_path(row: Mapping[str, Any]) -> str:
    value = row.get("source_audio_path")
    if value:
        return repo_display_path(project_path(str(value)))
    audio = row.get("audio")
    if isinstance(audio, Mapping):
        value = audio.get("path")
        if value:
            return repo_display_path(project_path(str(value)))
    value = row.get("audio_path")
    if value:
        return repo_display_path(project_path(str(value)))
    return ""


def label_text_present_count(row: Mapping[str, Any]) -> int:
    counts = row.get("text_observation_counts")
    if not isinstance(counts, Mapping):
        return 0
    try:
        return int(counts.get("text_present") or 0)
    except (TypeError, ValueError):
        return 0


def cluster_seed_drop_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_text_present: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    labels: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        schema = str(row.get("schema") or "")
        if schema and schema != CLUSTER_LABEL_SCHEMA:
            raise ValueError(f"unsupported cluster label schema {schema!r}")
        cluster_id = str(row.get("cluster_id") or "").strip()
        seed_action = str(row.get("seed_action") or "").strip()
        decision = str(row.get("display_decision") or "").strip()
        if not cluster_id:
            counts["missing_cluster_id"] += 1
            excluded.append({"reason": "missing_cluster_id"})
            continue
        if seed_action != SEED_USE:
            counts[f"excluded_seed_action:{seed_action or 'empty'}"] += 1
            excluded.append({"cluster_id": cluster_id, "reason": "seed_action_not_use_seed", "seed_action": seed_action})
            continue
        if row.get("training_label_included") is False:
            counts["excluded_training_label_false"] += 1
            excluded.append({"cluster_id": cluster_id, "reason": "training_label_included_false"})
            continue
        if decision != DISPLAY_DROP:
            counts[f"excluded_display:{decision or 'empty'}"] += 1
            excluded.append({"cluster_id": cluster_id, "reason": "display_decision_not_drop", "display_decision": decision})
            continue
        text_present = label_text_present_count(row)
        if text_present > 0 and not allow_text_present:
            counts["excluded_text_present_cluster"] += 1
            excluded.append({"cluster_id": cluster_id, "reason": "text_present_cluster", "text_present": text_present})
            continue
        labels[cluster_id] = dict(row)
        counts["seed_drop_clusters"] += 1
    return labels, excluded, counts


def build_candidate(
    *,
    cluster_row: Mapping[str, Any],
    label_row: Mapping[str, Any],
    cluster_labels_path: Path,
    clusters_path: Path,
) -> dict[str, Any]:
    sample_id = str(cluster_row.get("sample_id") or "").strip()
    start = row_float(cluster_row, "start")
    end = row_float(cluster_row, "end", start)
    duration_s = max(0.0, end - start)
    cluster_id = str(cluster_row.get("cluster_id") or label_row.get("cluster_id") or "")
    source_audio = source_audio_path(cluster_row)
    confidence = row_float(label_row, "confidence_avg", row_float(cluster_row, "cluster_confidence", 1.0))
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_id": f"cueqc-clusterseed-drop-hardcase-{sample_id}",
        "source": "cueqc_cluster_seed_drop_audit",
        "source_label_paths": [repo_display_path(cluster_labels_path)],
        "source_cluster_path": repo_display_path(clusters_path),
        "source_label_count": 1,
        "source_evidence": [
            {
                "source_label_path": repo_display_path(cluster_labels_path),
                "cluster_id": cluster_id,
                "seed_action": str(label_row.get("seed_action") or ""),
                "display_decision": str(label_row.get("display_decision") or ""),
                "training_label_included": label_row.get("training_label_included", True),
                "updated_at": str(label_row.get("updated_at") or ""),
                "cluster_count": int(row_float(label_row, "count", 0.0)),
                "text_observation_counts": dict(label_row.get("text_observation_counts") or {}),
            }
        ],
        "sample_id": sample_id,
        "audit_id": sample_id,
        "video_id": str(cluster_row.get("video_id") or ""),
        "video_label": str(cluster_row.get("video_label") or ""),
        "chunk_index": int(row_float(cluster_row, "chunk_index", 0.0)),
        "start": round(start, 6),
        "end": round(end, 6),
        "duration_s": round(duration_s, 6),
        "duration_bucket": duration_bucket(duration_s),
        "text": str(cluster_row.get("text") or ""),
        "raw_text": str(cluster_row.get("raw_text") or ""),
        "text_observation_bucket": text_observation_bucket(cluster_row),
        "manual_decision": "cluster_seed_drop",
        "reason_tags": ["cluster_seed_drop", "empty_text_cluster"],
        "notes": str(label_row.get("notes") or ""),
        "display_prob_drop_min": round(confidence, 6),
        "display_prob_drop_max": round(confidence, 6),
        "display_prob_drop_mean": round(confidence, 6),
        "display_prob_keep_mean": 0.0,
        "candidate_route": FRAME_NEGATIVE_ROUTE,
        "route_reason": (
            "cueqc cluster seed_action=use_seed and display_decision=drop; "
            "mixed/skip and text-present clusters abstain by default"
        ),
        "hard_negative_status": "candidate_requires_audio_materialization",
        "required_conversion": "convert to SpeechBoundary-JA frame-negative labels",
        "cluster_id": cluster_id,
        "cluster_confidence": row_float(cluster_row, "cluster_confidence", confidence),
        "source_audio_path": source_audio,
        "audio_id": str(cluster_row.get("audio_id") or ""),
    }


def duration_bucket(duration_s: float) -> str:
    if duration_s < 0.5:
        return "<0.5s"
    if duration_s < 1.0:
        return "0.5-1s"
    if duration_s < 2.0:
        return "1-2s"
    if duration_s < 4.0:
        return "2-4s"
    return ">=4s"


def export_cueqc_cluster_seed_hardcases(
    *,
    cluster_labels_path: Path,
    clusters_path: Path,
    output_dir: Path,
    allow_text_present: bool = False,
    max_per_cluster: int = 0,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    cluster_labels_path = cluster_labels_path.resolve()
    clusters_path = clusters_path.resolve()
    output_dir = output_dir.resolve()
    if not cluster_labels_path.exists():
        raise FileNotFoundError(f"missing cluster labels: {repo_display_path(cluster_labels_path)}")
    if not clusters_path.exists():
        raise FileNotFoundError(f"missing clusters: {repo_display_path(clusters_path)}")

    label_rows = read_jsonl(cluster_labels_path)
    cluster_rows = read_jsonl(clusters_path)
    seed_labels, excluded_labels, label_counts = cluster_seed_drop_labels(
        label_rows,
        allow_text_present=allow_text_present,
    )

    per_cluster_seen: defaultdict[str, int] = defaultdict(int)
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in cluster_rows:
        sample_id = str(row.get("sample_id") or "").strip()
        cluster_id = str(row.get("cluster_id") or "").strip()
        if not sample_id:
            skipped.append({"reason": "missing_sample_id", "cluster_id": cluster_id})
            continue
        label = seed_labels.get(cluster_id)
        if label is None:
            continue
        if max_per_cluster > 0 and per_cluster_seen[cluster_id] >= max_per_cluster:
            skipped.append({"sample_id": sample_id, "cluster_id": cluster_id, "reason": "max_per_cluster"})
            continue
        candidate = build_candidate(
            cluster_row=row,
            label_row=label,
            cluster_labels_path=cluster_labels_path,
            clusters_path=clusters_path,
        )
        if not candidate["source_audio_path"]:
            skipped.append({"sample_id": sample_id, "cluster_id": cluster_id, "reason": "missing_source_audio_path"})
            continue
        candidates.append(candidate)
        per_cluster_seen[cluster_id] += 1

    candidates.sort(
        key=lambda row: (
            str(row.get("video_id") or ""),
            float(row.get("start") or 0.0),
            str(row.get("sample_id") or ""),
        )
    )
    if require_nonempty and not candidates:
        raise ValueError("no cluster seed drop candidates were exported")

    candidates_path = output_dir / "cueqc_confirmed_drop_candidates.jsonl"
    skipped_path = output_dir / "cueqc_cluster_seed_drop_skipped.json"
    summary_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    write_jsonl(candidates_path, candidates)
    write_json(skipped_path, {"labels": excluded_labels, "clusters": skipped})

    route_counts = Counter(str(row.get("candidate_route") or "") for row in candidates)
    video_counts = Counter(str(row.get("video_id") or "") for row in candidates)
    cluster_candidate_counts = Counter(str(row.get("cluster_id") or "") for row in candidates)
    duration_counts = Counter(str(row.get("duration_bucket") or "") for row in candidates)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cluster_labels_path": repo_display_path(cluster_labels_path),
        "clusters_path": repo_display_path(clusters_path),
        "output_dir": repo_display_path(output_dir),
        "allow_text_present": bool(allow_text_present),
        "max_per_cluster": int(max_per_cluster),
        "outputs": {
            "confirmed_drop_candidates": repo_display_path(candidates_path),
            "skipped": repo_display_path(skipped_path),
            "summary_json": repo_display_path(summary_path),
            "summary_md": repo_display_path(summary_md_path),
        },
        "counts": {
            "cluster_label_rows": len(label_rows),
            "cluster_rows": len(cluster_rows),
            "seed_drop_clusters": len(seed_labels),
            "exported_candidates": len(candidates),
            "skipped_clusters": len(skipped),
            "excluded_label_rows": len(excluded_labels),
        },
        "label_filter_counts": dict(label_counts),
        "candidate_route_counts": dict(route_counts),
        "candidate_video_counts": dict(video_counts),
        "candidate_cluster_counts": dict(cluster_candidate_counts),
        "candidate_duration_bucket_counts": dict(duration_counts),
        "speech_boundary_hard_negative": {
            "frame_negative_candidates_emitted": True,
            "direct_boundary_refiner_dataset_emitted": False,
            "policy": (
                "Only high-precision CueQC cluster seed drops are exported. "
                "mixed_skip/skip and text-present clusters abstain by default."
            ),
            "next_conversion": [
                "run tools.boundary.prepare_cueqc_drop_hard_negative_sources",
                "pass speech_boundary_negative_manifest.json to tools.boundary.ja.build_scorer_v5_native_dataset",
            ],
        },
    }
    write_json(summary_path, summary)
    summary_md_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# CueQC Cluster Seed Drop Hard-Case Export",
        "",
        f"- Cluster labels: `{summary['cluster_labels_path']}`",
        f"- Clusters: `{summary['clusters_path']}`",
        f"- Output: `{summary['output_dir']}`",
        f"- Allow text-present clusters: `{summary['allow_text_present']}`",
        "",
        "## Counts",
        "",
        f"- Cluster label rows: `{counts['cluster_label_rows']}`",
        f"- Cluster rows: `{counts['cluster_rows']}`",
        f"- Seed drop clusters: `{counts['seed_drop_clusters']}`",
        f"- Exported candidates: `{counts['exported_candidates']}`",
        f"- Excluded label rows: `{counts['excluded_label_rows']}`",
        "",
        "## Outputs",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in summary["outputs"].items())
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Emits SpeechBoundary-JA hard-negative candidates only.",
            "- It does not emit Boundary Refiner rows.",
            "- `mixed_skip`, `skip`, disabled training labels, keep labels, and text-present clusters abstain by default.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export high-precision CueQC cluster seed drop labels as SpeechBoundary-JA hard-negative "
            "candidates. Default policy excludes mixed/skip and text-present clusters."
        )
    )
    parser.add_argument(
        "--cluster-labels",
        default="",
        help="cueqc_cluster_labels.jsonl. Defaults to latest agents/audits/*/cueqc_cluster_labels.jsonl.",
    )
    parser.add_argument(
        "--clusters",
        default="",
        help="cueqc_clusters.jsonl. Defaults to source_clusters in the audit summary next to --cluster-labels.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to agents/temp/YYYYMMDD_HHMMSS_speech-boundary-hard-negative-candidates-from-cueqc-cluster-seed.",
    )
    parser.add_argument("--allow-text-present", action="store_true")
    parser.add_argument("--max-per-cluster", type=int, default=0)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args(argv)
    if args.max_per_cluster < 0:
        parser.error("--max-per-cluster must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cluster_labels_path = project_path(args.cluster_labels) if args.cluster_labels else discover_latest_cluster_labels()
    clusters_path = project_path(args.clusters) if args.clusters else discover_clusters_from_audit_summary(cluster_labels_path)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT
        / "agents"
        / "temp"
        / f"{local_timestamp()}_speech-boundary-hard-negative-candidates-from-cueqc-cluster-seed"
    )
    summary = export_cueqc_cluster_seed_hardcases(
        cluster_labels_path=cluster_labels_path,
        clusters_path=clusters_path,
        output_dir=output_dir,
        allow_text_present=args.allow_text_present,
        max_per_cluster=args.max_per_cluster,
        require_nonempty=not args.allow_empty,
    )
    print(f"output_dir={summary['output_dir']}")
    print(f"candidates={summary['outputs']['confirmed_drop_candidates']}")
    print(
        "seed_drop_clusters={clusters} candidates={candidates}".format(
            clusters=summary["counts"]["seed_drop_clusters"],
            candidates=summary["counts"]["exported_candidates"],
        )
    )
    print(f"label_filter_counts={json.dumps(summary['label_filter_counts'], ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
