#!/usr/bin/env python3
"""Label pre-ASR clips with Grok STT and compile CTC training examples.

The provider call is resumable and content-addressed.  A preflight computes the
maximum bill from unique audio duration before any request is sent, and the
dispatcher never reserves work beyond ``--budget-usd``.  Provider-reported
``usage.cost`` is appended and flushed after every completed request so a
resumed run continues from actual spend rather than starting a fresh budget
counter.  The append-only hot path avoids repeatedly rewriting a growing
results snapshot; results are compacted into manifest order at clean exits.

Grok's word units are not used as literal frame labels.  The audited compiler
first merges tokenizer-scale gaps, ignores unresolved short islands and edge
jitter, then emits two ordinary CTC sample types: lexical crops with text, and
reliable non-word crops with an empty target.  The latter teach the existing CTC
blank; their sampling share is capped later by ``train_ctc_aligner.py``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment import (  # noqa: E402
    acoustic_text,
    minimum_ctc_frames,
    normalize_text,
)
from tools.audits.grok_stt_smoke_audit import (  # noqa: E402
    MODEL,
    _first_config_value,
    _sha256,
    call_openrouter_stt,
    compile_frame_supervision,
    normalize_response,
    read_jsonl,
    write_jsonl,
)
from tools.audits.audit_nav import audit_generated_at  # noqa: E402
from tools.omni.openai_compat import load_env_file  # noqa: E402


RESULT_SCHEMA = "grok_ctc_teacher_result_v1"
MANIFEST_SCHEMA = "grok_ctc_teacher_manifest_v1"
SUMMARY_SCHEMA = "grok_ctc_teacher_summary_v1"
DEFAULT_PRICE_PER_HOUR_USD = 0.10


def _append_jsonl_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Durably append complete JSONL records without rewriting prior results."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _source_id(row: Mapping[str, Any], position: int) -> str:
    return str(
        row.get("candidate_id")
        or row.get("sample_id")
        or row.get("audio_id")
        or f"source-{position:06d}"
    )


def _usage_cost(response: Mapping[str, Any], fallback: float) -> float:
    try:
        value = float((response.get("usage") or {}).get("cost"))
    except (TypeError, ValueError):
        return float(fallback)
    return value if value >= 0.0 else float(fallback)


def _is_transient(error: BaseException) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "transport error",
            "timed out",
            "timeout",
            "connection reset",
        )
    )


def _provider_call(
    task: Mapping[str, Any],
    *,
    api_key: str,
    base_url: str,
    model: str,
    language: str,
    timeout_s: float,
    attempts: int,
) -> dict[str, Any]:
    """Execute one unique-audio request; retry transient failures only."""

    audio_path = Path(str(task["audio_path"]))
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            raw, headers = call_openrouter_stt(
                audio_path=audio_path,
                api_key=api_key,
                base_url=base_url,
                model=model,
                language=language,
                timeout_s=timeout_s,
            )
            normalized = normalize_response(
                raw, fallback_duration_s=float(task["duration_s"])
            )
            return {
                "raw_response": raw,
                "response": normalized,
                "response_headers": headers,
                "latency_s": round(time.perf_counter() - started, 6),
                "attempts": attempt,
            }
        except Exception as error:  # noqa: BLE001
            if attempt >= attempts or not _is_transient(error):
                raise
            time.sleep(min(3.0, float(attempt)))
    raise AssertionError("provider retry loop exhausted without result")


def _split_span(start_s: float, end_s: float, maximum_s: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    cursor = float(start_s)
    while end_s - cursor > maximum_s:
        spans.append((cursor, cursor + maximum_s))
        cursor += maximum_s
    if end_s > cursor:
        spans.append((cursor, float(end_s)))
    return spans


def _teacher_acoustic_characters(
    words: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand provider units to timed pronounceable characters.

    Grok normally returns one Japanese character per unit, but multi-character
    units are legal.  Their internal boundary is not known, so it is interpolated
    only for matching the unit to canonical text; emitted crop boundaries still
    use the provider unit/island edges.
    """

    characters: list[dict[str, Any]] = []
    for unit_index, word in enumerate(words):
        text, _ = acoustic_text(normalize_text(str(word.get("text") or "")))
        if not text:
            continue
        start_s = float(word.get("start_s") or 0.0)
        end_s = float(word.get("end_s") or 0.0)
        if end_s <= start_s:
            continue
        step = (end_s - start_s) / len(text)
        for offset, char in enumerate(text):
            characters.append(
                {
                    "char": char,
                    "start_s": start_s + step * offset,
                    "end_s": start_s + step * (offset + 1),
                    "unit_index": unit_index,
                }
            )
    return characters


def _edit_alignment(
    source: str, target: str
) -> tuple[dict[int, int], dict[int, int], int]:
    """Equal and diagonal links from one deterministic Levenshtein alignment."""

    rows, columns = len(source) + 1, len(target) + 1
    distance = [[0] * columns for _ in range(rows)]
    for index in range(rows):
        distance[index][0] = index
    for index in range(columns):
        distance[0][index] = index
    for i in range(1, rows):
        for j in range(1, columns):
            substitution = distance[i - 1][j - 1] + (source[i - 1] != target[j - 1])
            distance[i][j] = min(
                substitution,
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
            )

    equal_links: dict[int, int] = {}
    diagonal_links: dict[int, int] = {}
    i, j = len(source), len(target)
    while i or j:
        if (
            i
            and j
            and source[i - 1] == target[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            equal_links[i - 1] = j - 1
            diagonal_links[i - 1] = j - 1
            i -= 1
            j -= 1
            continue
        # Prefer a substitution over a delete/insert tie.  It keeps neighbouring
        # exact matches anchored to the same local phrase instead of drifting to
        # a repeated character elsewhere in the sentence.
        if i and j and distance[i][j] == distance[i - 1][j - 1] + 1:
            diagonal_links[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            i -= 1
        elif j:
            j -= 1
        else:  # pragma: no cover - the initialized matrix always has a move
            raise AssertionError("invalid edit-distance backtrace")
    return equal_links, diagonal_links, distance[-1][-1]


_SMALL_KANA = frozenset("ゃゅょぁぃぅぇぉっゎャュョァィゥェォッヮー")


def _canonical_teacher_islands(
    characters: Sequence[Mapping[str, Any]], *, merge_gap_s: float
) -> list[dict[str, Any]]:
    """Phrase islands for canonical crops, less brittle than token-scale gaps.

    The generic frame compiler intentionally uses a tight 154 ms threshold for
    deciding safe non-word regions.  A training crop asks a different question:
    it must not split `絶対`, `注ぎ` or a small kana pair merely because Grok's
    timestamp units have a 200 ms hole.  A 350 ms phrase threshold preserves
    those words; punctuation-sized pauses remain separate.  Small kana may
    attach across up to one second because splitting `じゃ` is never a useful
    Japanese target, even when the provider timestamp is visibly fragmented.
    """

    islands: list[dict[str, Any]] = []
    for index, item in enumerate(characters):
        if not islands:
            islands.append(
                {
                    "start_s": float(item["start_s"]),
                    "end_s": float(item["end_s"]),
                    "teacher_indices": [index],
                }
            )
            continue
        previous = characters[index - 1]
        gap = float(item["start_s"]) - float(previous["end_s"])
        attach_small_kana = str(item["char"]) in _SMALL_KANA and gap <= 1.0
        if gap <= merge_gap_s or attach_small_kana:
            islands[-1]["end_s"] = float(item["end_s"])
            islands[-1]["teacher_indices"].append(index)
        else:
            islands.append(
                {
                    "start_s": float(item["start_s"]),
                    "end_s": float(item["end_s"]),
                    "teacher_indices": [index],
                }
            )
    return islands


def _compile_canonical_text_crops(
    result: Mapping[str, Any],
    *,
    context_s: float,
    maximum_cer: float,
    minimum_teacher_match_share: float,
    minimum_canonical_match_share: float,
    minimum_island_match_share: float,
    minimum_crop_chars: int,
    minimum_crop_s: float,
    max_crops_per_source: int,
    canonical_merge_gap_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use Grok only as time evidence; dataset text remains the CTC target."""

    duration_s = float(result["source_duration_s"])
    canonical_text = normalize_text(str(result.get("canonical_text") or ""))
    canonical, _ = acoustic_text(canonical_text)
    response = result.get("response") or {}
    words = list(response.get("words") or [])
    teacher_chars = _teacher_acoustic_characters(words)
    teacher = "".join(str(item["char"]) for item in teacher_chars)
    if not canonical:
        return [], {"reason": "canonical_acoustic_empty"}
    if not teacher:
        return [], {"reason": "teacher_acoustic_empty"}

    equal_links, diagonal_links, edit_distance = _edit_alignment(teacher, canonical)
    matched = len(equal_links)
    cer = edit_distance / max(1, len(canonical))
    teacher_match_share = matched / max(1, len(teacher))
    canonical_match_share = len(set(equal_links.values())) / max(1, len(canonical))
    metrics = {
        "canonical_characters": len(canonical),
        "teacher_characters": len(teacher),
        "matched_characters": matched,
        "cer": round(cer, 6),
        "teacher_match_share": round(teacher_match_share, 6),
        "canonical_match_share": round(canonical_match_share, 6),
    }
    if cer > maximum_cer:
        return [], {**metrics, "reason": "cer_above_maximum"}
    if teacher_match_share < minimum_teacher_match_share:
        return [], {**metrics, "reason": "teacher_match_below_minimum"}
    if canonical_match_share < minimum_canonical_match_share:
        return [], {**metrics, "reason": "canonical_match_below_minimum"}

    islands = _canonical_teacher_islands(
        teacher_chars, merge_gap_s=canonical_merge_gap_s
    )
    source_id = str(result["source_id"])
    group = str(result.get("source_group") or result.get("video_id") or source_id)
    common = {
        "schema": MANIFEST_SCHEMA,
        "audio": str(result["audio"]),
        "group": group,
        "video_id": group,
        "partition": str(result.get("partition") or ""),
        "source_id": source_id,
        "source_label": "galgame_canonical",
        "teacher_model": str(result.get("model") or MODEL),
        "teacher_audio_sha256": str(result["audio_sha256"]),
        "canonical_text": canonical_text,
        "canonical_acoustic_text": canonical,
    }

    candidates: list[dict[str, Any]] = []
    previous_canonical_end = 0
    for island_index, island in enumerate(islands):
        island_start = float(island["start_s"])
        island_end = float(island["end_s"])
        teacher_indices = list(island["teacher_indices"])
        equal_linked = [
            equal_links[index] for index in teacher_indices if index in equal_links
        ]
        diagonal_linked = [
            diagonal_links[index]
            for index in teacher_indices
            if index in diagonal_links
        ]
        if len(diagonal_linked) < minimum_crop_chars:
            continue
        island_match_share = len(equal_linked) / max(1, len(teacher_indices))
        if island_match_share < minimum_island_match_share:
            continue
        canonical_start = min(diagonal_linked)
        canonical_end = max(diagonal_linked) + 1
        # A global monotonic alignment should already order islands.  Refuse an
        # overlap rather than emitting the same canonical character twice.
        if canonical_start < previous_canonical_end:
            continue
        target = canonical[canonical_start:canonical_end]
        if len(target) < minimum_crop_chars:
            continue
        # Large unmatched holes mean the two equal endpoints accidentally
        # bracketed a different phrase.  At most two substitutions/insertions
        # may be carried inside one teacher island.
        if len(target) > len(diagonal_linked) + 2:
            continue

        crop_start = max(0.0, island_start - context_s)
        crop_end = min(duration_s, island_end + context_s)
        if island_index:
            previous = islands[island_index - 1]
            crop_start = max(
                crop_start,
                (float(previous["end_s"]) + island_start) / 2.0,
            )
        if island_index + 1 < len(islands):
            following = islands[island_index + 1]
            crop_end = min(
                crop_end,
                (island_end + float(following["start_s"])) / 2.0,
            )
        crop_duration = crop_end - crop_start
        if crop_duration < minimum_crop_s:
            continue
        if minimum_ctc_frames(target) > crop_duration * 13.0:
            continue
        candidates.append(
            {
                **common,
                "audio_id": f"{source_id}-canonical-{island_index:03d}",
                "text": target,
                "target_kind": "canonical_text_crop",
                "source_start_s": round(crop_start, 6),
                "source_end_s": round(crop_end, 6),
                "duration_s": round(crop_duration, 6),
                "teacher_start_s": island_start,
                "teacher_end_s": island_end,
                "canonical_start_index": canonical_start,
                "canonical_end_index": canonical_end,
                "teacher_island_match_share": round(island_match_share, 6),
                "teacher_clip_cer": round(cer, 6),
            }
        )
        previous_canonical_end = canonical_end

    if max_crops_per_source > 0 and len(candidates) > max_crops_per_source:
        candidates = sorted(
            candidates,
            key=lambda row: (
                -len(str(row["text"])),
                -float(row["teacher_island_match_share"]),
                float(row["source_start_s"]),
            ),
        )[:max_crops_per_source]
        candidates.sort(key=lambda row: float(row["source_start_s"]))
    return candidates, {
        **metrics,
        "reason": "accepted" if candidates else "no_eligible_island",
        "crops": len(candidates),
    }


def compile_ctc_manifest(
    results: Sequence[Mapping[str, Any]],
    *,
    context_s: float = 0.25,
    minimum_blank_s: float = 0.50,
    maximum_blank_s: float = 10.0,
    maximum_cer: float = 0.30,
    minimum_teacher_match_share: float = 0.70,
    minimum_canonical_match_share: float = 0.70,
    minimum_island_match_share: float = 0.60,
    minimum_crop_chars: int = 2,
    minimum_crop_s: float = 0.50,
    max_crops_per_source: int = 0,
    canonical_merge_gap_s: float = 0.35,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Turn audited Grok islands into cropped text and blank-only examples."""

    examples: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    canonical_metrics: list[dict[str, Any]] = []
    canonical_status: Counter[str] = Counter()
    for result in results:
        source_id = str(result["source_id"])
        source_label = str(result.get("source_label") or "")
        source_counts[source_label] += 1
        if str(result.get("canonical_text") or ""):
            canonical_examples, canonical_report = _compile_canonical_text_crops(
                result,
                context_s=context_s,
                maximum_cer=maximum_cer,
                minimum_teacher_match_share=minimum_teacher_match_share,
                minimum_canonical_match_share=minimum_canonical_match_share,
                minimum_island_match_share=minimum_island_match_share,
                minimum_crop_chars=minimum_crop_chars,
                minimum_crop_s=minimum_crop_s,
                max_crops_per_source=max_crops_per_source,
                canonical_merge_gap_s=canonical_merge_gap_s,
            )
            examples.extend(canonical_examples)
            reason = str(canonical_report.get("reason") or "unknown")
            canonical_status[reason] += 1
            if reason != "accepted":
                skipped[f"canonical_{reason}"] += 1
            canonical_metrics.append(canonical_report)
            continue
        duration_s = float(result["source_duration_s"])
        response = result.get("response") or {}
        compiled = compile_frame_supervision(
            list(response.get("words") or []), duration_s
        )
        islands = list(compiled["lexical_islands"])
        group = str(result.get("video_id") or source_id)
        common = {
            "schema": MANIFEST_SCHEMA,
            "audio": str(result["audio"]),
            "group": group,
            "video_id": group,
            "source_id": source_id,
            "source_label": source_label,
            "teacher_model": str(result.get("model") or MODEL),
            "teacher_audio_sha256": str(result["audio_sha256"]),
        }

        for index, island in enumerate(islands):
            text = normalize_text(str(island.get("text") or ""))
            if not text:
                skipped["lexical_island_normalized_empty"] += 1
                continue
            start_s = max(0.0, float(island["start_s"]) - context_s)
            end_s = min(duration_s, float(island["end_s"]) + context_s)
            # Context from neighbouring islands may meet at the midpoint but
            # never overlaps, so the same acoustic frame cannot carry two
            # incompatible positive transcripts.
            if index:
                previous = islands[index - 1]
                midpoint = (float(previous["end_s"]) + float(island["start_s"])) / 2.0
                start_s = max(start_s, midpoint)
            if index + 1 < len(islands):
                following = islands[index + 1]
                midpoint = (float(island["end_s"]) + float(following["start_s"])) / 2.0
                end_s = min(end_s, midpoint)
            if end_s <= start_s:
                skipped["lexical_crop_empty"] += 1
                continue
            examples.append(
                {
                    **common,
                    "audio_id": f"{source_id}-word-{index:03d}",
                    "text": text,
                    "target_kind": "text",
                    "source_start_s": round(start_s, 6),
                    "source_end_s": round(end_s, 6),
                    "duration_s": round(end_s - start_s, 6),
                    "teacher_start_s": float(island["start_s"]),
                    "teacher_end_s": float(island["end_s"]),
                }
            )

        # An entirely empty answer contradicting the upstream keep judgment is
        # not strong enough to turn a whole spoken clip into a negative.  Empty
        # drop/ambiguous clips and internal non-word regions around detected
        # words remain useful blank supervision.
        allow_blank = bool(islands) or source_label != "definite_keep"
        if not allow_blank:
            skipped["teacher_empty_definite_keep"] += 1
            continue
        blank_index = 0
        for span in compiled["frame_supervision"]:
            if span["label"] != "non_word":
                continue
            start_s = float(span["start_s"])
            end_s = float(span["end_s"])
            if end_s - start_s < minimum_blank_s:
                skipped["blank_below_minimum"] += 1
                continue
            for piece_start, piece_end in _split_span(start_s, end_s, maximum_blank_s):
                if piece_end - piece_start < minimum_blank_s:
                    skipped["blank_remainder_below_minimum"] += 1
                    continue
                examples.append(
                    {
                        **common,
                        "audio_id": f"{source_id}-blank-{blank_index:03d}",
                        "text": "",
                        "target_kind": "blank",
                        "source_start_s": round(piece_start, 6),
                        "source_end_s": round(piece_end, 6),
                        "duration_s": round(piece_end - piece_start, 6),
                    }
                )
                blank_index += 1

    kinds = Counter(str(row["target_kind"]) for row in examples)
    hours = Counter()
    for row in examples:
        hours[str(row["target_kind"])] += float(row["duration_s"])
    summary = {
        "examples": len(examples),
        "examples_by_target_kind": dict(kinds),
        "hours_by_target_kind": {
            kind: round(seconds / 3600.0, 6) for kind, seconds in sorted(hours.items())
        },
        "source_labels": dict(source_counts),
        "skipped": dict(skipped),
        "context_s": context_s,
        "minimum_blank_s": minimum_blank_s,
        "maximum_blank_s": maximum_blank_s,
    }
    if canonical_metrics:
        accepted = [row for row in canonical_metrics if row.get("reason") == "accepted"]
        summary["canonical"] = {
            "source_rows": len(canonical_metrics),
            "accepted_sources": len(accepted),
            "acceptance_share": round(len(accepted) / len(canonical_metrics), 6),
            "mean_cer_accepted": round(
                sum(float(row["cer"]) for row in accepted) / len(accepted), 6
            )
            if accepted
            else None,
            "maximum_cer": maximum_cer,
            "minimum_teacher_match_share": minimum_teacher_match_share,
            "minimum_canonical_match_share": minimum_canonical_match_share,
            "minimum_island_match_share": minimum_island_match_share,
            "minimum_crop_chars": minimum_crop_chars,
            "minimum_crop_s": minimum_crop_s,
            "max_crops_per_source": max_crops_per_source,
            "canonical_merge_gap_s": canonical_merge_gap_s,
            "status": dict(canonical_status),
        }
    return examples, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(manifest_path)
    if args.limit and args.limit < len(source_rows):
        rng = random.Random(args.seed)
        positions = sorted(rng.sample(range(len(source_rows)), args.limit))
        source_rows = [source_rows[position] for position in positions]

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, row in enumerate(source_rows):
        source_id = _source_id(row, position)
        if source_id in seen_ids:
            raise ValueError(f"duplicate source id: {source_id}")
        seen_ids.add(source_id)
        audio_path = Path(str(row["audio"]))
        if not audio_path.is_absolute():
            audio_path = PROJECT_ROOT / audio_path
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        duration_s = float(row.get("duration_s") or 0.0)
        if duration_s <= 0.0:
            raise ValueError(f"invalid duration for {source_id}: {duration_s}")
        selected.append(
            {
                "source_id": source_id,
                "audio": str(audio_path),
                "audio_sha256": _sha256(audio_path),
                "source_duration_s": duration_s,
                "source_label": str(row.get("label") or ""),
                "video_id": str(row.get("video_id") or source_id),
                "source_group": str(row.get("source_group") or row.get("video_id") or source_id),
                "source_index": int(row.get("source_index") if row.get("source_index") is not None else position),
                "partition": str(row.get("partition") or ""),
                "canonical_text": normalize_text(
                    str(row.get(args.canonical_text_field) or "")
                )
                if args.canonical_text_field
                else "",
            }
        )
    write_jsonl(output_dir / "selection.jsonl", selected)

    existing_by_id: dict[str, dict[str, Any]] = {}
    completed_by_sha: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(output_dir / "results.jsonl"):
        if row.get("schema") != RESULT_SCHEMA or row.get("model") != args.model:
            continue
        existing_by_id[str(row["source_id"])] = row
        completed_by_sha.setdefault(str(row["audio_sha256"]), row)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row["audio_sha256"]), []).append(row)
    unique_audio_s = sum(float(rows[0]["source_duration_s"]) for rows in grouped.values())
    preflight_cost = unique_audio_s / 3600.0 * float(args.price_per_hour_usd)
    if float(args.prior_spend_usd) + preflight_cost > float(args.budget_usd) + 1e-12:
        raise RuntimeError(
            "preflight budget refused: "
            f"prior ${args.prior_spend_usd:.6f} + planned ${preflight_cost:.6f} "
            f"> cap ${args.budget_usd:.6f}"
        )

    config = load_env_file(Path(args.env_file).expanduser())
    api_key = _first_config_value(
        config,
        ("OPENROUTER_API_KEY", "OMNI_API_KEY", "OPENAI_API_KEY", "API_KEY"),
    )
    if not api_key:
        raise RuntimeError("OpenRouter API key not found in the configured env file")
    base_url = _first_config_value(config, ("OPENROUTER_BASE_URL", "OMNI_BASE_URL"))

    # Actual historical cost is counted once per unique provider call.  Older
    # output rows duplicated by SHA share provider_call_sha and are not billed
    # twice here.
    actual_by_sha = {
        sha: _usage_cost(row.get("response") or {}, float(row.get("estimated_cost_usd") or 0.0))
        for sha, row in completed_by_sha.items()
    }
    spent = float(args.prior_spend_usd) + sum(actual_by_sha.values())
    errors: list[dict[str, Any]] = []
    result_lock = threading.Lock()
    results_path = output_dir / "results.jsonl"
    errors_path = output_dir / "errors.jsonl"

    def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        temporary = path.with_name(path.name + ".tmp")
        write_jsonl(temporary, rows)
        temporary.replace(path)

    def compact_snapshots() -> None:
        with result_lock:
            ordered = [existing_by_id[source["source_id"]] for source in selected if source["source_id"] in existing_by_id]
            atomic_write_jsonl(results_path, ordered)
            atomic_write_jsonl(errors_path, errors)

    def append_results(rows: Sequence[Mapping[str, Any]]) -> None:
        with result_lock:
            _append_jsonl_rows(results_path, rows)

    def append_error(row: Mapping[str, Any]) -> None:
        with result_lock:
            errors.append(dict(row))
            _append_jsonl_rows(errors_path, [row])

    tasks: list[dict[str, Any]] = []
    for sha, rows in grouped.items():
        if sha in completed_by_sha:
            template = completed_by_sha[sha]
            for source in rows:
                if source["source_id"] not in existing_by_id:
                    existing_by_id[source["source_id"]] = {
                        **template,
                        **source,
                        "source_id": source["source_id"],
                    }
            continue
        source = rows[0]
        tasks.append(
            {
                "sha": sha,
                "rows": rows,
                "audio_path": source["audio"],
                "duration_s": source["source_duration_s"],
                "estimated_cost_usd": float(source["source_duration_s"])
                / 3600.0
                * float(args.price_per_hour_usd),
            }
        )
    # Compact once before dispatch so interrupted prior runs cannot leave
    # duplicates.  During dispatch each new record is an O(1) durable append.
    compact_snapshots()

    completed_calls = len(completed_by_sha)
    task_cursor = 0
    reserved = 0.0
    inflight: dict[Future, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        while task_cursor < len(tasks) or inflight:
            while task_cursor < len(tasks) and len(inflight) < max(1, int(args.workers)):
                task = tasks[task_cursor]
                estimate = float(task["estimated_cost_usd"])
                if spent + reserved + estimate > float(args.budget_usd) + 1e-12:
                    break
                future = executor.submit(
                    _provider_call,
                    task,
                    api_key=api_key,
                    base_url=base_url,
                    model=args.model,
                    language=args.language,
                    timeout_s=float(args.timeout_s),
                    attempts=int(args.attempts),
                )
                inflight[future] = task
                reserved += estimate
                task_cursor += 1
            if not inflight:
                break
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                task = inflight.pop(future)
                estimate = float(task["estimated_cost_usd"])
                reserved -= estimate
                try:
                    provider = future.result()
                except Exception as error:  # noqa: BLE001
                    append_error(
                        {
                            "audio_sha256": task["sha"],
                            "source_ids": [row["source_id"] for row in task["rows"]],
                            "error": f"{type(error).__name__}: {error}",
                            "transient": _is_transient(error),
                        }
                    )
                    continue
                cost = _usage_cost(provider["response"], estimate)
                spent += cost
                completed_calls += 1
                completed_rows: list[dict[str, Any]] = []
                for source in task["rows"]:
                    completed_row = {
                        "schema": RESULT_SCHEMA,
                        **source,
                        "model": args.model,
                        "language_hint": args.language,
                        "estimated_cost_usd": round(estimate, 9),
                        "provider_call_sha": task["sha"],
                        "created_at": audit_generated_at(),
                        **provider,
                    }
                    existing_by_id[source["source_id"]] = completed_row
                    completed_rows.append(completed_row)
                append_results(completed_rows)
                if completed_calls % 25 == 0 or completed_calls == len(grouped):
                    print(
                        f"teacher_progress calls={completed_calls}/{len(grouped)} "
                        f"rows={len(existing_by_id)}/{len(selected)} "
                        f"spent_with_prior=${spent:.6f}",
                        flush=True,
                    )

    compact_snapshots()
    ordered_results = [
        existing_by_id[row["source_id"]]
        for row in selected
        if row["source_id"] in existing_by_id
    ]
    examples, compile_summary = compile_ctc_manifest(
        ordered_results,
        context_s=float(args.context_s),
        minimum_blank_s=float(args.minimum_blank_s),
        maximum_blank_s=float(args.maximum_blank_s),
        maximum_cer=float(args.maximum_cer),
        minimum_teacher_match_share=float(args.minimum_teacher_match_share),
        minimum_canonical_match_share=float(args.minimum_canonical_match_share),
        minimum_island_match_share=float(args.minimum_island_match_share),
        minimum_crop_chars=int(args.minimum_crop_chars),
        minimum_crop_s=float(args.minimum_crop_s),
        max_crops_per_source=int(args.max_crops_per_source),
        canonical_merge_gap_s=float(args.canonical_merge_gap_s),
    )
    write_jsonl(output_dir / "ctc_manifest.jsonl", examples)
    accepted_source_ids = {
        str(row["source_id"])
        for row in examples
        if str(row.get("target_kind") or "") == "canonical_text_crop"
    }
    full_examples = [
        {
            "schema": "galgame_ctc_teacher_full_manifest_v1",
            "audio_id": f"{row['source_id']}-full",
            "audio": row["audio"],
            "text": row["canonical_text"],
            "target_kind": "canonical_full_text",
            "source_id": row["source_id"],
            "source_group": row["source_group"],
            "group": row["source_group"],
            "partition": row["partition"],
            "duration_s": row["source_duration_s"],
        }
        for row in selected
        if row["source_id"] in accepted_source_ids and row.get("canonical_text")
    ]
    write_jsonl(output_dir / "full_manifest.jsonl", full_examples)
    current_actual = sum(
        _usage_cost(row.get("response") or {}, float(row.get("estimated_cost_usd") or 0.0))
        for row in {
            str(result["provider_call_sha"]): result for result in ordered_results
        }.values()
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "generated_at": audit_generated_at(),
        "model": args.model,
        "manifest": str(manifest_path),
        "selected_rows": len(selected),
        "unique_audio": len(grouped),
        "completed_rows": len(ordered_results),
        "failed_unique_audio": len(errors),
        "unique_audio_hours": round(unique_audio_s / 3600.0, 6),
        "price_per_hour_usd": float(args.price_per_hour_usd),
        "preflight_cost_usd": round(preflight_cost, 9),
        "provider_actual_cost_usd": round(current_actual, 9),
        "prior_spend_usd": float(args.prior_spend_usd),
        "cost_with_prior_usd": round(float(args.prior_spend_usd) + current_actual, 9),
        "budget_usd": float(args.budget_usd),
        "budget_remaining_usd": round(
            float(args.budget_usd) - float(args.prior_spend_usd) - current_actual, 9
        ),
        "compiler": compile_summary,
        "results": "results.jsonl",
        "ctc_manifest": "ctc_manifest.jsonl",
        "full_manifest": "full_manifest.jsonl",
        "accepted_full_rows": len(full_examples),
        "errors": "errors.jsonl",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-file", default="~/.config/omni/openrouter")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--price-per-hour-usd", type=float, default=DEFAULT_PRICE_PER_HOUR_USD)
    parser.add_argument("--budget-usd", type=float, default=0.50)
    parser.add_argument("--prior-spend-usd", type=float, default=0.0)
    parser.add_argument("--context-s", type=float, default=0.25)
    parser.add_argument("--minimum-blank-s", type=float, default=0.50)
    parser.add_argument("--maximum-blank-s", type=float, default=10.0)
    parser.add_argument(
        "--canonical-text-field",
        default="",
        help="manifest field holding trusted dataset text; Grok is then used only for timing",
    )
    parser.add_argument("--maximum-cer", type=float, default=0.30)
    parser.add_argument("--minimum-teacher-match-share", type=float, default=0.70)
    parser.add_argument("--minimum-canonical-match-share", type=float, default=0.70)
    parser.add_argument("--minimum-island-match-share", type=float, default=0.60)
    parser.add_argument("--minimum-crop-chars", type=int, default=2)
    parser.add_argument("--minimum-crop-s", type=float, default=0.50)
    parser.add_argument("--max-crops-per-source", type=int, default=0)
    parser.add_argument("--canonical-merge-gap-s", type=float, default=0.35)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
