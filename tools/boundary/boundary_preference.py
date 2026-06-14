from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CANDIDATE_SCHEMA = "boundary_preference_candidate_v1"
BLIND_ITEM_SCHEMA = "boundary_preference_blind_item_v1"
ANSWER_KEY_SCHEMA = "boundary_preference_answer_key_v1"
LABEL_SCHEMA = "boundary_preference_label_v1"
SUMMARY_SCHEMA = "boundary_preference_summary_v1"

PRIMARY_LABELS = {"a_better", "b_better", "tie", "both_bad", "uncertain"}
USABLE_LABELS = {"a_better", "b_better", "tie", "both_bad"}
DECISIVE_LABELS = {"a_better", "b_better"}
AXES = ("right.start", "left.end")
OFFSETS_MS = (-160, -80, 80, 160)

_PUNCT_RE = re.compile(r"[\s\u3000、。！？!?.,，．…・「」『』（）()【】\[\]~〜ー]+")
_NONLEXICAL_RE = re.compile(
    r"^[\s\u3000、。！？!?.,，．…・「」『』（）()【】\[\]~〜ー"
    r"ぁ-ぉゃゅょっァ-ォャュョッ"
    r"あいうえおアイウエオ"
    r"んンっッはぁハァふぅフゥへぇヘェほぉホォ"
    r"うぅウゥえぇエェおぉオォあぁアァ"
    r"くぅクゥぐぅグゥむぅムゥ"
    r"はっハッふっフッへっヘッほっホッ"
    r"ー]+$"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def stable_hash(*values: Any) -> str:
    payload = "\x1f".join(str(value) for value in values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def aligned_offset_frames(offset_ms: int, frame_hop_s: float) -> int:
    if offset_ms == 0:
        raise ValueError("offset_ms must be non-zero")
    if frame_hop_s <= 0.0:
        raise ValueError("frame_hop_s must be positive")
    frames = int(round((offset_ms / 1000.0) / frame_hop_s))
    if frames == 0:
        raise ValueError("offset is smaller than one feature frame")
    aligned_s = frames * frame_hop_s
    if not math.isclose(aligned_s, offset_ms / 1000.0, abs_tol=1e-6):
        raise ValueError(
            f"offset {offset_ms}ms is not aligned to feature frame hop {frame_hop_s:.6f}s"
        )
    return frames


def perturbation_category(axis: str, offset_frames: int) -> str:
    if axis not in AXES:
        raise ValueError(f"unsupported axis: {axis!r}")
    if offset_frames == 0:
        raise ValueError("offset_frames must be non-zero")
    direction = "plus" if offset_frames > 0 else "minus"
    scale = "large" if abs(offset_frames) >= 8 else "small"
    return f"{axis}:{direction}:{scale}"


def candidate_id(
    *,
    video_id: str,
    boundary_index: int,
    axis: str,
    offset_frames: int,
) -> str:
    axis_token = axis.replace(".", "_")
    sign = "p" if offset_frames > 0 else "m"
    return f"{video_id}-b{boundary_index:05d}-{axis_token}-{sign}{abs(offset_frames):02d}f"


def compact_text(value: Any) -> str:
    return _PUNCT_RE.sub("", str(value or "")).lower()


def text_similarity(left: Any, right: Any) -> float:
    a = compact_text(left)
    b = compact_text(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def is_nonlexical_text(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text and _NONLEXICAL_RE.fullmatch(text))


def qc_rank(value: Any) -> int:
    return {"": 0, "ok": 0, "warn": 1, "reject": 2}.get(str(value or "").lower(), 1)


def alignment_rank(result: Mapping[str, Any]) -> int:
    quality = str(result.get("alignment_quality") or "")
    subtype = str(result.get("fallback_subtype") or "")
    if quality == "forced" and subtype in {"", "none"}:
        return 0
    if quality == "nonlexical" or subtype == "nonlexical_text":
        return 1
    if subtype == "proportional_after_sentinel" or bool(result.get("sentinel")):
        return 3
    if str(result.get("fallback_type") or "") not in {"", "none"}:
        return 2
    return 1


def repeat_risk(result: Mapping[str, Any]) -> bool:
    profile = result.get("repeat_profile")
    if isinstance(profile, Mapping):
        run = int(profile.get("run") or 0)
        ratio = float(profile.get("ratio") or 0.0)
        if run >= 4 and ratio >= 0.35:
            return True
    return bool(result.get("chunk_repetition_risk"))


def candidate_observables(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
) -> dict[str, Any]:
    similarity = text_similarity(baseline.get("text"), challenger.get("text"))
    baseline_qc = qc_rank(baseline.get("asr_qc_severity"))
    challenger_qc = qc_rank(challenger.get("asr_qc_severity"))
    baseline_alignment = alignment_rank(baseline)
    challenger_alignment = alignment_rank(challenger)
    baseline_repeat = repeat_risk(baseline)
    challenger_repeat = repeat_risk(challenger)
    baseline_nonlexical = bool(
        baseline.get("nonlexical_text") or is_nonlexical_text(baseline.get("text"))
    )
    challenger_nonlexical = bool(
        challenger.get("nonlexical_text") or is_nonlexical_text(challenger.get("text"))
    )
    baseline_cps = float(baseline.get("cue_density_cps") or baseline.get("chars_per_sec") or 0.0)
    challenger_cps = float(
        challenger.get("cue_density_cps") or challenger.get("chars_per_sec") or 0.0
    )
    gap_crossing = bool(candidate.get("gap_crossing"))

    score_parts = {
        "asr_text_instability": round(1.0 - similarity, 6),
        "asr_qc_delta": abs(challenger_qc - baseline_qc),
        "asr_qc_risk": max(baseline_qc, challenger_qc),
        "repeat_nonlexical_delta": int(baseline_repeat != challenger_repeat)
        + int(baseline_nonlexical != challenger_nonlexical),
        "repeat_nonlexical_risk": int(baseline_repeat or challenger_repeat)
        + int(baseline_nonlexical or challenger_nonlexical),
        "alignment_delta": abs(challenger_alignment - baseline_alignment),
        "alignment_risk": max(baseline_alignment, challenger_alignment),
        "cue_density_delta": round(abs(challenger_cps - baseline_cps), 6),
        "cue_density_risk": round(max(baseline_cps, challenger_cps), 6),
        "gap_crossing": gap_crossing,
    }
    score = (
        score_parts["asr_text_instability"] * 6.0
        + score_parts["asr_qc_delta"] * 2.5
        + score_parts["asr_qc_risk"] * 0.8
        + score_parts["repeat_nonlexical_delta"] * 2.0
        + score_parts["repeat_nonlexical_risk"] * 0.6
        + score_parts["alignment_delta"] * 2.5
        + score_parts["alignment_risk"] * 0.7
        + min(4.0, score_parts["cue_density_delta"]) * 0.7
        + max(0.0, score_parts["cue_density_risk"] - 4.0) * 0.2
        + (2.0 if gap_crossing else 0.0)
    )
    risk_bucket = "stable_control"
    if gap_crossing:
        risk_bucket = "gap_crossing"
    elif score_parts["alignment_delta"] or score_parts["asr_qc_delta"]:
        risk_bucket = "qc_alignment_change"
    elif score_parts["repeat_nonlexical_delta"] or score_parts["repeat_nonlexical_risk"]:
        risk_bucket = "repeat_nonlexical"
    elif score_parts["cue_density_delta"] >= 1.0 or score_parts["cue_density_risk"] >= 6.0:
        risk_bucket = "cue_density"
    elif score_parts["asr_text_instability"] >= 0.18:
        risk_bucket = "text_changed"
    return {
        "selection_score": round(score, 6),
        "risk_bucket": risk_bucket,
        "observable_signals": score_parts,
    }


def _category_extra_set(video_index: int) -> set[str]:
    sets = (
        {
            "right.start:minus:small",
            "right.start:plus:large",
            "left.end:plus:small",
            "left.end:minus:large",
        },
        {
            "right.start:plus:small",
            "right.start:minus:large",
            "left.end:minus:small",
            "left.end:plus:large",
        },
    )
    return sets[video_index % len(sets)]


def category_quotas(
    *,
    video_ids: Sequence[str],
    per_video: int = 36,
) -> dict[str, dict[str, int]]:
    categories = [
        perturbation_category(axis, frames)
        for axis in AXES
        for frames in (-8, -4, 4, 8)
    ]
    base, remainder = divmod(per_video, len(categories))
    if remainder not in {0, 4}:
        raise ValueError("per_video must divide into 8 categories with a balanced remainder of 0 or 4")
    result: dict[str, dict[str, int]] = {}
    for video_index, video_id in enumerate(video_ids):
        extras = _category_extra_set(video_index) if remainder else set()
        result[video_id] = {
            category: base + int(category in extras)
            for category in categories
        }
    return result


def _pick_diverse(rows: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("risk_bucket") or "stable_control")].append(row)
    for values in buckets.values():
        values.sort(
            key=lambda item: (
                -float(item.get("selection_score") or 0.0),
                str(item.get("candidate_id") or ""),
            )
        )
    bucket_order = sorted(
        buckets,
        key=lambda bucket: (
            bucket == "stable_control",
            -float(buckets[bucket][0].get("selection_score") or 0.0),
            bucket,
        ),
    )
    selected: list[dict[str, Any]] = []
    while bucket_order and len(selected) < limit:
        next_order: list[str] = []
        for bucket in bucket_order:
            values = buckets[bucket]
            if values and len(selected) < limit:
                selected.append(values.pop(0))
            if values:
                next_order.append(bucket)
        bucket_order = next_order
    return selected


def select_balanced_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    video_ids: Sequence[str],
    per_video: int = 36,
) -> list[dict[str, Any]]:
    quotas = category_quotas(video_ids=video_ids, per_video=per_video)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        grouped[
            (
                str(row.get("video_id") or ""),
                str(row.get("perturbation_category") or ""),
            )
        ].append(row)

    selected: list[dict[str, Any]] = []
    for video_id in video_ids:
        for category, quota in quotas[video_id].items():
            available = grouped[(video_id, category)]
            if len(available) < quota:
                raise ValueError(
                    f"not enough candidates for {video_id} {category}: "
                    f"need={quota} available={len(available)}"
                )
            selected.extend(_pick_diverse(available, quota))
    selected.sort(
        key=lambda row: (
            video_ids.index(str(row.get("video_id") or "")),
            float(row.get("boundary_time_s") or 0.0),
            str(row.get("candidate_id") or ""),
        )
    )
    for index, row in enumerate(selected, start=1):
        row["unique_index"] = index
    return selected


def _side_payload(candidate: Mapping[str, Any], identity: str) -> dict[str, Any]:
    result = candidate[f"{identity}_result"]
    interval = candidate[f"{identity}_interval"]
    return {
        "start": round(float(interval["start"]), 6),
        "end": round(float(interval["end"]), 6),
        "duration_s": round(float(interval["end"]) - float(interval["start"]), 6),
        "text": str(result.get("text") or ""),
        "raw_text": str(result.get("raw_text") or ""),
        "asr_qc_severity": str(result.get("asr_qc_severity") or ""),
        "asr_qc_reasons": list(result.get("asr_qc_reasons") or []),
        "alignment_quality": str(result.get("alignment_quality") or ""),
        "fallback_subtype": str(result.get("fallback_subtype") or ""),
        "sentinel": bool(result.get("sentinel")),
        "nonlexical_text": bool(result.get("nonlexical_text")),
        "repeat_profile": dict(result.get("repeat_profile") or {}),
        "cue_density_cps": float(result.get("cue_density_cps") or 0.0),
    }


def build_blind_items(
    unique_candidates: Sequence[Mapping[str, Any]],
    *,
    hidden_duplicate_count: int = 12,
    seed: int = 20260611,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if hidden_duplicate_count < 0:
        raise ValueError("hidden_duplicate_count must be non-negative")
    rng = random.Random(seed)
    canonical_rows: list[tuple[Mapping[str, Any], str]] = []
    used_item_ids: set[str] = set()
    for candidate in unique_candidates:
        stable_id = f"bp-{stable_hash(candidate.get('candidate_id'))[:12]}"
        if stable_id in used_item_ids:
            raise ValueError(f"duplicate blind item id for candidate {candidate.get('candidate_id')!r}")
        used_item_ids.add(stable_id)
        canonical_rows.append((candidate, stable_id))

    by_video: dict[str, list[tuple[Mapping[str, Any], str]]] = defaultdict(list)
    for item in canonical_rows:
        by_video[str(item[0].get("video_id") or "")].append(item)
    duplicate_sources: list[tuple[Mapping[str, Any], str]] = []
    if hidden_duplicate_count:
        video_ids = sorted(by_video)
        cursor = {video_id: 0 for video_id in video_ids}
        shuffled = {}
        for video_id, values in by_video.items():
            by_category: dict[str, list[tuple[Mapping[str, Any], str]]] = defaultdict(list)
            for item in values:
                by_category[str(item[0].get("perturbation_category") or "")].append(item)
            for category_values in by_category.values():
                rng.shuffle(category_values)
            interleaved: list[tuple[Mapping[str, Any], str]] = []
            category_order = sorted(by_category)
            round_index = 0
            while any(round_index < len(by_category[category]) for category in category_order):
                for category in category_order:
                    category_values = by_category[category]
                    if round_index < len(category_values):
                        interleaved.append(category_values[round_index])
                round_index += 1
            shuffled[video_id] = interleaved
        while len(duplicate_sources) < hidden_duplicate_count:
            made_progress = False
            for video_id in video_ids:
                values = shuffled[video_id]
                if cursor[video_id] >= len(values):
                    continue
                duplicate_sources.append(values[cursor[video_id]])
                cursor[video_id] += 1
                made_progress = True
                if len(duplicate_sources) >= hidden_duplicate_count:
                    break
            if not made_progress:
                raise ValueError("not enough unique candidates for hidden duplicates")

    review_rows: list[tuple[Mapping[str, Any], str, bool, str]] = [
        (candidate, item_id, False, item_id)
        for candidate, item_id in canonical_rows
    ]
    for duplicate_index, (candidate, canonical_id) in enumerate(duplicate_sources, start=1):
        item_id = (
            "bpdup-"
            + stable_hash(seed, candidate.get("candidate_id"), duplicate_index)[:12]
        )
        review_rows.append((candidate, item_id, True, canonical_id))
    rng.shuffle(review_rows)

    blind_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    for display_index, (candidate, item_id, is_duplicate, canonical_id) in enumerate(
        review_rows,
        start=1,
    ):
        a_identity, b_identity = (
            ("baseline", "challenger")
            if rng.random() < 0.5
            else ("challenger", "baseline")
        )
        blind_rows.append(
            {
                "schema": BLIND_ITEM_SCHEMA,
                "item_id": item_id,
                "display_index": display_index,
                "video_id": str(candidate.get("video_id") or ""),
                "video_label": str(candidate.get("video_label") or ""),
                "media_path": str(candidate.get("media_path") or ""),
                "boundary_time_s": round(float(candidate.get("boundary_time_s") or 0.0), 6),
                "context_start": round(float(candidate.get("context_start") or 0.0), 6),
                "context_end": round(float(candidate.get("context_end") or 0.0), 6),
                "a": _side_payload(candidate, a_identity),
                "b": _side_payload(candidate, b_identity),
            }
        )
        answer_rows.append(
            {
                "schema": ANSWER_KEY_SCHEMA,
                "item_id": item_id,
                "display_index": display_index,
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "canonical_item_id": canonical_id,
                "is_hidden_duplicate": is_duplicate,
                "a_identity": a_identity,
                "b_identity": b_identity,
                "video_id": str(candidate.get("video_id") or ""),
                "perturbation_category": str(
                    candidate.get("perturbation_category") or ""
                ),
                "axis": str(candidate.get("axis") or ""),
                "offset_frames": int(candidate.get("offset_frames") or 0),
                "offset_ms": int(candidate.get("offset_ms") or 0),
            }
        )
    return blind_rows, answer_rows


def label_by_item(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        item_id = str(row.get("item_id") or row.get("sample_id") or "")
        if not item_id:
            continue
        primary = str(row.get("primary_label") or row.get("label") or "")
        if primary not in PRIMARY_LABELS:
            primary = ""
        row["primary_label"] = primary
        result[item_id] = row
    return result


def normalized_preference(
    label: Mapping[str, Any] | None,
    answer: Mapping[str, Any],
) -> str:
    primary = str((label or {}).get("primary_label") or "")
    if primary == "a_better":
        return str(answer.get("a_identity") or "")
    if primary == "b_better":
        return str(answer.get("b_identity") or "")
    return primary


def summarize_preferences(
    label_rows: Sequence[Mapping[str, Any]],
    answer_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    labels = label_by_item(label_rows)
    answers = {
        str(row.get("item_id") or ""): dict(row)
        for row in answer_rows
        if row.get("item_id")
    }
    primary_counts: Counter[str] = Counter()
    for item_id in answers:
        primary = str(labels.get(item_id, {}).get("primary_label") or "")
        primary_counts[primary or "missing"] += 1

    usable_count = sum(primary_counts[label] for label in USABLE_LABELS)
    decisive_count = sum(primary_counts[label] for label in DECISIVE_LABELS)
    decisive_ratio = decisive_count / usable_count if usable_count else 0.0

    hidden_pairs_total = 0
    hidden_pairs_usable = 0
    hidden_pairs_consistent = 0
    for answer in answers.values():
        if not bool(answer.get("is_hidden_duplicate")):
            continue
        hidden_pairs_total += 1
        duplicate_id = str(answer.get("item_id") or "")
        canonical_id = str(answer.get("canonical_item_id") or "")
        duplicate_label = labels.get(duplicate_id)
        canonical_label = labels.get(canonical_id)
        if (
            str((duplicate_label or {}).get("primary_label") or "") not in USABLE_LABELS
            or str((canonical_label or {}).get("primary_label") or "") not in USABLE_LABELS
        ):
            continue
        hidden_pairs_usable += 1
        canonical_answer = answers.get(canonical_id)
        if canonical_answer is None:
            continue
        if normalized_preference(duplicate_label, answer) == normalized_preference(
            canonical_label,
            canonical_answer,
        ):
            hidden_pairs_consistent += 1
    hidden_consistency = (
        hidden_pairs_consistent / hidden_pairs_usable
        if hidden_pairs_usable
        else 0.0
    )

    challenger_wins = 0
    challenger_categories: Counter[str] = Counter()
    unique_labeled = 0
    for item_id, answer in answers.items():
        if bool(answer.get("is_hidden_duplicate")):
            continue
        label = labels.get(item_id)
        if str((label or {}).get("primary_label") or "") in USABLE_LABELS:
            unique_labeled += 1
        if normalized_preference(label, answer) == "challenger":
            challenger_wins += 1
            challenger_categories[str(answer.get("perturbation_category") or "")] += 1

    category_coverage = sum(1 for count in challenger_categories.values() if count > 0)
    gate_checks = {
        "hidden_duplicate_consistency_ge_0_80": hidden_consistency >= 0.80,
        "usable_labels_ge_90": usable_count >= 90,
        "decisive_ratio_gt_0_50": decisive_ratio > 0.50,
        "challenger_wins_gt_25": challenger_wins > 25,
        "challenger_category_coverage_gt_3": category_coverage > 3,
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "label_schema": LABEL_SCHEMA,
        "review_item_count": len(answers),
        "received_label_rows": len(label_rows),
        "primary_label_counts": dict(primary_counts),
        "usable_label_count": usable_count,
        "decisive_label_count": decisive_count,
        "decisive_ratio": round(decisive_ratio, 6),
        "unique_labeled_count": unique_labeled,
        "hidden_duplicate_pairs_total": hidden_pairs_total,
        "hidden_duplicate_pairs_usable": hidden_pairs_usable,
        "hidden_duplicate_pairs_consistent": hidden_pairs_consistent,
        "hidden_duplicate_consistency": round(hidden_consistency, 6),
        "challenger_wins": challenger_wins,
        "challenger_win_category_counts": dict(challenger_categories),
        "challenger_win_category_coverage": category_coverage,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "gate_policy": {
            "hidden_duplicate_consistency": ">=0.80",
            "usable_labels": ">=90",
            "decisive_ratio": ">0.50",
            "challenger_wins": ">25",
            "challenger_category_coverage": ">3",
        },
    }
