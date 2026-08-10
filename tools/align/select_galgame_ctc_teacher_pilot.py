#!/usr/bin/env python3
"""Select a deterministic, budget-bounded Galgame timing-Teacher pilot.

The dataset text remains the canonical transcript.  This selector only chooses
audio for a timing Teacher, stratifying duration so the pilot contains enough
multi-phrase clips for timestamps to matter without spending the budget on the
longest tail of the corpus.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment import acoustic_text, minimum_ctc_frames, normalize_text  # noqa: E402


SCHEMA = "galgame_ctc_teacher_pilot_v1"
SUMMARY_SCHEMA = "galgame_ctc_teacher_pilot_summary_v1"
DEFAULT_BINS = (
    (2.0, 4.0, 2000),
    (4.0, 7.0, 1500),
    (7.0, 10.0, 1000),
    (10.0, 15.0, 500),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _partition(group: str, *, seed: int, val_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "val" if value < val_fraction else "train"


def select_pilot(
    rows: Sequence[Mapping[str, Any]],
    *,
    bins: Sequence[tuple[float, float, int]] = DEFAULT_BINS,
    minimum_acoustic_chars: int = 4,
    group_block: int = 200,
    val_fraction: float = 0.10,
    seed: int = 20260808,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if group_block <= 0:
        raise ValueError("group_block must be positive")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    pools: list[list[dict[str, Any]]] = [[] for _ in bins]
    skipped: Counter[str] = Counter()
    for position, raw in enumerate(rows):
        duration_s = float(raw.get("duration_s") or 0.0)
        canonical_text = normalize_text(str(raw.get("text") or ""))
        canonical_acoustic, _ = acoustic_text(canonical_text)
        if len(canonical_acoustic) < minimum_acoustic_chars:
            skipped["too_few_acoustic_characters"] += 1
            continue
        # The crop cache is extracted at 13 fps before the head's x2 upsample.
        # Keeping the native-rate feasibility rule makes every selected full
        # sentence usable by both A/B arms without zero_infinity hiding rows.
        if minimum_ctc_frames(canonical_text) > duration_s * 13.0:
            skipped["text_denser_than_native_frames"] += 1
            continue
        bin_index = next(
            (
                index
                for index, (start_s, end_s, _count) in enumerate(bins)
                if start_s <= duration_s < end_s
            ),
            None,
        )
        if bin_index is None:
            skipped["outside_duration_bins"] += 1
            continue
        source_index = int(raw.get("index") if raw.get("index") is not None else position)
        source_group = f"source-block-{source_index // group_block:05d}"
        pools[bin_index].append(
            {
                **dict(raw),
                "schema": SCHEMA,
                "canonical_text": canonical_text,
                "canonical_acoustic_text": canonical_acoustic,
                "source_index": source_index,
                "source_group": source_group,
                "partition": _partition(
                    source_group, seed=seed, val_fraction=val_fraction
                ),
            }
        )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    by_bin: dict[str, dict[str, Any]] = {}
    for pool, (start_s, end_s, wanted) in zip(pools, bins):
        if len(pool) < wanted:
            raise ValueError(
                f"duration bin [{start_s}, {end_s}) has {len(pool)} eligible rows; "
                f"need {wanted}"
            )
        chosen = rng.sample(pool, wanted)
        selected.extend(chosen)
        by_bin[f"{start_s:g}-{end_s:g}"] = {
            "eligible": len(pool),
            "selected": len(chosen),
            "audio_hours": round(
                sum(float(row["duration_s"]) for row in chosen) / 3600.0, 6
            ),
        }

    selected.sort(key=lambda row: int(row["source_index"]))
    audio_s = sum(float(row["duration_s"]) for row in selected)
    partitions = Counter(str(row["partition"]) for row in selected)
    groups_by_partition = {
        partition: len(
            {
                str(row["source_group"])
                for row in selected
                if row["partition"] == partition
            }
        )
        for partition in sorted(partitions)
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "selected_rows": len(selected),
        "audio_hours": round(audio_s / 3600.0, 6),
        "partitions": dict(partitions),
        "groups_by_partition": groups_by_partition,
        "source_groups": len({str(row["source_group"]) for row in selected}),
        "duration_bins": by_bin,
        "minimum_acoustic_chars": minimum_acoustic_chars,
        "group_block": group_block,
        "val_fraction": val_fraction,
        "seed": seed,
        "skipped": dict(skipped),
    }
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--minimum-acoustic-chars", type=int, default=4)
    parser.add_argument("--group-block", type=int, default=200)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--price-per-hour-usd", type=float, default=0.10)
    parser.add_argument("--budget-usd", type=float, default=1.0)
    args = parser.parse_args()

    output = Path(args.output)
    selected, summary = select_pilot(
        read_jsonl(Path(args.manifest)),
        minimum_acoustic_chars=args.minimum_acoustic_chars,
        group_block=args.group_block,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    estimated_cost = summary["audio_hours"] * float(args.price_per_hour_usd)
    if estimated_cost > float(args.budget_usd) + 1e-12:
        raise SystemExit(
            f"pilot would cost ${estimated_cost:.6f}, above ${args.budget_usd:.6f}"
        )
    write_jsonl(output, selected)
    summary.update(
        {
            "manifest": str(Path(args.manifest)),
            "output": str(output),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "price_per_hour_usd": float(args.price_per_hour_usd),
            "estimated_cost_usd": round(estimated_cost, 9),
            "budget_usd": float(args.budget_usd),
        }
    )
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
