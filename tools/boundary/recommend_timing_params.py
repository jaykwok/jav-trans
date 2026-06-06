#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.timing_formula import recommend_boundary_timing_params


def _load_duration_stats(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "duration_stats" in data:
        stats = data["duration_stats"]
    else:
        stats = data
    if not isinstance(stats, dict):
        raise ValueError(f"{path} does not contain a duration_stats object")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recommend SpeechBoundary-JA timing params from clean speech-island duration stats.",
    )
    parser.add_argument(
        "summary_json",
        type=Path,
        help="summary.json or a JSON object containing duration_stats.percentiles_s",
    )
    parser.add_argument(
        "--target-domain-speedup",
        type=float,
        default=1.5,
        help="Source-domain speech duration divided by target-domain speech duration. Default: 1.5",
    )
    parser.add_argument(
        "--max-padded-cap-s",
        type=float,
        default=9.0,
        help="Hard cap for padded ASR chunk context. Default: 9.0",
    )
    parser.add_argument(
        "--env",
        action="store_true",
        help="Print .env lines instead of full JSON.",
    )
    args = parser.parse_args()

    result = recommend_boundary_timing_params(
        _load_duration_stats(args.summary_json),
        target_domain_speedup=args.target_domain_speedup,
        max_padded_cap_s=args.max_padded_cap_s,
    )
    if args.env:
        for key, value in result["env"].items():
            print(f"{key}={value}")
        return 0
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
