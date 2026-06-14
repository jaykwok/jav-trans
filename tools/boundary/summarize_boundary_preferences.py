#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.boundary.boundary_preference import (  # noqa: E402
    read_jsonl,
    summarize_preferences,
)


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize blind Boundary Refiner preference labels and evaluate the v5.1 gate."
    )
    parser.add_argument("--labels", required=True, help="JSONL exported by the audit page.")
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels_path = project_path(args.labels)
    answer_key_path = project_path(args.answer_key)
    output_path = project_path(args.output_json)
    summary = summarize_preferences(
        read_jsonl(labels_path),
        read_jsonl(answer_key_path),
    )
    summary.update(
        {
            "labels_path": str(labels_path),
            "answer_key_path": str(answer_key_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"summary={output_path}")
    print(f"gate_passed={summary['gate_passed']}")
    print(
        "usable={usable} decisive={decisive} decisive_ratio={ratio:.3f} "
        "hidden_consistency={consistency:.3f} challenger_wins={wins} coverage={coverage}".format(
            usable=summary["usable_label_count"],
            decisive=summary["decisive_label_count"],
            ratio=summary["decisive_ratio"],
            consistency=summary["hidden_duplicate_consistency"],
            wins=summary["challenger_wins"],
            coverage=summary["challenger_win_category_coverage"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
