from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.generate_galgame_ctc_teacher_audit_html import _spread  # noqa: E402


def test_spread_is_deterministic_and_keeps_extremes() -> None:
    rows = [
        {"source_id": f"s-{index}", "duration_s": index + 1, "cer": index / 10}
        for index in range(10)
    ]

    first = _spread(rows, 4)
    second = _spread(list(reversed(rows)), 4)

    assert [row["source_id"] for row in first] == [row["source_id"] for row in second]
    assert first[0]["source_id"] == "s-0"
    assert first[-1]["source_id"] == "s-9"
