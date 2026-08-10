from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.select_galgame_ctc_teacher_pilot import select_pilot  # noqa: E402


def _row(index: int, duration_s: float, text: str = "これは台詞です") -> dict:
    return {
        "index": index,
        "audio_id": f"clip-{index}",
        "audio": f"audio/{index}.ogg",
        "duration_s": duration_s,
        "text": text,
    }


def test_selection_is_stratified_deterministic_and_group_partitioned() -> None:
    rows = [
        _row(0, 2.5),
        _row(1, 3.5),
        _row(200, 4.5),
        _row(201, 5.5),
        _row(400, 7.5),
        _row(401, 8.5),
        _row(600, 10.5),
        _row(601, 12.5),
    ]
    bins = ((2.0, 4.0, 1), (4.0, 7.0, 1), (7.0, 10.0, 1), (10.0, 15.0, 1))

    first, summary = select_pilot(rows, bins=bins, group_block=200, seed=9)
    second, _ = select_pilot(rows, bins=bins, group_block=200, seed=9)

    assert [row["audio_id"] for row in first] == [row["audio_id"] for row in second]
    assert len(first) == 4
    assert summary["selected_rows"] == 4
    assert all(row["canonical_text"] == "これは台詞です" for row in first)
    assert all(row["canonical_acoustic_text"] == "これは台詞です" for row in first)
    assert all(row["partition"] in {"train", "val"} for row in first)


def test_punctuation_only_rows_are_not_teacher_candidates() -> None:
    rows = [_row(0, 2.5, "……！？"), _row(1, 2.6)]
    selected, summary = select_pilot(
        rows,
        bins=((2.0, 4.0, 1),),
        minimum_acoustic_chars=4,
    )

    assert [row["audio_id"] for row in selected] == ["clip-1"]
    assert summary["skipped"]["too_few_acoustic_characters"] == 1
