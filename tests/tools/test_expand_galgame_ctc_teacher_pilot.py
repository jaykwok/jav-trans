from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.expand_galgame_ctc_teacher_pilot import expand_pilot  # noqa: E402


def _rows(per_bin: int = 12) -> list[dict]:
    rows: list[dict] = []
    for bin_index, duration in enumerate((2.5, 4.5, 7.5, 10.5)):
        for offset in range(per_bin):
            index = bin_index * 100 + offset
            rows.append(
                {
                    "index": index,
                    "audio_id": f"id-{index}",
                    "audio": f"audio-{index}.ogg",
                    "duration_s": duration,
                    "text": "これは十分な文章です",
                }
            )
    return rows


def test_expansion_retains_paid_base_rows_and_adds_only_new_ids() -> None:
    source = _rows()
    tiny_bins = ((2.0, 4.0, 2), (4.0, 7.0, 2), (7.0, 10.0, 2), (10.0, 15.0, 2))
    # Build a base-shaped manifest manually; expand_pilot uses production bin
    # ratios, so monkeypatching DEFAULT_BINS keeps this test small and focused.
    import tools.align.expand_galgame_ctc_teacher_pilot as module

    original = module.DEFAULT_BINS
    module.DEFAULT_BINS = tiny_bins
    try:
        from tools.align.select_galgame_ctc_teacher_pilot import select_pilot

        base, _ = select_pilot(source, bins=tiny_bins, seed=7)
        expanded, summary = expand_pilot(source, base, multiplier=2, seed=7)
    finally:
        module.DEFAULT_BINS = original

    base_ids = {row["audio_id"] for row in base}
    expanded_ids = {row["audio_id"] for row in expanded}
    assert base_ids <= expanded_ids
    assert len(expanded) == 16
    assert summary["base_rows"] == 8
    assert summary["added_rows"] == 8
    assert summary["base_rows_retained"] == 8
    assert sum(summary["partitions"].values()) == 16
