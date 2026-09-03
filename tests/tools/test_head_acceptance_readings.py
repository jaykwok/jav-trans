"""Pooling two readings of the same cues: blank share, and frame classes.

The acceptance report is the only thing that decides whether a retrain ships, so
the v2 frame reading had to enter it without disturbing the reading every
previous head was judged on. It does that as extra *rows* - `<head>~frame` - and
these tests pin the two properties that make the comparison mean anything: the
scores must come from the same cues, and a v1 head must not acquire an empty
frame row just because a v2 head is in the same run.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.aggregate_head_acceptance import operating_point  # noqa: E402


def _film_report(path: Path, *, with_frame: bool) -> None:
    """One synthetic film: moaning cues score high, dialogue cues score low.

    The frame reading is made deliberately *better* separated than the blank one
    so a swap between the two columns cannot pass unnoticed.
    """
    cues = []
    for index in range(40):
        moaning = index < 20
        # The blank reading overlaps on purpose - a quarter of the dialogue cues
        # score above the quietest moans, which is the breathy-speech confusion
        # blank cannot resolve. The frame reading separates the same cues
        # cleanly, so a swap between the two columns cannot pass unnoticed.
        blank = (0.90 if index % 4 else 0.97) if moaning else (
            0.98 if index % 4 == 0 else 0.80
        )
        cue = {
            "index": index,
            "group": "vocalisation_dropped" if moaning else "dialogue_lexical",
            "text": "あっ" if moaning else "本当にそうですね",
            "duration_s": 2.0,
            "shipped": blank,
            "v3": blank,
        }
        if with_frame:
            cue["v3:speech"] = 0.02 if moaning else 0.90
            cue["v3:vocalisation"] = 0.90 if moaning else 0.05
        cues.append(cue)
    payload = {
        "schema": "alignment_head_acceptance_v1",
        "heads": {
            "shipped": {"frame_head_available": False},
            "v3": {"frame_head_available": bool(with_frame)},
        },
        "cues": cues,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _run(tmp_path: Path, *reports: Path) -> dict:
    out = tmp_path / "pooled.json"
    command = [sys.executable, "tools/align/aggregate_head_acceptance.py", "--out", str(out)]
    for index, report in enumerate(reports):
        command += ["--report", f"film-{index}={report.relative_to(PROJECT_ROOT)}"]
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


def test_a_v2_head_is_pooled_twice_and_a_v1_head_once(tmp_path: Path) -> None:
    work = PROJECT_ROOT / "agents" / "temp" / "pytest-head-acceptance"
    work.mkdir(parents=True, exist_ok=True)
    report = work / f"{tmp_path.name}.json"
    _film_report(report, with_frame=True)

    pooled = _run(tmp_path, report)["pooled"]

    assert set(pooled) == {"shipped", "v3", "v3~frame"}
    # The frame reading separates these cues perfectly; the blank one does not.
    assert pooled["v3~frame"]["auc_blank"] == 1.0
    assert pooled["v3"]["auc_blank"] < 1.0
    report.unlink()


def test_a_head_without_frame_classes_gets_no_frame_row(tmp_path: Path) -> None:
    """Otherwise the table would carry a row of nulls for every v1 head and the
    pooled verdict would have to special-case it downstream."""
    work = PROJECT_ROOT / "agents" / "temp" / "pytest-head-acceptance"
    work.mkdir(parents=True, exist_ok=True)
    report = work / f"{tmp_path.name}-plain.json"
    _film_report(report, with_frame=False)

    pooled = _run(tmp_path, report)["pooled"]

    assert set(pooled) == {"shipped", "v3"}
    report.unlink()


def test_both_readings_are_scored_on_the_same_cues(tmp_path: Path) -> None:
    """The pairing is the whole point: a difference between the two AUCs has to
    be the class system, not a different sample of cues."""
    work = PROJECT_ROOT / "agents" / "temp" / "pytest-head-acceptance"
    work.mkdir(parents=True, exist_ok=True)
    report = work / f"{tmp_path.name}-paired.json"
    _film_report(report, with_frame=True)

    pooled = _run(tmp_path, report)["pooled"]

    for label in ("v3", "v3~frame"):
        assert pooled[label]["cues_vocalisation"] == 20
        assert pooled[label]["cues_dialogue"] == 20
    report.unlink()


def test_the_operating_point_is_the_best_recall_inside_the_budget() -> None:
    """Heads put their mass at different absolute levels, so a shared threshold
    would compare calibration rather than capability.

    The walk goes *down*, so the answer is the lowest threshold still inside the
    budget - the most recall the head can buy for the false drops allowed, not
    the first threshold that fits.
    """
    positive = [0.99] * 8 + [0.80] * 2
    negative = [0.70] * 19 + [0.995]

    point = operating_point(positive, negative, 0.05)

    assert point is not None
    cut, recall, false_drop = point
    assert false_drop == 0.05
    assert recall == 1.0
    # It keeps walking past 0.80 - recall is already 1.0 there - and stops one
    # step above the negatives' own level, where the budget would blow out.
    assert 0.70 < cut <= 0.80

    # A budget tighter than a single negative cue is worth (1/20 = 5%) does not
    # fail - it reports the head buying nothing, which is the honest answer and
    # the one that makes a too-strict budget visible instead of absent.
    tight = operating_point(positive, negative, 0.01)
    assert tight is not None
    assert tight[1] == 0.0 and tight[2] == 0.0

    # None is reserved for "there was nothing to measure".
    assert operating_point([], negative, 0.05) is None
