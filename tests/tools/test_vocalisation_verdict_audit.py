"""The listening audit's void condition, and what `unsure` may not become.

Two things in this evaluator carry the result. The control stratum decides
whether the page measured anything at all - kanji-bearing cues certainly contain
words, so if the ear did not hear words there, "no words" elsewhere means
nothing and the honest outcome is a void run rather than a negative one. And
`unsure` was offered as the exit for an unjudgeable clip, so folding it into
either answer afterwards would undo the reason it exists.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run(
    tmp_path: Path,
    key: list[dict],
    verdicts: list[dict],
    *,
    strata: tuple[str, str, str] | None = None,
) -> dict:
    key_path = tmp_path / "answer_key.jsonl"
    verdict_path = tmp_path / "manual_verdicts.jsonl"
    out = tmp_path / "result.json"
    _write(key_path, key)
    _write(verdict_path, verdicts)
    extra: list[str] = []
    if strata is not None:
        under_test, control_words, control_no_words = strata
        extra = [
            "--under-test", under_test,
            "--control-words", control_words,
            "--control-no-words", control_no_words,
        ]
    result = subprocess.run(
        [
            sys.executable,
            "tools/audits/evaluate_vocalisation_verdict_audit.py",
            "--verdicts", str(verdict_path),
            "--answer-key", str(key_path),
            "--out", str(out),
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


def _key(row_id: str, stratum: str) -> dict:
    return {
        "row_id": row_id,
        "stratum": stratum,
        "text": "…",
        "speech": 0.01,
        "vocalisation": 0.9,
        "speech_run_s": 0.0,
        "start_s": 0.0,
        "end_s": 2.0,
    }


def _verdict(row_id: str, answer: str) -> dict:
    return {"row_id": row_id, "verdict": answer, "note": ""}


def test_a_control_the_ear_failed_voids_the_run(tmp_path: Path) -> None:
    """The result that must not be reported as a success.

    If kanji-bearing cues do not come back as words, then "no words" on the
    strata under test is not evidence about those strata - it is evidence the
    question or the listening did not work.
    """
    key = [_key(f"c{i}", "dialogue_control") for i in range(10)]
    key += [_key(f"t{i}", "new_isolated") for i in range(10)]
    verdicts = [_verdict(f"c{i}", "no_words") for i in range(10)]
    verdicts += [_verdict(f"t{i}", "no_words") for i in range(10)]

    report = _run(tmp_path, key, verdicts)

    assert report["valid"] is False
    assert report["control_has_words_share"] == 0.0
    # The measurement is still reported - void is a statement about whether it
    # can be read, not a reason to hide it.
    assert report["w6_words_in_newly_dropped"] == 0


def test_a_passing_control_lets_the_result_be_read(tmp_path: Path) -> None:
    key = [_key(f"c{i}", "dialogue_control") for i in range(10)]
    key += [_key(f"t{i}", "new_kana") for i in range(10)]
    verdicts = [_verdict(f"c{i}", "has_words") for i in range(9)]
    verdicts += [_verdict("c9", "no_words")]
    verdicts += [_verdict(f"t{i}", "no_words") for i in range(10)]

    report = _run(tmp_path, key, verdicts)

    assert report["valid"] is True
    assert report["control_has_words_share"] == pytest.approx(0.9)
    assert report["w6_words_in_newly_dropped"] == 0
    assert report["w6_judged"] == 10


def test_unsure_is_excluded_from_both_answers(tmp_path: Path) -> None:
    """A forced guess becomes evidence, which is why the exit exists; averaging
    it back in afterwards would put the guess back."""
    key = [_key(f"c{i}", "dialogue_control") for i in range(10)]
    key += [_key(f"t{i}", "new_isolated") for i in range(10)]
    verdicts = [_verdict(f"c{i}", "has_words") for i in range(10)]
    verdicts += [_verdict(f"t{i}", "no_words") for i in range(7)]
    verdicts += [_verdict(f"t{i}", "unsure") for i in range(7, 10)]

    report = _run(tmp_path, key, verdicts)

    assert report["w6_judged"] == 7
    assert report["w6_unsure"] == 3
    assert report["by_stratum"]["new_isolated"]["unsure"] == 3


def test_a_verdict_with_no_key_row_stops_the_run(tmp_path: Path) -> None:
    """Joining by `row_id` is the only link between the blind page and the
    answers; a miss means the two files describe different samples."""
    key_path = tmp_path / "answer_key.jsonl"
    verdict_path = tmp_path / "manual_verdicts.jsonl"
    _write(key_path, [_key("a", "new_kana")])
    _write(verdict_path, [_verdict("b", "no_words")])

    result = subprocess.run(
        [
            sys.executable,
            "tools/audits/evaluate_vocalisation_verdict_audit.py",
            "--verdicts", str(verdict_path),
            "--answer-key", str(key_path),
            "--out", str(tmp_path / "result.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "no key row" in result.stderr


def test_the_split_page_reuses_the_same_arithmetic(tmp_path: Path) -> None:
    """Its strata have different names and its key has no per-cue posteriors.

    Both would have forced a second copy of this evaluator, and a second copy is
    a second place for the `unsure` rule - the one thing here that is easy to get
    wrong - to drift.
    """
    key = [
        {"row_id": "k1", "stratum": "split_kept", "text": "だめ", "cue_index": 1,
         "start_s": 0.0, "end_s": 2.0},
        {"row_id": "k2", "stratum": "split_kept", "text": "やめて", "cue_index": 2,
         "start_s": 0.0, "end_s": 2.0},
        {"row_id": "d1", "stratum": "already_dropped", "text": "あっ", "cue_index": 9,
         "start_s": 0.0, "end_s": 2.0},
        {"row_id": "r1", "stratum": "split_removed", "text": "あっ、あんっ",
         "cue_index": 1, "start_s": 0.0, "end_s": 2.0},
        {"row_id": "r2", "stratum": "split_removed", "text": "はぁ、はぁ",
         "cue_index": 2, "start_s": 0.0, "end_s": 2.0},
    ]
    verdicts = [
        _verdict("k1", "has_words"),
        _verdict("k2", "has_words"),
        _verdict("d1", "no_words"),
        _verdict("r1", "no_words"),
        _verdict("r2", "unsure"),
    ]

    report = _run(
        tmp_path,
        key,
        verdicts,
        strata=("split_removed", "split_kept", "already_dropped"),
    )

    assert report["valid"] is True
    assert report["under_test"] == ["split_removed"]
    # The unsure clip is not folded into either answer, here as anywhere else.
    assert report["w6_words_in_newly_dropped"] == 0
    assert report["w6_judged"] == 1
    assert report["w6_unsure"] == 1


def test_a_key_without_posteriors_does_not_crash_the_report(tmp_path: Path) -> None:
    """A split key carries the fragment, not a per-cue speech share. Which clip
    a word was heard in is the finding, and it does not depend on those."""
    key = [
        {"row_id": "k1", "stratum": "split_kept", "text": "だめ", "start_s": 0.0,
         "end_s": 2.0},
        {"row_id": "d1", "stratum": "already_dropped", "text": "あっ", "start_s": 0.0,
         "end_s": 2.0},
        {"row_id": "r1", "stratum": "split_removed", "text": "あっ", "start_s": 0.0,
         "end_s": 2.0},
    ]
    verdicts = [
        _verdict("k1", "has_words"),
        _verdict("d1", "no_words"),
        _verdict("r1", "has_words"),
    ]

    report = _run(
        tmp_path,
        key,
        verdicts,
        strata=("split_removed", "split_kept", "already_dropped"),
    )

    assert report["w6_words_in_newly_dropped"] == 1
    hit = report["words_heard"]["split_removed"][0]
    assert hit["row_id"] == "r1" and hit["speech"] is None
