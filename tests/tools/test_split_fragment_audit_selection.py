"""What the split-fragment audit page may and may not reveal.

A blind listening test is the only evidence there is for a sub-cue split, and
the page prints each row's id. So the id has to carry nothing: the first version
of this selector built it from the cue index plus a per-stratum suffix
(`-r0` for a removed fragment, `-k` for what survived, `-d` for the reference),
which handed the auditor the grouping the whole page exists to hide.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _cue(index: int, *, text: str, start: float, split: dict | None = None) -> dict:
    cue = {
        "ja_text": text,
        "start": start,
        "end": start + 3.0,
        "acoustic_start": start,
        "acoustic_end": start + 3.0,
    }
    if split is not None:
        cue["vocalisation_split"] = split
    return cue


def _write_cues(path: Path, cues: list[dict]) -> None:
    path.write_text(json.dumps({"blocks": cues}, ensure_ascii=False), encoding="utf-8")


def _run(tmp_path: Path) -> dict:
    split_cues = [
        _cue(
            index,
            text="だめだって言ってるでしょ",
            start=10.0 * index,
            split={
                "removed_prefix": "あっ、あんっ、",
                "removed_suffix": "",
                "removed_seconds": 2.0,
                "removed_spans": [[10.0 * index - 2.0, 10.0 * index]],
            },
        )
        for index in range(1, 6)
    ]
    unfiltered = split_cues + [
        _cue(100 + index, text="あっ、あんっ", start=500.0 + 10.0 * index)
        for index in range(6)
    ]
    _write_cues(tmp_path / "cues.json", split_cues)
    _write_cues(tmp_path / "unfiltered.json", unfiltered)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools/audits/select_split_fragment_audit.py"),
            "--cues", str(tmp_path / "cues.json"),
            "--unfiltered-cues", str(tmp_path / "unfiltered.json"),
            "--audio", str(audio),
            "--film-alias", "sample-z",
            "--out-dir", str(out_dir),
            "--per-stratum", "4",
            "--per-reference-stratum", "3",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return {
        "manifest": [
            json.loads(line)
            for line in (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        "key": [
            json.loads(line)
            for line in (out_dir / "answer_key.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
    }


def test_the_row_id_does_not_encode_the_stratum(tmp_path: Path) -> None:
    rows = _run(tmp_path)["manifest"]

    assert len(rows) == 10
    ids = [row["row_id"] for row in rows]
    assert ids == [f"sample-z-{i:04d}" for i in range(1, 11)], (
        "ids must be a running number over the shuffled list, not derived "
        "from the cue or its group"
    )


def test_the_manifest_carries_nothing_but_the_clip(tmp_path: Path) -> None:
    """The shell refuses extra fields, and that refusal is the blinding."""
    for row in _run(tmp_path)["manifest"]:
        assert set(row) == {"schema", "row_id", "audio", "start_s", "end_s"}


def test_the_answer_key_can_still_trace_a_row_back(tmp_path: Path) -> None:
    """Blinding the page must not make a disagreement unreviewable."""
    result = _run(tmp_path)
    by_id = {row["row_id"]: row for row in result["key"]}

    assert set(by_id) == {row["row_id"] for row in result["manifest"]}
    for row in by_id.values():
        assert row["stratum"] in {"split_removed", "split_kept", "already_dropped"}
        assert isinstance(row["cue_index"], int)


def test_the_reference_stratum_comes_from_the_unfiltered_arm(tmp_path: Path) -> None:
    """Cues the shipped rule deletes are absent from the filtered file, so
    drawing them from it would silently produce a different population."""
    key = _run(tmp_path)["key"]
    dropped = [row for row in key if row["stratum"] == "already_dropped"]

    assert dropped, "the control stratum must not be empty"
    # The fixture's unfiltered file is the five split cues followed by six
    # all-vocalisation ones. Only the latter may appear here: the filtered file
    # holds indices 0-4 and none of those decompose.
    assert all(row["cue_index"] >= 5 for row in dropped)
    assert all(row["start_s"] >= 500.0 for row in dropped)
