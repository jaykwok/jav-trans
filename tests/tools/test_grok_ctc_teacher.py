from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.run_grok_ctc_teacher import (  # noqa: E402
    _append_jsonl_rows,
    _is_transient,
    compile_ctc_manifest,
)


def _result(*, label: str, duration: float, words: list[dict]) -> dict:
    return {
        "source_id": "clip-1",
        "source_label": label,
        "source_duration_s": duration,
        "audio": "D:/audio.wav",
        "audio_sha256": "abc",
        "video_id": "video-a",
        "model": "x-ai/grok-stt-1.0",
        "response": {"words": words},
    }


def _canonical_result(*, canonical: str, words: list[dict]) -> dict:
    return {
        **_result(label="galgame", duration=4.0, words=words),
        "canonical_text": canonical,
        "source_group": "source-block-00001",
        "partition": "val",
    }


def test_append_jsonl_rows_preserves_existing_records(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    _append_jsonl_rows(output, [{"source_id": "a", "text": "あ"}])
    _append_jsonl_rows(output, [{"source_id": "b", "text": "い"}])

    assert output.read_text(encoding="utf-8").splitlines() == [
        '{"source_id":"a","text":"あ"}',
        '{"source_id":"b","text":"い"}',
    ]


def test_text_islands_and_nonword_regions_become_ordinary_ctc_examples() -> None:
    rows, summary = compile_ctc_manifest(
        [
            _result(
                label="definite_keep",
                duration=3.0,
                words=[
                    {"text": "こん", "start_s": 1.0, "end_s": 1.2},
                    {"text": "にちは", "start_s": 1.25, "end_s": 1.7},
                ],
            )
        ]
    )

    text = [row for row in rows if row["target_kind"] == "text"]
    blank = [row for row in rows if row["target_kind"] == "blank"]
    assert len(text) == 1
    assert text[0]["text"] == "こんにちは"
    assert text[0]["source_start_s"] == pytest.approx(0.75)
    assert text[0]["source_end_s"] == pytest.approx(1.95)
    assert blank
    assert all(row["text"] == "" for row in blank)
    assert all(row["group"] == "video-a" for row in rows)
    assert summary["examples_by_target_kind"] == {
        "text": 1,
        "blank": len(blank),
    }


def test_empty_teacher_answer_cannot_turn_a_definite_keep_clip_into_blank() -> None:
    rows, summary = compile_ctc_manifest(
        [_result(label="definite_keep", duration=4.0, words=[])]
    )

    assert rows == []
    assert summary["skipped"]["teacher_empty_definite_keep"] == 1


def test_empty_drop_clip_becomes_bounded_blank_chunks() -> None:
    rows, _ = compile_ctc_manifest(
        [_result(label="definite_drop", duration=24.0, words=[])],
        maximum_blank_s=10.0,
    )

    assert [row["duration_s"] for row in rows] == [10.0, 10.0, 4.0]
    assert all(row["target_kind"] == "blank" for row in rows)


def test_positive_contexts_from_neighbouring_islands_do_not_overlap() -> None:
    rows, _ = compile_ctc_manifest(
        [
            _result(
                label="definite_keep",
                duration=2.0,
                words=[
                    {"text": "あ", "start_s": 0.40, "end_s": 0.60},
                    {"text": "い", "start_s": 0.80, "end_s": 1.00},
                ],
            )
        ],
        context_s=0.25,
    )
    text = [row for row in rows if row["target_kind"] == "text"]
    assert len(text) == 2
    assert text[0]["source_end_s"] <= text[1]["source_start_s"]


def test_canonical_mode_uses_dataset_text_and_never_emits_blank_targets() -> None:
    rows, summary = compile_ctc_manifest(
        [
            _canonical_result(
                canonical="今日は猫です。",
                words=[
                    {"text": char, "start_s": 0.5 + index * 0.16, "end_s": 0.62 + index * 0.16}
                    for index, char in enumerate("今日は犬です")
                ],
            )
        ],
        max_crops_per_source=1,
    )

    assert len(rows) == 1
    assert rows[0]["text"] == "今日は猫です"
    assert rows[0]["target_kind"] == "canonical_text_crop"
    assert rows[0]["group"] == "source-block-00001"
    assert rows[0]["partition"] == "val"
    assert summary["canonical"]["accepted_sources"] == 1
    assert "blank" not in summary["examples_by_target_kind"]


def test_canonical_mode_rejects_a_teacher_transcript_that_does_not_match() -> None:
    rows, summary = compile_ctc_manifest(
        [
            _canonical_result(
                canonical="今日は猫です",
                words=[
                    {"text": char, "start_s": 0.5 + index * 0.16, "end_s": 0.62 + index * 0.16}
                    for index, char in enumerate("全然違う文")
                ],
            )
        ]
    )

    assert rows == []
    assert summary["canonical"]["accepted_sources"] == 0
    assert summary["skipped"]["canonical_cer_above_maximum"] == 1


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("OpenRouter STT HTTP 429: busy", True),
        ("OpenRouter STT HTTP 503: unavailable", True),
        ("OpenRouter STT transport error: reset", True),
        ("OpenRouter STT HTTP 400: invalid request", False),
        ("OpenRouter STT HTTP 402: no credit", False),
    ],
)
def test_only_transient_provider_errors_are_retryable(message: str, expected: bool) -> None:
    assert _is_transient(RuntimeError(message)) is expected
