from __future__ import annotations

import base64
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.audits import grok_stt_smoke_audit as audit  # noqa: E402


def _response() -> dict:
    return {
        "text": "こんにちは",
        "language": "ja",
        "duration": 2.0,
        "words": [
            {"word": "こん", "start": 0.2, "end": 0.5},
            {"text": "にちは", "start": 0.8, "end": 1.4},
        ],
        "segments": [{"id": 0, "start": 0, "end": 2, "text": "こんにちは"}],
        "usage": {"seconds": 2, "cost": 0.0001},
    }


def _result() -> dict:
    return {
        "schema": audit.RESULT_SCHEMA,
        "audit_id": "grok-001",
        "source_row_id": "pause-0028",
        "source_class": "definite_drop",
        "audio": "media/grok-001.wav",
        "source_duration_s": 2.0,
        "latency_s": 1.25,
        "response": audit.normalize_response(_response(), fallback_duration_s=2.0),
    }


def test_request_explicitly_asks_openrouter_for_word_timestamps() -> None:
    payload = audit.build_request_payload(b"RIFFaudio", language="ja")
    assert payload["response_format"] == "verbose_json"
    assert payload["timestamp_granularities"] == ["word"]
    assert payload["provider"]["require_parameters"] is True
    assert payload["provider"]["data_collection"] == "allow"
    assert payload["provider"]["zdr"] is False
    assert base64.b64decode(payload["input_audio"]["data"]) == b"RIFFaudio"
    assert not payload["input_audio"]["data"].startswith("data:")


def test_response_normalizes_both_openrouter_word_key_shapes() -> None:
    normalized = audit.normalize_response(_response(), fallback_duration_s=99)
    assert normalized["transcript"] == "こんにちは"
    assert normalized["duration_s"] == 2.0
    assert normalized["words"] == [
        {"text": "こん", "start_s": 0.2, "end_s": 0.5},
        {"text": "にちは", "start_s": 0.8, "end_s": 1.4},
    ]
    assert normalized["diagnostics"]["word_count"] == 2
    assert normalized["diagnostics"]["timed_coverage_s"] == 0.9
    assert normalized["gaps"] == [
        {"start_s": 0.0, "end_s": 0.2},
        {"start_s": 0.5, "end_s": 0.8},
        {"start_s": 1.4, "end_s": 2.0},
    ]


def test_invalid_word_spans_are_reported_not_painted() -> None:
    response = _response()
    response["words"].append({"word": "坏", "start": 1.5, "end": 1.4})
    normalized = audit.normalize_response(response, fallback_duration_s=2.0)
    assert normalized["diagnostics"]["invalid_word_count"] == 1
    assert [word["text"] for word in normalized["words"]] == ["こん", "にちは"]


def test_fragmented_units_merge_but_punctuation_does_not_become_speech() -> None:
    words = [
        {"text": "こ", "start_s": 0.2, "end_s": 0.3},
        {"text": "ん", "start_s": 0.36, "end_s": 0.5},
        {"text": "。", "start_s": 0.5, "end_s": 0.9},
        {"text": "あ", "start_s": 1.0, "end_s": 1.2},
    ]
    islands = audit.merge_lexical_islands(words)
    assert islands == [
        {"text": "こん", "start_s": 0.2, "end_s": 0.5, "unit_count": 2},
        {"text": "あ", "start_s": 1.0, "end_s": 1.2, "unit_count": 1},
    ]


def test_frame_supervision_is_complete_and_keeps_boundary_uncertainty() -> None:
    compiled = audit.compile_frame_supervision(
        [{"text": "声", "start_s": 0.3, "end_s": 0.6}], 1.0
    )
    spans = compiled["frame_supervision"]
    assert spans[0]["start_frame"] == 0
    assert spans[-1]["end_frame"] == compiled["frame_count"]
    assert [span["label"] for span in spans] == [
        "non_word",
        "ignore",
        "word",
        "ignore",
        "non_word",
    ]
    assert compiled["merge_gap_frames"] == 4
    assert compiled["minimum_lexical_island_frames"] == 2
    assert compiled["boundary_ignore_frames"] == 2


def test_sub_two_frame_island_becomes_ignore_not_word() -> None:
    compiled = audit.compile_frame_supervision(
        [{"text": "あ", "start_s": 0.50, "end_s": 0.52}], 1.0
    )
    assert compiled["lexical_islands"] == []
    assert compiled["ignored_short_islands"] == [
        {"text": "あ", "start_s": 0.5, "end_s": 0.52, "unit_count": 1}
    ]
    labels = [span["label"] for span in compiled["frame_supervision"]]
    assert "word" not in labels
    assert "ignore" in labels


def test_audit_page_hides_sampling_truth_and_offers_two_verdicts(
    tmp_path: Path,
) -> None:
    summary = audit.build_page([_result()], tmp_path)
    text = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert summary["shows_source_class"] is False
    assert "こんにちは" in text
    assert "Grok 时间单元" in text
    assert "合并语音岛" in text
    assert "短岛 ignore" in text
    assert "最终帧监督" in text
    assert "潜在切点区域" in text
    assert "转写正确" in text
    assert "边界准确" in text
    assert "本应无词（幻听）" in text
    assert "manual_verdicts.jsonl" in text
    assert "createAuditReviewCore" in text
    assert "灰色不等于静音或安全切点" in text
    assert "definite_drop" not in text
    assert "pause-0028" not in text


def test_script_payload_escapes_a_closing_script_tag(tmp_path: Path) -> None:
    result = _result()
    result["response"]["transcript"] = "</script><b>unsafe</b>"
    audit.build_page([result], tmp_path)
    text = (tmp_path / "index.html").read_text(encoding="utf-8")
    embedded = text.split("const ROWS=", 1)[1].split(";const STORAGE_KEY", 1)[0]
    assert "</script>" not in embedded
    assert json.loads(embedded)[0]["transcript"] == "</script><b>unsafe</b>"
