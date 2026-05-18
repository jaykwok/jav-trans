from __future__ import annotations

import json

from testset_quality_eval import (
    DatasetCase,
    SubtitleSegment,
    clean_subtitle_text,
    discover_dataset_cases,
    evaluate_prediction_against_reference,
    load_index,
    load_or_discover_cases,
    normalize_eval_text,
    parse_reference_segments,
    parse_srt_segments,
    write_index,
)


def test_clean_subtitle_text_strips_tags_and_newlines():
    assert clean_subtitle_text(r"{\fad(1,2)}你好\N世界{\pos(1,2)}") == "你好\n世界"
    assert normalize_eval_text("[F] 你，好！") == "你好"


def test_parse_ass_reference_segments_filters_staff_and_styles(tmp_path):
    reference_path = tmp_path / "sample.ass"
    reference_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "ScriptType: v4.00+",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                r"Dialogue: 0,0:00:01.00,0:00:02.20,Default,,0,0,0,,{\pos(1,2)}你好\N世界",
                "Dialogue: 0,0:00:03.00,0:00:04.00,staff,,0,0,0,,字幕组",
                "Dialogue: 0,0:00:05.00,0:00:06.00,title,,0,0,0,,标题",
            ]
        ),
        encoding="utf-8",
    )

    segments = parse_reference_segments(reference_path)

    assert len(segments) == 1
    assert segments[0].start == 1.0
    assert segments[0].end == 2.2
    assert segments[0].text == "你好\n世界"
    assert segments[0].source.endswith("/sample.ass")
    assert segments[0].style == "Default"


def test_parse_srt_segments_prefers_chinese_line(tmp_path):
    srt_path = tmp_path / "sample.srt"
    srt_path.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n你好\n\n",
        encoding="utf-8",
    )

    segments = parse_srt_segments(srt_path)

    assert len(segments) == 1
    assert segments[0].text == "你好"


def test_evaluate_prediction_against_reference_reports_quality_and_hallucination_proxy():
    reference = [
        SubtitleSegment(0.0, 1.0, "你好"),
        SubtitleSegment(2.0, 3.0, "世界"),
    ]
    prediction = [
        SubtitleSegment(0.0, 1.0, "你好"),
        SubtitleSegment(2.0, 3.0, "世"),
        SubtitleSegment(5.0, 6.0, "哈哈哈哈哈哈"),
    ]

    metrics = evaluate_prediction_against_reference(reference, prediction)

    assert metrics["reference_count"] == 2
    assert metrics["prediction_count"] == 3
    assert metrics["matched_reference_count"] == 2
    assert metrics["timeline_recall"] == 1.0
    assert metrics["timeline_precision"] == 0.666667
    assert metrics["unsupported_prediction_count"] == 1
    assert metrics["repeated_prediction_count"] == 1
    assert metrics["zh_deletion_rate"] > 0
    assert 0 < metrics["zh_char_f1"] < 1


def test_discover_dataset_cases_uses_reference_metadata_and_one_reference_fallback(tmp_path):
    dataset = tmp_path / "test"
    first = dataset / "first"
    first.mkdir(parents=True)
    video_a = first / "video-a[720p].mp4"
    video_a.write_bytes(b"video")
    reference_a = first / "subtitle-a.ass"
    reference_a.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Video File: video-a[720p].mp4",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,你好",
            ]
        ),
        encoding="utf-8",
    )
    second = dataset / "second"
    second.mkdir()
    video_b = second / "only-video.mkv"
    video_b.write_bytes(b"video")
    reference_b = second / "only-sub.srt"
    reference_b.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n世界\n\n",
        encoding="utf-8",
    )

    cases = discover_dataset_cases(dataset)

    assert len(cases) == 2
    first_case = next(case for case in cases if case.video_path.endswith(video_a.name))
    assert isinstance(first_case, DatasetCase)
    assert first_case.reference_path.endswith(reference_a.name)
    assert first_case.match_reason == "reference_metadata_exact"
    second_case = next(case for case in cases if case.video_path.endswith(video_b.name))
    assert second_case.reference_path.endswith(reference_b.name)
    assert second_case.match_reason == "only_reference_in_directory"


def test_index_roundtrip_and_load_or_discover_prefers_index(tmp_path):
    dataset = tmp_path / "test"
    dataset.mkdir()
    video = dataset / "case.mp4"
    reference = dataset / "case.ass"
    video.write_bytes(b"video")
    reference.write_text(
        "\n".join(
            [
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,你好",
            ]
        ),
        encoding="utf-8",
    )
    cases = discover_dataset_cases(dataset)
    index_path = dataset / "index.json"

    write_index(index_path, cases, dataset)
    loaded = load_index(index_path)
    loaded_via_default = load_or_discover_cases(
        dataset_root=dataset,
        index_path=index_path,
        use_index=True,
    )

    assert [case.video_path for case in loaded] == [case.video_path for case in cases]
    assert loaded_via_default == loaded
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["case_count"] == 1
