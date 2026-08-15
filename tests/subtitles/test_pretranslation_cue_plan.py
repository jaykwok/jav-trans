import json
from pathlib import Path

import pytest

import main
from asr import alignment
from asr.local_backend import LocalAsrBackend
from helpers import make_job_context
from pipeline.artifacts import AsrArtifacts
from subtitles.options import SubtitleOptions


def _artifacts(tmp_path: Path, segments: list[dict], *, bilingual: bool = False) -> AsrArtifacts:
    job_temp_dir = tmp_path / "jobs" / "clip"
    output_dir = tmp_path / "out"
    job_temp_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    return AsrArtifacts(
        segments=segments,
        audio_path=str(job_temp_dir / "audio.wav"),
        job_temp_dir=str(job_temp_dir),
        asr_details={"transcript_chunks": [], "stage_timings": {}},
        aligned_segments_path=str(job_temp_dir / "clip.aligned_segments.json"),
        transcript_path=str(job_temp_dir / "clip.transcript.json"),
        asr_manifest_path=str(job_temp_dir / "clip.asr_manifest.json"),
        pipeline_timings={},
        logger=None,
        run_log_path=None,
        audio_cache_key="audio-key",
        video_stem="clip",
        output_dir=str(output_dir),
        srt_path=str(output_dir / "clip.srt"),
        bilingual_json_path=str(job_temp_dir / "clip.bilingual.json"),
        quality_report_path="",
        bilingual=bilingual,
        timings_path=str(job_temp_dir / "clip.timings.json"),
        translation_cache_path=str(job_temp_dir / "translation_cache.jsonl"),
        asr_log=[],
        audio_cached=True,
        device="cpu",
        backend_label="mock_asr",
        video_duration_s=3.0,
        pipeline_started=0.0,
        job_id="clip",
        aligned_cache_signature={"version": 2},
    )


def test_translation_uses_pre_normalized_cues(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    segments = [
        {
            "start": 0.0,
            "end": 1.2,
            "text": "私",
            "words": [{"word": "私", "start": 0.0, "end": 1.2}],
        },
        {
            "start": 1.0,
            "end": 2.0,
            "text": "君",
            "words": [{"word": "君", "start": 1.0, "end": 2.0}],
        },
    ]
    artifacts = _artifacts(tmp_path, segments)
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        tmp_path / "jobs",
        subtitle_mode="zh",
        translation_max_workers=1,
        keep_temp_files=True,
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(main, "_print_timing_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(main.translator_module, "generate_global_context", lambda items: "")

    def fake_translate_segments(items, **_kwargs):
        seen["items"] = [dict(item) for item in items]
        return [f"zh-{index}" for index, _item in enumerate(items)], [], []

    monkeypatch.setattr(main.translator_module, "translate_segments", fake_translate_segments)

    main.run_translation_and_write(str(video_path), artifacts, ctx=ctx, job_id="clip")

    translated = seen["items"]
    assert len(translated) == 2
    expected_end = translated[1]["start"] - SubtitleOptions().frame_gap_s
    assert translated[0]["end"] == pytest.approx(expected_end)
    assert translated[0]["end"] + SubtitleOptions().frame_gap_s <= translated[1]["start"]
    assert translated[0]["text"] == "私"
    assert translated[1]["text"] == "君"

    srt_content = (tmp_path / "out" / "clip.srt").read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:00,916" in srt_content

    sidecar = json.loads(
        (tmp_path / "jobs" / "clip" / "clip.bilingual.json").read_text(encoding="utf-8")
    )
    assert sidecar["blocks"][0]["end"] == pytest.approx(expected_end)
    assert sidecar["blocks"][0]["zh_text"] == "zh-0"
    assert sidecar["blocks"][0]["display_clamped_to_max"] is False

    timings = json.loads(
        (tmp_path / "jobs" / "clip" / "clip.timings.json").read_text(encoding="utf-8")
    )
    assert timings["counts"]["segments"] == 2
    assert timings["counts"]["translation_cues"] == 2
    assert timings["asr_details"]["subtitle_cue_plan"]["stage"] == "pre_translation"
    assert timings["asr_details"]["subtitle_cue_plan"]["layout_diagnostics"][
        "display_clamped_to_max"
    ] == 0



def test_pretranslation_cue_plan_preserves_model_routed_cues(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    segments = [
        {"start": 0.0, "end": 0.4, "text": "私", "words": [{"word": "私", "start": 0.0, "end": 0.4}]},
        {"start": 0.5, "end": 0.9, "text": "私", "words": [{"word": "私", "start": 0.5, "end": 0.9}]},
        {"start": 1.0, "end": 1.4, "text": "私", "words": [{"word": "私", "start": 1.0, "end": 1.4}]},
        {"start": 2.0, "end": 3.0, "text": "今日はいい天気ですね", "words": [{"word": "今日はいい天気ですね", "start": 2.0, "end": 3.0}]},
    ]
    artifacts = _artifacts(tmp_path, segments)
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        tmp_path / "jobs",
        subtitle_mode="zh",
        translation_max_workers=1,
        keep_temp_files=True,
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(main, "_print_timing_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(main.translator_module, "generate_global_context", lambda items: "")

    def fake_translate_segments(items, **_kwargs):
        seen["items"] = [dict(item) for item in items]
        return [f"zh-{index}" for index, _item in enumerate(items)], [], []

    monkeypatch.setattr(main.translator_module, "translate_segments", fake_translate_segments)

    main.run_translation_and_write(str(video_path), artifacts, ctx=ctx, job_id="clip")

    translated = seen["items"]
    assert len(translated) == 4
    assert [item["text"] for item in translated] == ["私", "私", "私", "今日はいい天気ですね"]

    aligned_payload = json.loads(
        (tmp_path / "jobs" / "clip" / "clip.aligned_segments.json").read_text(encoding="utf-8")
    )
    assert "subtitle_display_policy" not in aligned_payload["asr_details"]
    plan = aligned_payload["asr_details"]["subtitle_cue_plan"]
    assert plan["segments_before"] == 4
    assert plan["cues_before"] == 4
    assert plan["cues_after"] == 4


def test_cue_summary_exposes_measured_map_skip_without_display_clamp():
    text = "この文字列は十分に長いので表示時間による分割が必要になります"
    measured_text = text.replace("文字", "")
    words = [
        {
            "word": char,
            "start": index * 0.5,
            "end": index * 0.5 + 0.4,
            "timestamp_kind": "ctc_forced_alignment",
        }
        for index, char in enumerate(measured_text)
    ]

    cues, summary = main._prepare_translation_cues(
        [{"start": 0.0, "end": 20.0, "text": text, "words": words}],
        subtitle_options=SubtitleOptions(),
        bilingual=True,
    )

    assert len(cues) == 1
    diagnostics = summary["layout_diagnostics"]
    assert diagnostics["subtitle_layout_split_skipped"] == {
        "measured_word_text_map_incomplete": 1
    }
    assert diagnostics["display_clamped_to_max"] == 0
    assert diagnostics["proportional_fallback_used"] == 0
    # The soft limit remains visible, but the source timeline is untouched.
    assert diagnostics["duration_soft_cap_violation"] == 1
    assert diagnostics["duration_violation"] == 1
    plan = main._subtitle_cue_plan_summary(
        segments_before=1,
        mode="bilingual",
        cue_summary=summary,
    )
    assert plan["schema"] == "subtitle_cue_summary_v2"
    assert plan["layout_diagnostics"] == diagnostics


def test_local_ctc_words_stay_completely_mapped_through_cue_planning(monkeypatch):
    backend = LocalAsrBackend("cpu")
    spans = [
        alignment.CharSpan(
            char="先",
            index=0,
            start_frame=0,
            end_frame=1,
            start_s=0.0,
            end_s=0.5,
            score=-0.1,
        ),
        alignment.CharSpan(
            char="後",
            index=1,
            start_frame=390,
            end_frame=391,
            start_s=15.0,
            end_s=15.5,
            score=-0.1,
        ),
    ]
    monkeypatch.setattr(
        backend,
        "_align_characters",
        lambda *_args, **_kwargs: (spans, (0.0, 15.5)),
    )
    monkeypatch.setattr(
        backend,
        "_resolve_shadow_alignment_head",
        lambda _log: None,
    )
    result, _log = backend._use_boundary_timing_result(
        master_text="先後",
        raw_master_text="先後",
        duration=15.5,
        detected_language="Japanese",
        normalized_path="unused.wav",
        timing_start=0.0,
        timing_end=15.5,
        timing_window_source="chunk",
        log=[],
    )
    assert result["alignment_mode"] == "ctc_forced_alignment"
    assert "".join(word["word"] for word in result["words"]) == result["text"]

    cues, summary = main._prepare_translation_cues(
        [
            {
                "start": 0.0,
                "end": 15.5,
                "text": result["text"],
                "words": result["words"],
            }
        ],
        subtitle_options=SubtitleOptions(),
        bilingual=True,
    )

    assert [cue["ja_text"] for cue in cues] == ["先", "後"]
    assert summary["layout_diagnostics"]["subtitle_layout_split_skipped"] == {}
    assert summary["layout_diagnostics"]["subtitle_layout_split_source"] == {
        "measured_safe_boundary_dp": 2
    }


def test_postgate_flags_survive_from_segment_to_cue_plan():
    """The flags existed and reached nothing.

    `_annotate_segments_with_postgate` puts them on the segment, but the block
    builder rebuilt each block from a fixed key list, so the mark died one step
    before the layout. Pieces of a split segment inherit it, because a chunk the
    audio did not support does not become supported halfway through.
    """
    long_text = "この文字列は十分に長いので表示時間による分割が必要になります"
    words = [
        {
            "word": char,
            "start": index * 0.5,
            "end": index * 0.5 + 0.4,
            "timestamp_kind": "ctc_forced_alignment",
        }
        for index, char in enumerate(long_text)
    ]
    segments = [
        {
            "start": 0.0,
            "end": words[-1]["end"],
            "text": long_text,
            "words": words,
            "postgate_flags": ["repeated_unit"],
        },
        {
            "start": 40.0,
            "end": 41.0,
            "text": "普通の台詞",
            "words": [
                {
                    "word": char,
                    "start": 40.0 + index * 0.2,
                    "end": 40.0 + index * 0.2 + 0.2,
                    "timestamp_kind": "ctc_forced_alignment",
                }
                for index, char in enumerate("普通の台詞")
            ],
        },
    ]

    cues, summary = main._prepare_translation_cues(
        segments,
        subtitle_options=SubtitleOptions(),
        bilingual=False,
    )

    flagged = [cue for cue in cues if cue.get("postgate_flags")]
    assert flagged, [cue.get("ja_text") for cue in cues]
    assert all(cue["postgate_flags"] == ["repeated_unit"] for cue in flagged)
    assert summary["layout_diagnostics"]["postgate_flagged_cues"] == len(flagged)
    assert summary["layout_diagnostics"]["postgate_cue_flags"] == {
        "repeated_unit": len(flagged)
    }
    # The unflagged segment stays unflagged; the union is per segment, not global.
    assert any(not cue.get("postgate_flags") for cue in cues)


def test_postgate_flags_reach_the_bilingual_sidecar(monkeypatch, tmp_path):
    """A count nobody can resolve to lines is not locatable evidence.

    The quality report says how many flagged cues survived into the finished
    file; the sidecar is the only artifact that can say *which*. The translated
    path rebuilt each block from a fixed key list, so the mark died at the last
    step. Written only where it is set, so a grep lands on the flagged cues.
    """
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    segments = [
        {
            "start": 0.0,
            "end": 1.2,
            "text": "今日はいい天気ですね",
            "words": [{"word": "今日はいい天気ですね", "start": 0.0, "end": 1.2}],
            "postgate_flags": ["repeated_unit"],
        },
        {
            "start": 2.0,
            "end": 3.0,
            "text": "そうですね",
            "words": [{"word": "そうですね", "start": 2.0, "end": 3.0}],
        },
    ]
    artifacts = _artifacts(tmp_path, segments)
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        tmp_path / "jobs",
        subtitle_mode="zh",
        translation_max_workers=1,
        keep_temp_files=True,
    )

    monkeypatch.setattr(main, "_print_timing_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(main.translator_module, "generate_global_context", lambda items: "")
    monkeypatch.setattr(
        main.translator_module,
        "translate_segments",
        lambda items, **_kwargs: ([f"zh-{index}" for index, _ in enumerate(items)], [], []),
    )

    main.run_translation_and_write(str(video_path), artifacts, ctx=ctx, job_id="clip")

    blocks = json.loads(
        (tmp_path / "jobs" / "clip" / "clip.bilingual.json").read_text(encoding="utf-8")
    )["blocks"]
    flagged = [block for block in blocks if "postgate_flags" in block]
    assert [block["ja_text"] for block in flagged] == ["今日はいい天気ですね"]
    assert flagged[0]["postgate_flags"] == ["repeated_unit"]
    assert flagged[0]["zh_text"] == "zh-0"
    # Absent, not an empty list: the clean cues stay out of the way of a grep.
    assert sum("postgate_flags" in block for block in blocks) == 1
