import json
from pathlib import Path

import main
from helpers import make_job_context
from pipeline.artifacts import AsrArtifacts


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
        video_fps=25.0,
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
            "text": "あ",
            "words": [{"word": "あ", "start": 0.0, "end": 1.2}],
        },
        {
            "start": 1.0,
            "end": 2.0,
            "text": "い",
            "words": [{"word": "い", "start": 1.0, "end": 2.0}],
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
    assert translated[0]["end"] == 0.92
    assert translated[0]["end"] + (2 / 25.0) <= translated[1]["start"]
    assert translated[0]["text"] == "あ"
    assert translated[1]["text"] == "い"

    srt_content = (tmp_path / "out" / "clip.srt").read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:00,920" in srt_content

    sidecar = json.loads(
        (tmp_path / "jobs" / "clip" / "clip.bilingual.json").read_text(encoding="utf-8")
    )
    assert sidecar["blocks"][0]["end"] == 0.92
    assert sidecar["blocks"][0]["zh_text"] == "zh-0"

    timings = json.loads(
        (tmp_path / "jobs" / "clip" / "clip.timings.json").read_text(encoding="utf-8")
    )
    assert timings["counts"]["segments"] == 2
    assert timings["counts"]["translation_cues"] == 2
    assert timings["asr_details"]["subtitle_cue_plan"]["stage"] == "pre_translation"


def test_display_policy_compacts_cues_without_mutating_aligned_segments(
    monkeypatch,
    tmp_path,
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    segments = [
        {
            "start": 0.0,
            "end": 0.4,
            "text": "あ",
            "words": [{"word": "あ", "start": 0.0, "end": 0.4}],
            "fallback_subtype": "nonlexical_text",
        },
        {
            "start": 0.5,
            "end": 0.9,
            "text": "あ",
            "words": [{"word": "あ", "start": 0.5, "end": 0.9}],
            "fallback_subtype": "nonlexical_text",
        },
        {
            "start": 1.0,
            "end": 1.4,
            "text": "あ",
            "words": [{"word": "あ", "start": 1.0, "end": 1.4}],
            "fallback_subtype": "nonlexical_text",
        },
        {
            "start": 2.0,
            "end": 3.0,
            "text": "今日はいい天気ですね",
            "words": [{"word": "今日はいい天気ですね", "start": 2.0, "end": 3.0}],
            "alignment_quality": "boundary",
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
    assert translated[0]["display_decision"] == "compact"
    assert translated[0]["source_segment_ids"] == [0, 1, 2]
    assert translated[1]["text"] == "今日はいい天気ですね"

    sidecar = json.loads(
        (tmp_path / "jobs" / "clip" / "clip.bilingual.json").read_text(encoding="utf-8")
    )
    assert [block["display_decision"] for block in sidecar["blocks"]] == [
        "compact",
        "keep",
    ]
    assert sidecar["blocks"][0]["raw_texts"] == ["あ", "あ", "あ"]

    aligned_payload = json.loads(
        (tmp_path / "jobs" / "clip" / "clip.aligned_segments.json").read_text(
            encoding="utf-8"
        )
    )
    assert [segment["text"] for segment in aligned_payload["segments"]] == [
        "あ",
        "あ",
        "あ",
        "今日はいい天気ですね",
    ]
    assert "display_decision" not in aligned_payload["segments"][0]

    policy = aligned_payload["asr_details"]["subtitle_display_policy"]
    assert policy["cues_before"] == 4
    assert policy["cues_after"] == 2
    assert policy["counts"]["compact"] == 1
    assert policy["counts"]["drop"] == 2

    plan = aligned_payload["asr_details"]["subtitle_cue_plan"]
    assert plan["segments_before"] == 4
    assert plan["cues_before_display_policy"] == 4
    assert plan["cues_after"] == 2
