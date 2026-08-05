import json

from pipeline import artifacts as artifacts_module
from pipeline.artifacts import (
    AsrArtifacts,
    serialize_asr_artifacts,
    write_translation_artifacts_snapshot,
)


def test_translation_resume_snapshot_keeps_asr_details(tmp_path):
    artifacts = AsrArtifacts(
        segments=[],
        audio_path=str(tmp_path / "audio.wav"),
        job_temp_dir=str(tmp_path),
        asr_details={
            "transcript_chunks": [{"text": "ok"}],
        },
        aligned_segments_path=str(tmp_path / "aligned.json"),
        transcript_path=str(tmp_path / "transcript.json"),
        asr_manifest_path=str(tmp_path / "manifest.json"),
        pipeline_timings={},
        logger=None,
        run_log_path="",
        audio_cache_key="key",
        video_stem="clip",
        output_dir=str(tmp_path),
        srt_path=str(tmp_path / "clip.srt"),
        bilingual_json_path=str(tmp_path / "bilingual.json"),
        quality_report_path="",
        bilingual=False,
        timings_path=str(tmp_path / "timings.json"),
        translation_cache_path=str(tmp_path / "translation.jsonl"),
        asr_log=[],
        audio_cached=False,
        device="cuda:0",
        backend_label="mock",
        video_duration_s=1.0,
        pipeline_started=0.0,
        job_id="job",
    )

    payload = serialize_asr_artifacts(artifacts)

    assert payload["asr_details"]["transcript_chunks"] == [{"text": "ok"}]
    # The snapshot copies asr_details verbatim; the pre-ASR candidate
    # compaction was dropped with the chain that produced those candidates.
    assert "pre_asr_candidate_count" not in payload["asr_details"]


def test_snapshot_only_resolves_declared_path_fields(tmp_path, monkeypatch):
    artifacts = AsrArtifacts(
        segments=[{"text": "これは字幕であり、パスではありません。"}],
        audio_path=str(tmp_path / "audio.wav"),
        job_temp_dir=str(tmp_path),
        asr_details={
            "transcript_chunks": [
                {
                    "text": "日本語テキスト",
                    "feature_names": [f"ptm_feature_{index}" for index in range(3000)],
                }
            ],
        },
        aligned_segments_path=str(tmp_path / "aligned.json"),
        transcript_path=str(tmp_path / "transcript.json"),
        asr_manifest_path=str(tmp_path / "manifest.json"),
        pipeline_timings={},
        logger=None,
        run_log_path="",
        audio_cache_key="key",
        video_stem="clip",
        output_dir=str(tmp_path),
        srt_path=str(tmp_path / "clip.srt"),
        bilingual_json_path=str(tmp_path / "bilingual.json"),
        quality_report_path="",
        bilingual=False,
        timings_path=str(tmp_path / "timings.json"),
        translation_cache_path=str(tmp_path / "translation.jsonl"),
        asr_log=[],
        audio_cached=False,
        device="cuda:0",
        backend_label="mock",
        video_duration_s=1.0,
        pipeline_started=0.0,
        job_id="job",
    )
    calls: list[str] = []
    original = artifacts_module._project_relative

    def counted(value):
        calls.append(str(value))
        return original(value)

    monkeypatch.setattr(artifacts_module, "_project_relative", counted)
    snapshot_path = write_translation_artifacts_snapshot(artifacts)
    payload = json.loads(artifacts_module.Path(snapshot_path).read_text(encoding="utf-8"))

    assert len(calls) == len(artifacts_module.ASR_ARTIFACT_PATH_FIELDS)
    assert payload["segments"][0]["text"] == "これは字幕であり、パスではありません。"
    assert payload["asr_details"]["transcript_chunks"][0]["feature_names"][-1] == "ptm_feature_2999"


def _snapshot_artifacts(tmp_path, run_log_path) -> AsrArtifacts:
    return AsrArtifacts(
        segments=[],
        audio_path=str(tmp_path / "audio.wav"),
        job_temp_dir=str(tmp_path),
        asr_details={},
        aligned_segments_path=str(tmp_path / "aligned.json"),
        transcript_path=str(tmp_path / "transcript.json"),
        asr_manifest_path=str(tmp_path / "manifest.json"),
        pipeline_timings={},
        logger=None,
        run_log_path=run_log_path,
        audio_cache_key="key",
        video_stem="clip",
        output_dir=str(tmp_path),
        srt_path=str(tmp_path / "clip.srt"),
        bilingual_json_path=str(tmp_path / "bilingual.json"),
        quality_report_path="",
        bilingual=False,
        timings_path=str(tmp_path / "timings.json"),
        translation_cache_path=str(tmp_path / "translation.jsonl"),
        asr_log=[],
        audio_cached=False,
        device="cuda:0",
        backend_label="mock",
        video_duration_s=1.0,
        pipeline_started=0.0,
        job_id="job",
    )


def test_a_retried_translation_gets_its_run_log_back(tmp_path):
    """A snapshot cannot carry a FileHandler, so the retry used to log nowhere:
    the run log ended at the failure and the retry left no trace at all."""
    import main as pipeline_main

    log_path = tmp_path / "log" / "run.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("2026-08-03 14:15:58 [INFO] stage_start translation\n", encoding="utf-8")
    artifacts = _snapshot_artifacts(tmp_path, log_path)

    pipeline_main._reopen_snapshot_run_logger(artifacts)
    try:
        assert artifacts.logger is not None
        pipeline_main._log_stage(artifacts.logger, "translation_settings reasoning_effort=none")
    finally:
        pipeline_main._close_artifacts_logger(artifacts)

    text = log_path.read_text(encoding="utf-8")
    assert "stage_start translation" in text  # appended, not truncated
    assert "reasoning_effort=none" in text
    assert artifacts.logger is None


def test_reopening_leaves_a_live_logger_and_a_pathless_snapshot_alone(tmp_path):
    import logging

    import main as pipeline_main

    existing = logging.getLogger("test.run.logger.kept")
    artifacts = _snapshot_artifacts(tmp_path, tmp_path / "log" / "run.log")
    artifacts.logger = existing
    pipeline_main._reopen_snapshot_run_logger(artifacts)
    assert artifacts.logger is existing

    without_path = _snapshot_artifacts(tmp_path, "")
    pipeline_main._reopen_snapshot_run_logger(without_path)
    assert without_path.logger is None
