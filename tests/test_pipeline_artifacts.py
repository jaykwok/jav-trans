import json

from pipeline import artifacts as artifacts_module
from pipeline.artifacts import (
    AsrArtifacts,
    serialize_asr_artifacts,
    write_translation_artifacts_snapshot,
)


def test_translation_resume_snapshot_drops_large_pre_asr_features(tmp_path):
    artifacts = AsrArtifacts(
        segments=[],
        audio_path=str(tmp_path / "audio.wav"),
        job_temp_dir=str(tmp_path),
        asr_details={
            "pre_asr_candidates": [
                {"pre_asr_ptm_pooled_features": [0.1] * 4096},
                {"pre_asr_ptm_pooled_features": [0.2] * 4096},
            ],
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

    assert "pre_asr_candidates" not in payload["asr_details"]
    assert payload["asr_details"]["pre_asr_candidate_count"] == 2
    assert payload["asr_details"]["transcript_chunks"] == [{"text": "ok"}]
    assert "pre_asr_candidates" in artifacts.asr_details


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
