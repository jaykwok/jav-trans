from __future__ import annotations

import json
from pathlib import Path

from pipeline.quality import load_global_glossary_pairs, write_quality_report


def _write_json_atomic(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class _Console:
    def print(self, *_args, **_kwargs) -> None:
        return None


def _env_flag(name: str) -> bool:
    import os

    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def test_quality_report_default_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("QUALITY_REPORT_ENABLED", raising=False)
    monkeypatch.setenv("QUALITY_REPORT_DIR", str(tmp_path / "reports"))

    path = write_quality_report(
        video_stem="sample",
        job_temp_dir=str(tmp_path),
        aligned_segments=[],
        asr_details={},
        project_root=tmp_path,
        console=_Console(),
        write_json_atomic=_write_json_atomic,
        env_flag=_env_flag,
        video_duration_s=60.0,
    )

    assert path is None
    assert not (tmp_path / "reports").exists()


def test_quality_report_enabled_defaults_to_video_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("QUALITY_REPORT_ENABLED", "1")
    monkeypatch.delenv("QUALITY_REPORT_DIR", raising=False)

    path = write_quality_report(
        video_stem="sample",
        job_temp_dir=str(tmp_path),
        aligned_segments=[
            {
                "start": 0.0,
                "end": 1.0,
                "text": "こんにちは",
                "ja": "こんにちは",
                "zh": "你好",
            }
        ],
        asr_details={},
        project_root=tmp_path,
        console=_Console(),
        write_json_atomic=_write_json_atomic,
        env_flag=_env_flag,
        video_duration_s=60.0,
    )

    expected = tmp_path / "video" / "sample" / "sample.quality_report.md"
    json_sidecar = tmp_path / "video" / "sample" / "sample.quality_report.json"
    assert path == str(expected)
    assert expected.exists()
    assert "Quality Report: sample" in expected.read_text(encoding="utf-8")
    payload = json.loads(json_sidecar.read_text(encoding="utf-8"))
    assert payload["per_min_subtitle_count"] == 1.0
    assert "warnings" in payload


def test_quality_report_relative_dir_override_resolves_from_project_root(monkeypatch, tmp_path):
    monkeypatch.setenv("QUALITY_REPORT_ENABLED", "1")
    monkeypatch.setenv("QUALITY_REPORT_DIR", "reports")

    path = write_quality_report(
        video_stem="relative",
        job_temp_dir=str(tmp_path),
        aligned_segments=[],
        asr_details={},
        project_root=tmp_path,
        console=_Console(),
        write_json_atomic=_write_json_atomic,
        env_flag=_env_flag,
        video_duration_s=60.0,
    )

    assert Path(path) == tmp_path / "reports" / "relative.quality_report.md"
    assert (tmp_path / "reports" / "relative.quality_report.json").exists()


def test_quality_report_keeps_subtitle_timing_fallback_observation_without_warning(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("QUALITY_REPORT_ENABLED", "1")
    monkeypatch.setenv("QUALITY_REPORT_DIR", str(tmp_path / "reports"))

    path = write_quality_report(
        video_stem="chunk-denominator",
        job_temp_dir=str(tmp_path),
        aligned_segments=[
            {
                "start": 0.0,
                "end": 1.0,
                "text": "こんにちは",
                "ja": "こんにちは",
                "zh": "你好",
            }
        ],
        asr_details={"fallback_count": 25, "chunk_count": 100},
        project_root=tmp_path,
        console=_Console(),
        write_json_atomic=_write_json_atomic,
        env_flag=_env_flag,
        video_duration_s=60.0,
    )

    payload = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["alignment_fallback_count"] == 25
    assert payload["alignment_fallback_total"] == 100
    assert payload["alignment_fallback_ratio"] == 0.25
    assert not any("alignment_fallback_ratio" in item for item in payload["warnings"])


def test_load_global_glossary_pairs_reads_hashed_translation_glossary(tmp_path):
    glossary_path = tmp_path / "translation_global_glossary.abc123.json"
    glossary_path.write_text(
        json.dumps({"terms": [{"ja": "あなた", "zh": "你"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    pairs = load_global_glossary_pairs(
        str(tmp_path),
        "sample",
        console=_Console(),
    )

    assert pairs == [("あなた", "你")]
