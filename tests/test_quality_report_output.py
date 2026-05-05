from __future__ import annotations

import json
from pathlib import Path

import main


def test_quality_report_default_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("QUALITY_REPORT_ENABLED", raising=False)
    monkeypatch.setenv("QUALITY_REPORT_DIR", str(tmp_path / "reports"))

    path = main._write_quality_report(
        video_stem="sample",
        job_temp_dir=str(tmp_path),
        aligned_segments=[],
        asr_details={},
        video_duration_s=60.0,
    )

    assert path is None
    assert not (tmp_path / "reports").exists()


def test_quality_report_enabled_writes_to_project_reports(monkeypatch, tmp_path):
    report_dir = tmp_path / "reports"
    monkeypatch.setenv("QUALITY_REPORT_ENABLED", "1")
    monkeypatch.setenv("QUALITY_REPORT_DIR", str(report_dir))

    path = main._write_quality_report(
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
        video_duration_s=60.0,
    )

    expected = report_dir / "sample.quality_report.json"
    assert path == str(expected)
    payload = json.loads(expected.read_text(encoding="utf-8"))
    assert payload["per_min_subtitle_count"] == 1.0
    assert "warnings" in payload


def test_quality_report_relative_dir_resolves_from_project_root(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("QUALITY_REPORT_ENABLED", "1")
    monkeypatch.setenv("QUALITY_REPORT_DIR", "reports")

    path = main._write_quality_report(
        video_stem="relative",
        job_temp_dir=str(tmp_path),
        aligned_segments=[],
        asr_details={},
        video_duration_s=60.0,
    )

    assert Path(path) == tmp_path / "reports" / "relative.quality_report.json"
