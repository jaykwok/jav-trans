from __future__ import annotations

import json
from pathlib import Path

from pipeline.quality import write_quality_report


class _Console:
    def print(self, *_args, **_kwargs) -> None:
        return None


def _write_json_atomic(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _env_flag(_name: str) -> bool:
    return False


def _segments() -> list[dict]:
    return [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "こんにちは",
            "ja": "こんにちは",
            "zh": "你好",
        }
    ]


def test_quality_report_explicit_report_dir_is_per_job(monkeypatch, tmp_path):
    monkeypatch.setenv("QUALITY_REPORT_DIR", str(tmp_path / "env-reports"))

    first_dir = tmp_path / "job-a-reports"
    second_dir = tmp_path / "job-b-reports"
    first_path = write_quality_report(
        video_stem="sample-a",
        job_temp_dir=str(tmp_path / "job-a"),
        aligned_segments=_segments(),
        asr_details={},
        project_root=tmp_path,
        console=_Console(),
        write_json_atomic=_write_json_atomic,
        env_flag=_env_flag,
        video_duration_s=60.0,
        enabled=True,
        report_dir=first_dir,
    )
    second_path = write_quality_report(
        video_stem="sample-b",
        job_temp_dir=str(tmp_path / "job-b"),
        aligned_segments=_segments(),
        asr_details={},
        project_root=tmp_path,
        console=_Console(),
        write_json_atomic=_write_json_atomic,
        env_flag=_env_flag,
        video_duration_s=60.0,
        enabled=True,
        report_dir=second_dir,
    )

    assert Path(first_path) == first_dir / "sample-a.quality_report.md"
    assert Path(second_path) == second_dir / "sample-b.quality_report.md"
    assert (first_dir / "sample-a.quality_report.md").exists()
    assert (second_dir / "sample-b.quality_report.md").exists()
    assert (first_dir / "sample-a.quality_report.json").exists()
    assert (second_dir / "sample-b.quality_report.json").exists()
    assert not (tmp_path / "env-reports").exists()


def test_quality_report_explicit_hard_fail_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv("QC_HARD_FAIL", "1")

    report_dir = tmp_path / "reports"
    path = write_quality_report(
        video_stem="warnings-allowed",
        job_temp_dir=str(tmp_path / "job"),
        aligned_segments=[],
        asr_details={},
        project_root=tmp_path,
        console=_Console(),
        write_json_atomic=_write_json_atomic,
        env_flag=_env_flag,
        video_duration_s=60.0,
        enabled=True,
        report_dir=report_dir,
        hard_fail=False,
    )

    assert Path(path) == report_dir / "warnings-allowed.quality_report.md"
    assert (report_dir / "warnings-allowed.quality_report.json").exists()
