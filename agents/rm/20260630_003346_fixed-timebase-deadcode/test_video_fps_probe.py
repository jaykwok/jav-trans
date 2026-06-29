import json
import subprocess

import pytest

from pipeline import audio as pipeline_audio


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("30000/1001", 30000 / 1001),
        ("24000/1001", 24000 / 1001),
        ("30/1", 30.0),
        ("24", 24.0),
        ("0/0", None),
        ("N/A", None),
        ("bad", None),
    ],
)
def test_parse_frame_rate(raw, expected):
    parsed = pipeline_audio._parse_frame_rate(raw)
    if expected is None:
        assert parsed is None
    else:
        assert parsed == pytest.approx(expected)


def test_probe_video_fps_prefers_avg_frame_rate(monkeypatch):
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "avg_frame_rate": "30000/1001",
                            "r_frame_rate": "60/1",
                        }
                    ]
                }
            ),
        )

    monkeypatch.setattr(pipeline_audio.subprocess, "run", fake_run)

    assert pipeline_audio.probe_video_fps("sample.mp4") == pytest.approx(30000 / 1001)


def test_probe_video_fps_falls_back_to_r_frame_rate(monkeypatch):
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "avg_frame_rate": "0/0",
                            "r_frame_rate": "24000/1001",
                        }
                    ]
                }
            ),
        )

    monkeypatch.setattr(pipeline_audio.subprocess, "run", fake_run)

    assert pipeline_audio.probe_video_fps("sample.mp4") == pytest.approx(24000 / 1001)
