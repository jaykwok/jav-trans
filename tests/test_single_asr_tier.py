"""The 0.6B ASR tier was dropped on 2026-07-31; it must not creep back.

It was unmaintained, and the CTC alignment head is only trained against the
1.7B encoder, so a second tier meant every setting, registry and OOM message
carried a per-repo dimension that no one was validating. These tests pin the
removal across all four surfaces it used to touch: the repo registry, the
default settings, the job schema, and the browser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from asr.backends import qwen
from core import config

import web as _web_package

ROOT = Path(__file__).resolve().parents[1]
_SRC_WEB = ROOT / "src" / "web"
if str(_SRC_WEB) not in _web_package.__path__:
    _web_package.__path__.append(str(_SRC_WEB))

from web.models import JobSpec  # noqa: E402

DROPPED_REPO = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"


def test_only_the_17b_repo_is_registered() -> None:
    assert set(qwen.QWEN_ASR_BACKEND_REPOS) == {qwen.QWEN_ASR_17B_REPO_ID}
    assert qwen.DEFAULT_QWEN_ASR_BACKEND == qwen.QWEN_ASR_17B_REPO_ID


def test_dropped_repo_is_rejected_rather_than_silently_remapped() -> None:
    # A stale `.env` is the realistic way this id comes back. Resolving it to
    # the 1.7B model would transcribe with a model the user did not choose.
    with pytest.raises(ValueError, match="Unsupported Qwen ASR backend"):
        qwen.qwen_asr_repo_id(DROPPED_REPO)


def test_no_default_setting_mentions_the_dropped_repo() -> None:
    for key, value in config.DEFAULT_SETTINGS.items():
        assert "0.6B" not in str(value), key


def test_job_spec_rejects_the_dropped_backend() -> None:
    with pytest.raises(ValueError):
        JobSpec(video_paths=["sample.mp4"], asr_backend=DROPPED_REPO)


def test_browser_cannot_offer_the_dropped_backend() -> None:
    static = ROOT / "src" / "web" / "static"
    sources = [static / "index.html"] + sorted((static / "js").glob("*.js"))
    for path in sources:
        assert "0.6B" not in path.read_text(encoding="utf-8"), path


def test_oom_guidance_does_not_point_at_a_model_that_is_gone() -> None:
    from pipeline import gpu_worker

    detail = gpu_worker._terminal_oom_detail(
        env={"ASR_BACKEND": qwen.QWEN_ASR_17B_REPO_ID},
        detail="CUDA out of memory",
        batch_size=1,
        retry_records=[],
    )

    assert "0.6B" not in detail
    assert not hasattr(gpu_worker, "LOW_VRAM_ASR_BACKEND")
