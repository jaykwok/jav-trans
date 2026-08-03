"""Every ASR-stage setting has to be able to reach the ASR stage.

Two things hang off the one predicate `main._is_asr_stage_advanced_key`:

* what the web "参数调优" box is allowed to forward into the stage worker, and
* what `_asr_stage_config_signature_for_env` folds into the aligned-segments
  cache key.

So a setting left out of it is invisible twice - unreachable *and* absent from
the cache signature, which means changing it replays the previous run's output.
That has now happened three times: the loop-guard settings were missing from the
ASR result-cache signature (2026-08-02), the whole `ASR_DECODE_` family was
unreachable, and `ASR_INVALID_SEGMENT_DURATION` /
`ASR_MIN_REPAIRED_SEGMENT_DURATION` move segment end times while sitting in
neither list (both 2026-08-03).

This test does not know which settings matter; it only refuses to let a *new* one
be added without a decision being recorded here.
"""

from __future__ import annotations

import re
from pathlib import Path

import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STAGE_SOURCES = (
    sorted((PROJECT_ROOT / "src" / "asr").rglob("*.py"))
    + [PROJECT_ROOT / "src" / "pipeline" / "gpu_worker.py"]
    + [PROJECT_ROOT / "src" / "pipeline" / "batch_profile.py"]
)
_READ = re.compile(
    r"""(?:os\.)?getenv\(\s*["']([A-Z0-9_]+)["']|_env_\w+\(\s*["']([A-Z0-9_]+)["']"""
)
# Names the stage reads that deliberately do not travel through the advanced box.
_DELIBERATELY_NOT_FORWARDED = {
    # Chosen in its own UI field, and already special-cased into the aligned
    # cache signature by `_asr_stage_config_signature_for_env`.
    "ASR_BACKEND",
    # Test scaffolding: makes the worker answer without loading a model.
    "ASR_STAGE_WORKER_MOCK",
}


def _stage_env_names() -> set[str]:
    names: set[str] = set()
    for path in _STAGE_SOURCES:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _READ.finditer(text):
            names.add(match.group(1) or match.group(2))
    return {
        name
        for name in names
        if name.startswith(("ASR_", "KEEP_ASR", "GPU_BATCH_PROFILE", "TRANSCRIPTION_"))
    }


def test_every_asr_stage_setting_is_reachable():
    unreachable = sorted(
        name
        for name in _stage_env_names()
        if not main._is_asr_stage_advanced_key(name)
        and name not in _DELIBERATELY_NOT_FORWARDED
    )
    assert not unreachable, (
        "these settings are read by the ASR stage but can neither be forwarded to "
        "it nor participate in the aligned-segments cache signature: "
        f"{unreachable}. Add them to main._ASR_STAGE_ADVANCED_KEYS (plus "
        "_ASR_STAGE_CACHE_NEUTRAL_KEYS if they cannot change the output), or to "
        "_DELIBERATELY_NOT_FORWARDED here with the reason."
    )


def test_a_setting_cannot_be_neutral_without_being_reachable():
    """The neutral list says "forwarded, but keep it out of the cache key". An
    entry that is not forwardable in the first place is a contradiction, and it
    was one: the result-cache switches sat there while being unreachable."""
    orphans = sorted(
        name
        for name in main._ASR_STAGE_CACHE_NEUTRAL_KEYS
        if not main._is_asr_stage_advanced_key(name)
    )
    assert not orphans, orphans


def test_the_segment_repair_knobs_are_in_the_cache_signature(monkeypatch):
    """They move segment end times, so the aligned-segments cache must not serve
    the previous value across a change to either one."""
    monkeypatch.setenv("ASR_BACKEND", "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf")
    for name, value in (
        ("ASR_INVALID_SEGMENT_DURATION", "0.5"),
        ("ASR_MIN_REPAIRED_SEGMENT_DURATION", "1.5"),
    ):
        monkeypatch.delenv(name, raising=False)
        base = main._asr_stage_config_signature_for_env()
        monkeypatch.setenv(name, value)
        assert main._asr_stage_config_signature_for_env() != base, name
        monkeypatch.delenv(name, raising=False)
