"""A mistyped setting is told to the user, once, where they can still see it.

Parsing no longer raises out of whichever module happened to read the field
first: the value is recorded and the default applies. That is only an
improvement if something says so - otherwise a `.env` typo becomes a silent
change of behaviour. Startup is the place: it is the console the launcher opens,
and it is before any job has spent time on the wrong value.
"""

from __future__ import annotations

from pathlib import Path

import web as _web_package

_SRC_WEB = Path(__file__).resolve().parents[2] / "src" / "web"
if str(_SRC_WEB) not in _web_package.__path__:
    _web_package.__path__.append(str(_SRC_WEB))

from core import typed_config
from web import app as web_app


def test_every_bad_field_is_listed_at_startup(monkeypatch, capsys):
    typed_config.clear_configuration_problems()
    monkeypatch.setenv("SUBTITLE_MAX_SOURCE_CHARS", "twenty")
    monkeypatch.setenv("ASR_DECODE_TOKENS_PER_SECOND", "-3")
    monkeypatch.setenv("GPU_BATCH_PROFILE_MAX_ENTRIES", "lots")
    try:
        web_app._warn_configuration_problems()
        printed = capsys.readouterr().out
    finally:
        typed_config.clear_configuration_problems()

    # All three, not just the first one to be read.
    assert "SUBTITLE_MAX_SOURCE_CHARS" in printed
    assert "ASR_DECODE_TOKENS_PER_SECOND" in printed
    assert "GPU_BATCH_PROFILE_MAX_ENTRIES" in printed


def test_a_clean_configuration_says_nothing(monkeypatch, capsys):
    typed_config.clear_configuration_problems()
    for name in (
        "SUBTITLE_MAX_SOURCE_CHARS",
        "ASR_DECODE_TOKENS_PER_SECOND",
        "GPU_BATCH_PROFILE_MAX_ENTRIES",
    ):
        monkeypatch.delenv(name, raising=False)
    try:
        web_app._warn_configuration_problems()
        printed = capsys.readouterr().out
    finally:
        typed_config.clear_configuration_problems()

    assert printed.strip() == ""


def test_a_refused_version_stamp_is_reported_not_swallowed(monkeypatch, capsys):
    """The stamp is the one field that is refused rather than corrected, and it
    raises where it is read. Startup has to survive that and still say why."""
    typed_config.clear_configuration_problems()
    monkeypatch.setenv("SUBTITLE_LAYOUT_ENGINE", "anchor_aware_dp_v2")
    try:
        web_app._warn_configuration_problems()
        printed = capsys.readouterr().out
    finally:
        typed_config.clear_configuration_problems()

    assert "SUBTITLE_LAYOUT_ENGINE" in printed


def test_ports_and_backend_limits_are_reported_together(monkeypatch, capsys):
    bad = {
        "JAV_TRANS_PORT": "bad", "JAV_TRANS_EVENTS_PORT": "inf",
        "LLAMACPP_CTX_SIZE": "nan", "LOCAL_BACKEND_WAIT_TIMEOUT_S": "wait",
        "LOCAL_BACKEND_CLOSE_WAIT_S": "-4", "LOCAL_RETIRE_WAIT_S": "inf",
    }
    typed_config.clear_configuration_problems()
    for name, value in bad.items():
        monkeypatch.setenv(name, value)
    try:
        web_app._warn_configuration_problems()
        output = capsys.readouterr().out
        assert all(name in output for name in bad)
        from llm.backends import _admission_timeout_s
        assert _admission_timeout_s() == 600.0
    finally:
        typed_config.clear_configuration_problems()
