"""The job body the browser POSTs must validate against `JobSpec`.

`JobSpec` is `extra="forbid"`, so one stray key makes every submission fail with
422 and the whole web UI stops being able to start a job. That is exactly what
happened when the pluggable translation backend added `translation_backend` to
the reader `files.js` spreads into the job body: nothing in Python changed, no
test failed, and the button simply stopped working. These tests read the actual
JS so the two sides cannot drift apart again silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import web as _web_package

ROOT = Path(__file__).resolve().parents[2]
_SRC_WEB = ROOT / "src" / "web"
if str(_SRC_WEB) not in _web_package.__path__:
    _web_package.__path__.append(str(_SRC_WEB))

from web.models import JobSpec  # noqa: E402

JS = ROOT / "src" / "web" / "static" / "js"
INDEX = ROOT / "src" / "web" / "static" / "index.html"


def _object_literal_keys(source: str, start_marker: str) -> set[str]:
    start = source.index(start_marker) + len(start_marker)
    depth = 1
    index = start
    while depth:
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    body = source[start : index - 1]
    return set(re.findall(r"^\s*([a-z_][a-z0-9_]*)\s*:", body, flags=re.MULTILINE))


def _spread_calls(source: str, start_marker: str) -> set[str]:
    start = source.index(start_marker)
    end = source.index("};", start)
    return set(re.findall(r"\.\.\.([A-Za-z_][A-Za-z0-9_]*)\(", source[start:end]))


def _job_body_keys() -> set[str]:
    files_js = (JS / "files.js").read_text(encoding="utf-8")
    settings_js = (JS / "settings.js").read_text(encoding="utf-8")

    keys = _object_literal_keys(files_js, "const spec = {")
    spreads = _spread_calls(files_js, "const spec = {")
    assert spreads, "the job body no longer spreads a helper; update this test"
    for name in spreads:
        keys |= _object_literal_keys(settings_js, f"export function {name}() {{\n  return {{")

    # Assigned conditionally after the literal rather than inside it.
    keys |= set(re.findall(r"\bspec\.([a-z_][a-z0-9_]*)\s*=", files_js))
    return keys


def test_every_key_the_browser_sends_is_a_declared_job_spec_field() -> None:
    unknown = sorted(_job_body_keys() - set(JobSpec.model_fields))
    assert not unknown, f"JobSpec forbids extras; these would 422: {unknown}"


def test_translation_backend_choice_stays_out_of_the_job_body() -> None:
    # It is a settings-API field, persisted by syncSettingsFromFormForSubmit
    # before the job is created.
    body_keys = _job_body_keys()
    for settings_only in (
        "translation_backend",
        "llamacpp_server_path",
        "api_key",
        "base_url",
        "model",
    ):
        assert settings_only not in body_keys


def test_worker_field_says_concurrency_no_longer_moves_the_bill() -> None:
    """Until 2026-08-24 the batch size was `ceil(cues / (2 * workers))`, so this
    field really did price the job - and the hint correctly said so. Decoupling
    made that sentence false in the expensive direction: a user reading it would
    keep concurrency low to save money it no longer costs."""
    html = INDEX.read_text(encoding="utf-8")
    field = html[html.index('id="t-translation-max-workers"') :]
    field = field[: field.index("</label>")]
    assert "字幕总条数 ÷ 并发数 ÷ 2" not in field
    assert "不影响每批条数" in field
    assert "不会改变成本" in field


def test_reasoning_field_does_not_promise_an_escalated_repair() -> None:
    """The repair pass reissues at the base tier floored at `low`, so only `none`
    escalates. The hint used to name `low→high` explicitly."""
    html = INDEX.read_text(encoding="utf-8")
    field = html[html.index('id="api-reasoning-effort"') :]
    field = field[: field.index("</label>")]
    assert "low→high" not in field
    assert "只有 none 会升档" in field


def test_reasoning_field_explains_the_cascade_and_what_it_costs() -> None:
    """The selector is the single biggest lever on the bill, so the hint has to
    say both halves: the tier prices the whole film, and only flagged lines are
    escalated. It previously claimed DeepSeek maps low to high, which was the
    documentation error that hid a tenfold cost difference."""
    html = INDEX.read_text(encoding="utf-8")
    field = html[html.index('id="api-reasoning-effort"') :]
    field = field[: field.index("</label>")]
    assert "首轮全片按此强度翻译" in field
    assert "只对这些行集中复译" in field
    assert "输出的绝大部分是思维链" in field
    assert "映射为 high" not in field


def test_job_spec_accepts_a_full_browser_payload() -> None:
    payload: dict[str, object] = {
        "video_paths": ["sample.mp4"],
        "output_dir": "./out",
        "advanced": {"ASR_CHUNK_TARGET_S": "20.0"},
    }
    defaults: dict[str, object] = {
        "asr_backend": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
        "subtitle_mode": "zh",
        "skip_translation": False,
        "keep_quality_report": False,
        "translation_max_workers": 4,
        "keep_temp_files": False,
        "llm_reasoning_effort": "low",
        "llm_api_format": "chat",
        "target_lang": "简体中文",
        "translation_glossary": "",
        "resume_from_job_id": "",
    }
    for key in _job_body_keys():
        if key in payload:
            continue
        assert key in defaults, f"no sample value for browser-sent key {key!r}"
        payload[key] = defaults[key]

    spec = JobSpec(**payload)
    assert spec.video_paths == ["sample.mp4"]


def test_spec_with_a_settings_only_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        JobSpec(video_paths=["sample.mp4"], translation_backend="openai")


def test_open_folder_failure_is_not_silently_ignored() -> None:
    jobs_render = (JS / "jobsRender.js").read_text(encoding="utf-8")
    handler = jobs_render[jobs_render.index("const folder = e.target.closest('[data-folder]')") :]
    handler = handler[: handler.index("const retry = e.target.closest('[data-retry]')")]

    assert "const r = await fetch(`/api/open-folder" in handler
    assert "if (!r.ok) alert('打开文件夹失败：'" in handler
    assert "catch (error)" in handler
