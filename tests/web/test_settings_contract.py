from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
import re

from core.config import DEFAULT_SETTINGS
from web.models import JobSpec, MAX_TRANSLATION_WORKERS, SettingsUpdate
from web.routes import config as config_routes


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "web" / "static"


def _same_default(example: str, configured: str) -> bool:
    if example == configured:
        return True
    try:
        return Decimal(example) == Decimal(configured)
    except InvalidOperation:
        return False


def test_env_template_examples_are_live_and_do_not_drift_from_config() -> None:
    examples: dict[str, str] = {}
    for line in config_routes._initial_env_template_lines():
        match = re.fullmatch(r"# ([A-Z][A-Z0-9_]*)=(.*)\n", line)
        if match:
            examples[match.group(1)] = match.group(2)

    assert examples
    assert not (set(examples) - set(DEFAULT_SETTINGS))

    # These are intentionally useful override examples rather than restatements
    # of the built-in value. Every other example must track config.py exactly.
    intentional_overrides = {
        "ASR_ALIGNMENT_HEAD_PATH",
        "ASR_ALIGNMENT_SHADOW_HEAD_PATH",
        "PROXY_HOST",
        "PROXY_PORT",
    }
    drifted = {
        key: (value, DEFAULT_SETTINGS[key])
        for key, value in examples.items()
        if key not in intentional_overrides
        and not _same_default(value, DEFAULT_SETTINGS[key])
    }
    assert drifted == {}


def test_translation_settings_have_an_explicit_env_save_path() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    settings = (STATIC / "js" / "settings.js").read_text(encoding="utf-8")

    assert 'id="btn-save-translation"' in html
    assert "保存翻译设置到 .env" in html
    assert "$('btn-save-translation')?.addEventListener('click'" in settings
    assert "saveSettingsBody(buildSettingsBodyFromForm({ includeConnection: true }))" in settings


def test_custom_task_template_is_not_misrepresented_as_env_persistence() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    presets = (STATIC / "js" / "presets.js").read_text(encoding="utf-8")

    assert "保存自定义任务模板" in html
    assert "只保存在本机浏览器中，不写入" in html
    assert "不会写入 .env" in presets


def test_every_browser_translation_setting_is_accepted_by_settings_api() -> None:
    expected = {
        "api_key",
        "base_url",
        "model",
        "translation_backend",
        "llamacpp_server_path",
        "translation_glossary",
        "llm_reasoning_effort",
        "target_lang",
        "proxy_protocol",
        "proxy_host",
        "proxy_port",
    }
    assert expected <= set(SettingsUpdate.model_fields)
    assert SettingsUpdate.model_config.get("extra") == "forbid"


def test_env_owned_settings_are_not_overwritten_by_browser_memory() -> None:
    source = (STATIC / "js" / "formMemory.js").read_text(encoding="utf-8")
    excluded_block = source[
        source.index("const FORM_MEMORY_EXCLUDED") : source.index("]);", source.index("const FORM_MEMORY_EXCLUDED"))
    ]
    for control_id in (
        "translation-backend",
        "api-base-url",
        "api-model",
        "api-reasoning-effort",
        "api-target-lang",
        "api-glossary",
        "llamacpp-server-path",
        "proxy-enabled",
        "proxy-protocol",
        "proxy-host",
        "proxy-port",
    ):
        assert f"'{control_id}'" in excluded_block
    assert "delete data.controls[id]" in source


def test_frontend_defaults_match_backend_defaults_and_limits() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert DEFAULT_SETTINGS["TRANSLATION_BACKEND"] == "openai"
    assert '<option value="openai" selected>' in html
    assert (
        DEFAULT_SETTINGS["OPENAI_COMPATIBILITY_BASE_URL"]
        == "https://openrouter.ai/api/v1"
    )
    assert 'placeholder="https://openrouter.ai/api/v1"' in html
    assert "https://api.deepseek.com" in html
    assert DEFAULT_SETTINGS["LLM_REASONING_EFFORT"] == "low"
    # The option text carries a human-facing annotation (「medium（默认）」), so
    # only the value and the `selected` marker are the contract here.
    assert '<option value="low" selected>' in html
    assert DEFAULT_SETTINGS["TARGET_LANG"] == "简体中文"
    assert '<option value="简体中文">简体中文</option>' in html
    assert f'max="{MAX_TRANSLATION_WORKERS}"' in html


def test_standard_preset_matches_backend_job_defaults() -> None:
    """"标准" is applied unconditionally on every load for non-custom users
    (main.js's `applyPreset(state.activePreset)`), so a TUNING_FIELDS value
    that drifts from JobSpec silently overrides whatever /api/config just
    reported. translation_max_workers did exactly that once (shipped as 16
    against a backend default of 4) - this pins every "标准" field against
    the real JobSpec default so that class of bug fails the suite instead of
    showing up in a user's job list."""
    presets_js = (STATIC / "js" / "presets.js").read_text(encoding="utf-8")
    tuning_block = presets_js[
        presets_js.index("export const TUNING_FIELDS") :
        presets_js.index("};", presets_js.index("export const TUNING_FIELDS"))
    ]

    def js_value(field_id: str) -> str:
        match = re.search(rf"'{re.escape(field_id)}':\s*([^,\n]+),", tuning_block)
        assert match, f"{field_id} missing from presets.js TUNING_FIELDS"
        return match.group(1).strip()

    assert js_value("r-mode") == f"'{JobSpec.model_fields['subtitle_mode'].default}'"
    assert js_value("r-skip-translation") == str(JobSpec.model_fields["skip_translation"].default).lower()
    assert js_value("t-translation-max-workers") == f"'{JobSpec.model_fields['translation_max_workers'].default}'"
    assert js_value("t-quality-report") == str(JobSpec.model_fields["keep_quality_report"].default).lower()
    assert js_value("t-keep-temp") == str(JobSpec.model_fields["keep_temp_files"].default).lower()


def test_every_html_control_is_wired_and_every_direct_js_id_exists() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((STATIC / "js").glob("*.js"))
    )
    control_ids = set(
        re.findall(r'<(?:input|select|textarea|button)[^>]*\bid="([^"]+)"', html)
    )
    direct_js_ids = set(re.findall(r"\$\('([^']+)'\)", javascript))

    assert {control for control in control_ids if control not in javascript} == set()
    assert {control for control in direct_js_ids if f'id="{control}"' not in html} == set()
