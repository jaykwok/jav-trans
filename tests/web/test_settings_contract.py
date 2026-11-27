from __future__ import annotations

from pathlib import Path
import re

from core.config import DEFAULT_SETTINGS
from web.models import JobSpec, MAX_TRANSLATION_WORKERS, SettingsUpdate


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "web" / "static"


def test_translation_settings_have_an_explicit_config_save_path() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    settings = (STATIC / "js" / "settings.js").read_text(encoding="utf-8")

    assert 'id="btn-save-translation"' in html
    assert "保存翻译设置" in html
    assert ".env" not in html
    assert "$('btn-save-translation')?.addEventListener('click'" in settings
    assert "saveSettingsBody(buildSettingsBodyFromForm({ includeConnection: true }))" in settings
    assert "config/config.json" in settings


def test_presets_and_tuning_panel_are_removed_from_sidebar() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    javascript = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((STATIC / "js").glob("*.js"))
    )

    for removed in (
        "preset-chip",
        "section-tuning",
        "panel-tuning",
        "btn-save-custom",
        "r-skip-translation",
    ):
        assert removed not in html
        assert removed not in css
    assert "from './presets.js'" not in javascript
    assert "保存自定义任务模板" not in html


def test_every_browser_translation_setting_is_accepted_by_settings_api() -> None:
    expected = {
        "api_key",
        "base_url",
        "model",
        "translation_backend",
        "translation_glossary",
        "llm_reasoning_effort",
        "target_lang",
        "proxy_protocol",
        "proxy_host",
        "proxy_port",
        "disable_update_check",
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
        "proxy-enabled",
        "proxy-protocol",
        "proxy-host",
        "proxy-port",
        "update-auto-check",
    ):
        assert f"'{control_id}'" in excluded_block
    assert "delete data.controls[id]" in source


def test_frontend_defaults_match_backend_defaults_and_limits() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert DEFAULT_SETTINGS["TRANSLATION_BACKEND"] == "llamacpp"
    assert '<option value="llamacpp" selected>' in html
    assert '<div id="openai-backend-fields" style="display:none">' in html
    assert '<div id="llamacpp-backend-fields">' in html
    assert (
        DEFAULT_SETTINGS["OPENAI_COMPATIBILITY_BASE_URL"]
        == "https://openrouter.ai/api/v1"
    )
    assert 'placeholder="https://openrouter.ai/api/v1"' in html
    assert "https://api.deepseek.com" in html
    assert DEFAULT_SETTINGS["LLM_REASONING_EFFORT"] == "low"
    # The option text carries a human-facing annotation (「low（默认）」), so
    # only the value and the `selected` marker are the contract here.
    assert '<option value="low" selected>' in html
    assert DEFAULT_SETTINGS["TARGET_LANG"] == "简体中文"
    assert DEFAULT_SETTINGS["JAV_SUB_DISABLE_UPDATE_CHECK"] == "0"
    assert '<option value="简体中文">简体中文</option>' in html
    assert f'max="{MAX_TRANSLATION_WORKERS}"' in html


def test_sidebar_subtitle_output_and_concurrency_match_backend_defaults() -> None:
    """With presets/tuning removed, sidebar controls directly match JobSpec
    defaults and are remembered automatically by formMemory.js."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    form_memory = (STATIC / "js" / "formMemory.js").read_text(encoding="utf-8")

    # `#r-mode` lives in the video & subtitle panel with all 3 output modes.
    video_panel = html[html.index("📁 视频与字幕") : html.index('id="panel-model-requirements"')]
    assert 'id="r-mode"' in video_panel
    assert 'id="f-output-dir"' in video_panel
    mode_block = video_panel[video_panel.index('<select id="r-mode"') : video_panel.index("</select>", video_panel.index('<select id="r-mode"'))]
    assert set(re.findall(r'value="([^"]+)"', mode_block)) == {"zh", "bilingual", "ja_only"}
    assert f'<option value="{JobSpec.model_fields["subtitle_mode"].default}" selected>' in mode_block

    # `#t-translation-max-workers` lives inside `#openai-backend-fields` only.
    openai_block = html[
        html.index('<div id="openai-backend-fields"') : html.index('<div id="llamacpp-backend-fields"')
    ]
    assert 'id="t-translation-max-workers"' in openai_block
    assert f'value="{JobSpec.model_fields["translation_max_workers"].default}"' in openai_block
    assert f'max="{MAX_TRANSLATION_WORKERS}"' in openai_block
    assert "并发请求数" in openai_block

    # Per-job sidebar controls must NOT be excluded from browser form memory.
    excluded_block = form_memory[
        form_memory.index("const FORM_MEMORY_EXCLUDED") : form_memory.index("]);", form_memory.index("const FORM_MEMORY_EXCLUDED"))
    ]
    for remembered_id in ("r-mode", "f-output-dir", "t-translation-max-workers"):
        assert f"'{remembered_id}'" not in excluded_block


def test_retired_diagnostic_controls_are_removed_from_ui() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((STATIC / "js").glob("*.js"))
    )
    for retired_id in ("t-quality-report", "t-keep-temp", "t-env-override", "qc-overlay"):
        assert f'id="{retired_id}"' not in html
        assert retired_id not in javascript
    assert "data-gpu-diagnostics" not in javascript


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


def test_every_js_module_import_resolves_to_an_existing_file_and_export() -> None:
    """`node --check` only checks syntax, not ES module import paths. A broken
    relative import (like `./toast.js`) 404s at runtime and aborts `main.js`
    before any event listener or SSE connection attaches."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js_dir = STATIC / "js"
    for script_src in re.findall(r'<script[^>]*\bsrc="([^"]+)"', html):
        assert (STATIC / script_src).is_file(), f"Missing script in index.html: {script_src}"

    files = sorted(js_dir.glob("*.js"))
    sources = {path.name: path.read_text(encoding="utf-8") for path in files}
    direct_exports: dict[str, set[str]] = {}
    reexports: dict[str, list[tuple[str, str, str]]] = {}
    for name, text in sources.items():
        exported = set(
            re.findall(
                r"\bexport\s+(?:async\s+)?(?:function|const|let|var|class)\s+([A-Za-z0-9_$]+)",
                text,
            )
        )
        # Support `export { a, b as c };` (without `from`).
        for block in re.findall(r"\bexport\s+\{([^}]+)\}\s*(?!from\b)", text):
            for raw_item in block.split(","):
                item = raw_item.strip()
                if not item:
                    continue
                parts = [p.strip() for p in item.split(" as ")]
                exported.add(parts[-1])
        direct_exports[name] = exported

        reexport_entries: list[tuple[str, str, str]] = []
        for block, target in re.findall(
            r"""\bexport\s+\{([^}]+)\}\s+from\s+['"]\./([^'"]+)['"]""",
            text,
        ):
            for raw_item in block.split(","):
                item = raw_item.strip()
                if not item:
                    continue
                parts = [p.strip() for p in item.split(" as ")]
                orig_name = parts[0]
                alias_name = parts[-1]
                reexport_entries.append((orig_name, alias_name, target))
        reexports[name] = reexport_entries

    exports_by_file: dict[str, set[str]] = {
        name: set(names) for name, names in direct_exports.items()
    }
    broken: list[str] = []
    # Resolve re-exports (supports transitive chains).
    for _ in range(len(files) + 1):
        changed = False
        for name, entries in reexports.items():
            for orig_name, alias_name, target in entries:
                if target in exports_by_file and orig_name in exports_by_file[target]:
                    if alias_name not in exports_by_file[name]:
                        exports_by_file[name].add(alias_name)
                        changed = True
        if not changed:
            break

    for name, entries in reexports.items():
        for orig_name, _alias_name, target in entries:
            if target not in exports_by_file:
                broken.append(f"{name}: missing re-export module ./{target}")
            elif orig_name not in exports_by_file[target]:
                broken.append(f"{name}: re-exported {orig_name} is not exported by ./{target}")

    for path in files:
        text = sources[path.name]
        for target in re.findall(r"""\bfrom\s+['"]\./([^'"]+)['"]""", text):
            if target not in exports_by_file:
                broken.append(f"{path.name}: missing module ./{target}")
        for names, target in re.findall(
            r"""\bimport\s+\{([^}]+)\}\s+from\s+['"]\./([^'"]+)['"]""",
            text,
        ):
            if target not in exports_by_file:
                continue
            for raw_name in names.split(","):
                name = raw_name.strip().split(" as ")[0].strip()
                if name and name not in exports_by_file[target]:
                    broken.append(f"{path.name}: {name} is not exported by ./{target}")
    assert broken == []


def test_css_theme_tokens_meet_wcag_aa_contrast() -> None:
    """Guard core foreground/background tokens in style.css against AA regressions."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    root = css[css.index(":root {") : css.index("}", css.index(":root {"))]
    tokens = dict(re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", root))

    def parse_color(val: str) -> tuple[float, float, float, float]:
        val = val.strip()
        if val.startswith("#"):
            h = val[1:]
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 1.0)
        m = re.match(r"rgba?\(([^)]+)\)", val)
        assert m, f"Unsupported color value: {val!r}"
        parts = [float(p.strip()) for p in m.group(1).split(",")]
        return (parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else 1.0)

    def blend(fg: tuple[float, float, float, float], bg: tuple[float, float, float]) -> tuple[float, float, float]:
        a = fg[3]
        return (fg[0] * a + bg[0] * (1 - a), fg[1] * a + bg[1] * (1 - a), fg[2] * a + bg[2] * (1 - a))

    def luminance(rgb: tuple[float, float, float]) -> float:
        def ch(v: float) -> float:
            c = v / 255.0
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * ch(rgb[0]) + 0.7152 * ch(rgb[1]) + 0.0722 * ch(rgb[2])

    def contrast(fg: tuple[float, float, float], bg: tuple[float, float, float]) -> float:
        l1, l2 = luminance(fg), luminance(bg)
        return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

    bg = parse_color(tokens["--bg"])[:3]
    surface = blend(parse_color(tokens["--surface"]), bg)
    inset = blend(parse_color(tokens["--inset"]), surface)
    popover = blend(parse_color(tokens["--popover-bg"]), bg)

    checks = [
        ("text on surface", blend(parse_color(tokens["--text"]), surface), surface, 4.5),
        ("text-2 on surface", blend(parse_color(tokens["--text-2"]), surface), surface, 4.5),
        ("muted on surface", blend(parse_color(tokens["--muted"]), surface), surface, 4.5),
        ("placeholder on inset", blend(parse_color(tokens["--placeholder"]), inset), inset, 4.5),
        ("text on popover", blend(parse_color(tokens["--text"]), popover), popover, 4.5),
        ("text-2 on popover", blend(parse_color(tokens["--text-2"]), popover), popover, 4.5),
        ("white on btn-grad-a", (255.0, 255.0, 255.0), parse_color(tokens["--btn-grad-a"])[:3], 4.5),
        ("white on btn-grad-b", (255.0, 255.0, 255.0), parse_color(tokens["--btn-grad-b"])[:3], 4.5),
        ("white on btn-grad-c", (255.0, 255.0, 255.0), parse_color(tokens["--btn-grad-c"])[:3], 4.5),
        ("error on error-soft", blend(parse_color(tokens["--error"]), surface), blend(parse_color(tokens["--error-soft"]), surface), 4.5),
        ("warn on warn-soft", blend(parse_color(tokens["--warn"]), surface), blend(parse_color(tokens["--warn-soft"]), surface), 4.5),
    ]
    for label, fg_rgb, bg_rgb, minimum in checks:
        ratio = contrast(fg_rgb, bg_rgb)
        assert ratio >= minimum, f"{label}: contrast {ratio:.2f}:1 < {minimum}:1"


def test_enabling_llamacpp_settings_triggers_runtime_prefetch_and_rejects_server_path(
    monkeypatch, tmp_path
) -> None:
    import asyncio
    import httpx
    from llm.backends import llamacpp_runtime
    from web.app import create_app
    from web.routes import config as config_routes

    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setattr(config_routes, "PROJECT_ROOT", tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        llamacpp_runtime,
        "start_managed_llama_server_prefetch",
        lambda: calls.append("prefetch") or "started",
    )

    async def _run() -> None:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/api/settings", json={"translation_backend": "llamacpp"})
            assert r.status_code == 200
            assert r.json() == {"ok": True, "llama_runtime_prefetch": "started"}
            assert calls == ["prefetch"]

            r_bad = await client.post(
                "/api/settings", json={"llamacpp_server_path": "D:\\llama"}
            )
            assert r_bad.status_code == 422

    asyncio.run(_run())


def test_web_settings_save_refuses_to_overwrite_truncated_empty_config(
    monkeypatch, tmp_path
) -> None:
    import asyncio
    import httpx
    import bootstrap
    from web.app import create_app
    from web.routes import config as config_routes

    monkeypatch.setattr(config_routes, "PROJECT_ROOT", tmp_path)
    cfg = tmp_path / "config" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("", encoding="utf-8")
    retired_gguf = tmp_path / "models" / "hub" / "models--Qwen--Qwen3-8B" / "blobs" / "old.gguf"
    retired_gguf.parent.mkdir(parents=True, exist_ok=True)
    retired_gguf.write_bytes(b"weights")
    assert bootstrap.retired_model_cache_paths(tmp_path) == []

    async def _run() -> None:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/api/settings", json={"target_lang": "繁体中文"})
            assert r.status_code == 400
            assert "为空" in r.json()["detail"] or "JSON" in r.json()["detail"]

    asyncio.run(_run())
    assert cfg.read_text(encoding="utf-8") == ""
    assert bootstrap.retired_model_cache_paths(tmp_path) == []
