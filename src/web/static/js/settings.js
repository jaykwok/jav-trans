import { $, showToast, readErrorDetail } from './util.js';
import { addLog } from './log.js';
import { loadFormMemory, saveFormMemory, applyFormMemory } from './formMemory.js';
import { setModelComboboxOptions, installModelCombobox } from './modelCombobox.js';
import { refreshModelRequirements } from './modelRequirements.js';
import {
  updateProxyFieldsState,
  applyProxySettingsToForm,
  readProxySettingsFromForm,
  shouldIncludeProxyOnSubmit,
  installProxySettings,
} from './proxySettings.js';

export { refreshModelRequirements } from './modelRequirements.js';
export { updateProxyFieldsState } from './proxySettings.js';

export function subtitleModeLabel(mode, targetLang) {
  if (mode === 'ja_only') return '仅日文（不翻译）';
  const language = String(targetLang || '简体中文').trim() || '简体中文';
  if (mode === 'bilingual') return `双语（日文 + ${language}）`;
  return language;
}

export function updateSubtitleModeLabels() {
  const modeSel = $('r-mode');
  if (!modeSel) return;
  const targetLang = $('api-target-lang')?.value || '简体中文';
  for (const option of modeSel.options) {
    option.textContent = subtitleModeLabel(option.value, targetLang);
  }
}

export function updateSkipTransState() {
  const skip = $('r-mode')?.value === 'ja_only';
  const panel = $('panel-translation');
  if (panel) panel.classList.toggle('disabled', skip);
}

export function formatGlossaryForTextarea(raw) {
  const text = String(raw || '').replace(/\r\n?/g, '\n').trim();
  if (!text) return '';
  if (text.includes('\n')) {
    return text.split('\n').map(line => line.trim()).filter(Boolean).join('\n');
  }
  // Legacy single-line comma-separated glossary: split only where a comma starts
  // a new `source-target` pair so commas inside a translation target stay intact.
  return text
    .split(/,\s*(?=[^,\n]+(?:\s*(?:->|→|-)\s*[^,\n]+))/)
    .map(entry => entry.trim())
    .filter(Boolean)
    .join('\n');
}

function parseGlossaryLine(line) {
  const trimmed = String(line || '').trim();
  if (!trimmed) return { empty: true, normalized: '', convertedArrow: false, invalid: false };
  let separator = null;
  let sepIdx = -1;
  for (const candidate of ['→', '->', '-']) {
    const idx = trimmed.indexOf(candidate);
    if (idx >= 0 && (sepIdx < 0 || idx < sepIdx)) {
      sepIdx = idx;
      separator = candidate;
    }
  }
  if (!separator || sepIdx <= 0) {
    return { empty: false, normalized: '', convertedArrow: false, invalid: true };
  }
  const source = trimmed.slice(0, sepIdx).trim();
  const target = trimmed.slice(sepIdx + separator.length).trim();
  if (!source || !target) {
    return { empty: false, normalized: '', convertedArrow: false, invalid: true };
  }
  // Normalize fullwidth/ASCII commas inside target to Chinese dunhao only if
  // needed? No: store newline-separated so user wording survives round-trips.
  return {
    empty: false,
    normalized: `${source}-${target}`,
    convertedArrow: separator !== '-',
    invalid: false,
  };
}

export function inspectGlossaryText(raw) {
  const lines = String(raw || '').replace(/\r\n?/g, '\n').split('\n');
  const entries = [];
  const invalidLines = [];
  let convertedArrows = 0;
  for (const line of lines) {
    const parsed = parseGlossaryLine(line);
    if (parsed.empty) continue;
    if (parsed.invalid) {
      invalidLines.push(line.trim());
    } else {
      if (parsed.convertedArrow) convertedArrows += 1;
      entries.push(parsed.normalized);
    }
  }
  return {
    serialized: entries.join('\n'),
    invalidLines,
    convertedArrows,
  };
}

export async function loadConfig() {
  try {
    const r = await fetch('/api/config');
    if (!r.ok) {
      addLog(`加载默认配置失败（HTTP ${r.status}）：${await readErrorDetail(r)}`, 'stage-error');
      return;
    }
    const cfg = await r.json();

    const modeSel = $('r-mode');
    if (modeSel) {
      const modes = [...(cfg.subtitle_modes || ['zh', 'bilingual'])];
      if (!modes.includes('ja_only')) modes.push('ja_only');
      modeSel.innerHTML = '';
      for (const m of modes) {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = subtitleModeLabel(m, $('api-target-lang')?.value);
        modeSel.appendChild(opt);
      }
      if (cfg.defaults?.skip_translation) {
        modeSel.value = 'ja_only';
      } else if (cfg.defaults?.subtitle_mode && modes.includes(cfg.defaults.subtitle_mode)) {
        modeSel.value = cfg.defaults.subtitle_mode;
      }
    }

    const d = cfg.defaults ?? {};
    const workersEl = $('t-translation-max-workers');
    if (d.translation_max_workers != null && workersEl) workersEl.value = d.translation_max_workers;
    applyFormMemory();
    updateSubtitleModeLabels();
    updateSkipTransState();
    refreshModelRequirements();
  } catch (error) {
    addLog(`加载默认配置出错：${error?.message || String(error)}`, 'stage-error');
  }
}

export async function loadSettings({ previewsOnly = false } = {}) {
  try {
    const r = await fetch('/api/settings');
    if (!r.ok) {
      addLog(`加载翻译设置失败（HTTP ${r.status}）：${await readErrorDetail(r)}`, 'stage-error');
      return;
    }
    const s = await r.json();

    $('api-key-preview').textContent = s.api_key_preview
      ? '当前：' + s.api_key_preview
      : '当前：未设置';
    $('api-model-preview').textContent = s.model ? '当前：' + s.model : '当前：未设置';
    if (previewsOnly) return;

    const backendSel = $('translation-backend');
    if (backendSel && s.translation_backend) {
      backendSel.value = s.translation_backend;
      backendSel.dispatchEvent(new Event('change'));
    }

    if (s.base_url) $('api-base-url').value = s.base_url;
    if (s.model) {
      setModelComboboxOptions([s.model], { selected: s.model });
    }

    applyProxySettingsToForm(s);

    const effort = $('api-reasoning-effort');
    if (effort) effort.value = s.llm_reasoning_effort || 'low';
    const targetLang = $('api-target-lang');
    if (targetLang) targetLang.value = s.target_lang || '简体中文';
    updateSubtitleModeLabels();
    const glossary = $('api-glossary');
    if (glossary) {
      glossary.value = formatGlossaryForTextarea(s.translation_glossary);
    }

    const isConfigured = s.translation_backend === 'llamacpp'
      || !!(s.base_url && s.model && s.api_key_preview && !s.api_key_preview.includes('未设置'));
    const saved = loadFormMemory();
    if (!Object.hasOwn(saved.details, 'panel-translation')) {
      const pt = $('panel-translation');
      if (pt) pt.open = !isConfigured;
    }
  } catch (error) {
    addLog(`加载翻译设置出错：${error?.message || String(error)}`, 'stage-error');
  }
}

// JobSpec is `extra="forbid"`, so the job body may carry only fields it
// declares. Backend choice belongs to the settings API.
export function readJobTranslationSpecFromForm() {
  return {
    llm_reasoning_effort: $('api-reasoning-effort')?.value || 'low',
    target_lang:          $('api-target-lang')?.value || '简体中文',
    translation_glossary: inspectGlossaryText($('api-glossary')?.value || '').serialized,
  };
}

export function readTranslationSettingsFromForm() {
  const backend = $('translation-backend')?.value || 'llamacpp';
  return {
    translation_backend: backend,
    ...readJobTranslationSpecFromForm(),
  };
}

function buildSettingsBodyFromForm({ includeConnection = false, includeProxy = false, includeTranslation = true } = {}) {
  const body = includeTranslation ? readTranslationSettingsFromForm() : {};
  if (includeConnection) {
    const apiKey = $('api-key').value.trim();
    const baseUrl = $('api-base-url').value.trim();
    const model = $('api-model').value.trim();
    if (apiKey) body.api_key = apiKey;
    if (baseUrl) body.base_url = baseUrl;
    if (model) body.model = model;
  }
  if (includeProxy) {
    Object.assign(body, readProxySettingsFromForm());
  }
  return body;
}

function clearApiKeyInputIfSaved(body) {
  if (!body.api_key) return;
  if ($('api-key').value.trim() !== body.api_key) return;
  $('api-key').value = '';
  $('api-key').type = 'password';
  $('btn-show-key').textContent = '👁';
}

let _settingsSaveQueue = Promise.resolve();

export function saveSettingsBody(body) {
  const serialized = JSON.stringify(body);
  const saved = _settingsSaveQueue.then(async () => {
    const r = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: serialized,
    });
    if (!r.ok) throw new Error(await readErrorDetail(r));
    const data = typeof r.json === 'function' ? await r.json().catch(() => ({ ok: true })) : { ok: true };
    clearApiKeyInputIfSaved(body);
    return data;
  });
  _settingsSaveQueue = saved.catch(() => {});
  return saved;
}

export async function syncSettingsFromFormForSubmit() {
  const body = buildSettingsBodyFromForm({
    includeConnection: true,
    includeProxy: shouldIncludeProxyOnSubmit(),
  });
  await saveSettingsBody(body);
  saveFormMemory();
}

export function installSettingsPanel() {
  installModelCombobox();
  installProxySettings({ saveSettingsBody });

  $('btn-show-key').addEventListener('click', () => {
    const inp = $('api-key');
    const show = inp.type === 'password';
    inp.type = show ? 'text' : 'password';
    $('btn-show-key').textContent = show ? '🙈' : '👁';
  });

  $('btn-fetch-models').addEventListener('click', async () => {
    const baseUrl     = $('api-base-url').value.trim();
    const apiKeyInput = $('api-key').value.trim();
    const keyPreview  = $('api-key-preview').textContent || '';
    const hasStoredKey = keyPreview && !keyPreview.includes('未设置');

    if (!baseUrl) {
      alert('请先填写 API Base URL');
      return;
    }
    if (!apiKeyInput && !hasStoredKey) {
      alert('请先填写 API Key');
      return;
    }

    const btn = $('btn-fetch-models');
    btn.textContent = '获取中…';
    btn.disabled = true;
    try {
      try {
        await saveSettingsBody(buildSettingsBodyFromForm({ includeConnection: true }));
      } catch (e) {
        alert('保存 API 设置失败：' + e.message);
        return;
      }
      if (apiKeyInput) {
        await loadSettings({ previewsOnly: true });
      }

      const r = await fetch('/api/models');
      if (r.status === 400) {
        alert('配置不完整：' + await readErrorDetail(r));
        return;
      }
      if (r.status === 401 || r.status === 403) {
        alert('API Key 无效或无权限，请检查 Key 是否正确');
        return;
      }
      if (!r.ok) {
        alert('获取失败（' + r.status + '），请检查 Base URL：\n' + await readErrorDetail(r));
        return;
      }
      const { models } = await r.json();
      if ($('api-base-url').value.trim() !== baseUrl ||
          ($('api-key').value.trim() && $('api-key').value.trim() !== apiKeyInput)) {
        showToast('API 设置已变化，请重新获取模型');
        return;
      }
      if (!models.length) {
        alert('API 未返回任何模型，请确认 Base URL 和 Key 填写正确');
        return;
      }
      setModelComboboxOptions(models);
      const wrap = $('api-model-wrap');
      if (wrap) wrap.removeAttribute('title');
    } catch (e) {
      alert('获取模型出错，请检查网络或 Base URL：\n' + e.message);
    } finally {
      btn.textContent = '获取';
      btn.disabled = false;
    }
  });

  const updateTranslationBackendFields = () => {
    const backend = $('translation-backend')?.value || 'llamacpp';
    const openaiFields = $('openai-backend-fields');
    const llamacppFields = $('llamacpp-backend-fields');
    if (openaiFields) openaiFields.style.display = backend === 'openai' ? '' : 'none';
    if (llamacppFields) llamacppFields.style.display = backend === 'llamacpp' ? '' : 'none';
  };
  $('translation-backend')?.addEventListener('change', updateTranslationBackendFields);
  updateTranslationBackendFields();

  $('api-target-lang')?.addEventListener('change', updateSubtitleModeLabels);
  $('r-mode')?.addEventListener('change', updateSkipTransState);
  updateSkipTransState();

  $('btn-save-translation')?.addEventListener('click', async () => {
    const btn = $('btn-save-translation');
    if (!btn || btn.disabled) return;
    const prevText = btn.textContent;
    btn.disabled = true;
    btn.textContent = '保存中…';
    try {
      const glossaryEl = $('api-glossary');
      const glossaryCheck = inspectGlossaryText(glossaryEl?.value || '');
      if (glossaryCheck.convertedArrows > 0 && glossaryEl) {
        glossaryEl.value = glossaryCheck.serialized;
      }
      const result = await saveSettingsBody(buildSettingsBodyFromForm({ includeConnection: true }));
      saveFormMemory();
      await loadSettings({ previewsOnly: true });
      if (glossaryCheck.invalidLines.length > 0) {
        showToast(
          `翻译设置已保存到 config/config.json（已忽略 ${glossaryCheck.invalidLines.length} 行格式无效的术语，请使用“日文-译文”）`,
        );
      } else if (result?.llama_runtime_prefetch === 'started') {
        showToast('翻译设置已保存到 config/config.json，正在后台下载本地翻译引擎');
      } else {
        showToast('翻译设置已保存到 config/config.json');
      }
    } catch (e) {
      showToast('保存翻译设置失败：' + e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = prevText;
    }
  });
}
