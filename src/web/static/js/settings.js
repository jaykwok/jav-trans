import { state } from './state.js';
import { $ } from './util.js';
import { loadFormMemory, saveFormMemory, applyFormMemory } from './formMemory.js';
import { setActivePreset } from './presets.js';

const _BACKEND_LABELS = {
  'whisper-ja-anime-v0.3': 'whisper-ja-anime-v0.3（推荐）',
  'anime-whisper':          'AnimeWhisper',
  'qwen3-asr-1.7b':         'Qwen3-ASR-1.7B',
  'whisper-ja-1.5b':        'whisper-ja-1.5B',
};
const _SUBTITLE_MODE_LABELS = { zh: '中文字幕', bilingual: '中日双语' };

export async function loadConfig() {
  try {
    const r = await fetch('/api/config');
    if (!r.ok) return;
    const cfg = await r.json();

    const backendSel = $('r-backend');
    backendSel.innerHTML = '';
    for (const b of (cfg.backends || [])) {
      const opt = document.createElement('option');
      opt.value = b;
      opt.textContent = _BACKEND_LABELS[b] || b;
      backendSel.appendChild(opt);
    }
    if (cfg.defaults?.asr_backend && !Object.hasOwn(loadFormMemory().controls, 'r-backend')) {
      backendSel.value = cfg.defaults.asr_backend;
    }

    const modeSel = $('r-mode');
    modeSel.innerHTML = '';
    for (const m of (cfg.subtitle_modes || [])) {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = _SUBTITLE_MODE_LABELS[m] || m;
      modeSel.appendChild(opt);
    }
    if (cfg.defaults?.subtitle_mode) modeSel.value = cfg.defaults.subtitle_mode;

    const d = cfg.defaults ?? {};
    if (d.vad_threshold             != null) $('t-vad-threshold').value           = d.vad_threshold;
    if (d.translation_batch_size    != null) $('t-translation-batch-size').value  = d.translation_batch_size;
    if (d.translation_max_workers   != null) $('t-translation-max-workers').value = d.translation_max_workers;
    if (d.multi_cue_split           != null) $('t-multi-cue-split').checked       = !!d.multi_cue_split;
    if (d.show_gender               != null) $('t-show-gender').checked           = !!d.show_gender;
    if (d.asr_recovery              != null) $('t-asr-recovery').checked          = !!d.asr_recovery;
    if (d.skip_translation          != null) $('r-skip-translation').checked      = !!d.skip_translation;
    applyFormMemory();
    setActivePreset(state.activePreset);
  } catch {}
}

export async function loadSettings() {
  try {
    const r = await fetch('/api/settings');
    if (!r.ok) return;
    const s = await r.json();

    $('api-key-preview').textContent = s.api_key_preview
      ? '当前：' + s.api_key_preview
      : '当前：未设置';
    if (s.base_url) $('api-base-url').value = s.base_url;
    if (s.model) {
      const sel = $('api-model');
      sel.innerHTML = `<option value="${s.model}">${s.model}</option>`;
      sel.disabled = false;
      $('api-model-preview').textContent = '当前：' + s.model;
    }
    $('mirror-enabled').checked = s.hf_endpoint === 'https://hf-mirror.com';
    const asrContext = $('r-asr-context');
    if (asrContext) asrContext.value = s.asr_context || '';

    const effort = $('api-reasoning-effort');
    if (effort) effort.value = s.llm_reasoning_effort || 'xhigh';
    const apiFormat = $('api-format');
    if (apiFormat) apiFormat.value = s.llm_api_format || 'chat';
    const targetLang = $('api-target-lang');
    if (targetLang) targetLang.value = s.target_lang || '简体中文';
    const glossary = $('api-glossary');
    if (glossary) {
      glossary.value = (s.translation_glossary || '')
        .split(',').map(t => t.trim()).filter(Boolean).join('\n');
    }

    const isConfigured = !!(s.base_url && s.model && s.api_key_preview && !s.api_key_preview.includes('未设置'));
    const saved = loadFormMemory();
    if (!Object.hasOwn(saved.details, 'panel-translation')) {
      const pt = $('panel-translation');
      if (pt) pt.open = !isConfigured;
    }
  } catch {}
}

export function readTranslationSettingsFromForm() {
  return {
    llm_reasoning_effort: $('api-reasoning-effort').value || 'xhigh',
    llm_api_format:       $('api-format').value || 'chat',
    target_lang:          $('api-target-lang').value || '简体中文',
    translation_glossary: ($('api-glossary').value || '')
      .split('\n').map(t => t.trim()).filter(Boolean).join(', '),
  };
}

function buildSettingsBodyFromForm({ includeConnection = false, includeMirror = false } = {}) {
  const body = readTranslationSettingsFromForm();
  body.asr_context = $('r-asr-context')?.value.trim() || '';
  if (includeConnection) {
    const apiKey = $('api-key').value.trim();
    const baseUrl = $('api-base-url').value.trim();
    const model = $('api-model').value.trim();
    if (apiKey) body.api_key = apiKey;
    if (baseUrl) body.base_url = baseUrl;
    if (model) body.model = model;
  }
  if (includeMirror) {
    body.hf_endpoint = $('mirror-enabled').checked ? 'https://hf-mirror.com' : '';
  }
  return body;
}

function clearApiKeyInputIfSaved(body) {
  if (!body.api_key) return;
  $('api-key').value = '';
  $('api-key').type = 'password';
  $('btn-show-key').textContent = '👁';
}

export async function saveSettingsBody(body) {
  const r = await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  clearApiKeyInputIfSaved(body);
}

export async function syncSettingsFromFormForSubmit() {
  const body = buildSettingsBodyFromForm({
    includeConnection: true,
    includeMirror: true,
  });
  await saveSettingsBody(body);
  saveFormMemory();
}

export function installSettingsPanel() {
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
      if (apiKeyInput) {
        try {
          await saveSettingsBody(buildSettingsBodyFromForm({ includeConnection: true }));
        } catch (e) {
          alert('保存 API Key 失败：' + e.message);
          return;
        }
        await loadSettings();
      }

      const r = await fetch('/api/models');
      if (r.status === 400) {
        alert('配置不完整：' + await r.text());
        return;
      }
      if (r.status === 401 || r.status === 403) {
        alert('API Key 无效或无权限，请检查 Key 是否正确');
        return;
      }
      if (!r.ok) {
        alert('获取失败（' + r.status + '），请检查 Base URL：\n' + await r.text());
        return;
      }
      const { models } = await r.json();
      if (!models.length) {
        alert('API 未返回任何模型，请确认 Base URL 和 Key 填写正确');
        return;
      }
      const sel = $('api-model');
      const current = sel.value;
      sel.innerHTML = models.map(m =>
        `<option value="${m}"${m === current ? ' selected' : ''}>${m}</option>`
      ).join('');
      sel.disabled = false;
      const wrap = $('api-model-wrap');
      if (wrap) wrap.removeAttribute('title');
    } catch (e) {
      alert('获取模型出错，请检查网络或 Base URL：\n' + e.message);
    } finally {
      btn.textContent = '获取';
      btn.disabled = false;
    }
  });

}
