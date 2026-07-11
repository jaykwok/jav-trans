import { state } from './state.js';
import { $, escHtml, showToast } from './util.js';
import { loadFormMemory, saveFormMemory, applyFormMemory } from './formMemory.js';
import { setActivePreset } from './presets.js';

const _BACKEND_LABELS = {
  'jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf': 'Qwen3-ASR-0.6B-JA-Anime-Galgame-hf（低显存）',
  'jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf': 'Qwen3-ASR-1.7B-JA-Anime-Galgame-hf',
};
const _SUBTITLE_MODE_LABELS = { zh: '中文字幕', bilingual: '中日双语' };

function formatBackendLabel(repoId, recommendedRepoId) {
  const id = String(repoId || '');
  const label = _BACKEND_LABELS[id] || id.split('/').pop() || id;
  return id === recommendedRepoId ? `${label}（推荐）` : label;
}

let _modelRequirementsRequestSeq = 0;

function formatRequirementLabel(item) {
  const labels = Array.isArray(item.role_labels) ? item.role_labels.join('/') : '';
  const name = item.short_name || String(item.repo_id || '').split('/').pop() || item.repo_id || '';
  if (labels === name) return labels;
  return labels ? `${labels} ${name}` : name;
}

function renderModelRequirements(payload) {
  const notice = $('model-requirements-notice');
  if (!notice) return;
  const missingModels = (payload.required_models || []).filter(item => !item.present);
  const missingCheckpoints = (payload.required_checkpoints || []).filter(item => !item.present);
  const missing = [...missingModels, ...missingCheckpoints];
  const cuda = payload.cuda || {};
  const cudaProblem = cuda.status && cuda.status !== 'ok';
  if (!missing.length && !cudaProblem) {
    notice.hidden = true;
    notice.textContent = '';
    return;
  }

  const sections = [];
  if (missing.length) {
    let message;
    if (missingCheckpoints.length && missingModels.length) {
      message = `当前配置缺少 ${missingModels.length} 个基础模型和 ${missingCheckpoints.length} 个前置 checkpoint；基础模型可按配置下载，checkpoint 需要先准备。`;
    } else if (missingCheckpoints.length) {
      message = `当前 ASR 后端缺少 ${missingCheckpoints.length} 个前置 checkpoint，不能运行完整五模型链。`;
    } else if (payload.download_disabled && payload.needs_download) {
      message = `当前配置缺少 ${missingModels.length} 个模型；可自动下载的模型会在首次运行下载，已关闭自动下载的模型需要先准备。`;
    } else if (payload.download_disabled) {
      message = '当前配置缺少基础模型文件，且已关闭自动下载，需要先准备本地模型。';
    } else {
      message = `首次使用该配置需要下载 ${missingModels.length} 个模型；下载完成后会复用本地缓存。`;
    }
    const missingText = missing.map(formatRequirementLabel).join('、');
    sections.push(`${escHtml(message)}<br><strong>缺少：</strong>${escHtml(missingText)}`);
  }

  if (cudaProblem) {
    const smi = cuda.nvidia_smi || {};
    const runtime = cuda.torch_cuda_version ? `PyTorch CUDA ${cuda.torch_cuda_version}` : '';
    const driverCuda = smi.cuda_version ? `驱动 CUDA ${smi.cuda_version}` : '';
    const driver = smi.driver_version ? `驱动 ${smi.driver_version}` : '';
    const detail = [runtime, driverCuda, driver].filter(Boolean).join(' · ');
    const message = cuda.message || 'CUDA 环境不可用，请更新 NVIDIA 显卡驱动后重启应用。';
    sections.push(`${escHtml(message)}${detail ? `<br><strong>环境：</strong>${escHtml(detail)}` : ''}`);
  }

  notice.innerHTML = sections.join('<br>');
  notice.hidden = false;
}

export async function refreshModelRequirements() {
  const notice = $('model-requirements-notice');
  const backendSel = $('r-backend');
  if (!notice || !backendSel?.value) return;

  const requestSeq = ++_modelRequirementsRequestSeq;
  try {
    const r = await fetch(`/api/model-requirements?asr_backend=${encodeURIComponent(backendSel.value)}`);
    if (requestSeq !== _modelRequirementsRequestSeq) return;
    if (!r.ok) {
      notice.hidden = true;
      return;
    }
    renderModelRequirements(await r.json());
  } catch {
    if (requestSeq === _modelRequirementsRequestSeq) notice.hidden = true;
  }
}

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
      opt.dataset.repoId = b;
      opt.title = b;
      opt.textContent = formatBackendLabel(b, cfg.recommended_asr_backend);
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
    if (d.translation_max_workers   != null) $('t-translation-max-workers').value = d.translation_max_workers;
    if (d.skip_translation          != null) $('r-skip-translation').checked      = !!d.skip_translation;
    applyFormMemory();
    setActivePreset(state.activePreset);
    refreshModelRequirements();
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
      sel.innerHTML = '';
      const opt = document.createElement('option');
      opt.value = s.model;
      opt.textContent = s.model;
      sel.appendChild(opt);
      sel.disabled = false;
      $('api-model-preview').textContent = '当前：' + s.model;
    }
    const proxyProtocol = $('proxy-protocol');
    if (proxyProtocol) proxyProtocol.value = s.proxy_protocol || 'http';
    const proxyHost = $('proxy-host');
    if (proxyHost) proxyHost.value = s.proxy_host || '';
    const proxyPort = $('proxy-port');
    if (proxyPort) proxyPort.value = s.proxy_port || '';
    // Proxy is "on" exactly when a host and port are configured; reflect that
    // in the enable switch and the summary status pill, and disable the fields
    // when off so the state is obvious.
    const proxyOn = !!(s.proxy_host && s.proxy_port);
    const proxyEnabled = $('proxy-enabled');
    if (proxyEnabled) proxyEnabled.checked = proxyOn;
    updateProxyFieldsState();

    const effort = $('api-reasoning-effort');
    if (effort) effort.value = s.llm_reasoning_effort || 'medium';
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
  const normalizeGlossaryLine = line => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.includes('→') || trimmed.includes('->') || !trimmed.includes('-')) return '';
    const [source, ...rest] = trimmed.split('-');
    const target = rest.join('-').trim();
    const normalizedSource = source.trim();
    return normalizedSource && target ? `${normalizedSource}-${target}` : '';
  };

  return {
    llm_reasoning_effort: $('api-reasoning-effort').value || 'medium',
    llm_api_format:       $('api-format').value || 'chat',
    target_lang:          $('api-target-lang').value || '简体中文',
    translation_glossary: ($('api-glossary').value || '')
      .split('\n').map(normalizeGlossaryLine).filter(Boolean).join(', '),
  };
}

function updateProxyFieldsState() {
  const on = !!($('proxy-enabled')?.checked);
  for (const id of ['proxy-protocol', 'proxy-host', 'proxy-port']) {
    const el = $(id);
    if (el) el.disabled = !on;
  }
  const testBtn = $('btn-proxy-test');
  if (testBtn && !testBtn.dataset.testing) testBtn.disabled = !on;
  const tag = $('proxy-status-tag');
  if (tag) {
    tag.textContent = on ? '已启用' : '未启用';
    tag.dataset.on = on ? 'on' : 'off';
  }
}

function buildSettingsBodyFromForm({ includeConnection = false, includeProxy = false } = {}) {
  const body = readTranslationSettingsFromForm();
  if (includeConnection) {
    const apiKey = $('api-key').value.trim();
    const baseUrl = $('api-base-url').value.trim();
    const model = $('api-model').value.trim();
    if (apiKey) body.api_key = apiKey;
    if (baseUrl) body.base_url = baseUrl;
    if (model) body.model = model;
  }
  if (includeProxy) {
    // The enable switch is the single source of truth for on/off. Switch off
    // -> clear host/port on save (the backend tears down the proxy). Switch on
    // -> send the field values (port validated when present).
    const enabled = !!$('proxy-enabled')?.checked;
    const portText = ($('proxy-port')?.value || '').trim();
    const parsedPort = portText ? Number(portText) : null;
    if (enabled && portText && (!/^\d+$/.test(portText) || !Number.isInteger(parsedPort) || parsedPort < 1 || parsedPort > 65535)) {
      throw new Error('代理端口必须是 1-65535 的数字');
    }
    body.proxy_protocol = $('proxy-protocol')?.value || 'http';
    body.proxy_host = enabled ? ($('proxy-host')?.value || '').trim() : '';
    body.proxy_port = enabled ? parsedPort : null;
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
    includeProxy: true,
  });
  await saveSettingsBody(body);
  saveFormMemory();
}

export function installSettingsPanel() {
  $('r-backend')?.addEventListener('change', refreshModelRequirements);

  const saveProxySettings = async () => {
    try {
      await saveSettingsBody(buildSettingsBodyFromForm({ includeProxy: true }));
      saveFormMemory();
      const host = ($('proxy-host')?.value || '').trim();
      const port = ($('proxy-port')?.value || '').trim();
      showToast(host && port ? '代理设置已保存' : '代理设置已关闭');
    } catch (e) {
      showToast('保存代理设置失败：' + e.message);
    }
  };
  for (const id of ['proxy-protocol', 'proxy-host', 'proxy-port']) {
    const el = $(id);
    if (el) el.addEventListener('change', saveProxySettings);
  }
  $('proxy-enabled')?.addEventListener('change', () => {
    updateProxyFieldsState();
    saveProxySettings();
  });

  const proxyTestBtn = $('btn-proxy-test');
  proxyTestBtn?.addEventListener('click', async () => {
    if (proxyTestBtn.disabled) return;
    const resultEl = $('proxy-test-result');
    const prevText = proxyTestBtn.textContent;
    proxyTestBtn.dataset.testing = '1';
    proxyTestBtn.disabled = true;
    proxyTestBtn.textContent = '测试中…';
    if (resultEl) { resultEl.textContent = ''; resultEl.className = 'proxy-test-result'; }
    try {
      // Apply the proxy to the runtime env first, then ask the backend to try
      // reaching HuggingFace through it -- a wrong port fails loud here instead
      // of silently hanging later model downloads.
      await saveProxySettings();
      const r = await fetch('/api/proxy-test', { method: 'POST' });
      const data = await r.json().catch(() => ({}));
      if (resultEl) {
        if (data.ok) {
          resultEl.textContent = `✓ 经代理连通 HuggingFace（${data.elapsed_ms ?? '?'}ms）`;
          resultEl.className = 'proxy-test-result ok';
        } else {
          resultEl.textContent = '✗ ' + (data.error || '连接失败');
          resultEl.className = 'proxy-test-result fail';
        }
      }
    } catch (e) {
      if (resultEl) {
        resultEl.textContent = '✗ 测试失败：' + e.message;
        resultEl.className = 'proxy-test-result fail';
      }
    } finally {
      delete proxyTestBtn.dataset.testing;
      proxyTestBtn.textContent = prevText;
      updateProxyFieldsState();
    }
  });

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
        `<option value="${escHtml(m)}"${m === current ? ' selected' : ''}>${escHtml(m)}</option>`
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
