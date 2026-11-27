import { $, showToast, readErrorDetail } from './util.js';
import { saveFormMemory } from './formMemory.js';

export function updateProxyFieldsState() {
  const on = !!($('proxy-enabled')?.checked);
  const testBtn = $('btn-proxy-test');
  const testing = !!testBtn?.dataset.testing;
  const enabled = $('proxy-enabled');
  if (enabled) enabled.disabled = testing;
  for (const id of ['proxy-protocol', 'proxy-host', 'proxy-port']) {
    const el = $(id);
    if (el) el.disabled = !on || testing;
  }
  if (testBtn) testBtn.disabled = !on || testing;
  const tag = $('proxy-status-tag');
  if (tag) {
    tag.textContent = on ? '已启用' : '未启用';
    tag.dataset.on = on ? 'on' : 'off';
  }
}

export function applyProxySettingsToForm(s) {
  const proxyProtocol = $('proxy-protocol');
  if (proxyProtocol) proxyProtocol.value = s.proxy_protocol || 'http';
  const proxyHost = $('proxy-host');
  if (proxyHost) proxyHost.value = s.proxy_host || '';
  const proxyPort = $('proxy-port');
  if (proxyPort) proxyPort.value = s.proxy_port || '';
  const proxyOn = !!(s.proxy_host && s.proxy_port);
  const proxyEnabled = $('proxy-enabled');
  if (proxyEnabled) proxyEnabled.checked = proxyOn;
  updateProxyFieldsState();
}

export function readProxySettingsFromForm() {
  // The enable switch is the single source of truth for on/off. Switch off
  // -> clear host/port on save (the backend tears down the proxy). Switch on
  // -> require both address and a valid port before saving.
  const enabled = !!$('proxy-enabled')?.checked;
  const host = ($('proxy-host')?.value || '').trim();
  const portText = ($('proxy-port')?.value || '').trim();
  if (enabled && (!host || !portText)) throw new Error('请填写代理地址和端口');
  let parsedPort = null;
  if (enabled && portText) {
    if (!/^\d+$/.test(portText)) {
      throw new Error('代理端口必须是 1-65535 的数字');
    }
    parsedPort = Number(portText);
    if (!Number.isInteger(parsedPort) || parsedPort < 1 || parsedPort > 65535) {
      throw new Error('代理端口必须是 1-65535 的数字');
    }
  }
  return {
    proxy_protocol: $('proxy-protocol')?.value || 'http',
    proxy_host: enabled ? host : '',
    proxy_port: enabled && portText ? parsedPort : null,
  };
}

/**
 * Proxy changes auto-save on change. During job submission, ignore a disabled
 * or collapsed proxy panel so incomplete proxy inputs never block unrelated jobs.
 */
export function shouldIncludeProxyOnSubmit() {
  const enabled = !!$('proxy-enabled')?.checked;
  if (!enabled) return false;
  const panel = $('panel-proxy');
  return Boolean(panel?.open);
}

function canonicalProxyUrl(rawUrl) {
  const text = String(rawUrl || '').trim();
  const match = text.match(/^([a-zA-Z][a-zA-Z0-9+.-]*):\/\/(?:([^/\s]+)@)?(.+)$/);
  if (!match) return text.toLowerCase();
  const [, scheme, creds, hostPort] = match;
  return `${scheme.toLowerCase()}://${creds ? '***@' : ''}${hostPort.toLowerCase()}`;
}

export function installProxySettings({ saveSettingsBody }) {
  let proxyRevision = 0;
  const invalidateProxyTest = () => {
    proxyRevision += 1;
    const result = $('proxy-test-result');
    if (result) { result.textContent = ''; result.className = 'proxy-test-result'; }
  };
  const saveProxySettings = async ({ notify = true } = {}) => {
    const revision = proxyRevision;
    const body = readProxySettingsFromForm();
    await saveSettingsBody(body);
    saveFormMemory();
    if (notify && revision === proxyRevision) {
      showToast(body.proxy_host && body.proxy_port ? '代理设置已保存' : '代理已关闭');
    }
    return body;
  };
  const onProxyChange = async () => {
    invalidateProxyTest();
    const revision = proxyRevision;
    updateProxyFieldsState();
    try {
      await saveProxySettings();
      if (revision === proxyRevision) updateProxyFieldsState();
    } catch (e) {
      if (revision !== proxyRevision) return;
      showToast('保存代理设置失败：' + e.message);
      const tag = $('proxy-status-tag');
      if (tag) tag.textContent = '未保存';
    }
  };
  for (const id of ['proxy-protocol', 'proxy-host', 'proxy-port']) {
    const el = $(id);
    if (el) {
      el.addEventListener('input', invalidateProxyTest);
      el.addEventListener('change', onProxyChange);
    }
  }
  $('proxy-enabled')?.addEventListener('change', onProxyChange);

  const proxyTestBtn = $('btn-proxy-test');
  proxyTestBtn?.addEventListener('click', async () => {
    if (proxyTestBtn.disabled) return;
    const resultEl = $('proxy-test-result');
    const prevText = proxyTestBtn.textContent;
    const revision = proxyRevision;
    proxyTestBtn.dataset.testing = '1';
    updateProxyFieldsState();
    proxyTestBtn.textContent = '测试中…';
    if (resultEl) { resultEl.textContent = ''; resultEl.className = 'proxy-test-result'; }
    try {
      const saved = await saveProxySettings({ notify: false });
      const r = await fetch('/api/proxy-test', { method: 'POST' });
      if (!r.ok) throw new Error(await readErrorDetail(r));
      const data = await r.json().catch(() => ({}));
      const current = readProxySettingsFromForm();
      if (revision !== proxyRevision || JSON.stringify(current) !== JSON.stringify(saved)) {
        throw new Error('代理配置已变化，请重新测试');
      }
      const expectedUrl = canonicalProxyUrl(
        `${saved.proxy_protocol}://${saved.proxy_host}:${saved.proxy_port}`,
      );
      if (data.ok && canonicalProxyUrl(data.proxy_url) !== expectedUrl) {
        throw new Error('代理配置已变化，请重新测试');
      }
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
}
