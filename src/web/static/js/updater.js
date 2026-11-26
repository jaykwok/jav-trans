import { $, showToast } from './util.js';
import { addLog } from './log.js';

const POLL_ACTIVE_MS = 1200;
const POLL_STARTUP_MS = 2500;

let pollTimer = null;
let latestSnapshot = null;

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 MB';
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function renderStatus(data) {
  latestSnapshot = data;
  const trigger = $('btn-update-trigger');
  const versionInfo = $('update-version-info');
  const statusText = $('update-status-text');
  const progressWrap = $('update-progress-wrap');
  const progressBar = $('update-progress-bar');
  const notesEl = $('update-notes');
  const autoCheck = $('update-auto-check');
  const btnCheck = $('btn-update-check');
  const btnStart = $('btn-update-start');
  const btnCancel = $('btn-update-cancel');
  const btnOpen = $('btn-update-open');

  const current = data.current_version ? `v${data.current_version}` : 'v--';
  const latest = data.latest_version ? `v${data.latest_version}` : '';
  const hasNewer = Boolean(data.update_available && latest);
  const st = data.state || 'idle';
  const busy = st === 'checking' || st === 'downloading' || st === 'verifying' || st === 'restarting';

  if (autoCheck) autoCheck.checked = data.auto_check_enabled !== false;

  if (trigger) {
    if (st === 'available') {
      trigger.textContent = `↑ 新版本 ${latest}`;
      trigger.classList.add('has-update');
    } else if (st === 'downloading') {
      trigger.textContent = `↓ 下载更新 ${data.percent || 0}%`;
      trigger.classList.add('has-update');
    } else if (st === 'verifying' || st === 'restarting' || st === 'ready_to_restart') {
      trigger.textContent = '↻ 正在应用更新';
      trigger.classList.add('has-update');
    } else {
      trigger.textContent = current;
      trigger.classList.remove('has-update');
    }
  }

  if (versionInfo) {
    versionInfo.textContent = hasNewer && st !== 'up_to_date' && st !== 'idle'
      ? `当前版本：${current}  →  最新稳定版：${latest}`
      : `当前版本：${current}`;
  }

  if (statusText) {
    statusText.classList.remove('is-error', 'is-ok');
    if (st === 'checking') {
      statusText.textContent = '正在检查 GitHub 最新稳定版…';
    } else if (st === 'up_to_date') {
      statusText.textContent = `当前已是最新稳定版（${current}）。`;
      statusText.classList.add('is-ok');
    } else if (st === 'available') {
      const sizePart = data.asset_size ? `（约 ${formatBytes(data.asset_size)}）` : '';
      statusText.textContent = data.can_self_update
        ? `检测到新版本 ${latest}${sizePart}，可一键下载校验并覆盖升级。`
        : `检测到新版本 ${latest}${sizePart}。${data.cannot_update_reason || ''}`;
    } else if (st === 'downloading') {
      const total = data.total_bytes || data.asset_size || 0;
      const detail = total > 0
        ? `${formatBytes(data.downloaded_bytes)} / ${formatBytes(total)} (${data.percent || 0}%)`
        : `${formatBytes(data.downloaded_bytes)}`;
      statusText.textContent = `正在下载更新包：${detail}`;
    } else if (st === 'verifying') {
      statusText.textContent = '下载完成，正在核验 SHA-256 与发行清单…';
    } else if (st === 'restarting') {
      statusText.textContent = '更新包校验通过，正在关闭当前窗口并自动覆盖重启…';
      statusText.classList.add('is-ok');
    } else if (st === 'ready_to_restart') {
      statusText.textContent = '更新已校验并暂存完成，请重启 jav-sub.exe 自动应用更新。';
      statusText.classList.add('is-ok');
    } else if (st === 'error') {
      statusText.textContent = data.error || '更新失败，请检查网络或代理设置后重试。';
      statusText.classList.add('is-error');
    } else {
      statusText.textContent = '可点击「检查更新」查看 GitHub 最新稳定版。';
    }
  }

  if (progressWrap && progressBar) {
    const showBar = st === 'downloading' || st === 'verifying' || st === 'restarting';
    progressWrap.hidden = !showBar;
    const pct = st === 'verifying' || st === 'restarting' ? 100 : (data.percent || 0);
    progressBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  }

  if (notesEl) {
    const hasNotes = Boolean(data.release_notes) && hasNewer && st !== 'up_to_date' && st !== 'idle';
    notesEl.hidden = !hasNotes;
    notesEl.textContent = hasNotes ? data.release_notes : '';
  }

  if (btnCheck) btnCheck.disabled = busy;
  if (btnOpen) btnOpen.hidden = !data.release_url;
  if (btnCancel) btnCancel.hidden = !(st === 'downloading' || st === 'verifying');
  if (btnStart) {
    const canOfferStart =
      (st === 'available' || (st === 'error' && hasNewer)) && Boolean(data.can_self_update);
    btnStart.hidden = !canOfferStart;
    btnStart.disabled = busy;
  }

  schedulePoll(st);
}

function schedulePoll(st) {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  if (st === 'checking') {
    pollTimer = setTimeout(fetchUpdateStatus, POLL_STARTUP_MS);
  } else if (st === 'downloading' || st === 'verifying' || st === 'restarting') {
    pollTimer = setTimeout(fetchUpdateStatus, POLL_ACTIVE_MS);
  }
}

export async function fetchUpdateStatus() {
  try {
    const res = await fetch('/api/update/status');
    if (!res.ok) return null;
    const data = await res.json();
    renderStatus(data);
    return data;
  } catch {
    return null;
  }
}

function openModal() {
  const modal = $('update-modal');
  if (!modal) return;
  modal.hidden = false;
  fetchUpdateStatus();
}

function closeModal() {
  const modal = $('update-modal');
  if (modal) modal.hidden = true;
}

export function installUpdater() {
  const trigger = $('btn-update-trigger');
  const modal = $('update-modal');
  const btnClose = $('btn-update-close');
  const btnCheck = $('btn-update-check');
  const btnStart = $('btn-update-start');
  const btnCancel = $('btn-update-cancel');
  const btnOpen = $('btn-update-open');
  const autoCheck = $('update-auto-check');

  trigger?.addEventListener('click', openModal);
  btnClose?.addEventListener('click', closeModal);
  modal?.addEventListener('click', e => {
    if (e.target === modal) closeModal();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && modal && !modal.hidden) closeModal();
  });

  btnCheck?.addEventListener('click', async () => {
    btnCheck.disabled = true;
    if ($('update-status-text')) $('update-status-text').textContent = '正在检查 GitHub 最新稳定版…';
    try {
      const res = await fetch('/api/update/check', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      renderStatus(data);
    } catch (err) {
      showToast(`检查更新失败：${err.message}`);
    } finally {
      const st = latestSnapshot?.state;
      if (st !== 'checking' && st !== 'downloading' && st !== 'verifying' && st !== 'restarting') {
        btnCheck.disabled = false;
      }
    }
  });

  btnStart?.addEventListener('click', async () => {
    btnStart.disabled = true;
    try {
      const res = await fetch('/api/update/start', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      addLog(`[更新] 开始下载并校验 ${data.tag || data.latest_version || '新版本'}…`, 'info');
      renderStatus(data);
    } catch (err) {
      showToast(err.message);
      addLog(`[更新] 无法启动升级：${err.message}`, 'stage-error');
      btnStart.disabled = false;
    }
  });

  btnCancel?.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/update/cancel', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        renderStatus(data);
        if (data.state === 'restarting' || data.state === 'ready_to_restart') {
          showToast('校验已完成，正在重启应用更新');
        } else {
          showToast('已请求取消更新');
        }
      }
    } catch {
      /* ignore */
    }
  });

  btnOpen?.addEventListener('click', () => {
    if (latestSnapshot?.release_url) {
      window.open(latestSnapshot.release_url, '_blank', 'noopener,noreferrer');
    }
  });

  autoCheck?.addEventListener('change', async () => {
    const enabled = !!autoCheck.checked;
    autoCheck.disabled = true;
    try {
      const res = await fetch('/api/update/auto-check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      renderStatus(data);
      showToast(enabled ? '已开启启动时自动检测更新' : '已关闭启动时自动检测更新');
    } catch (err) {
      autoCheck.checked = !enabled;
      showToast(`保存失败：${err.message}`);
    } finally {
      autoCheck.disabled = false;
    }
  });

  fetchUpdateStatus();
}
