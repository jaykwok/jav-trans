import { state } from './state.js';
import { gpuAlert } from './dom.js';
import { escHtml, readErrorDetail } from './util.js';
import { addLog } from './log.js';

// Three states the user has to be able to tell apart: a stop that was asked for
// (the job badge says 停止中), a stop that is in progress, and a stop that could
// not be confirmed. The last one pauses GPU dispatch, so it must never be shown
// as if the machine were merely busy.
const STATE_LABEL = {
  cleaning: '正在清理 GPU 子进程',
  cleanup_failed: 'GPU 子进程未能停止，已暂停派发新的 GPU 任务',
};

function shouldShow(gpu) {
  if (!gpu) return false;
  if (gpu.state === 'cleaning' || gpu.state === 'cleanup_failed') return true;
  if (gpu.local_translation?.draining) return true;
  if (gpu.local_translation?.cleanup?.stuck) return true;
  return Boolean(gpu.handle_open) || (gpu.orphan_media_children || []).length > 0;
}

export function renderGpuState() {
  const gpu = state.gpuState;
  if (!gpuAlert) return;
  if (!shouldShow(gpu)) {
    gpuAlert.style.display = 'none';
    gpuAlert.classList.remove('danger');
    gpuAlert.innerHTML = '';
    gpuAlert._html = '';
    return;
  }
  // Cancelling a local request stops this app waiting; llama-server may still
  // be generating, and its VRAM is still gone. Saying so beats implying idle.
  const isDanger = gpu.state === 'cleanup_failed' || Boolean(gpu.local_translation?.cleanup?.stuck);
  const label = STATE_LABEL[gpu.state]
    || (gpu.local_translation?.cleanup?.stuck
      ? '本地翻译服务未能停止，显存未释放，暂不能加载本地模型'
      : gpu.local_translation?.draining
        ? '已取消的本地翻译请求可能仍在生成，显存尚未释放'
        : 'GPU 资源仍被占用');
  const busy = gpu.state === 'cleaning';
  const html = `
    <div class="gpu-alert-row">
      <span class="gpu-alert-label${isDanger ? ' danger' : ''}">${escHtml(label)}</span>
      <button class="btn-sm btn-outline" type="button" data-gpu-recheck>重新检查</button>
      <button class="btn-sm btn-outline" type="button" data-gpu-retry${busy ? ' disabled' : ''}>重试清理</button>
    </div>`;
  gpuAlert.classList.toggle('danger', isDanger);
  if (gpuAlert._html !== html) {
    gpuAlert.innerHTML = html;
    gpuAlert._html = html;
  }
  gpuAlert.style.display = 'block';
}

let lastGpuFetchError = '';

export async function fetchGpuState() {
  try {
    const r = await fetch('/api/gpu-state');
    if (!r.ok) {
      const detail = await readErrorDetail(r);
      const key = `http:${r.status}:${detail}`;
      if (key !== lastGpuFetchError) {
        lastGpuFetchError = key;
        addLog(`获取 GPU 状态失败（HTTP ${r.status}）：${detail}`, 'stage-error');
      }
      return;
    }
    lastGpuFetchError = '';
    state.gpuState = await r.json();
    renderGpuState();
  } catch (error) {
    const msg = error?.message || String(error);
    if (msg !== lastGpuFetchError) {
      lastGpuFetchError = msg;
      addLog(`获取 GPU 状态出错：${msg}`, 'stage-error');
    }
  }
}

export function installGpuState() {
  if (!gpuAlert) return;
  gpuAlert.addEventListener('click', async e => {
    if (e.target.closest('[data-gpu-recheck]')) {
      await fetchGpuState();
      return;
    }
    if (e.target.closest('[data-gpu-retry]')) {
      addLog('重试清理 GPU 子进程…', 'stage-start');
      try {
        const r = await fetch('/api/gpu-state/retry-cleanup', { method: 'POST' });
        if (r.ok) {
          state.gpuState = await r.json();
          renderGpuState();
          // "released" here means every owner confirmed - the ASR worker alone
          // saying so was how the banner claimed a card that was still busy.
          const clear = state.gpuState.state === 'released' && !state.gpuState.gpu_blocked;
          addLog(
            clear ? 'GPU 资源已全部释放，队列恢复派发' : 'GPU 资源仍未确认释放',
            clear ? 'stage-done' : 'stage-error',
          );
        } else {
          addLog(`重试清理失败（HTTP ${r.status}）：${await readErrorDetail(r)}`, 'stage-error');
        }
      } catch (error) {
        addLog(`重试清理失败：${error.message}`, 'stage-error');
      }
    }
  });
  // Slower than the job poll: this only changes when a process refuses to die,
  // and the backend is already retrying on its own backoff.
  setInterval(fetchGpuState, 5000);
}
