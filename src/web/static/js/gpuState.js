import { state } from './state.js';
import { gpuAlert } from './dom.js';
import { escHtml } from './util.js';
import { addLog } from './log.js';

// Three states the user has to be able to tell apart: a stop that was asked for
// (the job badge says 停止中), a stop that is in progress, and a stop that could
// not be confirmed. The last one pauses GPU dispatch, so it must never be shown
// as if the machine were merely busy.
const STATE_LABEL = {
  cleaning: '正在清理 GPU 子进程',
  cleanup_failed: 'GPU 子进程未能停止，已暂停派发新的 GPU 任务',
};

let diagnosticsOpen = false;

function shouldShow(gpu) {
  if (!gpu) return false;
  if (gpu.state === 'cleaning' || gpu.state === 'cleanup_failed') return true;
  if (gpu.local_translation?.draining) return true;
  if (gpu.local_translation?.cleanup?.stuck) return true;
  return Boolean(gpu.handle_open) || (gpu.orphan_media_children || []).length > 0;
}

function diagnosticsText(gpu) {
  const lines = [
    `状态：${gpu.state}`,
    `GPU 仍被占用：${gpu.gpu_blocked ? '是' : '否'}`,
    `作业对象句柄未关闭：${gpu.handle_open ? '是' : '否'}`,
    `下次自动重试：${gpu.next_retry_in_s ? `${Math.round(gpu.next_retry_in_s)} 秒后` : '（无）'}`,
  ];
  // Every owner of a resource still waiting for cleanup, not just the ASR one.
  (gpu.pending_resources || []).forEach(resource => {
    lines.push(
      `待清理资源：${resource.kind} ${resource.description || ''} `
        + `pid=${resource.pid ?? '（未知）'} 代次=${resource.generation} `
        + `进程=${resource.process_state} 句柄未关闭=${resource.handle_open ? '是' : '否'} `
        + `已重试 ${resource.attempts} 次 原因=${resource.last_error || '（无）'}`,
    );
  });
  if (gpu.local_translation?.draining) {
    lines.push('本地翻译服务：已取消的请求可能仍在生成（共享进程不会为取消而被杀）');
  }
  const localCleanup = gpu.local_translation?.cleanup;
  if (localCleanup?.stuck) {
    lines.push(
      `本地翻译服务未确认退出：pid=${localCleanup.pid ?? '（未知）'} `
        + `已重试 ${localCleanup.attempts ?? 0} 次 原因=${localCleanup.last_error || '（无）'}`,
    );
  }
  (gpu.orphan_media_children || []).forEach(child => {
    lines.push(
      `未确认释放的媒体子进程：${child.name} pid=${child.pid} 状态=${child.state} 原因=${child.reason}`,
    );
  });
  return lines.join('\n');
}

export function renderGpuState() {
  const gpu = state.gpuState;
  if (!gpuAlert) return;
  if (!shouldShow(gpu)) {
    gpuAlert.style.display = 'none';
    gpuAlert.innerHTML = '';
    gpuAlert._html = '';
    return;
  }
  // Cancelling a local request stops this app waiting; llama-server may still
  // be generating, and its VRAM is still gone. Saying so beats implying idle.
  const label = STATE_LABEL[gpu.state]
    || (gpu.local_translation?.cleanup?.stuck
      ? '本地翻译服务未能停止，显存未释放，暂不能加载本地模型'
      : gpu.local_translation?.draining
        ? '已取消的本地翻译请求可能仍在生成，显存尚未释放'
        : 'GPU 资源仍被占用');
  const busy = gpu.state === 'cleaning';
  const html = `
    <div class="gpu-alert-row">
      <span class="gpu-alert-label${gpu.state === 'cleanup_failed' || gpu.local_translation?.cleanup?.stuck ? ' danger' : ''}">${escHtml(label)}</span>
      <button class="btn-sm" data-gpu-recheck>重新检查</button>
      <button class="btn-sm" data-gpu-retry${busy ? ' disabled' : ''}>重试清理</button>
      <button class="btn-sm" data-gpu-diagnostics>${diagnosticsOpen ? '收起诊断信息' : '查看诊断信息'}</button>
    </div>
    ${diagnosticsOpen ? `<pre class="gpu-alert-diagnostics">${escHtml(diagnosticsText(gpu))}</pre>` : ''}`;
  if (gpuAlert._html !== html) {
    gpuAlert.innerHTML = html;
    gpuAlert._html = html;
  }
  gpuAlert.style.display = 'block';
}

export async function fetchGpuState() {
  try {
    const r = await fetch('/api/gpu-state');
    if (!r.ok) return;
    state.gpuState = await r.json();
    renderGpuState();
  } catch {}
}

export function installGpuState() {
  if (!gpuAlert) return;
  gpuAlert.addEventListener('click', async e => {
    if (e.target.closest('[data-gpu-diagnostics]')) {
      diagnosticsOpen = !diagnosticsOpen;
      renderGpuState();
      return;
    }
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
