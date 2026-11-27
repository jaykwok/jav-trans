import { state } from './state.js';
import { connDot } from './dom.js';
import { addLog } from './log.js';
import { renderJobs } from './jobsRender.js';
import { fetchJob, fetchAllJobs } from './jobsApi.js';
import { fetchGpuState } from './gpuState.js';

const INITIAL_RECONNECT_MS = 2000;
const MAX_RECONNECT_MS = 15000;
const REFRESH_PHASES = new Set(['start', 'progress', 'done', 'error', 'skip', 'degraded', 'blocked']);

let reconnectTimer = null;
let reconnectDelayMs = INITIAL_RECONNECT_MS;
let wasDisconnected = false;

export function reconnectSSEIfNeeded() {
  if (!state.sse && reconnectTimer !== null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
    reconnectDelayMs = INITIAL_RECONNECT_MS;
    connectSSE();
  }
}

export function connectSSE() {
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (state.sse) {
    state.sse.close();
    state.sse = null;
  }
  const es = new EventSource('/api/events');
  state.sse = es;

  es.addEventListener('message', e => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }

    if (ev.type === 'connected') {
      connDot.className = 'conn-dot ok';
      reconnectDelayMs = INITIAL_RECONNECT_MS;
      if (wasDisconnected) {
        wasDisconnected = false;
        addLog('实时事件流已恢复连接', 'stage-done');
        fetchAllJobs();
        fetchGpuState();
      }
      return;
    }

    if (ev.type === 'resync') {
      fetchAllJobs();
      fetchGpuState();
      return;
    }

    const phase = ev.phase ?? '';

    if (ev.stage === 'timing_summary' && phase === 'done') {
      const rows = Array.isArray(ev.extra?.rows) ? ev.extra.rows : [];
      addLog(ev.extra?.title || '阶段耗时', 'stage-done');
      rows.forEach(row => {
        const seconds = Number(row?.seconds);
        if (!row?.label || !Number.isFinite(seconds)) return;
        addLog(`  ${row.label}：${seconds.toFixed(2)}s`, 'stage-done');
      });
      return;
    }

    if (ev.stage === 'model_download' && ev.job_id) {
      const job = state.jobs[ev.job_id];
      if (job) {
        if (phase === 'done' || phase === 'error') {
          job._download = null;
        } else {
          const x = ev.extra || {};
          const prev = job._download || {};
          job._download = {
            file:    x.file    || prev.file    || '',
            sizeMb:  x.size_mb ?? prev.sizeMb  ?? null,
            pct:     x.pct     ?? prev.pct     ?? 0,
            speedMb: x.speed_mb ?? prev.speedMb ?? null,
          };
        }
        renderJobs();
      } else {
        fetchJob(ev.job_id);
      }
      const x = ev.extra || {};
      const dlInfo = phase === 'progress' && x.pct != null
        ? ` ${Math.round(x.pct)}%${x.speed_mb != null ? ` ${x.speed_mb.toFixed(1)}MB/s` : ''}`
        : '';
      addLog(
        `[model_download] ${phase}${dlInfo}`,
        phase === 'done' ? 'stage-done' : phase === 'error' ? 'stage-error' : 'stage-progress',
      );
      return;
    }

    const cls = phase === 'start' ? 'stage-start'
      : phase === 'done' ? 'stage-done'
      : phase === 'error' || phase === 'blocked' ? 'stage-error'
      : 'stage-progress';

    const label = ev.stage ? `[${ev.stage}] ${phase}` : JSON.stringify(ev);
    const extra = ev.extra ? ' ' + Object.entries(ev.extra).map(([k, v]) => `${k}=${v}`).join(' ') : '';
    addLog(label + extra, cls);

    if (ev.job_id && REFRESH_PHASES.has(phase)) {
      fetchJob(ev.job_id);
    }
  });

  es.onerror = () => {
    if (state.sse !== es) return;
    es.close();
    state.sse = null;
    connDot.className = 'conn-dot err';
    const delayMs = reconnectDelayMs;
    reconnectDelayMs = Math.min(MAX_RECONNECT_MS, reconnectDelayMs * 2);
    if (!wasDisconnected) {
      wasDisconnected = true;
      addLog(`实时事件流已断开，${Math.round(delayMs / 1000)} 秒后重连…`, 'stage-error');
    }
    if (reconnectTimer === null) {
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connectSSE();
        fetchAllJobs();
      }, delayMs);
    }
  };
}
