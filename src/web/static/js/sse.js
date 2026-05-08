import { state } from './state.js';
import { connDot } from './dom.js';
import { addLog } from './log.js';
import { renderJobs } from './jobsRender.js';
import { fetchJob, fetchAllJobs } from './jobsApi.js';

export function connectSSE() {
  if (state.sse) state.sse.close();
  const es = new EventSource('/api/events');
  state.sse = es;

  es.addEventListener('message', e => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }

    if (ev.type === 'connected') {
      connDot.className = 'conn-dot ok';
      return;
    }

    const phase = ev.phase ?? '';

    if (ev.stage === 'model_download' && ev.job_id) {
      const job = state.jobs[ev.job_id];
      if (job) {
        if (phase === 'done' || phase === 'error') {
          job._download = null;
        } else {
          const x = ev.extra || {};
          const prev = job._download || {};
          const fileChanged = x.file && x.file !== prev.file;
          const speedMb = x.speed_mb ?? prev.speedMb ?? null;
          let slowSince = fileChanged ? null : (prev.slowSince ?? null);
          if (speedMb != null) {
            if (speedMb < 1.0 && slowSince == null) slowSince = Date.now();
            else if (speedMb >= 1.0) slowSince = null;
          }
          job._download = {
            file:    x.file    || prev.file    || '',
            sizeMb:  x.size_mb ?? prev.sizeMb  ?? null,
            pct:     x.pct     ?? prev.pct     ?? 0,
            speedMb,
            slowSince,
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
      addLog(`[model_download] ${phase}${dlInfo}`,
        phase === 'done' ? 'stage-done' : phase === 'error' ? 'stage-error' : 'stage-progress');
      return;
    }

    const cls = phase === 'start' ? 'stage-start'
               : phase === 'done' ? 'stage-done'
               : phase === 'error' || phase === 'blocked' ? 'stage-error'
               : 'stage-progress';

    const label = ev.stage ? `[${ev.stage}] ${phase}` : JSON.stringify(ev);
    const extra = ev.extra ? ' ' + Object.entries(ev.extra).map(([k, v]) => `${k}=${v}`).join(' ') : '';
    addLog(label + extra, cls);

    if (ev.job_id && (phase === 'done' || phase === 'start' || phase === 'progress')) {
      fetchJob(ev.job_id);
    }
  });

  es.onerror = () => {
    connDot.className = 'conn-dot err';
    setTimeout(() => { connectSSE(); fetchAllJobs(); }, 3000);
  };
}
