import { state, ACTIVE_STATUSES } from './state.js';
import { renderJobs } from './jobsRender.js';

export { ACTIVE_STATUSES };

export async function fetchJob(id) {
  try {
    const r = await fetch(`/api/jobs/${id}`);
    if (!r.ok) return;
    const job = await r.json();
    const prev = state.jobs[id];
    state.jobs[id] = job;
    if (prev?._download && ACTIVE_STATUSES.has(job.status)) {
      state.jobs[id]._download = prev._download;
    }
    renderJobs();
  } catch {}
}

export async function fetchAllJobs() {
  try {
    const r = await fetch('/api/jobs');
    if (!r.ok) return;
    const jobs = await r.json();
    const prev = state.jobs;
    state.jobs = {};
    jobs.forEach(j => {
      state.jobs[j.id] = j;
      if (prev[j.id]?._download && ACTIVE_STATUSES.has(j.status)) {
        state.jobs[j.id]._download = prev[j.id]._download;
      }
    });
    renderJobs();
  } catch {}
}

export function startJobPolling() {
  setInterval(() => {
    const hasActive = Object.values(state.jobs).some(j => ACTIVE_STATUSES.has(j.status));
    if (hasActive) fetchAllJobs();
  }, 3000);
}
