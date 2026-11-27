import { state, ACTIVE_STATUSES } from './state.js';
import { renderJobs } from './jobsRender.js';
import { coalesce } from './coalesce.js';
import { addLog } from './log.js';
import { readErrorDetail } from './util.js';
import {
  NO_EPOCH,
  NO_REVISION,
  compareEpoch,
  recordDeletion,
  reduceDeletion,
  reduceJob,
  reduceJobList,
} from './jobsMerge.js';

const JOBS_REVISION_HEADER = 'X-Jobs-Revision';
const JOBS_EPOCH_HEADER = 'X-Jobs-Epoch';
const ACTIVE_POLL_MS = 3000;
const IDLE_POLL_MS = 15000;

function headerNumber(response, name, fallback) {
  const raw = response.headers.get(name);
  const value = Number(raw);
  return raw != null && Number.isFinite(value) ? value : fallback;
}

function listRevisionOf(response) {
  return headerNumber(response, JOBS_REVISION_HEADER, NO_REVISION);
}

function epochOf(response) {
  return headerNumber(response, JOBS_EPOCH_HEADER, NO_EPOCH);
}

const refreshJob = coalesce(requestJob);
const refreshAllJobs = coalesce(requestAllJobs);

export function fetchJob(id) {
  if (!id) return Promise.resolve();
  return refreshJob(id);
}

export function fetchAllJobs() {
  return refreshAllJobs('all');
}

let lastJobsApiError = '';

function reportJobsApiError(context, detail) {
  const key = `${context}:${detail}`;
  if (key === lastJobsApiError) return;
  lastJobsApiError = key;
  addLog(`${context}：${detail}`, 'stage-error');
}

async function requestJob(id) {
  try {
    const r = await fetch(`/api/jobs/${encodeURIComponent(id)}`);
    const epoch = epochOf(r);
    const era = compareEpoch(epoch, state.jobsEpoch);
    if (era > 0) {
      await fetchAllJobs();
      return;
    }
    if (era < 0) return;
    if (r.status === 410) {
      const before = state.jobs;
      const deletion = {
        deletedRevision: listRevisionOf(r), deletedEpoch: epoch, appliedEpoch: state.jobsEpoch,
      };
      state.jobsTombstones = recordDeletion(state.jobsTombstones, id, deletion);
      state.jobs = reduceDeletion(before, id, deletion);
      if (state.jobs !== before) renderJobs();
      return;
    }
    if (!r.ok) {
      if (r.status !== 404) {
        reportJobsApiError(`刷新任务 ${id} 失败（HTTP ${r.status}）`, await readErrorDetail(r));
      }
      return;
    }
    lastJobsApiError = '';
    const job = await r.json();
    const before = state.jobs;
    state.jobs = reduceJob(before, job, {
      activeStatuses: ACTIVE_STATUSES,
      appliedRevision: state.jobsRevision,
      jobEpoch: epoch,
      appliedEpoch: state.jobsEpoch,
      tombstones: state.jobsTombstones,
    });
    if (state.jobs === before) return;
    renderJobs();
  } catch (error) {
    reportJobsApiError(`刷新任务 ${id} 出错`, error?.message || String(error));
  }
}

async function requestAllJobs() {
  try {
    const r = await fetch('/api/jobs');
    if (!r.ok) {
      reportJobsApiError(`刷新任务列表失败（HTTP ${r.status}）`, await readErrorDetail(r));
      return;
    }
    lastJobsApiError = '';
    const jobs = await r.json();
    const merged = reduceJobList(state.jobs, jobs, {
      activeStatuses: ACTIVE_STATUSES,
      listRevision: listRevisionOf(r),
      appliedRevision: state.jobsRevision,
      listEpoch: epochOf(r),
      appliedEpoch: state.jobsEpoch,
      tombstones: state.jobsTombstones,
    });
    if (!merged.applied) return;
    state.jobs = merged.jobs;
    state.jobsRevision = merged.revision;
    state.jobsEpoch = merged.epoch;
    state.jobsTombstones = merged.tombstones;
    renderJobs();
  } catch (error) {
    reportJobsApiError('刷新任务列表出错', error?.message || String(error));
  }
}

export function startJobPolling() {
  let idleElapsed = 0;
  setInterval(() => {
    // Pause background polling while the tab is hidden; visibilitychange in
    // main.js triggers an immediate resync when the user returns.
    if (typeof document !== 'undefined' && document.hidden) return;
    const hasActive = Object.values(state.jobs).some(j => ACTIVE_STATUSES.has(j.status));
    if (hasActive) {
      idleElapsed = 0;
      fetchAllJobs();
      return;
    }
    idleElapsed += ACTIVE_POLL_MS;
    if (idleElapsed >= IDLE_POLL_MS) {
      idleElapsed = 0;
      fetchAllJobs();
    }
  }, ACTIVE_POLL_MS);
}
