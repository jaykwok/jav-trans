import { state, ACTIVE_STATUSES } from './state.js';
import { renderJobs } from './jobsRender.js';
import { coalesce } from './coalesce.js';
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
// Nothing is running, but the state can still change under the page: another
// tab deleting a job, or the service restarting. Slow enough to be free, often
// enough that a stale list is not permanent.
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

// Both refreshes are coalesced per subject: `fetchJob` per job id, `fetchAllJobs`
// under one shared key. Correctness does not depend on it - ordering is settled
// by revision in the reducer - so this only drops requests whose answer is
// already on its way.
const refreshJob = coalesce(requestJob);
const refreshAllJobs = coalesce(requestAllJobs);

export function fetchJob(id) {
  if (!id) return Promise.resolve();
  return refreshJob(id);
}

export function fetchAllJobs() {
  return refreshAllJobs('all');
}

async function requestJob(id) {
  try {
    const r = await fetch(`/api/jobs/${id}`);
    const epoch = epochOf(r);
    const era = compareEpoch(epoch, state.jobsEpoch);
    if (era > 0) {
      await fetchAllJobs();
      return;
    }
    if (era < 0) return;
    if (r.status === 410) {
      // Deleted, and the server still remembers when. A 404 could equally mean
      // "never existed", which is not a reason to remove a card.
      const before = state.jobs;
      const deletion = {
        deletedRevision: listRevisionOf(r), deletedEpoch: epoch, appliedEpoch: state.jobsEpoch,
      };
      state.jobsTombstones = recordDeletion(state.jobsTombstones, id, deletion);
      state.jobs = reduceDeletion(before, id, deletion);
      if (state.jobs !== before) renderJobs();
      return;
    }
    if (!r.ok) return;
    const job = await r.json();
    const before = state.jobs;
    state.jobs = reduceJob(before, job, {
      activeStatuses: ACTIVE_STATUSES,
      appliedRevision: state.jobsRevision,
      jobEpoch: epoch,
      appliedEpoch: state.jobsEpoch,
      tombstones: state.jobsTombstones,
    });
    if (state.jobs === before) return;  // a response older than what we hold
    renderJobs();
  } catch {}
}

async function requestAllJobs() {
  try {
    const r = await fetch('/api/jobs');
    if (!r.ok) return;
    const jobs = await r.json();
    const merged = reduceJobList(state.jobs, jobs, {
      activeStatuses: ACTIVE_STATUSES,
      listRevision: listRevisionOf(r),
      appliedRevision: state.jobsRevision,
      listEpoch: epochOf(r),
      appliedEpoch: state.jobsEpoch,
      tombstones: state.jobsTombstones,
    });
    if (!merged.applied) return;  // an older list, or an older server run
    state.jobs = merged.jobs;
    state.jobsRevision = merged.revision;
    state.jobsEpoch = merged.epoch;
    state.jobsTombstones = merged.tombstones;
    renderJobs();
  } catch {}
}

export function startJobPolling() {
  let idleElapsed = 0;
  setInterval(() => {
    // SSE progress can arrive just before the final job state is persisted.
    // Keep a lightweight reconciliation poll while jobs are active so a missed
    // or early final event cannot leave the UI stuck in an active state.
    const hasActive = Object.values(state.jobs).some(j => ACTIVE_STATUSES.has(j.status));
    if (hasActive) {
      idleElapsed = 0;
      fetchAllJobs();
      return;
    }
    // An all-terminal page used to stop polling entirely, so a job deleted in
    // another tab - or a restart that emptied the list - stayed on screen until
    // someone reloaded. It has nothing to wait for, so it just waits longer.
    idleElapsed += ACTIVE_POLL_MS;
    if (idleElapsed >= IDLE_POLL_MS) {
      idleElapsed = 0;
      fetchAllJobs();
    }
  }, ACTIVE_POLL_MS);
}
