// Merging job state that arrives out of order.
//
// No DOM, no fetch: it is a reducer, so it can be tested as one.

export const NO_REVISION = -1;
export const NO_EPOCH = -1;

export function revisionOf(job) {
  const value = Number(job?.revision);
  return Number.isFinite(value) ? value : NO_REVISION;
}

/**
 * Which run of the server issued each of these: -1, 0 or 1, and 0 whenever
 * either side is unknown - an unlabelled response is not evidence of anything.
 */
export function compareEpoch(incoming, applied) {
  if (incoming === NO_EPOCH || applied === NO_EPOCH) return 0;
  if (incoming < applied) return -1;
  return incoming > applied ? 1 : 0;
}

// `_download` is client-side progress for a model download, rebuilt from SSE
// and absent from every server payload. It survives a merge while the job is
// still active, the way it always has.
function carryClientState(previous, incoming, activeStatuses) {
  if (previous?._download && activeStatuses.has(incoming.status)) {
    return { ...incoming, _download: previous._download };
  }
  return incoming;
}

/**
 * One job's answer to "which of these two states is current?".
 * Returns the state to keep - possibly the one already held.
 */
export function mergeJob(previous, incoming, activeStatuses) {
  if (!incoming) return previous;
  if (!previous) return incoming;
  if (revisionOf(incoming) < revisionOf(previous)) return previous;
  return carryClientState(previous, incoming, activeStatuses);
}

/**
 * Fold a single-job response into the map held by the page.
 */
export function reduceJob(
  jobs,
  incoming,
  { activeStatuses, appliedRevision = NO_REVISION, jobEpoch = NO_EPOCH, appliedEpoch = NO_EPOCH, tombstones = {} } = {},
) {
  if (!incoming?.id) return jobs;
  if (compareEpoch(jobEpoch, appliedEpoch) !== 0) return jobs;
  const deleted = tombstones[incoming.id];
  if (deleted && compareEpoch(deleted.epoch, appliedEpoch) === 0 && revisionOf(incoming) <= deleted.revision) return jobs;
  const previous = jobs[incoming.id];
  if (!previous && appliedRevision !== NO_REVISION && revisionOf(incoming) <= appliedRevision) return jobs;
  return { ...jobs, [incoming.id]: mergeJob(previous, incoming, activeStatuses) };
}

export function recordDeletion(tombstones, id, {
  deletedRevision = NO_REVISION, deletedEpoch = NO_EPOCH, appliedEpoch = NO_EPOCH,
} = {}) {
  if (deletedRevision === NO_REVISION || compareEpoch(deletedEpoch, appliedEpoch) !== 0) return tombstones;
  const previous = tombstones[id];
  if (previous && previous.epoch === deletedEpoch && previous.revision >= deletedRevision) return tombstones;
  return { ...tombstones, [id]: { epoch: deletedEpoch, revision: deletedRevision } };
}

export function reduceDeletion(jobs, id, {
  deletedRevision = NO_REVISION, deletedEpoch = NO_EPOCH, appliedEpoch = NO_EPOCH,
} = {}) {
  if (compareEpoch(deletedEpoch, appliedEpoch) !== 0) return jobs;
  const held = jobs[id];
  if (!held) return jobs;
  if (deletedRevision !== NO_REVISION && revisionOf(held) > deletedRevision) return jobs;
  const next = { ...jobs };
  delete next[id];
  return next;
}

/**
 * Fold a whole-list response into the map held by the page.
 */
export function reduceJobList(
  jobs,
  incomingList,
  {
    activeStatuses,
    listRevision = NO_REVISION,
    appliedRevision = NO_REVISION,
    listEpoch = NO_EPOCH,
    appliedEpoch = NO_EPOCH,
    tombstones = {},
  } = {},
) {
  const unchanged = { jobs, revision: appliedRevision, epoch: appliedEpoch, tombstones, applied: false };
  if (!Array.isArray(incomingList)) return unchanged;

  const era = compareEpoch(listEpoch, appliedEpoch);
  if (era < 0) return unchanged;
  const rebuilding = era > 0;
  if (!rebuilding && listRevision !== NO_REVISION && listRevision <= appliedRevision) {
    return unchanged;
  }

  const held = rebuilding ? {} : jobs;
  const next = {};
  incomingList.forEach(incoming => {
    if (!incoming?.id) return;
    const deleted = rebuilding ? null : tombstones[incoming.id];
    if (deleted && compareEpoch(deleted.epoch, appliedEpoch) === 0 && revisionOf(incoming) <= deleted.revision) return;
    next[incoming.id] = mergeJob(held[incoming.id], incoming, activeStatuses);
  });

  // A job we hold that this list does not: keep it only when the list is dated
  // (`listRevision !== NO_REVISION`) and the held job is strictly newer than
  // that snapshot. An undated list (or one dated >= the job) is authoritative.
  Object.entries(held).forEach(([id, job]) => {
    if (next[id]) return;
    if (listRevision !== NO_REVISION && revisionOf(job) > listRevision) next[id] = job;
  });

  return {
    jobs: next,
    // Prune tombstones only once a dated list has crossed them.
    tombstones: rebuilding ? {} : Object.fromEntries(
      Object.entries(tombstones).filter(
        ([, deleted]) => listRevision === NO_REVISION || deleted.revision > listRevision,
      ),
    ),
    revision: listRevision === NO_REVISION ? appliedRevision : listRevision,
    epoch: listEpoch === NO_EPOCH ? appliedEpoch : listEpoch,
    applied: true,
  };
}
