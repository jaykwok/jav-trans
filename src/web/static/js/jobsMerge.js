// Merging job state that arrives out of order.
//
// The page learns about a job from two places at once: SSE progress triggers a
// GET per event, and a 3s poll refetches the whole list. Responses are not
// ordered - two GETs for the same job can land newest-first, and the older one
// used to win by arriving last, putting a `failed` card back over a job that
// had already been retried and was running again. Cancelling the earlier fetch
// does not help: the response had already arrived.
//
// So the server stamps every published change with a monotonic `revision`, and
// every list response carries the clock reading it was taken at. This module is
// the only place that decides what those numbers mean:
//
//   * a job update older than what we hold is dropped, never merged;
//   * a list older than one we have already applied is dropped whole;
//   * a job missing from a list is gone only if the list is newer than the job.
//
// Revisions are only comparable within one run of the server. The counter is
// resumed from the records that survived a restart, so deleting every job and
// restarting starts it over at 0 - and a page left open would then read every
// new number as older news and refuse the lot, including the empty list that
// explains why. So each response also carries the server's `epoch`, and the
// rules above apply only when it matches the one we already applied:
//
//   * an older epoch is an answer from a server run that has since ended;
//   * a newer epoch means the revisions we hold mean nothing, so the snapshot
//     is rebuilt from the response rather than merged into.
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
 *
 * `appliedRevision` is the newest list reading already applied. A response for
 * a job we do not hold, dated no later than that reading, describes a job that
 * list did not contain - it was deleted while this request was in flight, and
 * re-adding it is the resurrection this guard exists to prevent.
 */
export function reduceJob(
  jobs,
  incoming,
  { activeStatuses, appliedRevision = NO_REVISION, jobEpoch = NO_EPOCH, appliedEpoch = NO_EPOCH, tombstones = {} } = {},
) {
  if (!incoming?.id) return jobs;
  // A different epoch cannot be resolved one job at a time: an older one is an
  // answer from a server run that has ended, and a newer one invalidates every
  // revision held, which only a whole list can restate. Either way this response
  // is not merged - the caller refetches the list when the epoch moved forward.
  if (compareEpoch(jobEpoch, appliedEpoch) !== 0) return jobs;
  const deleted = tombstones[incoming.id];
  if (deleted && compareEpoch(deleted.epoch, appliedEpoch) === 0 && revisionOf(incoming) <= deleted.revision) return jobs;
  const previous = jobs[incoming.id];
  if (!previous && revisionOf(incoming) <= appliedRevision) return jobs;
  return { ...jobs, [incoming.id]: mergeJob(previous, incoming, activeStatuses) };
}

/**
 * Drop a job the server says was deleted, unless we hold something newer.
 *
 * `deletedRevision` is the reading the deletion was stamped with. A job we hold
 * that is newer than that reading is a *different* job with the same id only in
 * the sense that it was created after the delete - keeping it is the same rule
 * a list response follows.
 */
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
 *
 * Returns `{ jobs, revision, epoch, applied }`. `applied` is false when the
 * response was dated at or before one already applied, in which case nothing
 * changes: an older list can only tell us things we already know, or things
 * that have since been undone. A list from a newer epoch is applied whole,
 * replacing what is held rather than merging into it.
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
  if (era < 0) return unchanged;  // a server run that has since ended, answering late
  // Restarting the service restarts the revision counter, so across epochs the
  // numbers are not comparable and nothing held can outrank the response.
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

  // A job we hold that this list does not: keep it only if it is newer than the
  // list, i.e. it was created after the snapshot was taken. Otherwise the list
  // is the authority on membership and the job is gone.
  Object.entries(held).forEach(([id, job]) => {
    if (next[id]) return;
    if (revisionOf(job) > listRevision) next[id] = job;
  });

  return {
    jobs: next,
    // Once the list has crossed a deletion, its global watermark rejects all
    // responses that could resurrect that job. Until then retain the tombstone.
    tombstones: rebuilding ? {} : Object.fromEntries(
      Object.entries(tombstones).filter(([, deleted]) => deleted.revision > listRevision),
    ),
    revision: listRevision === NO_REVISION ? appliedRevision : listRevision,
    epoch: listEpoch === NO_EPOCH ? appliedEpoch : listEpoch,
    applied: true,
  };
}
