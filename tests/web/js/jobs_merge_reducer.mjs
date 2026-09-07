// Regression harness for the job-state reducer, run against the real module.
//
// Executed by tests/web/test_jobs_merge_reducer.py:
//     node jobs_merge_reducer.mjs <path to src/web/static/js/jobsMerge.js>
//
// Plain asserts rather than a framework: the module under test is dependency
// free on purpose, and this stays runnable by hand while debugging the page.

import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';

const [, , modulePath] = process.argv;
const { mergeJob, recordDeletion, reduceDeletion, reduceJob, reduceJobList, NO_REVISION, NO_EPOCH } = await import(
  pathToFileURL(modulePath).href
);

const ACTIVE = new Set(['queued', 'asr', 'translating', 'writing', 'cancelling']);
const merge = { activeStatuses: ACTIVE };

function job(id, revision, extra = {}) {
  return { id, revision, status: 'translating', run_id: 'run-1', ...extra };
}

const cases = {
  'a 410 blocks a late list and a late job without skipping unrelated updates'() {
    const deletion = { deletedRevision: 10, deletedEpoch: 2, appliedEpoch: 2 };
    const tombstones = recordDeletion({}, 'a', deletion);
    const held = reduceDeletion({ a: job('a', 7) }, 'a', deletion);
    const list = reduceJobList(held, [job('a', 7), job('b', 8)], {
      ...merge, tombstones, appliedEpoch: 2, listEpoch: 2, appliedRevision: 5, listRevision: 8,
    });
    assert.equal(list.jobs.a, undefined);
    assert.equal(list.jobs.b.revision, 8);
    assert.equal(list.revision, 8); // deletion must not advance the global list watermark
    assert.equal(list.tombstones.a.revision, 10);
    const late = reduceJob(list.jobs, job('a', 9), {
      ...merge, tombstones: list.tombstones, appliedEpoch: 2, jobEpoch: 2, appliedRevision: 8,
    });
    assert.equal(late.a, undefined);
    const crossed = reduceJobList(list.jobs, [job('b', 8)], {
      ...merge, tombstones: list.tombstones, appliedEpoch: 2, listEpoch: 2, appliedRevision: 8, listRevision: 11,
    });
    assert.deepEqual(crossed.tombstones, {});
    assert.equal(reduceJob(crossed.jobs, job('a', 9), {
      ...merge, tombstones: crossed.tombstones, appliedEpoch: 2, jobEpoch: 2, appliedRevision: crossed.revision,
    }).a, undefined);
  },

  'an old epoch deletion cannot remove a new job'() {
    const held = { a: job('a', 2) };
    const deletion = { deletedRevision: 100, deletedEpoch: 1, appliedEpoch: 2 };
    assert.equal(reduceDeletion(held, 'a', deletion), held);
    assert.deepEqual(recordDeletion({}, 'a', deletion), {});
  },

  'a new epoch clears deletion history and accepts its new revision clock'() {
    const tombstones = recordDeletion({}, 'a', { deletedRevision: 100, deletedEpoch: 1, appliedEpoch: 1 });
    const result = reduceJobList({}, [job('a', 2)], {
      ...merge, tombstones, appliedEpoch: 1, listEpoch: 2, appliedRevision: 90, listRevision: 3,
    });
    assert.equal(result.jobs.a.revision, 2);
    assert.deepEqual(result.tombstones, {});
  },

  // The reported failure: two GETs for one job, the older answer landing last.
  'an overtaken response never wins'() {
    const running = job('a', 12, { status: 'translating', run_id: 'run-2' });
    const stale = job('a', 7, { status: 'failed', run_id: 'run-1' });

    const jobs = reduceJob({ a: running }, stale, merge);

    assert.equal(jobs.a.status, 'translating');
    assert.equal(jobs.a.run_id, 'run-2');
    // Unchanged map identity, so the page does not re-render for nothing.
    assert.equal(jobs.a, running);
  },

  'a newer response replaces the held one'() {
    const jobs = reduceJob({ a: job('a', 7) }, job('a', 9, { status: 'done' }), merge);
    assert.equal(jobs.a.status, 'done');
  },

  'client-side download progress survives an accepted merge'() {
    const held = { ...job('a', 7), _download: { pct: 40 } };
    const jobs = reduceJob({ a: held }, job('a', 9), merge);
    assert.equal(jobs.a._download.pct, 40);
  },

  'a finished job drops the download progress'() {
    const held = { ...job('a', 7), _download: { pct: 40 } };
    const jobs = reduceJob({ a: held }, job('a', 9, { status: 'done' }), merge);
    assert.equal(jobs.a._download, undefined);
  },

  // A list is a statement about membership as of one moment, so an older one
  // cannot undo a newer one - including by re-adding what it deleted.
  'a list older than the applied one changes nothing'() {
    const held = { a: job('a', 20, { status: 'done' }) };
    const result = reduceJobList(held, [job('a', 5), job('b', 6)], {
      ...merge,
      listRevision: 8,
      appliedRevision: 20,
    });

    assert.equal(result.applied, false);
    assert.equal(result.jobs, held);
    assert.equal(result.revision, 20);
  },

  'a deleted job goes when a newer list omits it'() {
    const held = { a: job('a', 5, { status: 'done' }), b: job('b', 6) };
    const result = reduceJobList(held, [job('b', 6)], {
      ...merge,
      listRevision: 9,
      appliedRevision: 5,
    });

    assert.equal(result.applied, true);
    assert.equal(result.jobs.a, undefined);
    assert.ok(result.jobs.b);
    assert.equal(result.revision, 9);
  },

  'a job created after the snapshot survives that snapshot'() {
    const held = { fresh: job('fresh', 30) };
    const result = reduceJobList(held, [job('old', 4)], {
      ...merge,
      listRevision: 12,
      appliedRevision: NO_REVISION,
    });

    assert.ok(result.jobs.fresh, 'a job newer than the list is not membership news');
    assert.ok(result.jobs.old);
  },

  'an in-flight single-job response cannot resurrect a deleted job'() {
    // The GET was issued before the delete; its answer arrives after the list
    // that recorded the removal.
    const jobs = reduceJob({}, job('a', 5, { status: 'done' }), {
      ...merge,
      appliedRevision: 9,
    });

    assert.deepEqual(jobs, {});
  },

  'a job the page has never seen is still added'() {
    const jobs = reduceJob({}, job('new', 11), { ...merge, appliedRevision: 9 });
    assert.ok(jobs.new);
  },

  'a payload without a revision is treated as oldest'() {
    const held = job('a', 3);
    const kept = mergeJob(held, { id: 'a', status: 'failed' }, ACTIVE);
    assert.equal(kept, held);
  },

  // Deleting every job and restarting resumes the counter from what survived -
  // nothing - so the server's numbers restart below the page's. Without the
  // epoch the page reads that as old news and shows a list of jobs that are
  // gone, forever: nothing it receives can ever outrank what it holds.
  'a restarted server empties a list the page still holds'() {
    const held = { a: job('a', 100, { status: 'done' }) };
    const result = reduceJobList(held, [], {
      ...merge,
      listRevision: 0,
      appliedRevision: 101,
      listEpoch: 2,
      appliedEpoch: 1,
    });

    assert.equal(result.applied, true);
    assert.deepEqual(result.jobs, {});
    assert.equal(result.revision, 0);
    assert.equal(result.epoch, 2);
  },

  'a job created after a restart is not outranked by the old clock'() {
    const held = { a: job('a', 100, { status: 'done' }) };
    const result = reduceJobList(held, [job('fresh', 1, { status: 'queued' })], {
      ...merge,
      listRevision: 1,
      appliedRevision: 101,
      listEpoch: 2,
      appliedEpoch: 1,
    });

    assert.equal(result.applied, true);
    assert.ok(result.jobs.fresh);
    // The survival rule is a within-epoch rule: revision 100 says nothing about
    // where a new server's clock stands, so the job it dated is not kept.
    assert.equal(result.jobs.a, undefined);
  },

  'a list from a server run that has ended is ignored'() {
    const held = { a: job('a', 2) };
    const result = reduceJobList(held, [job('b', 500)], {
      ...merge,
      listRevision: 500,
      appliedRevision: 2,
      listEpoch: 1,
      appliedEpoch: 2,
    });

    assert.equal(result.applied, false);
    assert.equal(result.jobs, held);
    assert.equal(result.epoch, 2);
  },

  'a single-job response is never merged across epochs'() {
    const held = { a: job('a', 5) };
    const opts = { ...merge, appliedRevision: 5, appliedEpoch: 2 };

    // Older: an answer from the previous run of the service.
    assert.equal(reduceJob(held, job('a', 900, { status: 'failed' }), { ...opts, jobEpoch: 1 }), held);
    // Newer: true, but the revisions held are now meaningless, and one job
    // cannot restate a list. The caller refetches instead.
    assert.equal(reduceJob(held, job('a', 1, { status: 'done' }), { ...opts, jobEpoch: 3 }), held);
  },

  'a job the server says was deleted leaves the page'() {
    const held = { a: job('a', 5), b: job('b', 6) };

    const next = reduceDeletion(held, 'a', { deletedRevision: 7 });

    assert.deepEqual(Object.keys(next), ['b']);
  },

  'a job recreated after the delete is not removed by it'() {
    // Same id, higher revision: created after the deletion was stamped, so the
    // 410 is an answer about a job that no longer exists under that number.
    const held = { a: job('a', 9) };

    assert.equal(reduceDeletion(held, 'a', { deletedRevision: 7 }), held);
  },

  'an undated deletion still drops the card'() {
    // Past the tombstone bound the server can only say "gone", not when. That
    // is still better than leaving a card for a job nobody can open.
    const next = reduceDeletion({ a: job('a', 9) }, 'a', { deletedRevision: NO_REVISION });

    assert.deepEqual(Object.keys(next), []);
  },

  'a deletion for a card we do not hold changes nothing'() {
    const held = { a: job('a', 5) };

    assert.equal(reduceDeletion(held, 'zzz', { deletedRevision: 7 }), held);
  },

  'an unlabelled response is judged on revisions alone'() {
    // Every rule above is off when either side has no epoch, so a cached page
    // talking to a server that does not send the header behaves as before.
    const result = reduceJobList({ a: job('a', 5) }, [job('a', 9)], {
      ...merge,
      listRevision: 9,
      appliedRevision: 5,
      listEpoch: NO_EPOCH,
      appliedEpoch: 7,
    });

    assert.equal(result.applied, true);
    assert.equal(result.jobs.a.revision, 9);
    assert.equal(result.epoch, 7);
  },
};

let failures = 0;
for (const [name, run] of Object.entries(cases)) {
  try {
    run();
  } catch (error) {
    failures += 1;
    console.error(`FAIL ${name}\n${error.message}`);
  }
}

if (failures) {
  console.error(`${failures} reducer case(s) failed`);
  process.exit(1);
}
console.log(`ok - ${Object.keys(cases).length} reducer cases`);
