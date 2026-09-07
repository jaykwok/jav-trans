// Regression harness for per-subject request coalescing, run against the real
// module.
//
// Executed by tests/web/test_request_coalescing.py:
//     node request_coalescing.mjs <path to src/web/static/js/coalesce.js>

import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';

const [, , modulePath] = process.argv;
const { coalesce } = await import(pathToFileURL(modulePath).href);

/** A request whose completion the test decides. */
function deferred() {
  let settle;
  const promise = new Promise(resolve => { settle = resolve; });
  return { promise, settle };
}

/** Let every queued follow-up actually start. */
function flush() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

const cases = {
  async 'a burst during one request becomes one follow-up'() {
    // The reported cost: every SSE event issues a GET, and translation emits
    // them faster than a request completes.
    const opened = [];
    let pending = deferred();
    const refresh = coalesce(id => {
      opened.push(id);
      return pending.promise;
    });

    const first = refresh('a');
    for (let i = 0; i < 20; i += 1) refresh('a');
    assert.deepEqual(opened, ['a']);  // still one request open

    const settle = pending.settle;
    pending = deferred();
    settle();
    await first;
    await flush();

    // One follow-up, not twenty: the events that arrived while the first was
    // open could not have been answered any sooner than by its successor.
    assert.deepEqual(opened, ['a', 'a']);
  },

  async 'a quiet subject issues no follow-up'() {
    const opened = [];
    const refresh = coalesce(id => { opened.push(id); return Promise.resolve(); });

    await refresh('a');
    await flush();

    assert.deepEqual(opened, ['a']);
  },

  async 'two jobs refresh independently'() {
    const opened = [];
    const pending = deferred();
    const refresh = coalesce(id => { opened.push(id); return pending.promise; });

    refresh('a');
    refresh('b');

    // Coalescing is per subject: a slow answer about one job must not delay
    // the first question about another.
    assert.deepEqual(opened, ['a', 'b']);
    pending.settle();
  },

  async 'callers during a request join the one in flight'() {
    const pending = deferred();
    const refresh = coalesce(() => pending.promise.then(() => 'answer'));

    const first = refresh('a');
    const joined = refresh('a');

    assert.equal(joined, first);
    pending.settle();
    assert.equal(await joined, 'answer');
  },

  async 'a failed request does not wedge the subject'() {
    // `finally`, not `then`: a rejected refresh that left the key marked in
    // flight would silently stop the page updating that job forever.
    const opened = [];
    const refresh = coalesce(id => {
      opened.push(id);
      return Promise.reject(new Error('offline'));
    });

    await refresh('a').catch(() => {});
    await refresh('a').catch(() => {});

    assert.deepEqual(opened, ['a', 'a']);
  },

  async 'a synchronous throw does not wedge the subject'() {
    const opened = [];
    const refresh = coalesce(id => {
      opened.push(id);
      throw new Error('bad state');
    });

    await refresh('a').catch(() => {});
    await refresh('a').catch(() => {});

    assert.deepEqual(opened, ['a', 'a']);
  },
};

let failures = 0;
for (const [name, run] of Object.entries(cases)) {
  try {
    await run();
  } catch (error) {
    failures += 1;
    console.error(`FAIL ${name}\n${error.stack || error.message}`);
  }
}

if (failures) {
  console.error(`${failures} coalescing case(s) failed`);
  process.exit(1);
}
console.log(`ok - ${Object.keys(cases).length} coalescing cases`);
