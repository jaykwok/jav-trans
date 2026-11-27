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
  let reject;
  const promise = new Promise((resolve, rej) => {
    settle = resolve;
    reject = rej;
  });
  return { promise, settle, reject };
}

/** Let every queued follow-up actually start. */
function flush() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

const cases = {
  async 'a burst during one request becomes one follow-up'() {
    const opened = [];
    let pending = deferred();
    const refresh = coalesce(id => {
      opened.push(id);
      return pending.promise;
    });

    const first = refresh('a');
    for (let i = 0; i < 20; i += 1) refresh('a');
    assert.deepEqual(opened, ['a']);

    const settle = pending.settle;
    pending = deferred();
    settle();
    await first;
    await flush();

    assert.deepEqual(opened, ['a', 'a']);
    pending.settle();
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

    assert.deepEqual(opened, ['a', 'b']);
    pending.settle();
  },

  async 'trailing callers share one deferred promise that settles after the follow-up'() {
    let callCount = 0;
    const req1 = deferred();
    const req2 = deferred();
    const refresh = coalesce(() => {
      callCount += 1;
      return callCount === 1
        ? req1.promise.then(() => 'stale')
        : req2.promise.then(() => 'fresh');
    });

    const first = refresh('a');
    const trailing1 = refresh('a');
    const trailing2 = refresh('a');

    assert.notEqual(trailing1, first);
    assert.equal(trailing1, trailing2);

    let trailingSettled = false;
    trailing1.then(() => { trailingSettled = true; });

    req1.settle();
    assert.equal(await first, 'stale');
    await flush();
    // Must still be waiting for the follow-up request, not resolved by req1.
    assert.equal(trailingSettled, false);
    assert.equal(callCount, 2);

    req2.settle();
    assert.equal(await trailing1, 'fresh');
    assert.equal(await trailing2, 'fresh');
    assert.equal(trailingSettled, true);
  },

  async 'an unobserved trailing rejection does not trigger unhandledRejection or wedge the key'() {
    let unhandled = 0;
    const onUnhandled = () => { unhandled += 1; };
    process.on('unhandledRejection', onUnhandled);
    try {
      let attempt = 0;
      const req1 = deferred();
      const refresh = coalesce(() => {
        attempt += 1;
        if (attempt === 1) return req1.promise;
        if (attempt === 2) return Promise.reject(new Error('transient follow-up failure'));
        return Promise.resolve('recovered');
      });

      const first = refresh('a');
      // Fire-and-forget trailing call (not awaited).
      refresh('a');
      req1.settle();
      await first;
      await flush();

      assert.equal(unhandled, 0);
      assert.equal(await refresh('a'), 'recovered');
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  },

  async 'a failed request does not wedge the subject'() {
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
