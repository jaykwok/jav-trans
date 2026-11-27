// One request at a time per subject, and one more if anything changed while it
// was in flight.
//
// Ordering is decided by revision in `jobsMerge.js`. This module removes
// redundant overlapping requests while ensuring callers waiting on a trailing
// refresh resolve only after that follow-up request settles (avoiding stale
// intermediate states like a brief empty-list flash after submit/delete).

/**
 * Wrap `run(key)` so that calls for the same key never overlap.
 *
 * Callers arriving while a request is already in flight share one deferred
 * promise that settles when the trailing follow-up request finishes.
 */
export function coalesce(run) {
  const inFlight = new Map();
  const trailing = new Map();

  function createDeferred() {
    let resolve;
    let reject;
    const promise = new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    });
    // Suppress unhandledrejection if a trailing caller does not await/catch;
    // callers that DO await `promise` still observe the rejection normally.
    promise.catch(() => {});
    return { promise, resolve, reject };
  }

  function start(key, deferred = null) {
    let started;
    try {
      started = Promise.resolve(run(key));
    } catch (error) {
      started = Promise.reject(error);
    }

    const request = started.then(
      value => {
        if (deferred) deferred.resolve(value);
        return value;
      },
      error => {
        if (deferred) deferred.reject(error);
        throw error;
      },
    );

    const settled = request.finally(() => {
      inFlight.delete(key);
      const next = trailing.get(key);
      if (next) {
        trailing.delete(key);
        start(key, next).catch(() => {});
      }
    });

    inFlight.set(key, settled);
    return settled;
  }

  return function call(key) {
    if (inFlight.has(key)) {
      let next = trailing.get(key);
      if (!next) {
        next = createDeferred();
        trailing.set(key, next);
      }
      return next.promise;
    }
    return start(key);
  };
}
