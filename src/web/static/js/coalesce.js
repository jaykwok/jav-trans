// One request at a time per subject, and one more if anything changed while it
// was in flight.
//
// Every SSE event for a job triggers a GET for that job, and translation emits
// events far faster than a request completes: a long film produced hundreds of
// overlapping GETs for the same id, all asking the same question, most of them
// answered after the answer had already changed. The 3s poll did the same to
// the list.
//
// Debouncing would be the wrong fix - it delays the truth and, worse, it used to
// be the excuse for tolerating out-of-order responses. Ordering is decided by
// revision in `jobsMerge.js` and stays decided there. This only removes requests
// that are redundant *by construction*: while a request for a subject is open,
// later events about it cannot be answered any sooner than by the answer already
// coming, so they collapse into a single follow-up issued when it lands.
//
// The result is self-pacing - one request in flight plus at most one queued, so
// a fast server refreshes often and a slow one is not piled onto.

/**
 * Wrap `run(key)` so that calls for the same key never overlap.
 *
 * A call made while one is in flight returns that same promise and schedules
 * exactly one follow-up, however many times it is called. Keys are independent:
 * two different jobs still refresh in parallel.
 */
export function coalesce(run) {
  const inFlight = new Map();
  const trailing = new Set();

  function start(key) {
    let started;
    try {
      started = Promise.resolve(run(key));
    } catch (error) {
      // A synchronous throw is still a finished attempt, not a request that
      // never ends: turning it into a rejection keeps the bookkeeping below on
      // one path, so the key cannot be left permanently "in flight".
      started = Promise.reject(error);
    }
    // `finally` runs in a microtask even when `started` is already settled, so
    // `inFlight` is always populated before this can clear it.
    const request = started.finally(() => {
      inFlight.delete(key);
      if (trailing.delete(key)) start(key);
    });
    inFlight.set(key, request);
    return request;
  }

  return function call(key) {
    const existing = inFlight.get(key);
    if (existing) {
      trailing.add(key);
      return existing;
    }
    return start(key);
  };
}
