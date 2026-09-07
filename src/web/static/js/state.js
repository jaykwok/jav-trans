export const state = {
  files: [],
  jobs: {},
  // The newest list reading applied to `jobs` (see jobsMerge.js). Anything
  // older than this is a response that was overtaken on the way here - but only
  // within one epoch: `jobsRevision` restarts when the service does, so the
  // epoch below says which server run issued it.
  jobsRevision: -1,
  jobsEpoch: -1,
  jobsTombstones: {},
  logLines: [],
  sse: null,
  activePreset: 'standard',
  gpuState: null,
};

// 'cancelling' belongs here: the run is still going, so the page must keep
// polling it until whoever owns it reports that it actually stopped.
export const ACTIVE_STATUSES = new Set(['queued', 'asr', 'translating', 'writing', 'cancelling', 'saving']);
export const MAX_LOG = 200;
