export const state = {
  files: [],
  jobs: {},
  logLines: [],
  sse: null,
  activePreset: 'standard',
  gpuState: null,
};

// 'cancelling' belongs here: the run is still going, so the page must keep
// polling it until whoever owns it reports that it actually stopped.
export const ACTIVE_STATUSES = new Set(['queued', 'asr', 'translating', 'writing', 'cancelling']);
export const MAX_LOG = 200;
