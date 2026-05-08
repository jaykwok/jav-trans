export const state = {
  files: [],
  jobs: {},
  logLines: [],
  sse: null,
  activePreset: 'standard',
};

export const ACTIVE_STATUSES = new Set(['queued', 'asr', 'translating', 'writing']);
export const MAX_LOG = 200;
