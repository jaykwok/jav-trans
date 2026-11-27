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
  sse: null,
  gpuState: null,
};

// 'cancelling' belongs here: the run is still going, so the page must keep
// polling it until whoever owns it reports that it actually stopped.
export const ACTIVE_STATUSES = new Set([
  'queued', 'asr', 'translating', 'writing', 'cancelling', 'saving',
]);
export const CLEARABLE_STATUSES = new Set([
  'done', 'failed', 'cancelled', 'publish_failed', 'export_failed',
]);
export const ERROR_STATUSES = new Set([
  'failed', 'save_failed', 'publish_failed', 'export_failed',
]);
export const DELIVERY_FAILURE_STATUSES = new Set([
  'save_failed', 'publish_failed', 'export_failed',
]);
export const RETRYABLE_STATUSES = new Set([
  'failed', 'cancelled', 'save_failed', 'publish_failed', 'export_failed',
]);
export const CANCELLABLE_STATUSES = new Set([
  'queued', 'asr', 'translating', 'writing',
]);
export const SAVING_OR_DELIVERY_STATUSES = new Set([
  'saving', 'save_failed', 'publish_failed', 'export_failed',
]);
export const TRANSLATION_RETRY_STAGES = new Set([
  'translation_context', 'translation', 'write_output',
]);
export const MAX_LOG = 200;
