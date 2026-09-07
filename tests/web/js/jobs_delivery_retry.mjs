import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const elements = new Map();
function element() {
  return {
    children: [], listeners: {},
    addEventListener(name, handler) { this.listeners[name] = handler; },
    appendChild(child) { this.children.push(child); },
  };
}
globalThis.document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  },
  createElement: element,
};
const alerts = [];
globalThis.alert = message => alerts.push(message);
const requests = [];
globalThis.fetch = async (url, options) => {
  requests.push({ url, options });
  return { ok: true, json: async () => ({ id: 'sample-job' }) };
};
const [, , modulePath] = process.argv;
const { state } = await import(pathToFileURL(path.join(path.dirname(modulePath), 'state.js')));
const { installJobAreaHandlers } = await import(pathToFileURL(modulePath));
let syncs = 0;
let refreshes = 0;
installJobAreaHandlers(async () => { refreshes += 1; }, async () => {
  syncs += 1;
  throw new Error('invalid translation settings');
});
const click = elements.get('job-area').listeners.click;
const retryButton = { dataset: { retry: 'sample-job' } };
for (const status of ['save_failed', 'publish_failed', 'export_failed']) {
  state.jobs['sample-job'] = { id: 'sample-job', spec: {}, status };
  await click({ target: { closest: selector => selector === '[data-retry]' ? retryButton : null } });
}
assert.equal(syncs, 0);
assert.equal(alerts.length, 0);
assert.equal(refreshes, 3);
assert.deepEqual(requests.map(item => [item.url, item.options.method]),
  Array(3).fill(['/api/jobs/sample-job/retry', 'POST']));

state.jobs['sample-job'].status = 'failed';
await click({ target: { closest: selector => selector === '[data-retry]' ? retryButton : null } });
assert.equal(syncs, 1);
assert.equal(requests.length, 3);
assert.match(alerts[0], /invalid translation settings/);
console.log('ok - delivery retries bypass settings while computation retries validate them');
