import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const elements = new Map();
function makeNode(tag = 'div') {
  let _innerHTML = '';
  const node = {
    tagName: tag.toUpperCase(),
    className: '',
    dataset: {},
    style: {},
    children: [],
    parentNode: null,
    listeners: {},
    innerHTMLWrites: 0,
    _subEls: null,
    get innerHTML() { return _innerHTML; },
    set innerHTML(val) {
      _innerHTML = String(val);
      this.innerHTMLWrites += 1;
      const detailsMatch = _innerHTML.includes('job-error-wrap');
      this._subEls = {
        '.progress-fill': _innerHTML.includes('progress-fill') ? { style: {} } : null,
        '.dl-bar-fill': _innerHTML.includes('dl-bar-fill') ? { style: {} } : null,
        '.job-stage': _innerHTML.includes('job-stage') ? { textContent: '' } : null,
        '.dl-label': _innerHTML.includes('dl-label') ? { textContent: '' } : null,
        '.dl-info': _innerHTML.includes('dl-info') ? { textContent: '' } : null,
        '.job-error-wrap': detailsMatch ? { open: false } : null,
      };
    },
    classList: {
      contains(cls) { return node.className.split(/\s+/).includes(cls); },
    },
    addEventListener(name, handler) { this.listeners[name] = handler; },
    appendChild(child) {
      child.parentNode = this;
      this.children.push(child);
    },
    insertBefore(child, ref) {
      const existingIdx = this.children.indexOf(child);
      if (existingIdx >= 0) this.children.splice(existingIdx, 1);
      child.parentNode = this;
      if (ref && this.children.includes(ref)) {
        this.children.splice(this.children.indexOf(ref), 0, child);
      } else {
        this.children.push(child);
      }
    },
    remove() {
      if (!this.parentNode) return;
      const idx = this.parentNode.children.indexOf(this);
      if (idx >= 0) this.parentNode.children.splice(idx, 1);
      this.parentNode = null;
    },
    get previousElementSibling() {
      if (!this.parentNode) return null;
      const idx = this.parentNode.children.indexOf(this);
      return idx > 0 ? this.parentNode.children[idx - 1] : null;
    },
    get nextSibling() {
      if (!this.parentNode) return null;
      const idx = this.parentNode.children.indexOf(this);
      return idx >= 0 && idx + 1 < this.parentNode.children.length ? this.parentNode.children[idx + 1] : null;
    },
    querySelectorAll(selector) {
      if (selector === '.job-card') {
        return this.children.filter(c => c.classList.contains('job-card'));
      }
      return [];
    },
    querySelector(selector) {
      if (selector === '.job-card') {
        return this.children.find(c => c.classList.contains('job-card')) || null;
      }
      return this._subEls?.[selector] || null;
    },
  };
  return node;
}

globalThis.document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeNode('div'));
    return elements.get(id);
  },
  createElement: makeNode,
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
const { installJobAreaHandlers, renderJobs } = await import(pathToFileURL(modulePath));

// 1. Delivery retry bypasses settings sync; computation retry validates settings.
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
assert.deepEqual(
  requests.map(item => [item.url, item.options.method]),
  Array(3).fill(['/api/jobs/sample-job/retry', 'POST']),
);

state.jobs['sample-job'].status = 'failed';
await click({ target: { closest: selector => selector === '[data-retry]' ? retryButton : null } });
assert.equal(syncs, 1);
assert.equal(requests.length, 3);
assert.match(alerts[0], /invalid translation settings/);

// 2. Minimalist completed card renders ONLY ▶ 播放, 📂 文件夹, ✕ (no SRT open/download buttons).
state.jobs = {
  'done-job': {
    id: 'done-job',
    created_at: '2026-11-27T00:00:01.000Z',
    status: 'done',
    spec: { video_paths: ['D:\\Videos\\sample-a.mp4'] },
    output_target: 'D:\\Videos\\sample-a.srt',
    artifacts: ['sample-a.srt'],
    subtitle_cue_count: 10,
  },
  'export-failed-job': {
    id: 'export-failed-job',
    created_at: '2026-11-27T00:00:02.000Z',
    status: 'export_failed',
    spec: { video_paths: ['D:\\Videos\\sample-b.mp4'] },
    output_target: 'D:\\Videos\\sample-b.srt',
    artifacts: ['sample-b.srt'],
    error: 'x'.repeat(300),
  },
};
renderJobs();
const jobArea = elements.get('job-area');
const doneCard = jobArea.children.find(c => c.dataset.id === 'done-job');
assert.ok(doneCard);
assert.match(doneCard.innerHTML, /data-play="done-job"/);
assert.match(doneCard.innerHTML, /data-folder=/);
assert.match(doneCard.innerHTML, /data-remove="done-job"/);
assert.doesNotMatch(doneCard.innerHTML, /data-open-artifact/);
assert.doesNotMatch(doneCard.innerHTML, /data-dl/);
assert.doesNotMatch(doneCard.innerHTML, /下载 SRT/);

const exportCard = jobArea.children.find(c => c.dataset.id === 'export-failed-job');
assert.ok(exportCard);
assert.match(exportCard.innerHTML, /data-retry="export-failed-job"/);
assert.match(exportCard.innerHTML, /data-folder=/);
assert.match(exportCard.innerHTML, /data-remove="export-failed-job"/);
assert.doesNotMatch(exportCard.innerHTML, /data-open-artifact/);
assert.doesNotMatch(exportCard.innerHTML, /data-dl/);

// Preserves <details class="job-error-wrap"> open state on structural rewrite.
exportCard.querySelector('.job-error-wrap').open = true;
state.jobs['export-failed-job'].error = 'y'.repeat(310);
renderJobs();
assert.equal(exportCard.querySelector('.job-error-wrap').open, true);

// 3. Per-tick progress updates mutate textContent/style WITHOUT rewriting innerHTML.
state.jobs = {
  'run-job': {
    id: 'run-job',
    created_at: '2026-11-27T00:00:03.000Z',
    status: 'translating',
    current_stage: 'translation',
    spec: { video_paths: ['D:\\Videos\\sample-c.mp4'], skip_translation: false },
    progress: { stage: 'translation', translated: 10, expected: 100 },
    artifacts: [],
  },
};
renderJobs();
const runCard = jobArea.children.find(c => c.dataset.id === 'run-job');
assert.equal(runCard.innerHTMLWrites, 1);
assert.equal(runCard.querySelector('.job-stage').textContent, '翻译中 10/100');

state.jobs['run-job'].progress = { stage: 'translation', translated: 25, expected: 100 };
renderJobs();
assert.equal(runCard.innerHTMLWrites, 1);
assert.equal(runCard.querySelector('.job-stage').textContent, '翻译中 25/100');

// 4. skip_translation job never shows "翻译中".
state.jobs['run-job'].spec.skip_translation = true;
renderJobs();
assert.doesNotMatch(runCard.querySelector('.job-stage').textContent, /翻译中/);
assert.match(runCard.querySelector('.job-stage').textContent, /写入字幕/);

console.log('ok - delivery retries, minimalist cards, incremental progress updates, and skip_translation labels');
