const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(process.argv[2], 'utf8')
  .replace(/^import .*\r?\n/gm, '')
  .replace(/\bexport\s+/g, '');
const flush = () => new Promise(resolve => setImmediate(resolve));

function setup(fetchImpl) {
  const elements = new Map();
  const messages = [];
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: '', checked: false, disabled: false, textContent: '',
      className: '', dataset: {}, style: {}, handlers: new Map(),
      addEventListener(type, handler) {
        const handlers = this.handlers.get(type) || [];
        handlers.push(handler);
        this.handlers.set(type, handlers);
      },
    });
    return elements.get(id);
  };
  const sandbox = {
    $: element, state: {}, showToast: message => messages.push(message),
    installModelCombobox() {}, saveFormMemory() {},
    fetch: fetchImpl, setTimeout, clearTimeout,
  };
  element('proxy-enabled').checked = true;
  element('proxy-protocol').value = 'socks5';
  element('proxy-host').value = '127.0.0.1';
  element('proxy-port').value = '1080';
  vm.runInNewContext(source, sandbox);
  sandbox.installSettingsPanel();
  return {
    element, messages, sandbox,
    async dispatch(id, type) {
      await Promise.all((element(id).handlers.get(type) || []).map(fn => fn()));
      await flush();
    },
  };
}

async function main() {
  const requests = [];
  const failedSave = setup(async url => {
    requests.push(url);
    return { ok: false, text: async () => 'synthetic save failure' };
  });
  await failedSave.dispatch('btn-proxy-test', 'click');
  assert.deepEqual(requests, ['/api/settings']);
  assert.equal(failedSave.element('proxy-test-result').className, 'proxy-test-result fail');
  assert.equal(failedSave.element('proxy-enabled').disabled, false);

  let savedBody;
  const disabled = setup(async (_url, options) => {
    savedBody = JSON.parse(options.body);
    return { ok: true };
  });
  disabled.element('proxy-enabled').checked = false;
  await disabled.dispatch('proxy-enabled', 'change');
  assert.deepEqual(savedBody, { proxy_protocol: 'socks5', proxy_host: '', proxy_port: null });
  assert(disabled.messages.includes('代理已关闭'));
  assert.equal(disabled.element('proxy-status-tag').textContent, '未启用');

  let finishTest;
  const changed = setup(async url => {
    if (url === '/api/settings') return { ok: true };
    return await new Promise(resolve => { finishTest = resolve; });
  });
  const pending = changed.dispatch('btn-proxy-test', 'click');
  await flush();
  assert.equal(changed.element('proxy-port').disabled, true);
  assert.equal(changed.element('proxy-enabled').disabled, true);
  // A second settings refresh can still update controls programmatically.
  changed.element('proxy-port').value = '1081';
  finishTest({ ok: true, json: async () => ({ ok: true, elapsed_ms: 1, proxy_url: 'socks5://127.0.0.1:1080' }) });
  await pending;
  assert.equal(changed.element('proxy-test-result').className, 'proxy-test-result fail');
  assert.match(changed.element('proxy-test-result').textContent, /配置已变化/);
  assert.equal(changed.element('proxy-port').disabled, false);

  const success = setup(async url => url === '/api/settings' ? { ok: true } : {
    ok: true, json: async () => ({ ok: true, elapsed_ms: 1, proxy_url: 'socks5://127.0.0.1:1080' }),
  });
  await success.dispatch('btn-proxy-test', 'click');
  assert.equal(success.element('proxy-test-result').className, 'proxy-test-result ok');
  success.element('proxy-port').value = '1081';
  await success.dispatch('proxy-port', 'input');
  assert.equal(success.element('proxy-test-result').textContent, '');

  const invalid = setup(() => { throw new Error('invalid input must not reach fetch'); });
  invalid.element('proxy-port').value = 'invalid';
  await invalid.dispatch('btn-proxy-test', 'click');
  assert.match(invalid.element('proxy-test-result').textContent, /端口/);

  const pendingSaves = [];
  const ordered = setup((_url, options) => new Promise(resolve => pendingSaves.push({ resolve, body: JSON.parse(options.body) })));
  const first = ordered.sandbox.saveSettingsBody({ proxy_port: 1080 });
  const second = ordered.sandbox.saveSettingsBody({ proxy_port: 1081 });
  await flush();
  assert.equal(pendingSaves.length, 1);
  pendingSaves[0].resolve({ ok: true });
  await first;
  await flush();
  assert.equal(pendingSaves.length, 2);
  assert.equal(pendingSaves[1].body.proxy_port, 1081);
  pendingSaves[1].resolve({ ok: true });
  await second;

  let finishSave;
  const edited = setup(async (_url, options) => {
    if (options?.method === 'POST') return await new Promise(resolve => { finishSave = resolve; });
    return { ok: true, json: async () => ({ base_url: 'https://old.example.test/v1', model: 'old-model' }) };
  });
  edited.element('api-base-url').value = 'https://old.example.test/v1';
  edited.element('api-model').value = 'old-model';
  const saving = edited.dispatch('btn-save-translation', 'click');
  await flush();
  edited.element('api-base-url').value = 'https://new.example.test/v1';
  edited.element('api-model').value = 'new-model';
  edited.element('proxy-port').value = '1082';
  finishSave({ ok: true });
  await saving;
  assert.equal(edited.element('api-base-url').value, 'https://new.example.test/v1');
  assert.equal(edited.element('api-model').value, 'new-model');
  assert.equal(edited.element('proxy-port').value, '1082');
  assert(edited.messages.includes('翻译设置已保存到 .env'));
  console.log('ok - proxy failures, switching, stale results, save ordering, and preserving edits');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
