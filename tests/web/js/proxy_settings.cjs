const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

function stripModuleSyntax(code) {
  return code
    .replace(/^import\s+[\s\S]*?from\s+['"][^'"]+['"];?\s*\r?\n/gm, '')
    .replace(/^export\s+\{[\s\S]*?\}\s+from\s+['"][^'"]+['"];?\s*\r?\n/gm, '')
    .replace(/\bexport\s+/g, '');
}

const settingsPath = process.argv[2];
const jsDir = path.dirname(settingsPath);
const siblingFiles = ['modelRequirements.js', 'proxySettings.js'];
const parts = [];
for (const file of siblingFiles) {
  const full = path.join(jsDir, file);
  if (fs.existsSync(full)) {
    parts.push(stripModuleSyntax(fs.readFileSync(full, 'utf8')));
  }
}
parts.push(stripModuleSyntax(fs.readFileSync(settingsPath, 'utf8')));
const source = parts.join('\n\n');

const flush = () => new Promise(resolve => setImmediate(resolve));

async function readErrorDetail(response) {
  const raw = typeof response?.text === 'function' ? await response.text().catch(() => '') : '';
  try {
    const detail = JSON.parse(raw)?.detail;
    if (typeof detail === 'string' && detail.trim()) return detail.trim();
  } catch {}
  return raw.trim() || `HTTP ${response?.status ?? 'error'}`;
}

function setup(fetchImpl) {
  const elements = new Map();
  const messages = [];
  const element = id => {
    if (!elements.has(id)) {
      const classes = new Set();
      elements.set(id, {
        value: '', checked: false, disabled: false, open: false, hidden: false,
        textContent: '', innerHTML: '', className: '', dataset: {}, style: {},
        options: [],
        handlers: new Map(),
        classList: {
          add: name => classes.add(name),
          remove: name => classes.delete(name),
          contains: name => classes.has(name),
          toggle: (name, force) => {
            const next = force === undefined ? !classes.has(name) : Boolean(force);
            if (next) classes.add(name); else classes.delete(name);
            return next;
          },
        },
        addEventListener(type, handler) {
          const handlers = this.handlers.get(type) || [];
          handlers.push(handler);
          this.handlers.set(type, handlers);
        },
      });
    }
    return elements.get(id);
  };
  const sandbox = {
    $: element,
    state: {},
    showToast: message => messages.push(message),
    addLog() {},
    escHtml: s => String(s),
    readErrorDetail,
    installModelCombobox() {},
    setModelComboboxOptions() {},
    loadFormMemory: () => ({ controls: {}, details: {} }),
    saveFormMemory() {},
    applyFormMemory() {},
    fetch: fetchImpl,
    setTimeout,
    clearTimeout,
  };
  element('proxy-enabled').checked = true;
  element('proxy-protocol').value = 'socks5';
  element('proxy-host').value = '127.0.0.1';
  element('proxy-port').value = '1080';
  element('panel-proxy').open = true;
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
    return {
      ok: false,
      status: 422,
      text: async () => JSON.stringify({ detail: 'synthetic save failure' }),
    };
  });
  await failedSave.dispatch('btn-proxy-test', 'click');
  assert.deepEqual(requests, ['/api/settings']);
  assert.equal(failedSave.element('proxy-test-result').className, 'proxy-test-result fail');
  assert.match(failedSave.element('proxy-test-result').textContent, /synthetic save failure/);
  assert.doesNotMatch(failedSave.element('proxy-test-result').textContent, /\{"detail"/);
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

  // Authenticated proxy: backend returns redacted URL (`***@host:port`) and UI must accept it.
  const authSuccess = setup(async url => url === '/api/settings' ? { ok: true } : {
    ok: true, json: async () => ({ ok: true, elapsed_ms: 12, proxy_url: 'socks5://***@127.0.0.1:1080' }),
  });
  authSuccess.element('proxy-host').value = 'alice:s3cr3t@127.0.0.1';
  await authSuccess.dispatch('btn-proxy-test', 'click');
  assert.equal(authSuccess.element('proxy-test-result').className, 'proxy-test-result ok');
  assert.match(authSuccess.element('proxy-test-result').textContent, /12ms/);

  // Authenticated proxy with mixed-case hostname: backend normalizes hostname to lowercase via urlsplit.
  const authMixedCase = setup(async url => url === '/api/settings' ? { ok: true } : {
    ok: true, json: async () => ({ ok: true, elapsed_ms: 9, proxy_url: 'socks5://***@proxy.example:1080' }),
  });
  authMixedCase.element('proxy-host').value = 'alice:s3cr3t@Proxy.Example';
  await authMixedCase.dispatch('btn-proxy-test', 'click');
  assert.equal(authMixedCase.element('proxy-test-result').className, 'proxy-test-result ok');
  assert.match(authMixedCase.element('proxy-test-result').textContent, /9ms/);

  // Changing only credentials while the request is in flight still counts as changed.
  let finishAuthTest;
  const authChanged = setup(async url => {
    if (url === '/api/settings') return { ok: true };
    return await new Promise(resolve => { finishAuthTest = resolve; });
  });
  authChanged.element('proxy-host').value = 'alice:oldpass@127.0.0.1';
  const pendingAuth = authChanged.dispatch('btn-proxy-test', 'click');
  await flush();
  authChanged.element('proxy-host').value = 'alice:newpass@127.0.0.1';
  finishAuthTest({ ok: true, json: async () => ({ ok: true, elapsed_ms: 5, proxy_url: 'socks5://***@127.0.0.1:1080' }) });
  await pendingAuth;
  assert.equal(authChanged.element('proxy-test-result').className, 'proxy-test-result fail');
  assert.match(authChanged.element('proxy-test-result').textContent, /配置已变化/);

  const invalid = setup(() => { throw new Error('invalid input must not reach fetch'); });
  invalid.element('proxy-port').value = 'invalid';
  await invalid.dispatch('btn-proxy-test', 'click');
  assert.match(invalid.element('proxy-test-result').textContent, /端口/);

  // Collapsed or disabled proxy must NOT block unrelated job submission.
  const submitBodies = [];
  const collapsedProxy = setup(async (_url, options) => {
    submitBodies.push(JSON.parse(options.body));
    return { ok: true };
  });
  collapsedProxy.element('proxy-enabled').checked = true;
  collapsedProxy.element('proxy-host').value = '';
  collapsedProxy.element('proxy-port').value = '';
  collapsedProxy.element('panel-proxy').open = false;
  await collapsedProxy.sandbox.syncSettingsFromFormForSubmit();
  assert.equal(submitBodies.length, 1);
  assert.equal(Object.hasOwn(submitBodies[0], 'proxy_host'), false);

  // Glossary arrow conversion (`->`, `→`) and comma preservation across round-trip.
  const glossaryBodies = [];
  const glossaryCheck = setup(async (_url, options) => {
    if (options?.method === 'POST') {
      glossaryBodies.push(JSON.parse(options.body));
      return { ok: true, json: async () => ({ ok: true }) };
    }
    return { ok: true, json: async () => ({}) };
  });
  glossaryCheck.element('btn-save-translation').textContent = '保存翻译设置';
  glossaryCheck.element('api-glossary').value = 'ちんぽ -> 肉棒\nねこ→猫, 小猫\nbad line without sep';
  await glossaryCheck.dispatch('btn-save-translation', 'click');
  assert.equal(glossaryBodies[0].translation_glossary, 'ちんぽ-肉棒\nねこ-猫, 小猫');
  assert.equal(glossaryCheck.element('api-glossary').value, 'ちんぽ-肉棒\nねこ-猫, 小猫');
  assert.equal(glossaryCheck.element('btn-save-translation').disabled, false);
  assert.equal(glossaryCheck.element('btn-save-translation').textContent, '保存翻译设置');
  assert.match(glossaryCheck.messages[0], /已忽略 1 行/);
  assert.equal(
    glossaryCheck.sandbox.formatGlossaryForTextarea('ちんぽ-肉棒\nねこ-猫, 小猫'),
    'ちんぽ-肉棒\nねこ-猫, 小猫',
  );
  assert.equal(
    glossaryCheck.sandbox.formatGlossaryForTextarea('ちんぽ-肉棒, チンポ-肉棒'),
    'ちんぽ-肉棒\nチンポ-肉棒',
  );

  // refreshModelRequirements unhides panel on HTTP / network error.
  const failedReq = setup(async () => ({
    ok: false,
    status: 500,
    text: async () => JSON.stringify({ detail: 'probe error' }),
  }));
  failedReq.element('panel-model-requirements').hidden = true;
  await failedReq.sandbox.refreshModelRequirements();
  assert.equal(failedReq.element('panel-model-requirements').hidden, false);
  assert.match(failedReq.element('model-requirements-notice').innerHTML, /probe error/);

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
  edited.element('btn-save-translation').textContent = '保存翻译设置';
  edited.element('api-base-url').value = 'https://old.example.test/v1';
  edited.element('api-model').value = 'old-model';
  const saving = edited.dispatch('btn-save-translation', 'click');
  await flush();
  assert.equal(edited.element('btn-save-translation').disabled, true);
  assert.equal(edited.element('btn-save-translation').textContent, '保存中…');
  edited.element('api-base-url').value = 'https://new.example.test/v1';
  edited.element('api-model').value = 'new-model';
  edited.element('proxy-port').value = '1082';
  finishSave({ ok: true });
  await saving;
  assert.equal(edited.element('btn-save-translation').disabled, false);
  assert.equal(edited.element('btn-save-translation').textContent, '保存翻译设置');
  assert.equal(edited.element('api-base-url').value, 'https://new.example.test/v1');
  assert.equal(edited.element('api-model').value, 'new-model');
  assert.equal(edited.element('proxy-port').value, '1082');
  assert(edited.messages.includes('翻译设置已保存到 config/config.json'));

  // Translation backend toggle must show/hide local vs OpenAI fields cleanly.
  const toggle = setup(async () => ({ ok: true }));
  assert.equal(toggle.element('llamacpp-backend-fields').style.display, '');
  assert.equal(toggle.element('openai-backend-fields').style.display, 'none');
  toggle.element('translation-backend').value = 'openai';
  await toggle.dispatch('translation-backend', 'change');
  assert.equal(toggle.element('openai-backend-fields').style.display, '');
  assert.equal(toggle.element('llamacpp-backend-fields').style.display, 'none');
  toggle.element('translation-backend').value = 'llamacpp';
  await toggle.dispatch('translation-backend', 'change');
  assert.equal(toggle.element('llamacpp-backend-fields').style.display, '');
  assert.equal(toggle.element('openai-backend-fields').style.display, 'none');

  // Dynamic subtitle output labels follow target language, and ja_only dims translation panel.
  assert.equal(toggle.sandbox.subtitleModeLabel('zh', '简体中文'), '简体中文');
  assert.equal(toggle.sandbox.subtitleModeLabel('bilingual', '简体中文'), '双语（日文 + 简体中文）');
  assert.equal(toggle.sandbox.subtitleModeLabel('zh', '繁體中文'), '繁體中文');
  assert.equal(toggle.sandbox.subtitleModeLabel('bilingual', '繁體中文'), '双语（日文 + 繁體中文）');
  assert.equal(toggle.sandbox.subtitleModeLabel('zh', 'English'), 'English');
  assert.equal(toggle.sandbox.subtitleModeLabel('bilingual', 'English'), '双语（日文 + English）');
  assert.equal(toggle.sandbox.subtitleModeLabel('ja_only', 'English'), '仅日文（不翻译）');

  const modeEl = toggle.element('r-mode');
  modeEl.options = [
    { value: 'zh', textContent: '' },
    { value: 'bilingual', textContent: '' },
    { value: 'ja_only', textContent: '' },
  ];
  toggle.element('api-target-lang').value = 'English';
  await toggle.dispatch('api-target-lang', 'change');
  assert.deepEqual(
    modeEl.options.map(o => o.textContent),
    ['English', '双语（日文 + English）', '仅日文（不翻译）'],
  );

  modeEl.value = 'ja_only';
  await toggle.dispatch('r-mode', 'change');
  assert.equal(toggle.element('panel-translation').classList.contains('disabled'), true);
  modeEl.value = 'zh';
  await toggle.dispatch('r-mode', 'change');
  assert.equal(toggle.element('panel-translation').classList.contains('disabled'), false);

  console.log('ok - proxy failures, switching, stale results, save ordering, backend toggle, dynamic subtitle output, glossary, and preserving edits');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
