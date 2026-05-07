/* JAVTrans frontend — no build tools, plain ES2022 */

const $ = id => document.getElementById(id);

// ── State ────────────────────────────────────────────────────────────────────
// file entry: { type:'path', name, path }
const state = {
  files: [],
  jobs: {},
  logLines: [],
  sse: null,
};

// ── DOM refs ─────────────────────────────────────────────────────────────────
const dropZone    = $('drop-zone');
const fileList    = $('file-list');
const btnAddFolder  = $('btn-add-folder');
const btnSubmit     = $('btn-submit');
const jobArea       = $('job-area');
const jobAreaHeader = $('job-area-header');
const emptyState    = $('empty-state');
const logScroll     = $('log-scroll');
const connDot       = $('conn-dot');
const btnClearDone  = $('btn-clear-done');

// ── Form memory & paste menu ─────────────────────────────────────────────────
const FORM_MEMORY_KEY = 'javtrans.formMemory.v1';
const FORM_MEMORY_EXCLUDED_IDS = new Set(['s-apikey']);
const FORM_MEMORY_SELECTOR = 'input[id], select[id], textarea[id]';
const PASTEABLE_INPUT_TYPES = new Set(['', 'text', 'search', 'url', 'tel', 'email', 'password', 'number']);
let pasteTarget = null;
let pasteMenu = null;

function formMemoryControls() {
  return [...document.querySelectorAll(FORM_MEMORY_SELECTOR)].filter(el => {
    const type = (el.type || '').toLowerCase();
    return !FORM_MEMORY_EXCLUDED_IDS.has(el.id)
      && !['button', 'submit', 'reset', 'file', 'hidden'].includes(type);
  });
}

function loadFormMemory() {
  try {
    return JSON.parse(localStorage.getItem(FORM_MEMORY_KEY) || '{}');
  } catch {
    return {};
  }
}

function saveFormMemory() {
  const data = loadFormMemory();
  for (const el of formMemoryControls()) {
    data[el.id] = el.type === 'checkbox' ? !!el.checked : el.value;
  }
  try {
    localStorage.setItem(FORM_MEMORY_KEY, JSON.stringify(data));
  } catch {}
}

function applyFormMemory() {
  const data = loadFormMemory();
  for (const el of formMemoryControls()) {
    if (!Object.prototype.hasOwnProperty.call(data, el.id)) continue;
    if (el.type === 'checkbox') {
      el.checked = !!data[el.id];
    } else if (el.tagName === 'SELECT') {
      const value = String(data[el.id] ?? '');
      if ([...el.options].some(opt => opt.value === value)) el.value = value;
    } else {
      el.value = String(data[el.id] ?? '');
    }
  }
}

function installFormMemory() {
  document.addEventListener('input', e => {
    if (e.target.closest(FORM_MEMORY_SELECTOR)) saveFormMemory();
  });
  document.addEventListener('change', e => {
    if (e.target.closest(FORM_MEMORY_SELECTOR)) saveFormMemory();
  });
}

function isPasteableControl(el) {
  if (!el) return false;
  if (el.tagName === 'TEXTAREA') return true;
  if (el.tagName !== 'INPUT') return false;
  return PASTEABLE_INPUT_TYPES.has((el.type || '').toLowerCase());
}

function hidePasteMenu() {
  if (pasteMenu) pasteMenu.hidden = true;
  pasteTarget = null;
}

function createPasteMenu() {
  const menu = document.createElement('div');
  menu.className = 'paste-menu';
  menu.hidden = true;
  menu.innerHTML = '<button type="button">粘贴</button>';
  document.body.appendChild(menu);
  menu.addEventListener('click', async e => {
    e.preventDefault();
    const target = pasteTarget;
    hidePasteMenu();
    if (!isPasteableControl(target)) return;
    try {
      const text = await navigator.clipboard.readText();
      insertTextAtCursor(target, text);
    } catch {
      alert('无法读取剪贴板，请使用 Ctrl+V 粘贴');
    }
  });
  return menu;
}

function showPasteMenu(x, y, target) {
  if (!pasteMenu) pasteMenu = createPasteMenu();
  pasteTarget = target;
  pasteMenu.style.left = Math.min(x, window.innerWidth - 84) + 'px';
  pasteMenu.style.top = Math.min(y, window.innerHeight - 42) + 'px';
  pasteMenu.hidden = false;
}

function insertTextAtCursor(el, text) {
  el.focus();
  if (el.type === 'number') {
    const next = String(text ?? '').trim();
    if (next) el.value = next;
  } else if (typeof el.setRangeText === 'function' && el.selectionStart != null) {
    el.setRangeText(text, el.selectionStart, el.selectionEnd, 'end');
  } else {
    el.value += text;
  }
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
}

function installPasteMenu() {
  document.addEventListener('contextmenu', e => {
    const target = e.target.closest('input, textarea');
    if (!isPasteableControl(target)) return;
    e.preventDefault();
    showPasteMenu(e.clientX, e.clientY, target);
  });
  document.addEventListener('pointerdown', e => {
    if (pasteMenu && !pasteMenu.hidden && !pasteMenu.contains(e.target)) hidePasteMenu();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') hidePasteMenu();
  });
  window.addEventListener('scroll', hidePasteMenu, true);
  window.addEventListener('resize', hidePasteMenu);
}

// ── File handling ────────────────────────────────────────────────────────────
function renderFiles() {
  fileList.innerHTML = '';
  state.files.forEach((f, i) => {
    const chip = document.createElement('div');
    chip.className = 'file-chip';
    chip.innerHTML = `
      <span class="fname" title="${f.path}">${f.name}</span>
      <span class="rm" data-i="${i}">✕</span>`;
    fileList.appendChild(chip);
  });
  btnSubmit.disabled = state.files.length === 0;
}

fileList.addEventListener('click', e => {
  const rm = e.target.closest('.rm');
  if (rm) { state.files.splice(+rm.dataset.i, 1); renderFiles(); }
});

async function pickFiles() {
  try {
    const r = await fetch('/api/pick-files', { method: 'POST' });
    if (!r.ok) { alert('文件选择失败：' + await r.text()); return; }
    const { paths } = await r.json();
    for (const p of paths) {
      const name = p.split(/[\\/]/).pop();
      if (!state.files.find(x => x.path === p))
        state.files.push({ type: 'path', name, size: -1, path: p });
    }
    renderFiles();
  } catch (e) {
    alert('文件选择出错：' + e.message);
  }
}

async function pickFolder() {
  try {
    const r = await fetch('/api/pick-folder', { method: 'POST' });
    if (!r.ok) { alert('文件夹选择失败：' + await r.text()); return; }
    const { paths } = await r.json();
    if (!paths.length) return;
    for (const p of paths) {
      const name = p.split(/[\\/]/).pop();
      if (!state.files.find(x => x.path === p))
        state.files.push({ type: 'path', name, size: -1, path: p });
    }
    renderFiles();
  } catch (e) {
    alert('文件夹选择出错：' + e.message);
  }
}

btnAddFolder.addEventListener('click', pickFolder);

// PyWebView 6.x delivers drop paths via Python DOMEventHandler → evaluate_js
window.__pywebviewDrop = function(paths) {
  for (const p of paths) {
    const name = p.split(/[\\/]/).pop();
    if (!state.files.find(x => x.path === p))
      state.files.push({ type: 'path', name, size: -1, path: p });
  }
  renderFiles();
};

dropZone.addEventListener('click', pickFiles);
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const droppedFiles = Array.from(e.dataTransfer.files || []);
  // pywebview <6: pywebviewFullPath is injected on File objects directly
  const paths = droppedFiles.map(f => f.pywebviewFullPath || f.path).filter(Boolean);
  if (paths.length) {
    for (const p of paths) {
      const name = p.split(/[\\/]/).pop();
      if (!state.files.find(x => x.path === p))
        state.files.push({ type: 'path', name, size: -1, path: p });
    }
    renderFiles();
  } else if (!droppedFiles.length) {
    // No files in the drag payload at all — open native dialog
    pickFiles();
  }
  // pywebview 6+: files present but no path in JS — Python handler calls window.__pywebviewDrop
});

// ── Submit ───────────────────────────────────────────────────────────────────
btnSubmit.addEventListener('click', async () => {
  if (!state.files.length) return;
  btnSubmit.disabled = true;
  btnSubmit.textContent = '处理中…';

  const paths = state.files.map(f => f.path);

  const spec = {
    video_paths: paths,
    asr_backend: $('p-backend').value,
    subtitle_mode: $('p-mode').value,
    asr_context: $('p-asr-context').value.trim(),
    skip_translation: $('p-skip-trans').checked,
    multi_cue_split: $('p-multi-cue').checked,
    show_gender: $('p-gender').checked,
    asr_recovery: $('p-recovery').checked,
    keep_quality_report: $('p-quality-report').checked,
    vad_threshold: parseFloat($('p-vad').value) || 0.35,
    translation_batch_size: parseInt($('p-batch').value) || 100,
    translation_max_workers: parseInt($('p-max-workers').value) || 8,
    keep_temp_files: $('p-keep-temp').checked,
  };
  const outputDir = $('p-output-dir').value.trim();
  if (outputDir) spec.output_dir = outputDir;
  const advancedRaw = ($('p-advanced')?.value || '').trim();
  if (advancedRaw) {
    spec.advanced = {};
    for (const line of advancedRaw.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      const eq = trimmed.indexOf('=');
      if (eq > 0) spec.advanced[trimmed.slice(0, eq).trim()] = trimmed.slice(eq + 1).trim();
    }
  }

  try {
    const r = await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(spec),
    });
    if (!r.ok) throw new Error(await r.text());
    const { ids } = await r.json();
    addLog(`提交 ${ids.length} 个任务：${ids.join(', ')}`, 'stage-start');
    state.files = [];
    renderFiles();
    await fetchAllJobs();
  } catch (e) {
    alert('提交失败：' + e.message);
  }
  btnSubmit.disabled = state.files.length === 0;
  btnSubmit.textContent = '提交任务';
});

// ── Job rendering ─────────────────────────────────────────────────────────────
const STATUS_LABEL = {
  queued: '排队中', asr: 'ASR转写', translating: '翻译中',
  writing: '写入中', done: '完成', failed: '失败', cancelled: '已取消',
};

const STAGE_LABEL = {
  // job status values
  queued:              '排队等待',
  asr:                 'ASR 转写',
  translating:         '翻译中',
  writing:             '写入字幕',
  done:                '完成',
  failed:              '失败',
  cancelled:           '已取消',
  // detailed stage names from SSE events
  audio_prepare:       '音频提取',
  asr_alignment:       'ASR 转写 & 对齐',
  asr_text_transcribe: 'ASR 文本转写',
  alignment:           '强制对齐',
  f0_gender_detection: '性别检测',
  translation_context: '翻译上下文',
  translation:         '翻译中',
  write_output:        '写入字幕',
  model_download:      '模型下载',
};
const PROGRESS_PCT = { queued: 0, asr: 20, translating: 60, writing: 90, done: 100, failed: 100, cancelled: 0 };

function jobTitle(job) {
  if (!job.spec?.video_paths?.length) return job.id;
  const p = job.spec.video_paths[0];
  return p.split(/[\\/]/).pop() || job.id;
}

const CLEARABLE = new Set(['done', 'failed', 'cancelled']);

function renderJobs() {
  const ids = Object.keys(state.jobs);
  emptyState.style.display = ids.length ? 'none' : 'flex';

  // "清空已完成" button visibility
  const hasClearable = ids.some(id => CLEARABLE.has(state.jobs[id].status));
  jobAreaHeader.style.display = hasClearable ? 'flex' : 'none';

  [...jobArea.querySelectorAll('.job-card')].forEach(el => {
    if (!state.jobs[el.dataset.id]) el.remove();
  });

  ids.forEach(id => {
    const job = state.jobs[id];
    let card = jobArea.querySelector(`.job-card[data-id="${id}"]`);
    if (!card) {
      card = document.createElement('div');
      card.className = 'job-card';
      card.dataset.id = id;
      jobArea.insertBefore(card, jobArea.firstChild);
    }
    const translated = job.progress?.translated ?? job.progress?.extra?.translated;
    const expected = job.progress?.expected ?? job.progress?.extra?.expected;
    const translatedRatio = translated != null && expected
      ? Math.min(1, Math.max(0, translated / expected))
      : null;
    let pct = PROGRESS_PCT[job.status] ?? 0;
    if (job.progress?.stage === 'translation' && translatedRatio != null) {
      pct = Math.round(60 + translatedRatio * 30);
    }
    const fillClass = job.status === 'done' ? ' done' : job.status === 'failed' ? ' error' : '';
    const stage = STAGE_LABEL[job.current_stage] ?? STAGE_LABEL[job.status] ?? job.current_stage ?? job.status;
    const progressInfo = translated != null
      ? ` ${translated}/${expected ?? '?'}`
      : '';

    const isDone = job.status === 'done';
    const isRetryable = ['failed','cancelled'].includes(job.status);
    const isCancellable = ['queued','asr','translating','writing'].includes(job.status);
    const retryBtn = isRetryable
      ? `<button class="btn-sm btn-retry" data-retry="${id}" title="从当前缓存阶段重试">↺ 重试</button>`
      : '';

    // Artifacts: SRT files shown prominently; others in collapsible section
    const srtArtifacts = isDone ? job.artifacts.filter(p => /\.srt$/i.test(p)) : [];
    const otherArtifacts = isDone ? job.artifacts.filter(p => !/\.srt$/i.test(p)) : [];

    const srtBtns = srtArtifacts.map(p => {
      const name = p.split(/[\\/]/).pop();
      return `<button class="btn-sm btn-dl" data-dl="${id}" data-file="${name}" title="${name}">⬇ ${name}</button>`;
    }).join('');

    const otherSection = otherArtifacts.length ? `
      <details class="other-files">
        <summary>其他文件 (${otherArtifacts.length})</summary>
        ${otherArtifacts.map(p => {
          const name = p.split(/[\\/]/).pop();
          return `<button class="btn-sm btn-dl btn-dl-other" data-dl="${id}" data-file="${name}" title="${name}">⬇ ${name}</button>`;
        }).join('')}
      </details>` : '';

    const playBtn = isDone
      ? `<button class="btn-sm btn-play" data-play="${id}" title="用系统播放器打开视频">▶ 播放</button>`
      : '';

    const folderPath = isDone
      ? (srtArtifacts[0] || job.artifacts[0] || job.spec?.video_paths?.[0] || '')
      : '';
    const openFolderBtn = folderPath
      ? `<button class="btn-sm btn-folder" data-folder="${folderPath}" title="打开输出文件夹">📂 文件夹</button>`
      : '';

    const errorMsg = job.status === 'failed' && job.error
      ? job.error.length > 80
        ? `<details class="job-error-wrap"><summary class="job-error">${job.error.slice(0, 80)}…</summary><div class="job-error-full">${job.error}</div></details>`
        : `<div class="job-error">${job.error}</div>`
      : '';

    card.innerHTML = `
      <div class="job-header">
        <span class="job-title" title="${jobTitle(job)}">${jobTitle(job)}</span>
        <span class="badge badge-${job.status}">${STATUS_LABEL[job.status] ?? job.status}</span>
      </div>
      <div class="progress-bar"><div class="progress-fill${fillClass}" style="width:${pct}%"></div></div>
      <div class="job-footer">
        <span class="job-stage">${stage}${progressInfo}</span>
        ${playBtn}
        ${openFolderBtn}
        ${srtBtns}
        ${otherSection}
        ${retryBtn}
        ${isCancellable ? `<button class="btn-sm btn-del" data-cancel="${id}">取消</button>` : ''}
        ${CLEARABLE.has(job.status) ? `<button class="btn-sm btn-remove" data-remove="${id}" title="从列表删除">✕ 删除</button>` : ''}
      </div>
      ${errorMsg}`;
  });
}

jobArea.addEventListener('click', async e => {
  const dl = e.target.closest('[data-dl]');
  if (dl) {
    const url = `/api/output/${dl.dataset.dl}/${dl.dataset.file}`;
    const a = document.createElement('a');
    a.href = url; a.download = dl.dataset.file; a.click();
    return;
  }
  const play = e.target.closest('[data-play]');
  if (play) {
    const job = state.jobs[play.dataset.play];
    const videoPath = job?.spec?.video_paths?.[0];
    if (videoPath) {
      try {
        await fetch(`/api/open-video?path=${encodeURIComponent(videoPath)}`);
      } catch {}
    }
    return;
  }
  const folder = e.target.closest('[data-folder]');
  if (folder) {
    try {
      await fetch(`/api/open-folder?path=${encodeURIComponent(folder.dataset.folder)}`);
    } catch {}
    return;
  }
  const retry = e.target.closest('[data-retry]');
  if (retry) {
    const job = state.jobs[retry.dataset.retry];
    if (job?.spec) {
      try {
        const r = await fetch(`/api/jobs/${retry.dataset.retry}/retry`, { method: 'POST' });
        if (r.ok) {
          const retried = await r.json();
          addLog(`重试任务：${retried.id}`, 'stage-start');
          await fetchAllJobs();
        } else {
          alert('重试失败：' + await r.text());
        }
      } catch (e) {
        alert('重试出错：' + e.message);
      }
    }
    return;
  }
  const cancel = e.target.closest('[data-cancel]');
  if (cancel) {
    await fetch(`/api/jobs/${cancel.dataset.cancel}`, { method: 'DELETE' });
    await fetchAllJobs();
    return;
  }
  const remove = e.target.closest('[data-remove]');
  if (remove) {
    await fetch(`/api/jobs/${remove.dataset.remove}`, { method: 'DELETE' });
    await fetchAllJobs();
  }
});

btnClearDone.addEventListener('click', async () => {
  if (!confirm('确定清空所有已完成 / 失败 / 已取消的任务？')) return;
  await fetch('/api/jobs', { method: 'DELETE' });
  await fetchAllJobs();
});

// ── API helpers ───────────────────────────────────────────────────────────────
async function fetchAllJobs() {
  try {
    const r = await fetch('/api/jobs');
    if (!r.ok) return;
    const jobs = await r.json();
    state.jobs = {};
    jobs.forEach(j => state.jobs[j.id] = j);
    renderJobs();
  } catch {}
}

// ── SSE ───────────────────────────────────────────────────────────────────────
function connectSSE() {
  if (state.sse) state.sse.close();
  const es = new EventSource('/api/events');
  state.sse = es;

  es.addEventListener('message', e => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }

    if (ev.type === 'connected') {
      connDot.className = 'conn-dot ok';
      return;
    }

    const phase = ev.phase ?? '';
    const cls = phase === 'start' ? 'stage-start'
               : phase === 'done' ? 'stage-done'
               : phase === 'error' || phase === 'blocked' ? 'stage-error'
               : 'stage-progress';

    const label = ev.stage ? `[${ev.stage}] ${phase}` : JSON.stringify(ev);
    const extra = ev.extra ? ' ' + Object.entries(ev.extra).map(([k,v])=>`${k}=${v}`).join(' ') : '';
    addLog(label + extra, cls);

    if (ev.job_id && (phase === 'done' || phase === 'start' || phase === 'progress')) {
      fetchJob(ev.job_id);
    }
  });

  es.onerror = () => {
    connDot.className = 'conn-dot err';
    setTimeout(() => { connectSSE(); fetchAllJobs(); }, 3000);
  };
}

async function fetchJob(id) {
  try {
    const r = await fetch(`/api/jobs/${id}`);
    if (!r.ok) return;
    const job = await r.json();
    state.jobs[id] = job;
    renderJobs();
  } catch {}
}

// ── Log ───────────────────────────────────────────────────────────────────────
const MAX_LOG = 200;
function addLog(text, cls = '') {
  state.logLines.push({ text, cls });
  if (state.logLines.length > MAX_LOG) state.logLines.shift();
  const line = document.createElement('div');
  line.className = 'log-line' + (cls ? ' ' + cls : '');
  line.textContent = new Date().toLocaleTimeString() + '  ' + text;
  logScroll.appendChild(line);
  while (logScroll.children.length > MAX_LOG) logScroll.removeChild(logScroll.firstChild);
  logScroll.scrollTop = logScroll.scrollHeight;
}

$('log-clear').addEventListener('click', () => {
  state.logLines = [];
  logScroll.innerHTML = '';
});

// ── Settings ──────────────────────────────────────────────────────────────────
const _BACKEND_LABELS = {
  'whisper-ja-anime-v0.3': 'whisper-ja-anime-v0.3（推荐）',
};
const _SUBTITLE_MODE_LABELS = { zh: '中文字幕', bilingual: '中日双语' };

async function loadConfig() {
  try {
    const r = await fetch('/api/config');
    if (!r.ok) return;
    const cfg = await r.json();

    const backendSel = $('p-backend');
    backendSel.innerHTML = '';
    for (const b of (cfg.backends || [])) {
      const opt = document.createElement('option');
      opt.value = b;
      opt.textContent = _BACKEND_LABELS[b] || b;
      backendSel.appendChild(opt);
    }
    if (cfg.defaults?.asr_backend) backendSel.value = cfg.defaults.asr_backend;

    const modeSel = $('p-mode');
    modeSel.innerHTML = '';
    for (const m of (cfg.subtitle_modes || [])) {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = _SUBTITLE_MODE_LABELS[m] || m;
      modeSel.appendChild(opt);
    }
    if (cfg.defaults?.subtitle_mode) modeSel.value = cfg.defaults.subtitle_mode;

    const d = cfg.defaults ?? {};
    if (d.vad_threshold      != null) $('p-vad').value       = d.vad_threshold;
    if (d.translation_batch_size  != null) $('p-batch').value       = d.translation_batch_size;
    if (d.translation_max_workers != null) $('p-max-workers').value = d.translation_max_workers;
    if (d.multi_cue_split    != null) $('p-multi-cue').checked = !!d.multi_cue_split;
    if (d.show_gender        != null) $('p-gender').checked    = !!d.show_gender;
    if (d.asr_recovery       != null) $('p-recovery').checked  = !!d.asr_recovery;
    if (d.skip_translation   != null) $('p-skip-trans').checked = !!d.skip_translation;
    applyFormMemory();
  } catch {}
}

async function loadSettings() {
  try {
    const r = await fetch('/api/settings');
    if (!r.ok) return;
    const s = await r.json();
    $('s-key-preview').textContent = s.api_key_preview
      ? '当前：' + s.api_key_preview
      : '当前：未设置';
    if (s.base_url) $('s-baseurl').value = s.base_url;
    if (s.model) {
      const sel = $('s-model');
      sel.innerHTML = `<option value="${s.model}">${s.model}</option>`;
      sel.disabled = false;
      $('s-model-preview').textContent = '当前：' + s.model;
    }
    $('s-hf-mirror').checked = s.hf_endpoint === 'https://hf-mirror.com';
    const effort = $('s-reasoning-effort');
    if (effort) effort.value = s.llm_reasoning_effort || 'max';
    const apiFormat = $('s-api-format');
    if (apiFormat) apiFormat.value = s.llm_api_format || 'chat';
    const targetLang = $('s-target-lang');
    if (targetLang) targetLang.value = s.target_lang || '简体中文';
    const glossary = $('s-glossary');
    if (glossary) {
      glossary.value = (s.translation_glossary || '')
        .split(',').map(t => t.trim()).filter(Boolean).join('\n');
    }
    applyFormMemory();
  } catch {}
}

$('btn-fetch-models').addEventListener('click', async () => {
  // 前端预校验
  const baseUrl = $('s-baseurl').value.trim();
  if (!baseUrl) {
    alert('请先填写 API Base URL，保存后再获取模型');
    return;
  }
  const keyHint = $('s-key-preview').textContent;
  if (!keyHint || keyHint.includes('未设置')) {
    alert('请先填写并保存 API Key，再获取模型');
    return;
  }

  const btn = $('btn-fetch-models');
  btn.textContent = '获取中…';
  btn.disabled = true;
  try {
    const r = await fetch('/api/models');
    if (r.status === 400) {
      alert('配置不完整：' + await r.text() + '\n请先保存 API Key 和 Base URL');
      return;
    }
    if (r.status === 401 || r.status === 403) {
      alert('API Key 无效或无权限，请检查 Key 是否正确');
      return;
    }
    if (!r.ok) {
      alert('获取失败（' + r.status + '），请检查 Base URL 是否正确：\n' + await r.text());
      return;
    }
    const { models } = await r.json();
    if (!models.length) {
      alert('API 未返回任何模型，请确认 Base URL 和 Key 填写正确');
      return;
    }
    const sel = $('s-model');
    const current = sel.value;
    sel.innerHTML = models.map(m =>
      `<option value="${m}"${m === current ? ' selected' : ''}>${m}</option>`
    ).join('');
    sel.disabled = false;
    // 成功后移除 wrapper tooltip
    const wrap = $('s-model-wrap');
    if (wrap) wrap.removeAttribute('title');
  } catch (e) {
    alert('获取模型出错，请检查网络或 Base URL：\n' + e.message);
  } finally {
    btn.textContent = '获取';
    btn.disabled = false;
  }
});

function showSaveStatus(msg, type) {
  const el = $('save-status');
  el.textContent = msg;
  el.className = 'save-status ' + type;
  setTimeout(() => { el.textContent = ''; el.className = 'save-status'; }, 3000);
}

$('btn-eye').addEventListener('click', () => {
  const inp = $('s-apikey');
  const show = inp.type === 'password';
  inp.type = show ? 'text' : 'password';
  $('btn-eye').textContent = show ? '🙈' : '👁';
});

$('btn-save-settings').addEventListener('click', async () => {
  const apiKey    = $('s-apikey').value.trim();
  const baseUrl   = $('s-baseurl').value.trim();
  const model     = $('s-model').value.trim();
  const hfMirror  = $('s-hf-mirror').checked;
  $('btn-save-settings').disabled = true;
  try {
    const body = {};
    if (apiKey)  body.api_key  = apiKey;
    if (baseUrl) body.base_url = baseUrl;
    if (model)   body.model    = model;
    body.hf_endpoint = hfMirror ? 'https://hf-mirror.com' : '';
    body.llm_reasoning_effort = $('s-reasoning-effort').value || 'max';
    body.llm_api_format = $('s-api-format').value || 'chat';
    body.target_lang = $('s-target-lang').value || '简体中文';
    body.translation_glossary = ($('s-glossary').value || '')
      .split('\n').map(t => t.trim()).filter(Boolean).join(', ');
    const r = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    $('s-apikey').value = '';
    $('btn-eye').textContent = '👁';
    $('s-apikey').type = 'password';
    const mirrorNote = hfMirror ? '（hf-mirror 已开启，重启生效）' : '';
    showSaveStatus('✓ 已保存' + mirrorNote, 'ok');
    await loadSettings();
    saveFormMemory();
  } catch (e) {
    showSaveStatus('✗ 保存失败：' + e.message, 'error');
  }
  $('btn-save-settings').disabled = false;
});

// ── Init ──────────────────────────────────────────────────────────────────────
const ACTIVE_STATUSES = new Set(['queued', 'asr', 'translating', 'writing']);

setInterval(() => {
  const hasActive = Object.values(state.jobs).some(j => ACTIVE_STATUSES.has(j.status));
  if (hasActive) fetchAllJobs();
}, 3000);

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !btnSubmit.disabled) {
    btnSubmit.click();
  }
});

(async () => {
  installFormMemory();
  installPasteMenu();
  await loadConfig();
  await loadSettings();
  applyFormMemory();
  saveFormMemory();
  await fetchAllJobs();
  connectSSE();
})();
