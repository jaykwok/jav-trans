/* JAVTrans frontend — no build tools, plain ES2022 */

const $ = id => document.getElementById(id);
const escHtml = s => String(s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  files: [],
  jobs: {},
  logLines: [],
  sse: null,
};

// ── DOM refs ─────────────────────────────────────────────────────────────────
const dropZone      = $('drop-zone');
const fileList      = $('file-list');
const btnAddFolder  = $('btn-add-folder');
const btnSubmit     = $('btn-submit');
const jobArea       = $('job-area');
const jobAreaHeader = $('job-area-header');
const emptyState    = $('empty-state');
const logScroll     = $('log-scroll');
const connDot       = $('conn-dot');
const btnClearDone  = $('btn-clear-done');

// ── Form memory v3 ────────────────────────────────────────────────────────────
const FORM_MEMORY_KEY = 'javtrans.formMemory.v3';
const FORM_MEMORY_EXCLUDED = new Set(['api-key']);
const FORM_MEMORY_SELECTOR = 'input[id], select[id], textarea[id]';
const PASTEABLE_INPUT_TYPES = new Set(['', 'text', 'search', 'url', 'tel', 'email', 'password', 'number']);
let pasteTarget = null;
let pasteMenu = null;

function loadFormMemory() {
  try {
    const raw = JSON.parse(localStorage.getItem(FORM_MEMORY_KEY) || '{}');
    return { controls: raw.controls || {}, details: raw.details || {} };
  } catch {
    return { controls: {}, details: {} };
  }
}

function saveFormMemory() {
  const data = loadFormMemory();
  for (const el of document.querySelectorAll(FORM_MEMORY_SELECTOR)) {
    const type = (el.type || '').toLowerCase();
    if (FORM_MEMORY_EXCLUDED.has(el.id)) continue;
    if (['button', 'submit', 'reset', 'file', 'hidden'].includes(type)) continue;
    data.controls[el.id] = el.type === 'checkbox' ? !!el.checked : el.value;
  }
  for (const det of document.querySelectorAll('details[id]')) {
    data.details[det.id] = det.open;
  }
  data.activePreset = activePreset;
  try {
    localStorage.setItem(FORM_MEMORY_KEY, JSON.stringify(data));
  } catch {}
}

function applyFormMemory() {
  const data = loadFormMemory();
  for (const el of document.querySelectorAll(FORM_MEMORY_SELECTOR)) {
    const type = (el.type || '').toLowerCase();
    if (FORM_MEMORY_EXCLUDED.has(el.id)) continue;
    if (['button', 'submit', 'reset', 'file', 'hidden'].includes(type)) continue;
    if (!Object.hasOwn(data.controls, el.id)) continue;
    if (el.type === 'checkbox') {
      el.checked = !!data.controls[el.id];
    } else if (el.tagName === 'SELECT') {
      const value = String(data.controls[el.id] ?? '');
      if ([...el.options].some(opt => opt.value === value)) el.value = value;
    } else {
      el.value = String(data.controls[el.id] ?? '');
    }
  }
  for (const det of document.querySelectorAll('details[id]')) {
    if (Object.hasOwn(data.details, det.id)) det.open = !!data.details[det.id];
  }
  if (data.activePreset) activePreset = data.activePreset;
  if (!(activePreset in PRESETS) && activePreset !== 'custom') activePreset = 'standard';
}

function installFormMemory() {
  document.addEventListener('input', e => {
    if (e.target.closest(FORM_MEMORY_SELECTOR)) saveFormMemory();
  });
  document.addEventListener('change', e => {
    if (e.target.closest(FORM_MEMORY_SELECTOR)) saveFormMemory();
  });
  document.addEventListener('toggle', e => {
    if (e.target.id && e.target.tagName === 'DETAILS') saveFormMemory();
  }, true);
}

// ── Presets ───────────────────────────────────────────────────────────────────
const TUNING_FIELDS = {
  'r-mode':                    'zh',
  'r-skip-translation':        false,
  't-multi-cue-split':         true,
  't-show-gender':             true,
  't-asr-recovery':            false,
  't-vad-threshold':           '0.35',
  't-translation-batch-size':  '200',
  't-translation-max-workers': '4',
  't-quality-report':          false,
  't-keep-temp':               false,
};

const PRESETS = {
  standard: { fields: { ...TUNING_FIELDS } },
};

const CUSTOM_PRESET_KEY = 'javtrans.customPreset.v1';

function saveCustomPreset() {
  const fields = {};
  for (const id of Object.keys(TUNING_FIELDS)) {
    const el = $(id);
    if (!el) continue;
    fields[id] = el.type === 'checkbox' ? el.checked : el.value;
  }
  localStorage.setItem(CUSTOM_PRESET_KEY, JSON.stringify(fields));
}

function loadCustomPreset() {
  try { return JSON.parse(localStorage.getItem(CUSTOM_PRESET_KEY) || 'null'); }
  catch { return null; }
}

let activePreset = 'standard';

function setActivePreset(name) {
  activePreset = name;
  document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
  const target = document.querySelector(`.preset-chip[data-preset="${name}"]`);
  if (target) target.classList.add('active');
  const tuningSection = $('section-tuning');
  if (tuningSection) tuningSection.style.display = name === 'custom' ? '' : 'none';
}

function applyPreset(name) {
  if (name === 'custom') {
    const saved = loadCustomPreset();
    if (saved) {
      for (const [id, val] of Object.entries(saved)) {
        const el = $(id);
        if (!el) continue;
        if (el.type === 'checkbox') el.checked = !!val;
        else el.value = String(val);
      }
    }
    setActivePreset('custom');
  } else {
    const p = PRESETS[name];
    if (!p) return;
    for (const [id, val] of Object.entries(p.fields)) {
      const el = $(id);
      if (!el) continue;
      if (el.type === 'checkbox') el.checked = !!val;
      else el.value = String(val);
    }
    setActivePreset(name);
  }
  updateSkipTransState();
  saveFormMemory();
}

function installPresetChips() {
  const chips = document.querySelector('.preset-chips');
  if (chips) {
    chips.addEventListener('click', e => {
      const chip = e.target.closest('.preset-chip[data-preset]');
      if (!chip) return;
      applyPreset(chip.dataset.preset);
    });
  }
  $('btn-save-custom')?.addEventListener('click', () => {
    saveCustomPreset();
    // ensure saved as the active mode
    const data = loadFormMemory();
    data.activePreset = 'custom';
    try { localStorage.setItem(FORM_MEMORY_KEY, JSON.stringify(data)); } catch {}
    showToast('自定义设置已保存，下次启动将默认选中自定义模式');
  });
}

// ── Skip-translation linkage ──────────────────────────────────────────────────
function updateSkipTransState() {
  const skip = $('r-skip-translation')?.checked;
  const panel = $('panel-translation');
  if (panel) panel.classList.toggle('disabled', !!skip);
}

function installSkipTransLink() {
  $('r-skip-translation')?.addEventListener('change', updateSkipTransState);
}

// ── Paste menu ────────────────────────────────────────────────────────────────
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
  menu.innerHTML = `
    <button type="button" data-action="copy">复制</button>
    <button type="button" data-action="paste">粘贴</button>
  `;
  document.body.appendChild(menu);
  menu.addEventListener('click', async e => {
    e.preventDefault();
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const target = pasteTarget;
    hidePasteMenu();
    if (btn.dataset.action === 'copy') {
      try {
        await navigator.clipboard.writeText(window.getSelection()?.toString() || '');
      } catch {
        document.execCommand('copy');
      }
    } else if (btn.dataset.action === 'paste') {
      if (!isPasteableControl(target)) return;
      try {
        const text = await navigator.clipboard.readText();
        insertTextAtCursor(target, text);
      } catch {
        alert('无法读取剪贴板，请使用 Ctrl+V 粘贴');
      }
    }
  });
  return menu;
}

function showPasteMenu(x, y, target, { showCopy = false, showPaste = true } = {}) {
  if (!pasteMenu) pasteMenu = createPasteMenu();
  pasteTarget = target;
  pasteMenu.querySelector('[data-action="copy"]').hidden = !showCopy;
  pasteMenu.querySelector('[data-action="paste"]').hidden = !showPaste;
  pasteMenu.style.left = Math.min(x, window.innerWidth - 84) + 'px';
  pasteMenu.style.top = Math.min(y, window.innerHeight - 80) + 'px';
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
    const inputTarget = e.target.closest('input, textarea');
    const isInput = isPasteableControl(inputTarget);
    const sel = window.getSelection();
    const hasSelection = !!(sel && sel.toString().length);
    const inCopyArea = !!(e.target.closest('.log-scroll, .job-error-full, .job-error'));
    const showCopy = hasSelection && (isInput || inCopyArea);
    const showPaste = isInput;
    if (!showCopy && !showPaste) return;
    e.preventDefault();
    showPasteMenu(e.clientX, e.clientY, isInput ? inputTarget : null, { showCopy, showPaste });
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

// ── File handling ─────────────────────────────────────────────────────────────
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
  const paths = droppedFiles.map(f => f.pywebviewFullPath || f.path).filter(Boolean);
  if (paths.length) {
    for (const p of paths) {
      const name = p.split(/[\\/]/).pop();
      if (!state.files.find(x => x.path === p))
        state.files.push({ type: 'path', name, size: -1, path: p });
    }
    renderFiles();
  } else if (!droppedFiles.length) {
    pickFiles();
  }
});

// ── Submit ────────────────────────────────────────────────────────────────────
btnSubmit.addEventListener('click', async () => {
  if (!state.files.length) return;
  btnSubmit.disabled = true;
  btnSubmit.textContent = '处理中…';

  const paths = state.files.map(f => f.path);

  const spec = {
    video_paths:              paths,
    asr_backend:              $('r-backend').value,
    subtitle_mode:            $('r-mode').value,
    asr_context:              $('r-asr-context').value.trim(),
    skip_translation:         $('r-skip-translation').checked,
    multi_cue_split:          $('t-multi-cue-split').checked,
    show_gender:              $('t-show-gender').checked,
    asr_recovery:             $('t-asr-recovery').checked,
    keep_quality_report:      $('t-quality-report').checked,
    vad_threshold:            parseFloat($('t-vad-threshold').value) || 0.35,
    translation_batch_size:   parseInt($('t-translation-batch-size').value) || 200,
    translation_max_workers:  parseInt($('t-translation-max-workers').value) || 4,
    ...readTranslationSettingsFromForm(),
    keep_temp_files:          $('t-keep-temp').checked,
  };
  const outputDir = $('f-output-dir').value.trim();
  if (outputDir) spec.output_dir = outputDir;
  const advancedRaw = ($('t-env-override')?.value || '').trim();
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
    await syncSettingsFromFormForSubmit();
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
  queued:              '排队等待',
  asr:                 'ASR 转写',
  translating:         '翻译中',
  writing:             '写入字幕',
  done:                '完成',
  failed:              '失败',
  cancelled:           '已取消',
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
    const progressInfo = translated != null ? ` ${translated}/${expected ?? '?'}` : '';

    const isDone = job.status === 'done';
    const isRetryable = ['failed', 'cancelled'].includes(job.status);
    const isCancellable = ['queued', 'asr', 'translating', 'writing'].includes(job.status);
    const retryBtn = isRetryable
      ? `<button class="btn-sm btn-retry" data-retry="${id}" title="从当前缓存阶段重试">↺ 重试</button>`
      : '';

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
      ? job.error.length > 100
        ? `<details class="job-error-wrap">
            <summary class="job-error-summary"><span class="job-error-short">${escHtml(job.error.slice(0, 100))}…</span></summary>
            <pre class="job-error-full">${escHtml(job.error)}</pre>
          </details>`
        : `<div class="job-error">${escHtml(job.error)}</div>`
      : '';

    // Model download progress row
    const dl = job._download;
    let progressSection = `<div class="progress-bar"><div class="progress-fill${fillClass}" style="width:${pct}%"></div></div>`;
    if (dl) {
      const dlPct = dl.pct ?? 0;
      const fname = dl.file ? dl.file.split(/[\\/]/).pop().replace(/\.(safetensors|bin|pt|gguf)$/, '') : '模型';
      const downloadedMb = dl.sizeMb ? Math.round(dlPct / 100 * dl.sizeMb) : null;
      const sizeStr = downloadedMb != null && dl.sizeMb ? `${downloadedMb}/${dl.sizeMb}MB` : '';
      const speedStr = dl.speedMb != null ? `${dl.speedMb.toFixed(1)}MB/s` : '';
      const info = [sizeStr, speedStr].filter(Boolean).join(' · ');
      const slowDur = dl.slowSince ? Date.now() - dl.slowSince : 0;
      const mirrorOn = $('mirror-enabled')?.checked;
      const showMirrorBtn = slowDur >= 5000 && !mirrorOn;
      progressSection = `
        <div class="dl-row">
          <span class="dl-label">↓ ${fname}</span>
          <span class="dl-info">${info}</span>
          ${showMirrorBtn ? `<button class="btn-sm btn-enable-mirror" data-enable-mirror="${id}">启用镜像加速</button>` : ''}
        </div>
        <div class="dl-bar"><div class="dl-bar-fill" style="width:${dlPct}%"></div></div>`;
    }

    card.innerHTML = `
      <div class="job-header">
        <span class="job-title" title="${jobTitle(job)}">${jobTitle(job)}</span>
        <span class="badge badge-${job.status}">${STATUS_LABEL[job.status] ?? job.status}</span>
      </div>
      ${progressSection}
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
  const mirror = e.target.closest('[data-enable-mirror]');
  if (mirror) { await enableHfMirror(mirror.dataset.enableMirror || null); return; }

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
      try { await fetch(`/api/open-video?path=${encodeURIComponent(videoPath)}`); } catch {}
    }
    return;
  }
  const folder = e.target.closest('[data-folder]');
  if (folder) {
    try { await fetch(`/api/open-folder?path=${encodeURIComponent(folder.dataset.folder)}`); } catch {}
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
const ACTIVE_STATUSES = new Set(['queued', 'asr', 'translating', 'writing']);

async function fetchJob(id) {
  try {
    const r = await fetch(`/api/jobs/${id}`);
    if (!r.ok) return;
    const job = await r.json();
    const prev = state.jobs[id];
    state.jobs[id] = job;
    if (prev?._download && ACTIVE_STATUSES.has(job.status)) {
      state.jobs[id]._download = prev._download;
    }
    renderJobs();
  } catch {}
}

async function fetchAllJobs() {
  try {
    const r = await fetch('/api/jobs');
    if (!r.ok) return;
    const jobs = await r.json();
    const prev = state.jobs;
    state.jobs = {};
    jobs.forEach(j => {
      state.jobs[j.id] = j;
      if (prev[j.id]?._download && ACTIVE_STATUSES.has(j.status)) {
        state.jobs[j.id]._download = prev[j.id]._download;
      }
    });
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

    // model_download: update card progress, log compactly
    if (ev.stage === 'model_download' && ev.job_id) {
      const job = state.jobs[ev.job_id];
      if (job) {
        if (phase === 'done' || phase === 'error') {
          job._download = null;
        } else {
          const x = ev.extra || {};
          const prev = job._download || {};
          const fileChanged = x.file && x.file !== prev.file;
          const speedMb = x.speed_mb ?? prev.speedMb ?? null;
          let slowSince = fileChanged ? null : (prev.slowSince ?? null);
          if (speedMb != null) {
            if (speedMb < 1.0 && slowSince == null) slowSince = Date.now();
            else if (speedMb >= 1.0) slowSince = null;
          }
          job._download = {
            file:    x.file    || prev.file    || '',
            sizeMb:  x.size_mb ?? prev.sizeMb  ?? null,
            pct:     x.pct     ?? prev.pct     ?? 0,
            speedMb,
            slowSince,
          };
        }
        renderJobs();
      } else {
        fetchJob(ev.job_id);
      }
      const x = ev.extra || {};
      const dlInfo = phase === 'progress' && x.pct != null
        ? ` ${Math.round(x.pct)}%${x.speed_mb != null ? ` ${x.speed_mb.toFixed(1)}MB/s` : ''}`
        : '';
      addLog(`[model_download] ${phase}${dlInfo}`,
        phase === 'done' ? 'stage-done' : phase === 'error' ? 'stage-error' : 'stage-progress');
      return;
    }

    const cls = phase === 'start' ? 'stage-start'
               : phase === 'done' ? 'stage-done'
               : phase === 'error' || phase === 'blocked' ? 'stage-error'
               : 'stage-progress';

    const label = ev.stage ? `[${ev.stage}] ${phase}` : JSON.stringify(ev);
    const extra = ev.extra ? ' ' + Object.entries(ev.extra).map(([k, v]) => `${k}=${v}`).join(' ') : '';
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

$('log-copy-all').addEventListener('click', async () => {
  const lines = [...logScroll.querySelectorAll('.log-line')]
    .map(el => el.textContent).join('\n');
  try {
    await navigator.clipboard.writeText(lines);
    showToast('日志已复制');
  } catch {
    showToast('复制失败，请手动选择文本');
  }
});

logScroll.addEventListener('click', e => {
  const line = e.target.closest('.log-line');
  if (!line) return;
  line.classList.toggle('expanded');
});

// ── Config & Settings ─────────────────────────────────────────────────────────
const _BACKEND_LABELS = {
  'whisper-ja-anime-v0.3': 'whisper-ja-anime-v0.3（推荐）',
  'anime-whisper':          'AnimeWhisper',
  'qwen3-asr-1.7b':         'Qwen3-ASR-1.7B',
  'whisper-ja-1.5b':        'whisper-ja-1.5B',
};
const _SUBTITLE_MODE_LABELS = { zh: '中文字幕', bilingual: '中日双语' };

async function loadConfig() {
  try {
    const r = await fetch('/api/config');
    if (!r.ok) return;
    const cfg = await r.json();

    const backendSel = $('r-backend');
    backendSel.innerHTML = '';
    for (const b of (cfg.backends || [])) {
      const opt = document.createElement('option');
      opt.value = b;
      opt.textContent = _BACKEND_LABELS[b] || b;
      backendSel.appendChild(opt);
    }
    // only apply API default if user has no saved preference
    if (cfg.defaults?.asr_backend && !Object.hasOwn(loadFormMemory().controls, 'r-backend')) {
      backendSel.value = cfg.defaults.asr_backend;
    }

    const modeSel = $('r-mode');
    modeSel.innerHTML = '';
    for (const m of (cfg.subtitle_modes || [])) {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = _SUBTITLE_MODE_LABELS[m] || m;
      modeSel.appendChild(opt);
    }
    if (cfg.defaults?.subtitle_mode) modeSel.value = cfg.defaults.subtitle_mode;

    const d = cfg.defaults ?? {};
    if (d.vad_threshold             != null) $('t-vad-threshold').value           = d.vad_threshold;
    if (d.translation_batch_size    != null) $('t-translation-batch-size').value  = d.translation_batch_size;
    if (d.translation_max_workers   != null) $('t-translation-max-workers').value = d.translation_max_workers;
    if (d.multi_cue_split           != null) $('t-multi-cue-split').checked       = !!d.multi_cue_split;
    if (d.show_gender               != null) $('t-show-gender').checked           = !!d.show_gender;
    if (d.asr_recovery              != null) $('t-asr-recovery').checked          = !!d.asr_recovery;
    if (d.skip_translation          != null) $('r-skip-translation').checked      = !!d.skip_translation;
    applyFormMemory();
    setActivePreset(activePreset);
  } catch {}
}

async function loadSettings() {
  try {
    const r = await fetch('/api/settings');
    if (!r.ok) return;
    const s = await r.json();

    $('api-key-preview').textContent = s.api_key_preview
      ? '当前：' + s.api_key_preview
      : '当前：未设置';
    if (s.base_url) $('api-base-url').value = s.base_url;
    if (s.model) {
      const sel = $('api-model');
      sel.innerHTML = `<option value="${s.model}">${s.model}</option>`;
      sel.disabled = false;
      $('api-model-preview').textContent = '当前：' + s.model;
    }
    $('mirror-enabled').checked = s.hf_endpoint === 'https://hf-mirror.com';

    const effort = $('api-reasoning-effort');
    if (effort) effort.value = s.llm_reasoning_effort || 'max';
    const apiFormat = $('api-format');
    if (apiFormat) apiFormat.value = s.llm_api_format || 'chat';
    const targetLang = $('api-target-lang');
    if (targetLang) targetLang.value = s.target_lang || '简体中文';
    const glossary = $('api-glossary');
    if (glossary) {
      glossary.value = (s.translation_glossary || '')
        .split(',').map(t => t.trim()).filter(Boolean).join('\n');
    }

    // Auto-open panel-translation if unconfigured (only if localStorage has no preference)
    const isConfigured = !!(s.base_url && s.model && s.api_key_preview && !s.api_key_preview.includes('未设置'));
    const saved = loadFormMemory();
    if (!Object.hasOwn(saved.details, 'panel-translation')) {
      const pt = $('panel-translation');
      if (pt) pt.open = !isConfigured;
    }
  } catch {}
}

// ── Settings UI handlers ──────────────────────────────────────────────────────
$('btn-show-key').addEventListener('click', () => {
  const inp = $('api-key');
  const show = inp.type === 'password';
  inp.type = show ? 'text' : 'password';
  $('btn-show-key').textContent = show ? '🙈' : '👁';
});

$('btn-fetch-models').addEventListener('click', async () => {
  const baseUrl     = $('api-base-url').value.trim();
  const apiKeyInput = $('api-key').value.trim();
  const keyPreview  = $('api-key-preview').textContent || '';
  const hasStoredKey = keyPreview && !keyPreview.includes('未设置');

  if (!baseUrl) {
    alert('请先填写 API Base URL');
    return;
  }
  if (!apiKeyInput && !hasStoredKey) {
    alert('请先填写 API Key');
    return;
  }

  const btn = $('btn-fetch-models');
  btn.textContent = '获取中…';
  btn.disabled = true;
  try {
    // 有未保存的 key → 先自动保存，再拉列表
    if (apiKeyInput) {
      try {
        await saveSettingsBody(buildSettingsBodyFromForm({ includeConnection: true }));
      } catch (e) {
        alert('保存 API Key 失败：' + e.message);
        return;
      }
      await loadSettings();
    }

    const r = await fetch('/api/models');
    if (r.status === 400) {
      alert('配置不完整：' + await r.text());
      return;
    }
    if (r.status === 401 || r.status === 403) {
      alert('API Key 无效或无权限，请检查 Key 是否正确');
      return;
    }
    if (!r.ok) {
      alert('获取失败（' + r.status + '），请检查 Base URL：\n' + await r.text());
      return;
    }
    const { models } = await r.json();
    if (!models.length) {
      alert('API 未返回任何模型，请确认 Base URL 和 Key 填写正确');
      return;
    }
    const sel = $('api-model');
    const current = sel.value;
    sel.innerHTML = models.map(m =>
      `<option value="${m}"${m === current ? ' selected' : ''}>${m}</option>`
    ).join('');
    sel.disabled = false;
    const wrap = $('api-model-wrap');
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

function readTranslationSettingsFromForm() {
  return {
    llm_reasoning_effort: $('api-reasoning-effort').value || 'max',
    llm_api_format:       $('api-format').value || 'chat',
    target_lang:          $('api-target-lang').value || '简体中文',
    translation_glossary: ($('api-glossary').value || '')
      .split('\n').map(t => t.trim()).filter(Boolean).join(', '),
  };
}

function buildSettingsBodyFromForm({ includeConnection = false, includeMirror = false } = {}) {
  const body = readTranslationSettingsFromForm();
  if (includeConnection) {
    const apiKey = $('api-key').value.trim();
    const baseUrl = $('api-base-url').value.trim();
    const model = $('api-model').value.trim();
    if (apiKey) body.api_key = apiKey;
    if (baseUrl) body.base_url = baseUrl;
    if (model) body.model = model;
  }
  if (includeMirror) {
    body.hf_endpoint = $('mirror-enabled').checked ? 'https://hf-mirror.com' : '';
  }
  return body;
}

function clearApiKeyInputIfSaved(body) {
  if (!body.api_key) return;
  $('api-key').value = '';
  $('api-key').type = 'password';
  $('btn-show-key').textContent = '👁';
}

async function saveSettingsBody(body) {
  const r = await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  clearApiKeyInputIfSaved(body);
}

async function syncSettingsFromFormForSubmit() {
  const body = buildSettingsBodyFromForm({ includeConnection: true });
  await saveSettingsBody(body);
  saveFormMemory();
}

$('btn-save-api').addEventListener('click', async () => {
  $('btn-save-api').disabled = true;
  try {
    await saveSettingsBody(
      buildSettingsBodyFromForm({
        includeConnection: true,
        includeMirror: true,
      })
    );
    showSaveStatus('✓ 已保存', 'ok');
    await loadSettings();
    saveFormMemory();
  } catch (e) {
    showSaveStatus('✗ 保存失败：' + e.message, 'error');
  }
  $('btn-save-api').disabled = false;
});

// ── hf-mirror ─────────────────────────────────────────────────────────────────
async function _cancelAndRestartDownloadJob(jobId) {
  if (!jobId) return;
  const job = state.jobs[jobId];
  if (!job || !ACTIVE_STATUSES.has(job.status)) return;
  const spec = job.spec;
  await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
  // wait for backend to cancel + clean up partial files
  await new Promise(r => setTimeout(r, 1200));
  // re-submit same spec so download restarts with new endpoint
  try {
    await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(spec),
    });
  } catch {}
  await fetchAllJobs();
}

async function enableHfMirror(jobId = null) {
  try {
    const r = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hf_endpoint: 'https://hf-mirror.com' }),
    });
    if (!r.ok) throw new Error(await r.text());
    const mirrorChk = $('mirror-enabled');
    if (mirrorChk) mirrorChk.checked = true;
    saveFormMemory();

    if (jobId && state.jobs[jobId]?._download) {
      showToast('镜像已启用，正在取消当前下载并重新开始…');
      await _cancelAndRestartDownloadJob(jobId);
    } else {
      showToast('镜像已启用，对下次模型下载生效');
    }
  } catch (e) {
    showToast('启用失败：' + e.message);
  }
}

function installMirrorChangeHandler() {
  const chk = $('mirror-enabled');
  if (!chk) return;
  let _lastMirrorVal = chk.checked;

  chk.addEventListener('change', async () => {
    if (chk.checked === _lastMirrorVal) return;
    _lastMirrorVal = chk.checked;

    const endpoint = chk.checked ? 'https://hf-mirror.com' : '';
    try {
      const r = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hf_endpoint: endpoint }),
      });
      if (!r.ok) throw new Error(await r.text());
      saveFormMemory();
    } catch (e) {
      showToast('保存镜像设置失败：' + e.message);
      return;
    }

    // cancel + restart any active download
    const activeDownload = Object.values(state.jobs)
      .find(j => j._download && ACTIVE_STATUSES.has(j.status));
    if (activeDownload) {
      const label = chk.checked ? '镜像已启用' : '镜像已关闭';
      showToast(`${label}，正在取消当前下载并重新开始…`);
      await _cancelAndRestartDownloadJob(activeDownload.id);
    }
  });
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

// ── Init ──────────────────────────────────────────────────────────────────────
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
  installPresetChips();
  installSkipTransLink();
  await loadConfig();
  await loadSettings();
  applyFormMemory();
  if (activePreset !== 'custom') applyPreset(activePreset);
  else setActivePreset('custom');
  updateSkipTransState();
  saveFormMemory();
  installMirrorChangeHandler(); // after applyFormMemory so initial restore doesn't fire
  await fetchAllJobs();
  connectSSE();
})();
