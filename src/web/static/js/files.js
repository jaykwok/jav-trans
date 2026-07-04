import { state } from './state.js';
import { $ } from './util.js';
import { addLog } from './log.js';
import { btnAddFolder, btnSubmit, dropZone } from './dom.js';
import { fetchAllJobs } from './jobsApi.js';
import { renderJobs } from './jobsRender.js';
import { readTranslationSettingsFromForm, syncSettingsFromFormForSubmit } from './settings.js';

let nextPendingId = 1;

function addPathsToState(paths) {
  let added = 0;
  for (const p of paths) {
    const name = p.split(/[\\/]/).pop();
    if (!state.files.find(x => x.path === p)) {
      state.files.push({
        type: 'path',
        name,
        size: -1,
        path: p,
        pendingId: `pending-${nextPendingId++}`,
      });
      added += 1;
    }
  }
  renderPendingSelection();
  if (added) addLog(`已添加 ${added} 个待开始任务`, 'stage-progress');
}

function renderPendingSelection() {
  btnSubmit.disabled = state.files.length === 0;
  renderJobs();
}

function readTranslationMaxWorkers() {
  const value = parseInt($('t-translation-max-workers').value, 10);
  if (!Number.isFinite(value)) return 16;
  return Math.min(64, Math.max(1, value));
}

async function pickFiles() {
  try {
    const r = await fetch('/api/pick-files', { method: 'POST' });
    if (!r.ok) { alert('文件选择失败：' + await r.text()); return; }
    const { paths } = await r.json();
    addPathsToState(paths);
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
    addPathsToState(paths);
  } catch (e) {
    alert('文件夹选择出错：' + e.message);
  }
}

export function installFiles() {
  btnAddFolder.addEventListener('click', pickFolder);
  window.addEventListener('pending-files-changed', renderPendingSelection);

  window.__pywebviewDrop = function(paths) {
    addPathsToState(paths);
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
      addPathsToState(paths);
    } else if (!droppedFiles.length) {
      pickFiles();
    }
  });

  btnSubmit.addEventListener('click', async () => {
    if (!state.files.length) return;
    btnSubmit.disabled = true;
    btnSubmit.textContent = '启动中…';

    const pendingEntries = [...state.files];
    const paths = pendingEntries.map(f => f.path);
    const pendingIds = new Set(pendingEntries.map(f => f.pendingId));

    const spec = {
      video_paths:              paths,
      asr_backend:              $('r-backend').value,
      subtitle_mode:            $('r-mode').value,
      skip_translation:         $('r-skip-translation').checked,
      keep_quality_report:      $('t-quality-report').checked,
      translation_max_workers:  readTranslationMaxWorkers(),
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
      addLog(`开始 ${ids.length} 个任务：${ids.join(', ')}`, 'stage-start');
      state.files = state.files.filter(f => !pendingIds.has(f.pendingId));
      renderPendingSelection();
      await fetchAllJobs();
    } catch (e) {
      alert('启动失败：' + e.message);
    }
    btnSubmit.disabled = state.files.length === 0;
    btnSubmit.textContent = '开始任务';
  });
}
