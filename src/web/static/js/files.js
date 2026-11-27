import { state } from './state.js';
import { $, showToast, readErrorDetail } from './util.js';
import { addLog } from './log.js';
import { btnAddFolder, btnSubmit, dropZone } from './dom.js';
import { fetchAllJobs } from './jobsApi.js';
import { renderJobs } from './jobsRender.js';
import { readJobTranslationSpecFromForm, syncSettingsFromFormForSubmit } from './settings.js';

const VIDEO_EXT_RE = /\.(mp4|mkv|avi|ts|wmv|m2ts)$/i;

let nextPendingId = 1;
let picking = false;

function normalizePathKey(rawPath) {
  const text = String(rawPath || '').trim();
  // Windows drive or UNC paths are case-insensitive; normalize slashes + case.
  if (/^[a-zA-Z]:[\\/]/.test(text) || /^\\\\/.test(text)) {
    return text.replace(/\//g, '\\').toLowerCase();
  }
  return text;
}

function addPathsToState(paths, { filterVideos = false } = {}) {
  const existingKeys = new Set(state.files.map(item => normalizePathKey(item.path)));
  let added = 0;
  let ignoredNonVideo = 0;
  for (const raw of (paths || [])) {
    const p = String(raw || '').trim();
    if (!p) continue;
    if (filterVideos && !VIDEO_EXT_RE.test(p)) {
      ignoredNonVideo += 1;
      continue;
    }
    const key = normalizePathKey(p);
    if (!existingKeys.has(key)) {
      existingKeys.add(key);
      const name = p.split(/[\\/]/).pop();
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
  if (ignoredNonVideo > 0) {
    showToast(
      added > 0
        ? `已忽略 ${ignoredNonVideo} 个非视频文件（仅支持 mp4 / mkv / avi / ts / wmv / m2ts）`
        : '仅支持 mp4 / mkv / avi / ts / wmv / m2ts 视频文件',
    );
  }
}

function renderPendingSelection() {
  btnSubmit.disabled = state.files.length === 0;
  renderJobs();
}

function readTranslationMaxWorkers() {
  const value = parseInt($('t-translation-max-workers')?.value, 10);
  if (!Number.isFinite(value)) return 4;
  return Math.min(64, Math.max(1, value));
}

function setPickerBusy(busy) {
  picking = busy;
  if (btnAddFolder) btnAddFolder.disabled = busy;
  if (dropZone) dropZone.classList.toggle('picking', busy);
}

async function pickFiles() {
  if (picking) return;
  setPickerBusy(true);
  try {
    const r = await fetch('/api/pick-files', { method: 'POST' });
    if (!r.ok) { alert('文件选择失败：' + await readErrorDetail(r)); return; }
    const { paths } = await r.json();
    addPathsToState(paths);
  } catch (e) {
    alert('文件选择出错：' + e.message);
  } finally {
    setPickerBusy(false);
  }
}

async function pickFolder() {
  if (picking) return;
  setPickerBusy(true);
  try {
    const r = await fetch('/api/pick-folder', { method: 'POST' });
    if (!r.ok) { alert('文件夹选择失败：' + await readErrorDetail(r)); return; }
    const { paths, truncated } = await r.json();
    if (!paths.length) return;
    addPathsToState(paths);
    if (truncated) {
      showToast(`文件夹内视频数量较多，已先加载前 ${paths.length} 个`);
    }
  } catch (e) {
    alert('文件夹选择出错：' + e.message);
  } finally {
    setPickerBusy(false);
  }
}

export function installFiles() {
  btnAddFolder.addEventListener('click', pickFolder);
  window.addEventListener('pending-files-changed', renderPendingSelection);

  window.__pywebviewDrop = function(paths) {
    addPathsToState(paths, { filterVideos: true });
  };

  dropZone.addEventListener('click', pickFiles);
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const droppedFiles = Array.from(e.dataTransfer?.files || []);
    const paths = droppedFiles.map(f => f.pywebviewFullPath || f.path).filter(Boolean);
    if (paths.length) {
      addPathsToState(paths, { filterVideos: true });
    } else if (droppedFiles.length) {
      showToast('浏览器安全限制无法读取拖拽文件的本地路径，请点击拖拽区选择视频或点击「添加文件夹」');
    } else {
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

    const outputMode = $('r-mode')?.value || 'zh';
    const spec = {
      video_paths:              paths,
      subtitle_mode:            outputMode === 'bilingual' ? 'bilingual' : 'zh',
      skip_translation:         outputMode === 'ja_only',
      translation_max_workers:  readTranslationMaxWorkers(),
      ...readJobTranslationSpecFromForm(),
    };
    const outputDir = $('f-output-dir').value.trim();
    if (outputDir) spec.output_dir = outputDir;

    try {
      await syncSettingsFromFormForSubmit();
      const r = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(spec),
      });
      if (!r.ok) throw new Error(await readErrorDetail(r));
      const { ids } = await r.json();
      addLog(`开始 ${ids.length} 个任务：${ids.join(', ')}`, 'stage-start');
      state.files = state.files.filter(f => !pendingIds.has(f.pendingId));
      renderPendingSelection();
      await fetchAllJobs();
    } catch (e) {
      alert('启动失败：\n' + e.message);
    }
    btnSubmit.disabled = state.files.length === 0;
    btnSubmit.textContent = '开始任务';
  });
}
