import { state } from './state.js';
import { $ } from './util.js';
import { addLog } from './log.js';
import { fileList, btnAddFolder, btnSubmit, dropZone } from './dom.js';
import { fetchAllJobs } from './jobsApi.js';
import { readTranslationSettingsFromForm, syncSettingsFromFormForSubmit } from './settings.js';

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

export function installFiles() {
  fileList.addEventListener('click', e => {
    const rm = e.target.closest('.rm');
    if (rm) { state.files.splice(+rm.dataset.i, 1); renderFiles(); }
  });

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
}
