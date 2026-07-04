import { state } from './state.js';
import { escHtml } from './util.js';
import { jobArea, jobAreaHeader, emptyState, btnClearDone } from './dom.js';
import { addLog } from './log.js';

const STATUS_LABEL = {
  pending: '待开始', queued: '排队中', asr: 'ASR转写', translating: '翻译中',
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
  asr_alignment:       'ASR 转写 & 字幕时间轴',
  boundary_cache:      '边界缓存',
  speech_island_scorer:'语音岛检测',
  outer_edge_refiner:  '外边界精修',
  semantic_split_model:'语义切分判断',
  cut_edge_refiner:    '内部切点精修',
  pre_asr_cueqc:       'Pre-ASR CueQC',
  audio_chunk_export:  '导出 ASR 音频块',
  asr_text_transcribe: 'ASR 文本转写',
  subtitle_timing:     '字幕时间轴',
  translation_context: '翻译上下文',
  translation:         '翻译中',
  write_output:        '写入字幕',
  model_download:      '模型下载',
};

const PROGRESS_PCT = { queued: 0, asr: 20, translating: 60, writing: 90, done: 100, failed: 100, cancelled: 0 };
const STAGE_PCT = {
  audio_prepare: 3,
  boundary_cache: 7,
  speech_island_scorer: 11,
  outer_edge_refiner: 16,
  semantic_split_model: 21,
  cut_edge_refiner: 27,
  pre_asr_cueqc: 33,
  audio_chunk_export: 38,
  asr_text_transcribe: 43,
  subtitle_timing: 64,
  translation_context: 72,
  translation: 76,
  write_output: 97,
};

function clampPct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.min(100, Math.max(0, n));
}

export const CLEARABLE = new Set(['done', 'failed', 'cancelled']);

function jobTitle(job) {
  if (!job.spec?.video_paths?.length) return job.id;
  const p = job.spec.video_paths[0];
  return p.split(/[\\/]/).pop() || job.id;
}

export function renderJobs() {
  const ids = Object.keys(state.jobs);
  const pendingFiles = state.files;
  emptyState.style.display = ids.length || pendingFiles.length ? 'none' : 'flex';

  const hasClearable = ids.some(id => CLEARABLE.has(state.jobs[id].status));
  jobAreaHeader.style.display = hasClearable ? 'flex' : 'none';

  const visibleIds = new Set([
    ...ids,
    ...pendingFiles.map(file => file.pendingId),
  ]);
  [...jobArea.querySelectorAll('.job-card')].forEach(el => {
    if (!visibleIds.has(el.dataset.id)) el.remove();
  });

  ids.forEach(id => {
    const job = state.jobs[id];
    let card = jobArea.querySelector(`.job-card[data-id="${id}"]`);
    if (!card) {
      card = document.createElement('div');
      card.className = 'job-card';
      card.dataset.id = id;
    }
    const translated = job.progress?.translated ?? job.progress?.extra?.translated;
    const expected = job.progress?.expected ?? job.progress?.extra?.expected;
    const current = job.progress?.current ?? job.progress?.extra?.current;
    const total = job.progress?.total ?? job.progress?.extra?.total;
    const activeStage = job.progress?.stage || job.current_stage || job.status;
    const translatedRatio = translated != null && expected
      ? Math.min(1, Math.max(0, translated / expected))
      : null;
    const itemRatio = current != null && total
      ? Math.min(1, Math.max(0, current / total))
      : null;
    let pct = STAGE_PCT[activeStage] ?? PROGRESS_PCT[job.status] ?? 0;
    if (job.status === 'done') {
      pct = 100;
    } else if (activeStage === 'translation' && translatedRatio != null) {
      pct = Math.round(76 + translatedRatio * 19);
    } else if (activeStage === 'asr_text_transcribe' && itemRatio != null) {
      pct = Math.round(43 + itemRatio * 20);
    } else if (itemRatio != null && STAGE_PCT[activeStage] != null) {
      pct = Math.round(STAGE_PCT[activeStage] + itemRatio * 4);
    }
    pct = clampPct(pct);
    const fillClass = job.status === 'done' ? ' done' : job.status === 'failed' ? ' error' : '';
    const stage = STAGE_LABEL[activeStage] ?? STAGE_LABEL[job.status] ?? activeStage;
    const progressInfo = translated != null
      ? ` ${translated}/${expected ?? '?'}`
      : current != null ? ` ${current}/${total ?? '?'}` : '';

    const isDone = job.status === 'done';
    const isRetryable = ['failed', 'cancelled'].includes(job.status);
    const isCancellable = ['queued', 'asr', 'translating', 'writing'].includes(job.status);
    const retryStage = job.progress?.stage || job.current_stage || '';
    const translationRetry = ['translation_context', 'translation', 'write_output'].includes(retryStage);
    const retryBtn = isRetryable
      ? `<button class="btn-sm btn-retry" data-retry="${escHtml(id)}" title="${
          translationRetry
            ? '优先复用已完成的 ASR 产物，仅重试翻译/写出'
            : '重新进入 ASR 前置链，并复用仍然有效的边界与 ASR 缓存'
        }">↺ ${translationRetry ? '重试翻译' : '重试'}</button>`
      : '';

    const srtArtifacts = isDone ? job.artifacts.filter(p => /\.srt$/i.test(p)) : [];
    const otherArtifacts = isDone ? job.artifacts.filter(p => !/\.srt$/i.test(p)) : [];

    const srtBtns = srtArtifacts.map(p => {
      const name = p.split(/[\\/]/).pop() || '';
      return `<button class="btn-sm btn-open-artifact" data-open-artifact="${escHtml(id)}" data-file="${escHtml(name)}" title="用系统默认程序打开 ${escHtml(name)}">↗ ${escHtml(name)}</button>`;
    }).join('');

    const otherSection = otherArtifacts.length ? `
      <details class="other-files">
        <summary>其他文件 (${otherArtifacts.length})</summary>
        ${otherArtifacts.map(p => {
          const name = p.split(/[\\/]/).pop() || '';
          return `<button class="btn-sm btn-dl btn-dl-other" data-dl="${escHtml(id)}" data-file="${escHtml(name)}" title="${escHtml(name)}">⬇ ${escHtml(name)}</button>`;
        }).join('')}
      </details>` : '';

    const playBtn = isDone
      ? `<button class="btn-sm btn-play" data-play="${id}" title="用系统播放器打开视频">▶ 播放</button>`
      : '';

    const folderPath = isDone
      ? (srtArtifacts[0] || job.artifacts[0] || job.spec?.video_paths?.[0] || '')
      : '';
    const openFolderBtn = folderPath
      ? `<button class="btn-sm btn-folder" data-folder="${escHtml(folderPath)}" title="打开输出文件夹">📂 文件夹</button>`
      : '';

    const errorMsg = job.status === 'failed' && job.error
      ? job.error.length > 100
        ? `<details class="job-error-wrap">
            <summary class="job-error-summary"><span class="job-error-short">${escHtml(job.error.slice(0, 100))}…</span></summary>
            <pre class="job-error-full">${escHtml(job.error)}</pre>
          </details>`
        : `<div class="job-error">${escHtml(job.error)}</div>`
      : '';

    const dl = job._download;
    let progressSection = `<div class="progress-bar"><div class="progress-fill${fillClass}" style="width:${pct}%"></div></div>`;
    if (dl) {
      const dlPct = clampPct(dl.pct ?? 0);
      const fname = dl.file ? dl.file.split(/[\\/]/).pop().replace(/\.(safetensors|bin|pt|gguf)$/, '') : '模型';
      const downloadedMb = dl.sizeMb ? Math.round(dlPct / 100 * dl.sizeMb) : null;
      const sizeStr = downloadedMb != null && dl.sizeMb ? `${downloadedMb}/${dl.sizeMb}MB` : '';
      const speedStr = dl.speedMb != null ? `${dl.speedMb.toFixed(1)}MB/s` : '';
      const info = [sizeStr, speedStr].filter(Boolean).join(' · ');
      progressSection = `
        <div class="dl-row">
          <span class="dl-label">↓ ${escHtml(fname)}</span>
          <span class="dl-info">${escHtml(info)}</span>
        </div>
        <div class="dl-bar"><div class="dl-bar-fill" style="width:${dlPct}%"></div></div>`;
    }
    const title = jobTitle(job);

    card.innerHTML = `
      <div class="job-header">
        <span class="job-title" title="${escHtml(title)}">${escHtml(title)}</span>
        <span class="badge badge-${escHtml(job.status)}">${escHtml(STATUS_LABEL[job.status] ?? job.status)}</span>
      </div>
      ${progressSection}
      <div class="job-footer">
        <span class="job-stage">${escHtml(stage)}${escHtml(progressInfo)}</span>
        ${playBtn}
        ${openFolderBtn}
        ${srtBtns}
        ${otherSection}
        ${retryBtn}
        ${isCancellable ? `<button class="btn-sm btn-del" data-cancel="${escHtml(id)}">取消</button>` : ''}
        ${CLEARABLE.has(job.status) ? `<button class="btn-sm btn-remove" data-remove="${escHtml(id)}" title="从列表删除">✕ 删除</button>` : ''}
      </div>
      ${errorMsg}`;
    // Keep the visual order aligned with the API's FIFO job order.
    jobArea.appendChild(card);
  });

  pendingFiles.forEach(file => {
    let card = jobArea.querySelector(`.job-card[data-id="${file.pendingId}"]`);
    if (!card) {
      card = document.createElement('div');
      card.className = 'job-card job-card-pending';
      card.dataset.id = file.pendingId;
    }
    const title = file.name || file.path || file.pendingId;
    card.innerHTML = `
      <div class="job-header">
        <span class="job-title" title="${escHtml(file.path || title)}">${escHtml(title)}</span>
        <span class="badge badge-pending">${STATUS_LABEL.pending}</span>
      </div>
      <div class="progress-bar"><div class="progress-fill pending" style="width:0%"></div></div>
      <div class="job-footer">
        <span class="job-stage">等待点击“开始任务”</span>
        <button class="btn-sm btn-remove" data-remove-pending="${escHtml(file.pendingId)}" title="移出待开始列表">✕ 删除</button>
      </div>`;
    jobArea.appendChild(card);
  });
}

// fetchAllJobs is injected from main.js to avoid circular imports
export function installJobAreaHandlers(fetchAllJobs) {
  jobArea.addEventListener('click', async e => {
    const pending = e.target.closest('[data-remove-pending]');
    if (pending) {
      state.files = state.files.filter(file => file.pendingId !== pending.dataset.removePending);
      window.dispatchEvent(new Event('pending-files-changed'));
      return;
    }
    const openArtifact = e.target.closest('[data-open-artifact]');
    if (openArtifact) {
      try {
        const r = await fetch(
          `/api/open-artifact?job_id=${encodeURIComponent(openArtifact.dataset.openArtifact)}&path=${encodeURIComponent(openArtifact.dataset.file)}`,
          { method: 'POST' },
        );
        if (!r.ok) alert('打开字幕失败：' + await r.text());
      } catch (error) {
        alert('打开字幕失败：' + error.message);
      }
      return;
    }
    const dl = e.target.closest('[data-dl]');
    if (dl) {
      const url = `/api/output/${encodeURIComponent(dl.dataset.dl)}/${encodeURIComponent(dl.dataset.file)}`;
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
          await fetch(`/api/open-video?job_id=${encodeURIComponent(play.dataset.play)}&path=${encodeURIComponent(videoPath)}`, { method: 'POST' });
        } catch {}
      }
      return;
    }
    const folder = e.target.closest('[data-folder]');
    if (folder) {
      const card = folder.closest('.job-card');
      const jobId = card?.dataset?.id || '';
      try {
        await fetch(`/api/open-folder?job_id=${encodeURIComponent(jobId)}&path=${encodeURIComponent(folder.dataset.folder)}`, { method: 'POST' });
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
}
