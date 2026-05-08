import { state } from './state.js';
import { escHtml } from './util.js';
import { jobArea, jobAreaHeader, emptyState, btnClearDone } from './dom.js';
import { addLog } from './log.js';

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

export const CLEARABLE = new Set(['done', 'failed', 'cancelled']);

function jobTitle(job) {
  if (!job.spec?.video_paths?.length) return job.id;
  const p = job.spec.video_paths[0];
  return p.split(/[\\/]/).pop() || job.id;
}

export function renderJobs() {
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
      const mirrorOn = document.getElementById('mirror-enabled')?.checked;
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

// fetchAllJobs and enableHfMirror are injected from main.js to avoid circular imports
export function installJobAreaHandlers(fetchAllJobs, enableHfMirror) {
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
}
