import { state } from './state.js';
import { escHtml, readErrorDetail } from './util.js';
import { jobArea, jobAreaHeader, emptyState, btnClearDone } from './dom.js';
import { addLog } from './log.js';
import { openQcReport } from './qcReport.js';

const STATUS_LABEL = {
  pending: '待开始', queued: '排队中', asr: 'ASR转写', translating: '翻译中',
  writing: '写入中', done: '完成', failed: '失败', cancelled: '已取消',
};

const STAGE_LABEL = {
  queued:              '排队等待',
  asr:                 'ASR 转写',
  translating:         '翻译中',
  writing:             '写入字幕',
  done:                '已完成',
  failed:              '失败',
  cancelled:           '已取消',
  audio_prepare:       '音频提取',
  asr_alignment:       'ASR 转写 & 字幕时间轴',
  audio_chunking:      '音频切分',
  audio_chunk_export:  '导出 ASR 音频块',
  asr_text_transcribe: 'ASR 文本转写',
  subtitle_timing:     '字幕时间轴',
  translation_context: '翻译上下文',
  translation:         '翻译中',
  write_output:        '写入字幕',
  model_download:      '模型下载',
};

const PROGRESS_PCT = { queued: 0, asr: 20, translating: 60, writing: 90, done: 100, failed: 100, cancelled: 0 };
// Rebalanced on 2026-07-31: the five boundary stages that used to fill 3->38%
// no longer run, and chunking now costs one encoder pass instead of five models.
const STAGE_PCT = {
  audio_prepare: 3,
  audio_chunking: 8,
  audio_chunk_export: 14,
  asr_text_transcribe: 20,
  subtitle_timing: 55,
  translation_context: 72,
  translation: 76,
  write_output: 97,
};

const STAGE_ORDER = Object.keys(STAGE_PCT);

// How much of the bar a stage owns: the gap to whatever comes next, minus a
// point so an in-progress stage never reaches its successor's mark.
function stageSpan(stage) {
  const index = STAGE_ORDER.indexOf(stage);
  if (index < 0) return 0;
  const next = index + 1 < STAGE_ORDER.length ? STAGE_PCT[STAGE_ORDER[index + 1]] : 100;
  return Math.max(0, next - STAGE_PCT[stage] - 1);
}

function clampPct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.min(100, Math.max(0, n));
}

export const CLEARABLE = new Set(['done', 'failed', 'cancelled']);

// Our own messages name the missing setting and the panel that fills it in, so
// they are shown whole (wrapped, never ellipsised). Only a payload long enough
// to bury the card - a provider traceback, say - still gets folded away, and
// even then the first line stays visible.
const MAX_INLINE_ERROR_CHARS = 260;

function renderJobError(error) {
  const text = String(error ?? '').trim();
  if (!text) return '';
  if (text.length <= MAX_INLINE_ERROR_CHARS) {
    return `<div class="job-error">${escHtml(text)}</div>`;
  }
  const firstLine = text.split('\n')[0];
  const head = firstLine.length > MAX_INLINE_ERROR_CHARS
    ? `${firstLine.slice(0, MAX_INLINE_ERROR_CHARS)}…`
    : firstLine;
  return `<details class="job-error-wrap">
            <summary class="job-error-summary"><span class="job-error-short">${escHtml(head)}</span></summary>
            <pre class="job-error-full">${escHtml(text)}</pre>
          </details>`;
}

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

  // Re-appending an already-attached card restarts all of its CSS animations
  // (card-in entrance, progress flow/sheen), which showed up as a whole-card
  // blink on every poll. Cards are only (re)inserted when their order actually
  // changes, and innerHTML is only rewritten when the content differs.
  let prevCard = null;
  const placeCard = card => {
    const misplaced =
      card.parentNode !== jobArea ||
      (prevCard
        ? card.previousElementSibling !== prevCard
        : card.previousElementSibling?.classList?.contains('job-card'));
    if (misplaced) {
      const ref = prevCard ? prevCard.nextSibling : jobArea.querySelector('.job-card');
      jobArea.insertBefore(card, ref === card ? card.nextSibling : ref);
    }
    prevCard = card;
  };

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
    const terminalStage = CLEARABLE.has(job.status) ? job.status : null;
    const activeStage = terminalStage || job.progress?.stage || job.current_stage || job.status;
    const translatedRatio = translated != null && expected
      ? Math.min(1, Math.max(0, translated / expected))
      : null;
    const itemRatio = current != null && total
      ? Math.min(1, Math.max(0, current / total))
      : null;
    let pct = STAGE_PCT[activeStage] ?? PROGRESS_PCT[job.status] ?? 0;
    if (job.status === 'done') {
      pct = 100;
    } else {
      // Interpolate within the stage's own share of the bar, read from the
      // table rather than repeated as literals - the old code hardcoded the
      // 43..63 and 76..95 spans, so re-weighting a stage silently desynced them.
      const ratio = activeStage === 'translation' ? translatedRatio : itemRatio;
      if (ratio != null && STAGE_PCT[activeStage] != null) {
        pct = Math.round(STAGE_PCT[activeStage] + ratio * stageSpan(activeStage));
      }
    }
    pct = clampPct(pct);
    const fillClass = job.status === 'done' ? ' done' : job.status === 'failed' ? ' error' : '';
    const stage = STAGE_LABEL[activeStage] ?? STAGE_LABEL[job.status] ?? activeStage;
    const progressInfo = terminalStage
      ? ''
      : translated != null
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
            : '重新运行 ASR 转写与字幕时间轴，复用仍然有效的缓存'
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

    // The report is opt-in, so the button only exists when the run actually
    // wrote one - otherwise it would open a panel that can only apologise.
    const hasQualityReport = isDone && job.artifacts.some(p => /\.quality_report\.md$/i.test(p));
    const qcBtn = hasQualityReport
      ? `<button class="btn-sm btn-qc" data-qc="${escHtml(id)}" title="查看质量报告（切分、布局、复读、交付规格）">📊 质检</button>`
      : '';

    const folderPath = isDone
      ? (srtArtifacts[0] || job.artifacts[0] || job.spec?.video_paths?.[0] || '')
      : '';
    const openFolderBtn = folderPath
      ? `<button class="btn-sm btn-folder" data-folder="${escHtml(folderPath)}" title="打开输出文件夹">📂 文件夹</button>`
      : '';

    const errorMsg = job.status === 'failed' ? renderJobError(job.error) : '';

    const dl = job._download;
    const dlPct = dl ? clampPct(dl.pct ?? 0) : 0;
    let progressSection = `<div class="progress-bar"><div class="progress-fill${fillClass}"></div></div>`;
    if (dl) {
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
        <div class="dl-bar"><div class="dl-bar-fill"></div></div>`;
    }
    const title = jobTitle(job);

    const html = `
      <div class="job-header">
        <span class="job-title" title="${escHtml(title)}">${escHtml(title)}</span>
        <span class="badge badge-${escHtml(job.status)}">${escHtml(STATUS_LABEL[job.status] ?? job.status)}</span>
      </div>
      ${progressSection}
      <div class="job-footer">
        <span class="job-stage">${escHtml(stage)}${escHtml(progressInfo)}</span>
        ${playBtn}
        ${openFolderBtn}
        ${qcBtn}
        ${srtBtns}
        ${otherSection}
        ${retryBtn}
        ${isCancellable ? `<button class="btn-sm btn-del" data-cancel="${escHtml(id)}">取消</button>` : ''}
        ${CLEARABLE.has(job.status) ? `<button class="btn-sm btn-remove" data-remove="${escHtml(id)}" title="从列表删除">✕ 删除</button>` : ''}
      </div>
      ${errorMsg}`;
    if (card._html !== html) {
      card.innerHTML = html;
      card._html = html;
    }
    // Bar widths go through style so a pct-only change never rewrites the DOM
    // (rewriting restarts the flow/sheen animations mid-loop).
    const fill = card.querySelector('.progress-fill');
    if (fill) fill.style.width = `${pct}%`;
    const dlFill = card.querySelector('.dl-bar-fill');
    if (dlFill) dlFill.style.width = `${dlPct}%`;
    // Keep the visual order aligned with the API's FIFO job order.
    placeCard(card);
  });

  pendingFiles.forEach(file => {
    let card = jobArea.querySelector(`.job-card[data-id="${file.pendingId}"]`);
    if (!card) {
      card = document.createElement('div');
      card.className = 'job-card job-card-pending';
      card.dataset.id = file.pendingId;
    }
    const title = file.name || file.path || file.pendingId;
    const html = `
      <div class="job-header">
        <span class="job-title" title="${escHtml(file.path || title)}">${escHtml(title)}</span>
        <span class="badge badge-pending">${STATUS_LABEL.pending}</span>
      </div>
      <div class="progress-bar"><div class="progress-fill pending" style="width:0%"></div></div>
      <div class="job-footer">
        <span class="job-stage">等待点击“开始任务”</span>
        <button class="btn-sm btn-remove" data-remove-pending="${escHtml(file.pendingId)}" title="移出待开始列表">✕ 删除</button>
      </div>`;
    if (card._html !== html) {
      card.innerHTML = html;
      card._html = html;
    }
    placeCard(card);
  });
}

// fetchAllJobs and syncSettings are injected from main.js to avoid circular
// imports. syncSettings pushes the panel's current values to the server before a
// retry, so 重试 honours a setting the user just changed - which is usually why
// they are retrying at all.
export function installJobAreaHandlers(fetchAllJobs, syncSettings = null) {
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
    const qc = e.target.closest('[data-qc]');
    if (qc) {
      const jobId = qc.dataset.qc;
      await openQcReport(jobId, jobTitle(state.jobs[jobId] || { id: jobId }));
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
        const r = await fetch(`/api/open-folder?job_id=${encodeURIComponent(jobId)}&path=${encodeURIComponent(folder.dataset.folder)}`, { method: 'POST' });
        if (!r.ok) alert('打开文件夹失败：' + await readErrorDetail(r));
      } catch (error) {
        alert('打开文件夹失败：' + error.message);
      }
      return;
    }
    const retry = e.target.closest('[data-retry]');
    if (retry) {
      const job = state.jobs[retry.dataset.retry];
      if (job?.spec) {
        try {
          if (syncSettings) await syncSettings();
          const r = await fetch(`/api/jobs/${retry.dataset.retry}/retry`, { method: 'POST' });
          if (r.ok) {
            const retried = await r.json();
            addLog(`重试任务：${retried.id}`, 'stage-start');
            await fetchAllJobs();
          } else {
            alert('重试失败：\n' + await readErrorDetail(r));
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
