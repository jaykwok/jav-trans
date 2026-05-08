import { state, ACTIVE_STATUSES } from './state.js';
import { $, showToast } from './util.js';
import { saveFormMemory } from './formMemory.js';
import { fetchAllJobs } from './jobsApi.js';

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

export async function enableHfMirror(jobId = null) {
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

export function installMirrorChangeHandler() {
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

    const activeDownload = Object.values(state.jobs)
      .find(j => j._download && ACTIVE_STATUSES.has(j.status));
    if (activeDownload) {
      const label = chk.checked ? '镜像已启用' : '镜像已关闭';
      showToast(`${label}，正在取消当前下载并重新开始…`);
      await _cancelAndRestartDownloadJob(activeDownload.id);
    }
  });
}
