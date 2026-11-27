import { $, escHtml, readErrorDetail } from './util.js';
import { addLog } from './log.js';

let _modelRequirementsRequestSeq = 0;
let _modelRequirementsTimer = null;
let _lastCudaStatus = null;
let _hasMissingModels = false;

const MODEL_REQUIREMENTS_POLL_MS = 15000;

function scheduleModelRequirementsPoll(missingModels) {
  _hasMissingModels = Boolean(missingModels);
  if (_modelRequirementsTimer) {
    clearTimeout(_modelRequirementsTimer);
    _modelRequirementsTimer = null;
  }
  if (!_hasMissingModels) return;
  _modelRequirementsTimer = setTimeout(() => {
    _modelRequirementsTimer = null;
    if (typeof document !== 'undefined' && document.hidden) return;
    refreshModelRequirements({ includeCuda: false });
  }, MODEL_REQUIREMENTS_POLL_MS);
}

export function resumeModelRequirementsPollIfNeeded() {
  if (_hasMissingModels && !_modelRequirementsTimer) {
    refreshModelRequirements({ includeCuda: false });
  }
}

function renderModelRequirements(payload) {
  const notice = $('model-requirements-notice');
  const panel = $('panel-model-requirements');
  if (!notice) return;
  const missing = (payload.required_models || []).filter(item => !item.present);
  // Polls skip the CUDA probe (it spawns a torch import), so keep the last
  // verdict instead of reading its absence as "no problem".
  if (payload.cuda) _lastCudaStatus = payload.cuda;
  const cuda = payload.cuda || _lastCudaStatus || {};
  const cudaProblem = cuda.status && cuda.status !== 'ok';
  scheduleModelRequirementsPoll(missing.length > 0);
  if (!missing.length && !cudaProblem) {
    if (panel) panel.hidden = true;
    notice.textContent = '';
    return;
  }
  if (panel) panel.hidden = false;

  const sections = [];
  if (missing.length) {
    let message;
    if (payload.download_disabled) {
      message = '缺少本地模型文件，且已关闭自动下载，需要先准备本地模型。';
    } else {
      message = `首次使用需要下载 ${missing.length} 个模型；下载完成后会复用本地缓存。`;
    }
    const missingText = missing
      .map(item => item.short_name || item.repo_id || '')
      .filter(Boolean)
      .join('、');
    sections.push(`${escHtml(message)}<br><strong>缺少：</strong>${escHtml(missingText)}`);
  }

  if (cudaProblem) {
    const smi = cuda.nvidia_smi || {};
    const runtime = cuda.torch_cuda_version ? `PyTorch CUDA ${cuda.torch_cuda_version}` : '';
    const driverCuda = smi.cuda_version ? `驱动 CUDA ${smi.cuda_version}` : '';
    const driver = smi.driver_version ? `驱动 ${smi.driver_version}` : '';
    const detail = [runtime, driverCuda, driver].filter(Boolean).join(' · ');
    const message = cuda.message || 'CUDA 环境不可用，请更新 NVIDIA 显卡驱动后重启应用。';
    sections.push(`${escHtml(message)}${detail ? `<br><strong>环境：</strong>${escHtml(detail)}` : ''}`);
  }

  notice.innerHTML = sections.join('<br>');
  notice.hidden = false;
}

export async function refreshModelRequirements({ includeCuda = true } = {}) {
  const notice = $('model-requirements-notice');
  const panel = $('panel-model-requirements');
  if (!notice) return;

  const requestSeq = ++_modelRequirementsRequestSeq;
  const url = includeCuda
    ? '/api/model-requirements'
    : '/api/model-requirements?include_cuda=0';
  try {
    const r = await fetch(url);
    if (requestSeq !== _modelRequirementsRequestSeq) return;
    if (!r.ok) {
      const detail = await readErrorDetail(r);
      addLog(`检查运行环境与模型状态失败（HTTP ${r.status}）：${detail}`, 'stage-error');
      notice.innerHTML = escHtml(`运行环境/模型检测失败（HTTP ${r.status}）：${detail}`);
      notice.hidden = false;
      if (panel) panel.hidden = false;
      return;
    }
    renderModelRequirements(await r.json());
  } catch (error) {
    if (requestSeq === _modelRequirementsRequestSeq) {
      const msg = error?.message || String(error);
      addLog(`检查运行环境与模型状态出错：${msg}`, 'stage-error');
      notice.innerHTML = escHtml(`运行环境/模型检测失败：${msg}`);
      notice.hidden = false;
      if (panel) panel.hidden = false;
    }
  }
}
