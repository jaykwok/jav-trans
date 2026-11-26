import { state } from './state.js';
import { btnSubmit } from './dom.js';
import { installFormMemory, applyFormMemory, saveFormMemory } from './formMemory.js';
import { applyPreset, setActivePreset, installPresetChips, installSkipTransLink, updateSkipTransState } from './presets.js';
import { installPasteMenu } from './pasteMenu.js';
import { installLog, addLog } from './log.js';
import { installFiles } from './files.js';
import { installJobAreaHandlers } from './jobsRender.js';
import { fetchAllJobs, startJobPolling } from './jobsApi.js';
import { installGpuState, fetchGpuState } from './gpuState.js';
import { installUpdater } from './updater.js';
import { connectSSE } from './sse.js';
import {
  loadConfig,
  loadSettings,
  installSettingsPanel,
  syncSettingsFromFormForSubmit,
  updateSubtitleModeLabels,
} from './settings.js';

// Install all event listeners before any async work
installFormMemory();
installPasteMenu();
installPresetChips();
installSkipTransLink();
installLog();

window.addEventListener('error', e => {
  const detail = e.error?.message || e.message || '未知脚本错误';
  const where = e.filename ? ` (${e.filename.split('/').pop()}:${e.lineno})` : '';
  addLog(`[界面异常] ${detail}${where}`, 'stage-error');
  console.error('[jav-sub] unhandled error:', e.error || e);
});

window.addEventListener('unhandledrejection', e => {
  const reason = e.reason instanceof Error
    ? e.reason.message
    : String(e.reason ?? '未知 Promise 拒绝');
  addLog(`[界面异步异常] ${reason}`, 'stage-error');
  console.error('[jav-sub] unhandled rejection:', e.reason);
});

window.__javSubServerCrash = message => {
  addLog(`[后台服务异常退出] ${message}`, 'stage-error');
  alert(message);
};

installFiles();
installJobAreaHandlers(fetchAllJobs, syncSettingsFromFormForSubmit);
installSettingsPanel();
installGpuState();
installUpdater();
startJobPolling();

// Load config and settings, then restore form state
await loadConfig();
await loadSettings();
applyFormMemory();
updateSubtitleModeLabels();
if (state.activePreset !== 'custom') applyPreset(state.activePreset);
else setActivePreset('custom');
updateSkipTransState();
saveFormMemory();

await fetchAllJobs();
await fetchGpuState();
connectSSE();

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !btnSubmit.disabled) {
    btnSubmit.click();
  }
});
