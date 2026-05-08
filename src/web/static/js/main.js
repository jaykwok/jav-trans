import { state } from './state.js';
import { btnSubmit } from './dom.js';
import { installFormMemory, applyFormMemory, saveFormMemory } from './formMemory.js';
import { applyPreset, setActivePreset, installPresetChips, installSkipTransLink, updateSkipTransState } from './presets.js';
import { installPasteMenu } from './pasteMenu.js';
import { installLog } from './log.js';
import { installFiles } from './files.js';
import { installJobAreaHandlers } from './jobsRender.js';
import { fetchAllJobs, startJobPolling } from './jobsApi.js';
import { connectSSE } from './sse.js';
import { loadConfig, loadSettings, installSettingsPanel } from './settings.js';
import { installMirrorChangeHandler, enableHfMirror } from './hfMirror.js';

// Install all event listeners before any async work
installFormMemory();
installPasteMenu();
installPresetChips();
installSkipTransLink();
installLog();
installFiles();
installJobAreaHandlers(fetchAllJobs, enableHfMirror);
installSettingsPanel();
startJobPolling();

// Load config and settings, then restore form state
await loadConfig();
await loadSettings();
applyFormMemory();
if (state.activePreset !== 'custom') applyPreset(state.activePreset);
else setActivePreset('custom');
updateSkipTransState();
saveFormMemory();
// installMirrorChangeHandler must come after applyFormMemory so initial restore doesn't fire
installMirrorChangeHandler();

await fetchAllJobs();
connectSSE();

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !btnSubmit.disabled) {
    btnSubmit.click();
  }
});
