import { state } from './state.js';
import { $, showToast } from './util.js';
import { saveFormMemory, loadFormMemory } from './formMemory.js';

const CUSTOM_PRESET_KEY = 'javtrans.customPreset.v1';

export const TUNING_FIELDS = {
  'r-mode':                    'zh',
  'r-skip-translation':        false,
  't-multi-cue-split':         true,
  't-show-gender':             true,
  't-asr-recovery':            false,
  't-vad-threshold':           '0.35',
  't-translation-batch-size':  '200',
  't-translation-max-workers': '4',
  't-quality-report':          false,
  't-keep-temp':               false,
};

export const PRESETS = {
  standard: { fields: { ...TUNING_FIELDS } },
};

function saveCustomPreset() {
  const fields = {};
  for (const id of Object.keys(TUNING_FIELDS)) {
    const el = $(id);
    if (!el) continue;
    fields[id] = el.type === 'checkbox' ? el.checked : el.value;
  }
  localStorage.setItem(CUSTOM_PRESET_KEY, JSON.stringify(fields));
}

function loadCustomPreset() {
  try { return JSON.parse(localStorage.getItem(CUSTOM_PRESET_KEY) || 'null'); }
  catch { return null; }
}

export function setActivePreset(name) {
  state.activePreset = name;
  document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
  const target = document.querySelector(`.preset-chip[data-preset="${name}"]`);
  if (target) target.classList.add('active');
  const tuningSection = $('section-tuning');
  if (tuningSection) tuningSection.style.display = name === 'custom' ? '' : 'none';
}

export function updateSkipTransState() {
  const skip = $('r-skip-translation')?.checked;
  const panel = $('panel-translation');
  if (panel) panel.classList.toggle('disabled', !!skip);
}

export function applyPreset(name) {
  if (name === 'custom') {
    const saved = loadCustomPreset();
    if (saved) {
      for (const [id, val] of Object.entries(saved)) {
        const el = $(id);
        if (!el) continue;
        if (el.type === 'checkbox') el.checked = !!val;
        else el.value = String(val);
      }
    }
    setActivePreset('custom');
  } else {
    const p = PRESETS[name];
    if (!p) return;
    for (const [id, val] of Object.entries(p.fields)) {
      const el = $(id);
      if (!el) continue;
      if (el.type === 'checkbox') el.checked = !!val;
      else el.value = String(val);
    }
    setActivePreset(name);
  }
  updateSkipTransState();
  saveFormMemory();
}

export function installPresetChips() {
  const chips = document.querySelector('.preset-chips');
  if (chips) {
    chips.addEventListener('click', e => {
      const chip = e.target.closest('.preset-chip[data-preset]');
      if (!chip) return;
      applyPreset(chip.dataset.preset);
    });
  }
  $('btn-save-custom')?.addEventListener('click', () => {
    saveCustomPreset();
    state.activePreset = 'custom';
    saveFormMemory();
    showToast('自定义设置已保存，下次启动将默认选中自定义模式');
  });
}

export function installSkipTransLink() {
  $('r-skip-translation')?.addEventListener('change', updateSkipTransState);
}
