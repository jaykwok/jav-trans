import { state } from './state.js';

const FORM_MEMORY_KEY = 'jav-trans.formMemory.v3';
// These controls are owned by /api/settings and persisted in .env. Keeping a
// second copy in localStorage used to overwrite the freshly loaded backend
// value at the end of page startup, then write that stale browser value back
// to .env on the next job submission. Form memory is only for per-job UI state.
const FORM_MEMORY_EXCLUDED = new Set([
  'api-key',
  'translation-backend',
  'api-base-url',
  'api-model',
  'api-reasoning-effort',
  'api-target-lang',
  'api-glossary',
  'llamacpp-server-path',
  'proxy-enabled',
  'proxy-protocol',
  'proxy-host',
  'proxy-port',
]);
const FORM_MEMORY_SELECTOR = 'input[id], select[id], textarea[id]';

export function loadFormMemory() {
  try {
    const raw = JSON.parse(localStorage.getItem(FORM_MEMORY_KEY) || '{}');
    return {
      controls: raw.controls || {},
      details: raw.details || {},
      activePreset: raw.activePreset || 'standard',
    };
  } catch {
    return { controls: {}, details: {}, activePreset: 'standard' };
  }
}

export function saveFormMemory() {
  const data = loadFormMemory();
  for (const id of FORM_MEMORY_EXCLUDED) delete data.controls[id];
  for (const el of document.querySelectorAll(FORM_MEMORY_SELECTOR)) {
    const type = (el.type || '').toLowerCase();
    if (FORM_MEMORY_EXCLUDED.has(el.id)) continue;
    if (['button', 'submit', 'reset', 'file', 'hidden'].includes(type)) continue;
    data.controls[el.id] = el.type === 'checkbox' ? !!el.checked : el.value;
  }
  for (const det of document.querySelectorAll('details[id]')) {
    data.details[det.id] = det.open;
  }
  data.activePreset = state.activePreset;
  try {
    localStorage.setItem(FORM_MEMORY_KEY, JSON.stringify(data));
  } catch {}
}

export function applyFormMemory() {
  const data = loadFormMemory();
  for (const el of document.querySelectorAll(FORM_MEMORY_SELECTOR)) {
    const type = (el.type || '').toLowerCase();
    if (FORM_MEMORY_EXCLUDED.has(el.id)) continue;
    if (['button', 'submit', 'reset', 'file', 'hidden'].includes(type)) continue;
    if (!Object.hasOwn(data.controls, el.id)) continue;
    if (el.type === 'checkbox') {
      el.checked = !!data.controls[el.id];
    } else if (el.tagName === 'SELECT') {
      const value = String(data.controls[el.id] ?? '');
      if ([...el.options].some(opt => opt.value === value)) el.value = value;
    } else {
      el.value = String(data.controls[el.id] ?? '');
    }
  }
  for (const det of document.querySelectorAll('details[id]')) {
    if (Object.hasOwn(data.details, det.id)) det.open = !!data.details[det.id];
  }
  if (data.activePreset) state.activePreset = data.activePreset;
  // validate: only 'standard' and 'custom' are known presets
  if (state.activePreset !== 'standard' && state.activePreset !== 'custom') {
    state.activePreset = 'standard';
  }
}

export function installFormMemory() {
  document.addEventListener('input', e => {
    if (e.target.closest(FORM_MEMORY_SELECTOR)) saveFormMemory();
  });
  document.addEventListener('change', e => {
    if (e.target.closest(FORM_MEMORY_SELECTOR)) saveFormMemory();
  });
  document.addEventListener('toggle', e => {
    if (e.target.id && e.target.tagName === 'DETAILS') saveFormMemory();
  }, true);
}
