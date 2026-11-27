const FORM_MEMORY_KEY = 'jav-sub.formMemory.v3';
// These controls are owned by /api/settings and persisted in config/config.json. Keeping a
// second copy in localStorage used to overwrite the freshly loaded backend
// value at the end of page startup, then write that stale browser value back
// to config/config.json on the next job submission. Form memory is only for per-job UI state.
const FORM_MEMORY_EXCLUDED = new Set([
  'api-key',
  'translation-backend',
  'api-base-url',
  'api-model',
  'api-reasoning-effort',
  'api-target-lang',
  'api-glossary',
  'proxy-enabled',
  'proxy-protocol',
  'proxy-host',
  'proxy-port',
  'update-auto-check',
]);
const RETIRED_CONTROL_KEYS = ['r-skip-translation'];
const RETIRED_DETAIL_KEYS = ['panel-tuning'];
const FORM_MEMORY_SELECTOR = 'input[id], select[id], textarea[id]';
const SAVE_DEBOUNCE_MS = 200;

let memoryCache = null;
let saveDebounceTimer = null;

function readStorageFresh() {
  try {
    const raw = JSON.parse(localStorage.getItem(FORM_MEMORY_KEY) || '{}');
    return {
      controls: raw.controls || {},
      details: raw.details || {},
    };
  } catch {
    return { controls: {}, details: {} };
  }
}

export function loadFormMemory() {
  if (!memoryCache) memoryCache = readStorageFresh();
  return {
    controls: { ...memoryCache.controls },
    details: { ...memoryCache.details },
  };
}

export function saveFormMemory() {
  if (saveDebounceTimer !== null) {
    clearTimeout(saveDebounceTimer);
    saveDebounceTimer = null;
  }
  const data = memoryCache ? {
    controls: { ...memoryCache.controls },
    details: { ...memoryCache.details },
  } : readStorageFresh();
  for (const id of FORM_MEMORY_EXCLUDED) delete data.controls[id];
  for (const id of RETIRED_CONTROL_KEYS) delete data.controls[id];
  for (const id of RETIRED_DETAIL_KEYS) delete data.details[id];
  for (const el of document.querySelectorAll(FORM_MEMORY_SELECTOR)) {
    const type = (el.type || '').toLowerCase();
    if (FORM_MEMORY_EXCLUDED.has(el.id)) continue;
    if (['button', 'submit', 'reset', 'file', 'hidden'].includes(type)) continue;
    data.controls[el.id] = el.type === 'checkbox' ? !!el.checked : el.value;
  }
  for (const det of document.querySelectorAll('details[id]')) {
    data.details[det.id] = det.open;
  }
  memoryCache = data;
  try {
    localStorage.setItem(FORM_MEMORY_KEY, JSON.stringify(data));
  } catch {}
}

function scheduleSaveFormMemory() {
  if (saveDebounceTimer !== null) clearTimeout(saveDebounceTimer);
  saveDebounceTimer = setTimeout(() => {
    saveDebounceTimer = null;
    saveFormMemory();
  }, SAVE_DEBOUNCE_MS);
}

export function applyFormMemory() {
  memoryCache = readStorageFresh();
  const data = memoryCache;
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
  // Gracefully migrate users who previously toggled the retired "skip translation" switch.
  if (data.controls['r-skip-translation'] === true) {
    const modeEl = document.getElementById('r-mode');
    if (modeEl && [...modeEl.options].some(opt => opt.value === 'ja_only')) {
      modeEl.value = 'ja_only';
    }
  }
  for (const det of document.querySelectorAll('details[id]')) {
    if (Object.hasOwn(data.details, det.id)) det.open = !!data.details[det.id];
  }
}

export function installFormMemory() {
  document.addEventListener('input', e => {
    const el = e.target?.closest?.(FORM_MEMORY_SELECTOR);
    if (!el || FORM_MEMORY_EXCLUDED.has(el.id)) return;
    scheduleSaveFormMemory();
  });
  document.addEventListener('change', e => {
    const el = e.target?.closest?.(FORM_MEMORY_SELECTOR);
    if (!el || FORM_MEMORY_EXCLUDED.has(el.id)) return;
    saveFormMemory();
  });
  document.addEventListener('toggle', e => {
    if (e.target?.id && e.target.tagName === 'DETAILS') saveFormMemory();
  }, true);
  window.addEventListener('pagehide', () => {
    if (saveDebounceTimer !== null) saveFormMemory();
  });
}
