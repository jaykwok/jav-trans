import { $, escHtml } from './util.js';

// The `#api-model` input is the value everything else reads (settings.js
// saves it, loadSettings() pre-fills it) - this module only adds a filter
// dropdown on top of it. The dropdown never introduces free text: every
// commit and every blur is checked against `availableModels`, so the saved
// value can only ever be one of the ids the API actually returned.
let availableModels = [];
let lastValidValue = '';
let highlightIndex = -1;

const input = () => $('api-model');
const list = () => $('api-model-list');

function filteredModels(query) {
  const q = query.trim().toLowerCase();
  if (!q) return availableModels;
  return availableModels.filter(m => m.toLowerCase().includes(q));
}

function setHighlight(index, items) {
  highlightIndex = index;
  items.forEach((item, i) => item.classList.toggle('active', i === index));
  if (index >= 0) items[index].scrollIntoView({ block: 'nearest' });
}

function renderList(models) {
  const listEl = list();
  highlightIndex = -1;
  if (!models.length) {
    listEl.innerHTML = '<li class="model-combobox-empty">无匹配的模型</li>';
    return;
  }
  listEl.innerHTML = models
    .map(m => `<li class="model-combobox-item" data-value="${escHtml(m)}">${escHtml(m)}</li>`)
    .join('');
}

function openList() {
  const inputEl = input();
  if (inputEl.disabled || !availableModels.length) return;
  renderList(filteredModels(inputEl.value));
  list().hidden = false;
}

function closeList() {
  list().hidden = true;
  highlightIndex = -1;
}

function commitValue(value) {
  const inputEl = input();
  inputEl.value = value;
  lastValidValue = value;
  closeList();
}

// Only exact members of `availableModels` survive; anything else snaps back
// to the last confirmed value, which is how "type to filter" stays "pick
// from what was fetched" instead of turning into a free-text field.
function revertIfInvalid() {
  const inputEl = input();
  const value = inputEl.value.trim();
  if (availableModels.includes(value)) {
    lastValidValue = value;
  } else {
    inputEl.value = lastValidValue;
  }
}

export function setModelComboboxOptions(models, { selected } = {}) {
  availableModels = [...models];
  const inputEl = input();
  if (!inputEl) return;
  inputEl.disabled = availableModels.length === 0;
  // A fresh fetch with nothing pre-selected must leave the box empty, not
  // silently commit availableModels[0] - that value then doubles as the
  // filter query the next time the list opens, so the box appeared to have
  // "chosen" a model the user never picked, and browsing it only ever
  // showed that one match.
  const preferred = selected ?? inputEl.value;
  const next = availableModels.includes(preferred) ? preferred : '';
  inputEl.value = next;
  lastValidValue = next;
  inputEl.placeholder = availableModels.length
    ? '点击选择，或输入关键字筛选'
    : '-- 填好 Key 和 URL 后点获取 --';
}

export function installModelCombobox() {
  const inputEl = input();
  const listEl = list();
  if (!inputEl || !listEl) return;

  inputEl.addEventListener('focus', openList);
  inputEl.addEventListener('input', () => {
    renderList(filteredModels(inputEl.value));
    listEl.hidden = false;
  });
  inputEl.addEventListener('keydown', e => {
    if (listEl.hidden && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
      e.preventDefault();
      openList();
      return;
    }
    const items = [...listEl.querySelectorAll('.model-combobox-item')];
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (items.length) setHighlight((highlightIndex + 1) % items.length, items);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (items.length) setHighlight((highlightIndex - 1 + items.length) % items.length, items);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (highlightIndex >= 0 && items[highlightIndex]) {
        commitValue(items[highlightIndex].dataset.value);
      } else if (items.length === 1) {
        commitValue(items[0].dataset.value);
      } else {
        revertIfInvalid();
        closeList();
      }
    } else if (e.key === 'Escape') {
      revertIfInvalid();
      closeList();
    }
  });
  // mousedown, not click: it fires before the input's blur, so the value
  // commits before the blur handler below would otherwise revert it first.
  listEl.addEventListener('mousedown', e => {
    const item = e.target.closest('.model-combobox-item[data-value]');
    if (!item) return;
    e.preventDefault();
    commitValue(item.dataset.value);
  });
  inputEl.addEventListener('blur', () => {
    setTimeout(() => {
      revertIfInvalid();
      closeList();
    }, 0);
  });
  document.addEventListener('pointerdown', e => {
    if (!listEl.hidden && !listEl.contains(e.target) && e.target !== inputEl) closeList();
  });
}
