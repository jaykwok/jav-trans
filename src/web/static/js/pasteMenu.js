const PASTEABLE_INPUT_TYPES = new Set(['', 'text', 'search', 'url', 'tel', 'email', 'password', 'number']);
let pasteTarget = null;
let pasteMenu = null;

function isPasteableControl(el) {
  if (!el) return false;
  if (el.tagName === 'TEXTAREA') return true;
  if (el.tagName !== 'INPUT') return false;
  return PASTEABLE_INPUT_TYPES.has((el.type || '').toLowerCase());
}

function hidePasteMenu() {
  if (pasteMenu) pasteMenu.hidden = true;
  pasteTarget = null;
}

function createPasteMenu() {
  const menu = document.createElement('div');
  menu.className = 'paste-menu';
  menu.hidden = true;
  menu.innerHTML = `
    <button type="button" data-action="copy">复制</button>
    <button type="button" data-action="paste">粘贴</button>
  `;
  document.body.appendChild(menu);
  menu.addEventListener('click', async e => {
    e.preventDefault();
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const target = pasteTarget;
    hidePasteMenu();
    if (btn.dataset.action === 'copy') {
      try {
        await navigator.clipboard.writeText(window.getSelection()?.toString() || '');
      } catch {
        document.execCommand('copy');
      }
    } else if (btn.dataset.action === 'paste') {
      if (!isPasteableControl(target)) return;
      try {
        const text = await navigator.clipboard.readText();
        insertTextAtCursor(target, text);
      } catch {
        alert('无法读取剪贴板，请使用 Ctrl+V 粘贴');
      }
    }
  });
  return menu;
}

function showPasteMenu(x, y, target, { showCopy = false, showPaste = true } = {}) {
  if (!pasteMenu) pasteMenu = createPasteMenu();
  pasteTarget = target;
  pasteMenu.querySelector('[data-action="copy"]').hidden = !showCopy;
  pasteMenu.querySelector('[data-action="paste"]').hidden = !showPaste;
  pasteMenu.style.left = Math.min(x, window.innerWidth - 84) + 'px';
  pasteMenu.style.top = Math.min(y, window.innerHeight - 80) + 'px';
  pasteMenu.hidden = false;
}

function insertTextAtCursor(el, text) {
  el.focus();
  if (el.type === 'number') {
    const next = String(text ?? '').trim();
    if (next) el.value = next;
  } else if (typeof el.setRangeText === 'function' && el.selectionStart != null) {
    el.setRangeText(text, el.selectionStart, el.selectionEnd, 'end');
  } else {
    el.value += text;
  }
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
}

export function installPasteMenu() {
  document.addEventListener('contextmenu', e => {
    const inputTarget = e.target.closest('input, textarea');
    const isInput = isPasteableControl(inputTarget);
    const sel = window.getSelection();
    const hasSelection = !!(sel && sel.toString().length);
    const inCopyArea = !!(e.target.closest('.log-scroll, .job-error-full, .job-error'));
    const showCopy = hasSelection && (isInput || inCopyArea);
    const showPaste = isInput;
    if (!showCopy && !showPaste) return;
    e.preventDefault();
    showPasteMenu(e.clientX, e.clientY, isInput ? inputTarget : null, { showCopy, showPaste });
  });
  document.addEventListener('pointerdown', e => {
    if (pasteMenu && !pasteMenu.hidden && !pasteMenu.contains(e.target)) hidePasteMenu();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') hidePasteMenu();
  });
  window.addEventListener('scroll', e => {
    if (e.target?.id === 'log-scroll') return;
    hidePasteMenu();
  }, true);
  window.addEventListener('resize', hidePasteMenu);
}
