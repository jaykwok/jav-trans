import { MAX_LOG } from './state.js';
import { logScroll } from './dom.js';
import { showToast } from './util.js';

const AUTO_SCROLL_THRESHOLD_PX = 24;

function isNearBottom(container) {
  if (!container) return true;
  const remaining = container.scrollHeight - container.scrollTop - container.clientHeight;
  return remaining <= AUTO_SCROLL_THRESHOLD_PX;
}

export function addLog(text, cls = '') {
  const shouldScroll = isNearBottom(logScroll);
  const line = document.createElement('div');
  line.className = 'log-line' + (cls ? ' ' + cls : '');
  line.textContent = new Date().toLocaleTimeString() + '  ' + text;
  logScroll.appendChild(line);
  while (logScroll.children.length > MAX_LOG) logScroll.removeChild(logScroll.firstChild);
  if (shouldScroll) logScroll.scrollTop = logScroll.scrollHeight;
}

export function installLog() {
  document.getElementById('log-clear')?.addEventListener('click', () => {
    logScroll.innerHTML = '';
  });
  document.getElementById('log-copy-all')?.addEventListener('click', async () => {
    const lines = [...logScroll.querySelectorAll('.log-line')]
      .map(el => el.textContent).join('\n');
    try {
      await navigator.clipboard.writeText(lines);
      showToast('日志已复制');
    } catch {
      showToast('复制失败，请手动选择文本');
    }
  });
  logScroll.addEventListener('click', e => {
    const selection = window.getSelection?.()?.toString();
    if (selection && selection.length > 0) return;
    const line = e.target.closest('.log-line');
    if (!line) return;
    line.classList.toggle('expanded');
  });
}
