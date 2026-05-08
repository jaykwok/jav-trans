import { state, MAX_LOG } from './state.js';
import { logScroll } from './dom.js';
import { showToast } from './util.js';

export function addLog(text, cls = '') {
  state.logLines.push({ text, cls });
  if (state.logLines.length > MAX_LOG) state.logLines.shift();
  const line = document.createElement('div');
  line.className = 'log-line' + (cls ? ' ' + cls : '');
  line.textContent = new Date().toLocaleTimeString() + '  ' + text;
  logScroll.appendChild(line);
  while (logScroll.children.length > MAX_LOG) logScroll.removeChild(logScroll.firstChild);
  logScroll.scrollTop = logScroll.scrollHeight;
}

export function installLog() {
  document.getElementById('log-clear')?.addEventListener('click', () => {
    state.logLines = [];
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
    const line = e.target.closest('.log-line');
    if (!line) return;
    line.classList.toggle('expanded');
  });
}
