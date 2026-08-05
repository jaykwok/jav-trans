export const $ = id => document.getElementById(id);

export const escHtml = s => String(s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// FastAPI wraps every rejection as {"detail": ...}. Showing the raw body put
// the JSON punctuation in front of the instruction the user needs to read.
export async function readErrorDetail(response) {
  const raw = await response.text().catch(() => '');
  try {
    const detail = JSON.parse(raw)?.detail;
    if (typeof detail === 'string' && detail.trim()) return detail.trim();
    if (Array.isArray(detail)) {
      const lines = detail.map(item => item?.msg || JSON.stringify(item)).filter(Boolean);
      if (lines.length) return lines.join('\n');
    }
  } catch {}
  return raw.trim() || `HTTP ${response.status}`;
}

export function showToast(msg) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}
