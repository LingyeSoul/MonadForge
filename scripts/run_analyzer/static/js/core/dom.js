/* core/dom.js — 纯工具函数（无设计逻辑） */

export const $ = (sel, root) => (root || document).querySelector(sel);
export const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

export function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

export function fmtNum(v, d = 4) {
  if (v == null || Number.isNaN(v)) return '—';
  if (Math.abs(v) >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 0 });
  if (v !== 0 && Math.abs(v) < 1e-4) return v.toExponential(2);
  return Number(v).toFixed(d);
}

export function fmtLr(v) {
  if (v == null || Number.isNaN(v)) return '—';
  if (v === 0) return '0';
  if (v < 1e-4) return v.toExponential(2);
  return String(Number(v).toPrecision(4));
}

export function fmtDur(s) {
  if (s == null) return '—';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(sec).padStart(2, '0')}s`;
  return `${sec}s`;
}

export function fmtTs(ts) {
  if (ts == null) return '—';
  return new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false });
}

/* '#RRGGBB' → 'rgba(r,g,b,a)' */
export function hexA(hex, alpha) {
  const h = hex.replace('#', '');
  const n = parseInt(h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

export function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(t._tm);
  t._tm = setTimeout(() => t.classList.add('hidden'), 4000);
}

export function parseHash() {
  const h = location.hash || '#runs';
  if (h.startsWith('#run/')) {
    const raw = h.slice(5).split('/')[0];
    try { return { view: 'run', id: decodeURIComponent(raw) }; }
    catch (e) { return { view: 'run', id: raw }; }
  }
  if (h.startsWith('#monitor')) return { view: 'monitor' };
  if (h.startsWith('#compare')) return { view: 'compare' };
  return { view: 'runs' };
}

export function runHash(id) {
  return `#run/${encodeURIComponent(String(id))}`;
}

export async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.status;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* noop */ }
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json();
}
