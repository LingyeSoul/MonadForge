/* components/states.js — 空态 / 加载态（E18：坐标十字 + 逐字揭示） */

export function emptyHtml(text) {
  return `<div class="empty-state">
    <svg class="empty-cross" viewBox="0 0 40 40" aria-hidden="true">
      <line x1="3" y1="3" x2="37" y2="37"/>
      <line x1="37" y1="3" x2="3" y2="37"/>
    </svg>
    <div class="mono">${text}</div>
  </div>`;
}

export function loadingHtml() {
  return '<div class="empty-state"><span class="mono" id="loading-type">LOADING</span></div>';
}

/* 字符逐字揭示（~120ms/字，终端输出感；reduced-motion 由 CSS 缩短） */
export function startTyping(el, text, interval = 120) {
  if (!el) return;
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    el.textContent = text;
    return () => {};
  }
  let i = 0;
  el.textContent = '';
  const timer = setInterval(() => {
    i += 1;
    el.textContent = text.slice(0, i);
    if (i >= text.length) clearInterval(timer);
  }, interval);
  return () => clearInterval(timer);
}
