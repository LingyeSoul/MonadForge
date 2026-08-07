/* views/params.js — 参数视图 */

import { $, $$, esc } from '../core/dom.js';

export function renderParams(d) {
  const secs = d.params.sections || [];
  const keyline = d.params.keyline || [];
  $('#sec-params').innerHTML = `
    <div class="sec-head"><span class="sec-num">03</span><span class="sec-title">参数</span><span class="sec-sub">PARAMS · ${secs.length} SECTIONS</span></div>
    ${keyline.length ? `
    <div class="param-keyline">
      ${keyline.map(k => `<div class="pk-item"><span class="pk-k mono">${esc(k.k)}</span><span class="pk-v mono">${esc(String(k.v))}</span></div>`).join('')}
    </div>` : ''}
    <div class="param-tools">
      <label class="search-box" style="max-width:420px">
        <span class="search-ic mono">⌕</span>
        <input type="search" id="param-search" placeholder="搜索参数名" autocomplete="off">
      </label>
      <button class="text-btn" id="param-expand-all">全部展开</button>
      <button class="text-btn" id="param-collapse-all">全部收起</button>
      ${d.params.error ? `<span class="mono parse-err">parse error: ${esc(d.params.error)}</span>` : ''}
    </div>
    <div class="param-sections" id="param-sections"></div>`;

  const box = $('#param-sections');
  const render = (q) => {
    box.innerHTML = secs.map((s, i) => {
      const keys = s.keys.filter(k => !q || k.k.toLowerCase().includes(q));
      if (!keys.length && q) return '';
      return `
      <div class="param-sec" data-i="${i}">
        <div class="param-sec-head" role="button" tabindex="0">
          <span class="ps-num">${String(i + 1).padStart(2, '0')}</span>
          <span class="ps-name">${esc(s.source)}</span>
          <span class="ps-caret">▾</span>
          <span class="ps-count">${keys.length} KEYS</span>
        </div>
        <div class="ps-body">
          <table class="param-table">
            <tbody>${keys.map(k => `<tr><td class="pt-k">${esc(k.k)}</td><td class="pt-v">${esc(JSON.stringify(k.v))}</td></tr>`).join('')}</tbody>
          </table>
        </div>
      </div>`;
    }).join('') || '<div class="empty-state mono">NO MATCH</div>';
    $$('.param-sec .param-sec-head', box).forEach(h => h.addEventListener('click', () => {
      h.parentElement.classList.toggle('collapsed');
    }));
    $$('.param-sec .param-sec-head', box).forEach(h => h.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); h.parentElement.classList.toggle('collapsed'); }
    }));
  };
  render('');
  $('#param-search').addEventListener('input', e => render(e.target.value.toLowerCase()));
  $('#param-expand-all').addEventListener('click', () => $$('.param-sec', box).forEach(s => s.classList.remove('collapsed')));
  $('#param-collapse-all').addEventListener('click', () => $$('.param-sec', box).forEach(s => s.classList.add('collapsed')));
}
