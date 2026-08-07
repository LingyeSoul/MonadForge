/* components/dropdown.js — 自定义下拉（F2）
   设计：触发器 = 状态点 + 名称 + 元信息微标 + ▾；面板 = 纸白 1px ink 边框，
   选中行 = volt 左条 + g0 底；↑↓ 移动 / Enter 确认 / Esc 与外部点击关闭；
   aria: combobox / listbox / option / expanded / selected。 */

import { $, esc } from '../core/dom.js';
import { ST } from '../theme.js';

export function createDropdown({ root, onPick }) {
  let options = [];
  let value = null;
  let sig = '';
  let open = false;
  let cursor = -1;

  root.innerHTML = `
    <button class="mon-dd-trigger" type="button" aria-haspopup="listbox" aria-expanded="false">
      <span class="st-dot"><i></i></span>
      <span class="mdd-name">—</span>
      <span class="mdd-meta mono"></span>
      <span class="mdd-caret mono">▾</span>
    </button>
    <div class="mon-dd-panel hidden" role="listbox" aria-label="运行选择"></div>`;

  const trigger = $('.mon-dd-trigger', root);
  const panel = $('.mon-dd-panel', root);

  function rowHtml(o, i) {
    const st = ST[o.state] || { label: o.state, cls: 'st-stopped' };
    return `<div class="mdd-row ${o.id === value ? 'selected' : ''}" role="option" aria-selected="${o.id === value}"
        data-id="${esc(o.id)}" data-i="${i}" tabindex="-1">
      <span class="st-dot ${st.cls}"><i></i>${st.label}</span>
      <span class="mdd-r-name">${esc(o.name)}</span>
      <span class="mdd-r-meta">${esc(o.meta)}</span>
    </div>`;
  }

  function optionsSig(list) {
    return list.map(o => `${o.id}|${o.state}|${o.name}|${o.meta}`).join('\n');
  }

  function renderPanel() {
    const s = optionsSig(options);
    if (s !== sig) {
      sig = s;
      panel.innerHTML = options.map(rowHtml).join('') || '<div class="mdd-empty mono">NO OPTIONS</div>';
    }
    panel.querySelectorAll('.mdd-row').forEach((row, i) => {
      row.classList.toggle('selected', options[i] && options[i].id === value);
    });
  }

  function setValue(id, opts) {
    value = id;
    sig = ''; /* 强制重渲染选中态 */
    if (opts && opts.options) options = opts.options;
    renderPanel();
    const o = options.find(x => x.id === id);
    if (o) {
      const st = ST[o.state] || { label: o.state, cls: 'st-stopped' };
      trigger.querySelector('.st-dot').className = `st-dot ${st.cls}`;
      $('.mdd-name', trigger).textContent = o.name;
      $('.mdd-meta', trigger).textContent = o.meta;
    } else {
      $('.mdd-name', trigger).textContent = '—';
      $('.mdd-meta', trigger).textContent = '';
    }
    if (open) { cursor = Math.max(0, options.findIndex(x => x.id === id)); moveCursor(cursor, false); }
  }

  function setOptions(list, id) {
    options = list;
    if (id != null && list.some(o => o.id === id)) value = id;
    setValue(value);
  }

  function openPanel() {
    open = true;
    panel.classList.remove('hidden');
    trigger.setAttribute('aria-expanded', 'true');
    renderPanel();
    cursor = Math.max(0, options.findIndex(x => x.id === value));
    moveCursor(cursor, false);
  }

  function closePanel() {
    open = false;
    panel.classList.add('hidden');
    trigger.setAttribute('aria-expanded', 'false');
  }

  function moveCursor(i, scroll = true) {
    cursor = Math.max(0, Math.min(i, options.length - 1));
    panel.querySelectorAll('.mdd-row').forEach((row, idx) => {
      row.classList.toggle('mdd-cursor', idx === cursor);
    });
    const cur = panel.querySelectorAll('.mdd-row')[cursor];
    if (cur && scroll) cur.scrollIntoView({ block: 'nearest' });
  }

  function pick(i) {
    const o = options[i];
    if (!o) return;
    closePanel();
    setValue(o.id);
    onPick(o.id);
  }

  trigger.addEventListener('click', () => { open ? closePanel() : openPanel(); });
  trigger.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open ? closePanel() : openPanel(); }
    if (e.key === 'ArrowDown') { e.preventDefault(); open ? moveCursor(cursor + 1) : openPanel(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); open ? moveCursor(cursor - 1) : openPanel(); }
    if (e.key === 'Escape' && open) { e.preventDefault(); closePanel(); }
  });
  panel.addEventListener('click', e => {
    const row = e.target.closest('.mdd-row');
    if (row) pick(+row.dataset.i);
  });
  panel.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown') { e.preventDefault(); moveCursor(cursor + 1); }
    if (e.key === 'ArrowUp') { e.preventDefault(); moveCursor(cursor - 1); }
    if (e.key === 'Enter') { e.preventDefault(); pick(cursor); }
    if (e.key === 'Escape') { e.preventDefault(); closePanel(); trigger.focus(); }
  });
  document.addEventListener('click', e => {
    if (open && !root.contains(e.target)) closePanel();
  });

  return { setOptions, setValue, close: closePanel };
}
