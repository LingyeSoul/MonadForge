/* core/shortcuts.js — 全局快捷键（A2）
   键位：g+r / g+m / g+c 视图切换（800ms 序列窗口）、/ 聚焦档案搜索、
   ? 键位面板、Esc 关闭面板。输入框聚焦时不触发。 */

import { $ } from './dom.js';
import { focusRunSearch } from '../views/runs.js';

let pendingG = null;

export function openHelp() {
  $('#help-panel').classList.remove('hidden');
  $('#help-close').focus();
}

export function closeHelp() {
  $('#help-panel').classList.add('hidden');
}

export function initShortcuts() {
  $('#help-close').addEventListener('click', closeHelp);
  $('#help-panel').addEventListener('click', e => {
    if (e.target.id === 'help-panel') closeHelp();
  });

  document.addEventListener('keydown', e => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target;
    const typing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable);
    if (typing) {
      if (e.key === 'Escape' && !$('#help-panel').classList.contains('hidden')) closeHelp();
      return;
    }
    if (e.key === 'Escape') {
      if (!$('#help-panel').classList.contains('hidden')) { closeHelp(); e.preventDefault(); }
      return;
    }
    if (e.key === '?') { e.preventDefault(); toggleHelp(); return; }
    if (e.key === '/') { e.preventDefault(); goRuns(); focusRunSearch(); return; }
    if (e.key === 'g') {
      pendingG = { t: performance.now() };
      setTimeout(() => { pendingG = null; }, 800);
      return;
    }
    if (pendingG && performance.now() - pendingG.t <= 800) {
      pendingG = null;
      if (e.key === 'r') { e.preventDefault(); location.hash = '#runs'; }
      else if (e.key === 'm') { e.preventDefault(); location.hash = '#monitor'; }
      else if (e.key === 'c') { e.preventDefault(); location.hash = '#compare'; }
    }
  });
}

function goRuns() {
  if (!location.hash.startsWith('#runs')) location.hash = '#runs';
}

function toggleHelp() {
  if ($('#help-panel').classList.contains('hidden')) openHelp();
  else closeHelp();
}
