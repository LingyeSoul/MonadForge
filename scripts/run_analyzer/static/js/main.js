/* main.js — 启动 / 路由 / 全局行为 */

import { $, $$, parseHash } from './core/dom.js';
import { initLightbox } from './components/lightbox.js';
import { initShortcuts } from './core/shortcuts.js';
import { initRuns, loadRuns, refreshRuns } from './views/runs.js';
import { getDetail, loadDetail, stopDetailPolling } from './views/detail.js';
import { loadCompare, initCompare } from './views/compare.js';
import { loadMonitor } from './views/monitor.js';

export const views = { runs: '#view-runs', run: '#view-detail', compare: '#view-compare', monitor: '#view-monitor' };

function route() {
  const r = parseHash();
  if (r.view !== 'run') stopDetailPolling();
  Object.entries(views).forEach(([k, sel]) => $(sel).classList.toggle('hidden', k !== r.view));
  $$('.rail-item').forEach(a => a.classList.toggle('active', a.dataset.view === r.view));
  if (r.view === 'runs') loadRuns();
  if (r.view === 'run') loadDetail(r.id);
  if (r.view === 'compare') loadCompare();
  if (r.view === 'monitor') loadMonitor();
}

/* 索引轨展开（悬停/焦点 240ms） */
function initRail() {
  const rail = $('#rail');
  rail.addEventListener('mouseenter', () => document.body.classList.add('rail-open'));
  rail.addEventListener('mouseleave', () => document.body.classList.remove('rail-open'));
  rail.addEventListener('focusin', () => document.body.classList.add('rail-open'));
  rail.addEventListener('focusout', e => {
    if (!rail.contains(e.relatedTarget)) document.body.classList.remove('rail-open');
  });
}

/* 实时索引轮询（有运行中任务时 5s 刷新）+ 回到前台立即刷新 */
function initPolling() {
  setInterval(() => {
    if (parseHash().view !== 'runs') return;
    refreshRuns();
  }, 5000);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return;
    const r = parseHash();
    if (r.view === 'runs') { loadRuns(); }
    else if (r.view === 'monitor') { loadMonitor(); }
    else if (r.view === 'run') {
      const d = getDetail();
      if (d && (d.state === 'running' || d.state === 'queued')) loadDetail(d.id);
    }
  });
}

function main() {
  initRail();
  initRuns();
  initCompare();
  initLightbox();
  initShortcuts();
  initPolling();
  window.addEventListener('hashchange', route);
  route();
}

main();
