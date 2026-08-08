/* views/runs.js — 档案索引（B4 迷你曲线 / B5 勾选对比 / B6 状态计数） */

import { $, $$, api, esc, fmtNum, fmtDur, toast, parseHash, runHash } from '../core/dom.js';
import { ST } from '../theme.js';
import { loadState } from '../core/state.js';
import { cmpSel } from '../core/state.js';

let runsCache = [];
const runFilter = { q: '', st: 'all', sort: 'submitted_at', dir: -1 };
let visibleIds = [];   /* 当前筛选/排序后可见的 run id（供全选与浮动条使用） */

export function getRuns() { return runsCache; }

/* ── B4：迷你曲线 SVG（spark 数据 → polyline，null 断线）── */
function sparkSvg(spark) {
  if (!spark || !spark.length) return '<span class="mono spark-empty">—</span>';
  const vals = spark.filter(p => p != null).map(p => p[1]);
  if (!vals.length) return '<span class="mono spark-empty">—</span>';
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const span = hi - lo || 1;
  const W = 64, H = 22, pad = 2;
  let d = '';
  let pen = false;
  spark.forEach((p, i) => {
    if (p == null) { pen = false; return; }
    const x = pad + (i / (spark.length - 1 || 1)) * (W - pad * 2);
    const y = pad + (1 - (p[1] - lo) / span) * (H - pad * 2);
    d += `${pen ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
    pen = true;
  });
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true"><path d="${d}"/></svg>`;
}

/* ── B5：勾选 → 浮动对比条 ── */
function syncPickBar() {
  const bar = $('#cmp-bar');
  $('#cmp-bar-n').textContent = cmpSel.size;
  bar.classList.toggle('hidden', cmpSel.size < 1);
  const all = $('#pick-all');
  if (all) {
    const selVisible = visibleIds.filter(id => cmpSel.has(id)).length;
    all.checked = visibleIds.length > 0 && selVisible === visibleIds.length;
    all.indeterminate = selVisible > 0 && selVisible < visibleIds.length;
  }
}

function bindPick(tr, r) {
  const cb = tr.querySelector('.run-pick');
  cb.addEventListener('click', e => e.stopPropagation());
  cb.addEventListener('change', () => {
    if (cb.checked) { if (cmpSel.size < 8) cmpSel.add(r.id); else cb.checked = false; }
    else cmpSel.delete(r.id);
    syncPickBar();
    window.dispatchEvent(new CustomEvent('ra:compare-picked', { detail: { ids: [...cmpSel] } }));
  });
  tr.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      if (e.target === cb) return;
      e.preventDefault();
      location.hash = runHash(r.id);
    }
  });
}

export function renderRuns() {
  const q = runFilter.q.toLowerCase();
  let rows = runsCache.filter(r => {
    if (runFilter.st !== 'all' && r.state !== runFilter.st) return false;
    if (!q) return true;
    return [r.run_name, r.method, r.preset, r.id].some(s => String(s || '').toLowerCase().includes(q));
  });
  const dir = runFilter.dir;
  const num = k => r => (r[k] == null ? -Infinity : r[k]);
  const str = k => r => String(r[k] || '').toLowerCase();
  rows.sort((a, b) => {
    let av, bv;
    switch (runFilter.sort) {
      case 'final_avr_loss': case 'min_loss': case 'steps': case 'duration_s': case 'submitted_at':
        av = num(runFilter.sort)(a); bv = num(runFilter.sort)(b); break;
      case 'run_name': case 'method':
        av = str(runFilter.sort)(a); bv = str(runFilter.sort)(b); break;
      default: av = 0; bv = 0;
    }
    if (av === bv) return 0;
    return (av > bv ? 1 : -1) * dir;
  });
  $('#run-count').textContent = runsCache.length;
  $('#run-running').textContent = runsCache.filter(r => r.state === 'running' || r.state === 'queued').length;
  visibleIds = rows.map(r => r.id);
  /* B6：状态计数 */
  const countBy = st => runsCache.filter(r => r.state === st).length;
  $$('#status-chips .chip[data-st]').forEach(ch => {
    const st = ch.dataset.st;
    const n = st === 'all' ? runsCache.length : countBy(st);
    ch.textContent = `${st === 'all' ? '全部' : (ST[st] || { label: st }).label} ${n}`;
  });
  const tb = $('#runs-body');
  tb.innerHTML = '';
  $('#runs-empty').classList.toggle('hidden', rows.length > 0);
  const lastRunId = loadState().lastRun || null;
  rows.forEach((r, i) => {
    const st = ST[r.state] || { label: r.state, cls: 'st-stopped' };
    const srcs = [['jsonl', 'J', '结构化事件流（loss/lr 逐步数据）'], ['tensorboard', 'T', '张量板标量'], ['snapshot', 'S', '配置快照'], ['stdout', 'O', '标准输出']];
    const badges = srcs.map(([k, letter, desc]) =>
      `<span class="src-badge ${r.sources[k] ? 'on' : ''}" title="${desc}${r.sources[k] ? '' : '（该源无数据：训练在首个 step 前终止）'}">${letter}</span>`).join('');
    const tr = document.createElement('tr');
    tr.tabIndex = 0;
    if (r.id === lastRunId) tr.classList.add('active-row');
    tr.innerHTML = `
      <td class="ta-pick"><input type="checkbox" class="run-pick" ${cmpSel.has(r.id) ? 'checked' : ''} aria-label="选择 ${esc(r.run_name)}"></td>
      <td class="num">${String(i + 1).padStart(2, '0')}</td>
      <td><span class="ta-run-name">${esc(r.run_name)}</span><span class="ta-run-id">${esc(r.id)}</span></td>
      <td><span class="st-dot ${st.cls}"><i></i>${st.label}</span></td>
      <td class="num">${esc(r.method || '—')} · ${esc(r.preset || '—')}</td>
      <td class="num">${r.actual_epochs != null ? `${r.actual_epochs}/${r.total_epochs ?? '?'}` : '—'}</td>
      <td class="num">${r.steps ?? '—'}</td>
      <td class="num">${fmtNum(r.final_avr_loss)}</td>
      <td class="num">${fmtNum(r.min_loss)}</td>
      <td class="num">${fmtDur(r.duration_s)}</td>
      <td><span class="src-badges">${badges}</span></td>
      <td class="ta-spark">${sparkSvg(r.spark)}</td>`;
    tr.addEventListener('click', () => { location.hash = runHash(r.id); });
    bindPick(tr, r);
    tb.appendChild(tr);
  });
  syncPickBar();
}

export async function loadRuns() {
  try {
    const d = await api('/api/runs');
    runsCache = d.runs;
    renderRuns();
  } catch (e) { toast(String(e.message || e)); }
}

export async function refreshRuns() {
  try {
    const d = await api('/api/runs');
    runsCache = d.runs;
    renderRuns();
  } catch (e) { /* transient */ }
}

export function focusRunSearch() {
  const el = $('#run-search');
  if (el) { el.focus(); el.select(); }
}

export function initRuns() {
  $('#run-search').addEventListener('input', e => { runFilter.q = e.target.value; renderRuns(); });
  $$('#status-chips .chip').forEach(ch => ch.addEventListener('click', () => {
    $$('#status-chips .chip').forEach(x => x.classList.remove('active'));
    ch.classList.add('active');
    runFilter.st = ch.dataset.st;
    renderRuns();
  }));
  $$('#runs-table th[data-sort]').forEach(th => th.addEventListener('click', () => {
    const k = th.dataset.sort;
    if (runFilter.sort === k) runFilter.dir *= -1;
    else { runFilter.sort = k; runFilter.dir = -1; }
    renderRuns();
  }));
  /* B5：浮动条与全选（全选作用于当前可见行，受 8 项上限约束） */
  $('#pick-all').addEventListener('change', e => {
    if (e.target.checked) {
      for (const id of visibleIds) {
        if (cmpSel.size >= 8) break;
        cmpSel.add(id);
      }
    } else {
      visibleIds.forEach(id => cmpSel.delete(id));
    }
    renderRuns();
    window.dispatchEvent(new CustomEvent('ra:compare-picked', { detail: { ids: [...cmpSel] } }));
  });
  $('#cmp-bar-go').addEventListener('click', () => { location.hash = '#compare'; });
  $('#cmp-bar-clear').addEventListener('click', () => {
    runsCache.forEach(r => cmpSel.delete(r.id));
    renderRuns();
    window.dispatchEvent(new CustomEvent('ra:compare-picked', { detail: { ids: [] } }));
  });
  window.addEventListener('ra:compare-picked', () => {
    if (parseHash().view === 'runs') renderRuns();
  });
}
