/* views/monitor.js — 监控器 */

import { $, $$, api, esc, fmtNum, fmtLr, fmtDur, toast } from '../core/dom.js';
import { ST } from '../theme.js';
import { chartRegistry, mountChart, preserveZoom, zoomTo } from '../core/charts.js';
import { lossChartOption } from '../charts/loss.js';
import { lrChartOption } from '../charts/lr.js';
import { openLightbox } from '../components/lightbox.js';
import { createDropdown } from '../components/dropdown.js';
import { loadState, saveState } from '../core/state.js';

const _savedMon = loadState().monitor || {};
export const monUI = Object.assign(
  { smooth: 50, log: false, bands: true, raw: false, show: { 'loss/current': true, 'loss/average': true, 'loss/epoch_average': true } },
  _savedMon.ui || {}
);
let monRunId = _savedMon.runId || null;
let monTimer = null;
let monData = null;
let monBound = false;
let monMaxGs = 0;
let monDropdown = null;
let monQueued = 0;
let monLastSampleCount = 0;
let monFeedFilter = 'all';

function monitorRunOrder(runs) {
  const rank = { running: 0, queued: 1, error: 2, stopped: 3, done: 4, orphan: 5 };
  return [...runs].sort((a, b) =>
    ((rank[a.state] ?? 9) - (rank[b.state] ?? 9)) || ((b.submitted_at ?? 0) - (a.submitted_at ?? 0)));
}

function toDropdownOptions(runs) {
  return monitorRunOrder(runs).map(r => {
    const st = ST[r.state] || { label: r.state };
    const meta = `${st.label}${r.actual_epochs != null ? ` · E${r.actual_epochs}/${r.total_epochs ?? '?'}` : ''}${r.final_avr_loss != null ? ` · loss ${fmtNum(r.final_avr_loss)}` : ''}`;
    return { id: r.id, state: r.state, name: r.run_name, meta };
  });
}

function initMonDropdown() {
  if (monDropdown) return;
  monDropdown = createDropdown({
    root: $('#mon-dd'),
    onPick: (id) => {
      if (id !== monRunId) {
        monRunId = id;
        saveState({ monitor: { ui: monUI, runId: monRunId } });
        refreshMonitor();
      }
    },
  });
}

function bindMonitorControls() {
  if (monBound) return;
  monBound = true;
  const persist = () => saveState({ monitor: { ui: monUI, runId: monRunId } });
  /* D13：事件流过滤 */
  $$('#mon-feed-filter button').forEach(b => b.addEventListener('click', () => {
    $$('#mon-feed-filter button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    monFeedFilter = b.dataset.f;
    if (monData) renderMonFeed();
  }));
  const smooth = $('#mon-smooth');
  const onSmooth = () => {
    monUI.smooth = +smooth.value;
    $('#mon-smooth-v').textContent = smooth.value;
    persist();
    if (monData) refreshMonChart();
  };
  smooth.addEventListener('input', onSmooth);
  $$('#mon-seg-scale button').forEach(b => b.addEventListener('click', () => {
    $$('#mon-seg-scale button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    monUI.log = b.dataset.scale === '1';
    persist();
    if (monData) refreshMonChart();
  }));
  $('#mon-bands').addEventListener('change', e => { monUI.bands = e.target.checked; persist(); if (monData) refreshMonChart(); });
  $('#mon-raw').addEventListener('change', e => { monUI.raw = e.target.checked; persist(); if (monData) refreshMonChart(); });
  $$('#mon-chip-loss .sc-chip').forEach(ch => ch.addEventListener('click', () => {
    const tag = ch.dataset.tag;
    monUI.show[tag] = !monUI.show[tag];
    ch.classList.toggle('active', monUI.show[tag]);
    ch.classList.toggle('dim', !monUI.show[tag]);
    persist();
    if (monData) refreshMonChart();
  }));
}

function refreshMonChart() {
  if (!monData) return;
  const running = monData.state === 'running' || monData.state === 'queued';
  const { opt } = lossChartOption(monData, monUI, { scan: running });
  const ch = chartRegistry.get('mon-chart-loss');
  if (ch) { preserveZoom(ch, opt); ch.setOption(opt, { notMerge: true }); }
}

export async function loadMonitor() {
  if (monTimer) { clearInterval(monTimer); monTimer = null; }
  initMonDropdown();
  bindMonitorControls();
  let runs;
  try {
    runs = (await api('/api/runs')).runs;
  } catch (e) { toast(String(e.message || e)); return; }
  if (!runs.length) {
    $('#mon-status').textContent = '无训练记录';
    $('#mon-root').classList.add('hidden');
    $('#mon-empty').classList.remove('hidden');
    $('#mon-empty').textContent = 'NO RUNS — 尚无训练记录';
    return;
  }
  if (!monRunId || !runs.some(r => r.id === monRunId)) {
    const live = runs.find(r => r.state === 'running' || r.state === 'queued');
    monRunId = live ? live.id : monitorRunOrder(runs)[0].id;
  }
  monQueued = runs.filter(r => r.state === 'queued').length;
  monDropdown.setOptions(toDropdownOptions(runs), monRunId);
  $('#mon-empty').classList.add('hidden');
  $('#mon-root').classList.remove('hidden');
  await refreshMonitor();
}

function renderMonBar() {
  const k = monData.kpis;
  const running = monData.state === 'running' || monData.state === 'queued';
  const steps = k.steps ?? 0;
  const total = monData.total_steps;
  const pct = total ? Math.min(100, (steps / total) * 100) : 0;
  $('#mon-fill').style.width = pct + '%';
  const ep = k.actual_epochs ?? '—';
  const ips = k.duration_s ? steps / k.duration_s : null;
  $('#mon-left').innerHTML = `step <b>${steps}</b> / ${total ?? '?'} · E <b>${ep}</b> / ${monData.total_epochs ?? '?'}`;
  $('#mon-right').innerHTML =
    `${fmtDur(k.duration_s)}${ips != null && isFinite(ips) ? ` · ${ips.toFixed(3)} it/s` : ''}`;
  $('#mon-progress').textContent = running ? 'LIVE' : '';
  /* D12：进度条每轮刻度（epoch 边界线 + 当前轮段 volt 底衬） */
  const track = $('.mbp-track');
  if (track) {
    const spans = monData.epoch_spans || {};
    const curEp = k.actual_epochs;
    const maxGs = Math.max(monMaxGs, 1);
    let html = '';
    Object.keys(spans).sort((a, b) => a - b).forEach(ep => {
      const [s0, s1] = spans[ep];
      const l = (s0 / maxGs) * 100;
      const w = Math.max(((s1 - s0) / maxGs) * 100, 0.2);
      if (+ep === curEp && total) {
        html += `<span class="mbp-epoch" style="left:${l.toFixed(1)}%;width:${w.toFixed(1)}%"></span>`;
      }
      html += `<span class="mbp-tick" style="left:${l.toFixed(1)}%"></span>`;
    });
    track.innerHTML = `<span class="mbp-fill" id="mon-fill" style="width:${pct}%"></span>${html}`;
  }
  /* D15：队列微标 */
  const q = $('#mon-queued');
  q.classList.toggle('hidden', monQueued <= 0);
  q.innerHTML = `队列 <b>${monQueued}</b>`;
  /* 状态与 LIVE 指示 */
  const st = ST[monData.state] || { label: monData.state, cls: 'st-stopped' };
  const stateEl = $('#mon-state');
  stateEl.className = `st-dot ${st.cls}`;
  $('#mon-state-label').textContent = st.label;
  $('#mon-live').classList.toggle('hidden', !running);
  $('#mon-stage').classList.toggle('live', running);
  $('#mon-status').textContent = running ? `LIVE · ${esc(monData.run_name)}` : `${st.label} · ${esc(monData.run_name)}`;
}

function renderMonKpis() {
  const k = monData.kpis;
  const cur = (monData.series['loss/current'] || []).slice(-1)[0];
  const cells = [
    ['CURRENT LOSS', cur ? fmtNum(cur[1]) : '—', monData.state === 'running' ? 'volt' : ''],
    ['AVG LOSS', fmtNum(k.final_avr_loss), ''],
    ['MIN LOSS', fmtNum(k.min_loss), ''],
    ['EPOCH AVG', fmtNum(k.epoch_avg_last), ''],
    ['LR', fmtLr(k.lr_final), ''],
    ['CKPT/SAMPLE', `${k.ckpt_count} / ${k.sample_count}`, ''],
  ];
  const box = $('#mon-kpis');
  const newSig = cells.map(c => c[1]).join('|');
  const prevSig = box.dataset.v || '';
  box.innerHTML = cells.map(([l, v, cls]) => `
    <div class="kpi-cell"><div class="kpi-label">${l}</div><div class="kpi-value ${cls}">${v}</div></div>`).join('');
  /* E17：数值变化 → 300ms volt 淡出反馈 */
  if (prevSig && prevSig !== newSig) {
    box.querySelectorAll('.kpi-value').forEach(el => {
      el.classList.add('flash');
      setTimeout(() => el.classList.remove('flash'), 350);
    });
  }
  box.dataset.v = newSig;
}

function renderMonFeed() {
  const items = [];
  (monData.ckpts || []).forEach(c => items.push({ ts: c.ts, cls: 'tl-ckpt', text: `ckpt @ step ${c.global_step}` }));
  (monData.logs || []).forEach(l => items.push({ ts: l.ts, cls: l.level === 'ERROR' ? 'tl-err' : 'tl-warn', text: `${l.level} ${l.msg}` }));
  items.sort((a, b) => (b.ts ?? 0) - (a.ts ?? 0));
  const filtered = monFeedFilter === 'warn' ? items.filter(it => it.cls !== 'tl-ckpt') : items;
  const baseTs = monData.kpis && monData.kpis.duration_s != null ? (monData.firstStepTs ?? 0) : 0;
  $('#mon-feed').innerHTML = filtered.slice(0, 30).map(it =>
    `<div class="ev-row tl-row ${it.cls}">
      <span class="tl-ts">+${Math.round(it.ts ?? 0)}s</span>
      <span class="tl-dot"></span>
      <span class="tl-msg">${esc(it.text)}</span>
    </div>`).join('')
    || '<div class="ev-row"><span class="tl-msg">等待事件…</span></div>';
}

function renderMonFilm() {
  const box = $('#mon-film');
  const items = (monData.samples || []).slice();
  const prevCount = monLastSampleCount;
  const newCount = items.length > prevCount ? items.length - prevCount : 0;
  monLastSampleCount = items.length;
  const stickRight = newCount > 0 && prevCount > 0 && box.scrollLeft + box.clientWidth >= box.scrollWidth - 4;
  box.innerHTML = items.map((s, i) => {
    const fn = s.path ? s.path.split('/').pop() : '';
    const flash = newCount > 0 && i >= items.length - newCount ? ' flash' : '';
    const ar = s.w && s.h ? ` style="--ar: ${s.w} / ${s.h}"` : '';
    const size = s.w && s.h ? `<span class="g-size">${s.w}×${s.h}</span>` : '';
    return `<div class="g-item${flash}" tabindex="0" data-i="${i}" role="button" aria-label="sample"${ar}>
      <img src="/api/runs/${encodeURIComponent(monData.id)}/samples/${encodeURIComponent(fn)}" alt="sample" loading="lazy">
      <div class="g-cap"><span>${s.global_step != null ? `step ${s.global_step}` : ''}</span>${size}</div>
    </div>`;
  }).join('') || '<div class="g-cap" style="flex:1;color:#999">无采样</div>';
  if (stickRight) box.scrollLeft = box.scrollWidth;
  $$('.g-item', box).forEach(el => {
    const open = () => openLightbox(items[+el.dataset.i], monData.id);
    el.addEventListener('click', open);
    el.addEventListener('keydown', e => { if (e.key === 'Enter') open(); });
    const img = el.querySelector('img');
    img.addEventListener('load', () => {
      if (img.naturalWidth > 0 && img.naturalHeight > 0) {
        const want = `${img.naturalWidth} / ${img.naturalHeight}`;
        if (el.style.getPropertyValue('--ar') !== want) {
          el.style.setProperty('--ar', want);
          const cap = el.querySelector('.g-cap .g-size');
          if (cap) cap.textContent = `${img.naturalWidth}×${img.naturalHeight}`;
        }
      }
    });
  });
}

function renderMonEpochs() {
  const epochs = monData.epochs || {};
  const spans = monData.epoch_spans || {};
  const tb = $('#mon-epoch-body');
  tb.innerHTML = Object.keys(epochs).sort((a, b) => a - b).map(ep => {
    const s = epochs[ep];
    const sp = spans[ep];
    return `<tr tabindex="0" data-s0="${sp ? sp[0] : ''}" data-s1="${sp ? sp[1] : ''}">
      <td>E${String(ep).padStart(2, '0')}</td>
      <td>${fmtNum(s.min, 5)}</td><td>${fmtNum(s.max, 5)}</td>
      <td>${fmtNum(s.mean, 5)}</td><td>${fmtNum(s.std, 5)}</td>
      <td>${s.n}</td><td>${sp ? `${sp[0]}~${sp[1]}` : '—'}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" style="text-align:left;color:#999">等待数据…</td></tr>';
}

function zoomMonToEp(tr) {
  if (!tr || !tr.dataset.s1 || !monMaxGs) return;
  const charts = [chartRegistry.get('mon-chart-loss'), chartRegistry.get('mon-chart-lr')].filter(Boolean);
  zoomTo(charts, +tr.dataset.s0, +tr.dataset.s1, monMaxGs);
}

function refreshMonitor() {
  if (!monRunId) return Promise.resolve();
  return api(`/api/runs/${encodeURIComponent(monRunId)}`).then(d => {
    const wasRunning = monData && (monData.state === 'running' || monData.state === 'queued');
    monData = d;
    const running = d.state === 'running' || d.state === 'queued';
    /* 先构建图表以取得 maxGs（供进度条刻度定位），再渲染顶栏 */
    const { opt, maxGs } = lossChartOption(d, monUI, { scan: running });
    monMaxGs = maxGs;
    renderMonBar();
    renderMonKpis();
    renderMonFeed();
    renderMonFilm();
    renderMonEpochs();
    const ch = mountChart('mon-chart-loss', opt);
    preserveZoom(ch, opt);
    const tags = d.tags || {};
    if ((tags.lr || []).some(t => (d.series[t] || []).length)) {
      $('#mon-lr-title').textContent = 'LR // STEP';
      const lrOpt = lrChartOption(d, true);
      const lrCh = mountChart('mon-chart-lr', lrOpt);
      preserveZoom(lrCh, lrOpt);
    } else {
      $('#mon-lr-title').textContent = 'NO LR SERIES';
      const lrCh = chartRegistry.get('mon-chart-lr');
      if (lrCh) lrCh.clear();
    }
    /* 每轮统计行点击 → 联动缩放 */
    $$('#mon-epoch-body tr').forEach(tr => {
      tr.addEventListener('click', () => zoomMonToEp(tr));
      tr.addEventListener('keydown', e => { if (e.key === 'Enter') zoomMonToEp(tr); });
    });
    if (monTimer) { clearInterval(monTimer); monTimer = null; }
    if (running) {
      monTimer = setInterval(() => refreshMonitor().catch(() => { }), 2000);
    } else if (wasRunning) {
      toast(`${d.run_name} 训练已结束`);
    }
  }).catch(() => { });
}
