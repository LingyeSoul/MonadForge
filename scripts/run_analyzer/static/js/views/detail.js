/* views/detail.js — 详情编排：头部 / 章节导航 / 实时轮询 */

import { $, $$, api, esc, fmtTs, fmtNum, fmtDur, fmtLr } from '../core/dom.js';
import { ST } from '../theme.js';
import { saveState } from '../core/state.js';
import { loadingHtml, emptyHtml, startTyping } from '../components/states.js';
import { arcHtml } from '../components/kpi.js';
import { renderCurves } from './curves.js';
import { renderParams } from './params.js';
import { renderSamples } from './samples.js';
import { renderEvents } from './events.js';

let detail = null;
let detailTimer = null;
let detailObserver = null;

export function getDetail() { return detail; }

export function stopDetailPolling() {
  if (detailTimer) clearInterval(detailTimer);
  detailTimer = null;
}

function renderDetailHead() {
  const st = ST[detail.state] || { label: detail.state, cls: 'st-stopped' };
  const ov = detail.kpis;
  const root = $('#detail-root');
  root.innerHTML = `
    <div class="detail-top">
      <a class="back-link" href="#runs">← 返回档案</a>
    </div>
    <div class="detail-head">
      <div class="dh-index">R${detail.id.slice(-4)}</div>
      <div>
        <h1 class="dh-title">${esc(detail.run_name)}</h1>
        <div class="dh-meta">${esc(detail.id)} · submitted ${fmtTs(detail.submitted_at)}</div>
      </div>
      <div class="dh-badges">
        <div class="badge-row">
          <span class="badge">${esc(detail.method || '—')}</span>
          <span class="badge">PRESET ${esc(detail.preset || '—')}</span>
          <span class="badge ${detail.state === 'running' ? 'volt' : 'gray'}">${st.label}</span>
          ${detail.attempt_count > 1 ? `<span class="badge gray">已恢复 ${detail.attempt_count - 1} 次</span>` : ''}
        </div>
        <div class="badge-row">
          ${['jsonl', 'tensorboard', 'snapshot', 'stdout'].filter(k => detail.sources[k]).map(k => `<span class="badge gray">${k.toUpperCase()}</span>`).join('')}
        </div>
      </div>
    </div>
    <nav class="sec-nav" aria-label="章节">
      <a href="#sec-overview" class="active"><span class="sn-num">01</span>总览</a>
      <a href="#sec-curves"><span class="sn-num">02</span>曲线</a>
      <a href="#sec-params"><span class="sn-num">03</span>参数</a>
      <a href="#sec-samples"><span class="sn-num">04</span>采样</a>
      <a href="#sec-events"><span class="sn-num">05</span>事件</a>
    </nav>
    <div class="sec-block" id="sec-overview"></div>
    <div class="sec-block" id="sec-curves"></div>
    <div class="sec-block" id="sec-params"></div>
    <div class="sec-block" id="sec-samples"></div>
    <div class="sec-block" id="sec-events"></div>`;
  setupSecNav();
  renderOverview();
  renderCurves(detail);
  renderParams(detail);
  renderSamples(detail);
  renderEvents(detail);
}

function setupSecNav() {
  if (detailObserver) detailObserver.disconnect();
  const links = $$('.sec-nav a');
  const targets = links.map(a => $(a.getAttribute('href')));
  detailObserver = new IntersectionObserver(entries => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        const i = targets.indexOf(en.target);
        links.forEach(a => a.classList.toggle('active', a === links[i]));
      }
    });
  }, { rootMargin: '-40% 0px -55% 0px' });
  targets.forEach(t => detailObserver.observe(t));
  links.forEach(a => a.addEventListener('click', e => {
    e.preventDefault();
    $(a.getAttribute('href')).scrollIntoView({ behavior: 'smooth' });
  }));
}

/* ── 总览 ── */
function renderOverview() {
  const ov = detail.kpis;
  const running = detail.state === 'running' || detail.state === 'queued';
  const err = detail.state === 'error';
  const stageCls = `stage ${running ? 'live' : ''}`;
  const cells = [
    ['FINAL AVR LOSS', fmtNum(ov.final_avr_loss), running ? 'volt' : err ? 'volt' : '', `${fmtNum(ov.min_loss)} min / ${fmtNum(ov.max_loss)} max`],
    ['LOWEST LOSS', fmtNum(ov.min_loss), '', ''],
    ['STEPS', ov.steps != null ? `${ov.steps} / ${detail.total_steps ?? '?'}` : '—', '', '完成轮次 ' + (ov.actual_epochs ?? '—') + ' / ' + (detail.total_epochs ?? '—')],
    ['DURATION', fmtDur(ov.duration_s), '', ''],
    ['LR FINAL', fmtLr(ov.lr_final), '', '峰值 ' + fmtLr(ov.lr_max)],
    ['EPOCH AVG LAST', fmtNum(ov.epoch_avg_last), '', ''],
    ['CKPT/SAMPLE', `${ov.ckpt_count} / ${ov.sample_count}`, '', `${ov.val_count} val`],
    ['WARN / ERROR', `${ov.warn_count} / ${ov.error_count}`, ov.error_count > 0 ? 'volt' : '', ''],
  ];
  const meta = [
    ['job id', detail.id],
    ['current job', detail.current_job_id || detail.id],
    ['attempts', String(detail.attempt_count || 1)],
    ['kind', detail.kind],
    ['state', detail.state + (detail.stop_requested ? ' (stop requested)' : '')],
    ['started', fmtTs(detail.started_at)],
    ['ended', fmtTs(detail.ended_at)],
    ['run_end', detail.run_end && detail.run_end.status ? `${detail.run_end.status} @ step ${detail.run_end.final_step ?? '—'}` : '—'],
    ['rc', detail.rc != null ? String(detail.rc) : '—'],
    ['error', detail.error || (detail.run_end && detail.run_end.error) || '—'],
    ['ckpt path', detail.ckpt_path || '—'],
  ];
  const argv = detail.argv ? detail.argv.join(' ') : '—';
  const env = detail.extra_env ? Object.entries(detail.extra_env).map(([k, v]) => `${k}=${v}`).join('  ') : '—';
  $('#sec-overview').innerHTML = `
    <div class="sec-head"><span class="sec-num">01</span><span class="sec-title">总览</span><span class="sec-sub">OVERVIEW</span></div>
    <div class="ov-grid">
      <div class="${stageCls}">
        <div class="stage-head"><span class="stage-title mono">KPI // ${esc(detail.run_name)}</span></div>
        <div class="kpi-grid">
          ${cells.map(([l, v, cls, sub]) => `
            <div class="kpi-cell">
              <div class="kpi-label">${l}</div>
              <div class="kpi-value ${cls || ''}">${v}</div>
              ${sub ? `<div class="kpi-sub">${sub}</div>` : ''}
            </div>`).join('')}
        </div>
      </div>
      <div class="meta-card">
        <div class="meta-head">RUN // META</div>
        <div class="meta-arc">${arcHtml({ step: ov.steps, total: detail.total_steps, epoch: ov.actual_epochs, totalEpochs: detail.total_epochs })}</div>
        <div class="meta-list">
          ${meta.map(([k, v]) => `<div class="meta-row"><span class="meta-k">${k}</span><span class="meta-v">${esc(String(v))}</span></div>`).join('')}
          <div class="meta-row"><span class="meta-k">argv</span><span class="meta-v">${esc(argv)}</span></div>
          <div class="meta-row"><span class="meta-k">env</span><span class="meta-v">${esc(env)}</span></div>
        </div>
      </div>
    </div>`;
}

/* ── 加载与实时刷新 ── */
export async function loadDetail(id) {
  stopDetailPolling();
  const root = $('#detail-root');
  root.innerHTML = loadingHtml();
  startTyping($('#loading-type'), 'LOADING…');
  try {
    detail = await api(`/api/runs/${encodeURIComponent(id)}`);
  } catch (e) {
    root.innerHTML = emptyHtml(esc(String(e.message || e)));
    return;
  }
  saveState({ lastRun: detail.id });
  renderDetailHead();
  const running = detail.state === 'running' || detail.state === 'queued';
  $('#rail-status').textContent = running ? 'LIVE' : 'IDLE';
  if (running) {
    const polledId = detail.id;
    detailTimer = setInterval(async () => {
      try {
        const fresh = await api(`/api/runs/${encodeURIComponent(polledId)}`);
        const previous = JSON.stringify({
          steps: detail.kpis && detail.kpis.steps,
          state: detail.state,
          ended_at: detail.ended_at,
          error: detail.error,
          run_end: detail.run_end,
        });
        detail = fresh;
        const current = JSON.stringify({
          steps: fresh.kpis && fresh.kpis.steps,
          state: fresh.state,
          ended_at: fresh.ended_at,
          error: fresh.error,
          run_end: fresh.run_end,
        });
        if (current !== previous) {
          const scrollTop = document.documentElement.scrollTop;
          renderDetailHead();
          window.scrollTo(0, scrollTop);
        }
        if (fresh.state !== 'running' && fresh.state !== 'queued') {
          stopDetailPolling();
          $('#rail-status').textContent = 'IDLE';
        }
      } catch (e) { /* transient */ }
    }, 2000);
  }
}
