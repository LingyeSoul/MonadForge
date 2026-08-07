/* views/curves.js — 曲线视图（控件 / 每轮表 / 图构建） */

import { $, $$, fmtNum } from '../core/dom.js';
import { chartRegistry, mountChart, preserveZoom, zoomTo } from '../core/charts.js';
import { lossChartOption } from '../charts/loss.js';
import { lrChartOption } from '../charts/lr.js';
import { normChartOption } from '../charts/norm.js';
import { loadState, saveState } from '../core/state.js';
import { openLightbox } from '../components/lightbox.js';
import { emptyHtml } from '../components/states.js';

export const curveUI = Object.assign(
  { smooth: 50, log: false, bands: true, raw: false, show: { 'loss/current': true, 'loss/average': true, 'loss/epoch_average': true } },
  loadState().curves || {}
);

let data = null;
let detailCharts = [];
let zoomSyncLock = false;

function persistUI() {
  saveState({ curves: { smooth: curveUI.smooth, log: curveUI.log, bands: curveUI.bands, raw: curveUI.raw, show: curveUI.show } });
}

export function renderCurves(d) {
  data = d;
  const series = d.series || {};
  const hasLossData = (series['loss/current'] || []).length > 0 || (series['loss/average'] || []).length > 0;
  if (!hasLossData) {
    $('#sec-curves').innerHTML = `
      <div class="sec-head"><span class="sec-num">02</span><span class="sec-title">曲线</span><span class="sec-sub">CURVES</span></div>
      ${emptyHtml('NO CURVE DATA — 无曲线数据 · 训练在首个 step 前终止')}`;
    return;
  }
  const tags = d.tags || {};
  const hasLr = (tags.lr || []).length > 0;
  const hasNorm = (tags.norm || []).length > 0;
  $('#sec-curves').innerHTML = `
    <div class="sec-head"><span class="sec-num">02</span><span class="sec-title">曲线</span><span class="sec-sub">CURVES</span></div>
    <div class="chart-tools">
      <div class="ctl">
        <span class="ctl-label">SMOOTH</span>
        <input type="range" id="ctl-smooth" min="1" max="200" step="1" value="${curveUI.smooth}">
        <span class="ctl-val" id="ctl-smooth-v">${curveUI.smooth}</span>
      </div>
      <div class="ctl">
        <span class="ctl-label">Y</span>
        <div class="seg" id="seg-loss-scale">
          <button data-scale="0" class="${!curveUI.log ? 'active' : ''}">线性</button><button data-scale="1" class="${curveUI.log ? 'active' : ''}">对数</button>
        </div>
      </div>
      <label class="checkbox"><input type="checkbox" id="ctl-bands" ${curveUI.bands ? 'checked' : ''}><span></span>每轮范围</label>
      <label class="checkbox"><input type="checkbox" id="ctl-raw" ${curveUI.raw ? 'checked' : ''}><span></span>原始点线</label>
      <div class="chip-set" id="chip-loss">
        <button class="sc-chip ${curveUI.show['loss/current'] ? 'active' : 'dim'}" data-tag="loss/current">current</button>
        <button class="sc-chip ${curveUI.show['loss/average'] ? 'active' : 'dim'}" data-tag="loss/average">average</button>
        <button class="sc-chip ${curveUI.show['loss/epoch_average'] ? 'active' : 'dim'}" data-tag="loss/epoch_average">epoch avg</button>
      </div>
    </div>
    <div class="stage">
      <div class="stage-head"><span class="stage-title mono">LOSS // STEP</span><span class="stage-tools"><span class="mono stage-sub" id="loss-legend-info"></span></span></div>
      <div class="chart-pad"><div id="chart-loss" class="chart chart-h-lg"></div></div>
    </div>
    ${hasLr ? `
    <div class="chart-gap"></div>
    <div class="stage">
      <div class="stage-head"><span class="stage-title mono">LR // STEP</span>
        <div class="stage-tools"><div class="seg" id="seg-lr-scale"><button data-scale="0">线性</button><button data-scale="1" class="active">对数</button></div></div>
      </div>
      <div class="chart-pad"><div id="chart-lr" class="chart chart-h-sm"></div></div>
    </div>` : ''}
    ${hasNorm ? `
    <div class="chart-gap"></div>
    <div class="stage">
      <div class="stage-head"><span class="stage-title mono">NORM // STEP</span></div>
      <div class="chart-pad"><div id="chart-norm" class="chart chart-h-sm"></div></div>
    </div>` : ''}
    <div class="epoch-table-wrap">
      <table class="epoch-table">
        <thead><tr><th>轮</th><th>min</th><th>max</th><th>mean</th><th>std</th><th>点数</th><th>step 范围</th></tr></thead>
        <tbody id="epoch-table-body"></tbody>
      </table>
    </div>`;

  const { opt, maxGs } = lossChartOption(d, curveUI);
  const chart = mountChart('chart-loss', opt);
  if (hasLr) mountChart('chart-lr', lrChartOption(d, true));
  if (hasNorm) mountChart('chart-norm', normChartOption(d));
  detailCharts = [chart];
  if (hasLr) detailCharts.push(chartRegistry.get('chart-lr'));
  if (hasNorm) detailCharts.push(chartRegistry.get('chart-norm'));

  /* C8：点击采样点 → 打开对应 lightbox（先解绑防 live 刷新累积） */
  chart.off('click');
  chart.on('click', params => {
    if (params.seriesName !== 'sample') return;
    const gs = params.value && params.value[0];
    const s = (d.samples || []).find(x => x.global_step === gs);
    if (s) openLightbox(s, d.id);
  });
  /* C9：跨图 zoom 联动（loss → lr/norm，防循环） */
  chart.off('datazoom');
  chart.on('datazoom', () => {
    if (zoomSyncLock) return;
    zoomSyncLock = true;
    const dz = chart.getOption().dataZoom;
    if (dz && dz[1]) {
      detailCharts.forEach(c => {
        if (c !== chart) c.dispatchAction({ type: 'dataZoom', start: dz[1].start, end: dz[1].end, dataZoomIndex: 1 });
      });
    }
    setTimeout(() => { zoomSyncLock = false; }, 0);
  });

  /* 每轮明细表 */
  const spans = d.epoch_spans || {};
  const epochs = d.epochs || {};
  const tb = $('#epoch-table-body');
  tb.innerHTML = Object.keys(epochs).sort((a, b) => a - b).map(ep => {
    const s = epochs[ep], sp = spans[ep];
    return `<tr tabindex="0" data-ep="${ep}" data-s0="${sp ? sp[0] : ''}" data-s1="${sp ? sp[1] : ''}">
      <td>E${String(ep).padStart(2, '0')}</td>
      <td>${fmtNum(s.min, 5)}</td><td>${fmtNum(s.max, 5)}</td>
      <td>${fmtNum(s.mean, 5)}</td><td>${fmtNum(s.std, 5)}</td>
      <td>${s.n}</td><td>${sp ? `${sp[0]} ~ ${sp[1]}` : '—'}</td>
    </tr>`;
  }).join('');
  tb.addEventListener('click', e => {
    const tr = e.target.closest('tr[data-ep]');
    if (!tr || !tr.dataset.s1) return;
    zoomTo(detailCharts, +tr.dataset.s0, +tr.dataset.s1, maxGs);
  });
  tb.addEventListener('keydown', e => {
    if (e.key !== 'Enter') return;
    const tr = e.target.closest('tr[data-ep]');
    if (!tr || !tr.dataset.s1) return;
    zoomTo(detailCharts, +tr.dataset.s0, +tr.dataset.s1, maxGs);
  });

  /* 控件绑定 */
  const smooth = $('#ctl-smooth');
  const onSmooth = () => {
    curveUI.smooth = +smooth.value;
    $('#ctl-smooth-v').textContent = smooth.value;
    persistUI();
    refreshLossChart();
  };
  smooth.addEventListener('input', onSmooth);
  $$('#seg-loss-scale button').forEach(b => b.addEventListener('click', () => {
    $$('#seg-loss-scale button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    curveUI.log = b.dataset.scale === '1';
    persistUI();
    refreshLossChart();
  }));
  $('#ctl-bands').addEventListener('change', e => { curveUI.bands = e.target.checked; persistUI(); refreshLossChart(); });
  $('#ctl-raw').addEventListener('change', e => { curveUI.raw = e.target.checked; persistUI(); refreshLossChart(); });
  $$('#chip-loss .sc-chip').forEach(ch => ch.addEventListener('click', () => {
    const tag = ch.dataset.tag;
    curveUI.show[tag] = !curveUI.show[tag];
    ch.classList.toggle('active', curveUI.show[tag]);
    ch.classList.toggle('dim', !curveUI.show[tag]);
    persistUI();
    refreshLossChart();
  }));
  if (hasLr) {
    $$('#seg-lr-scale button').forEach(b => b.addEventListener('click', () => {
      $$('#seg-lr-scale button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      refreshLrChart(b.dataset.scale === '1');
    }));
  }
}

function refreshLossChart() {
  if (!data) return;
  const { opt } = lossChartOption(data, curveUI);
  const chart = chartRegistry.get('chart-loss');
  if (chart) { preserveZoom(chart, opt); chart.setOption(opt, { notMerge: true }); }
}

function refreshLrChart(log) {
  if (!data) return;
  const chart = chartRegistry.get('chart-lr');
  if (!chart) return;
  const opt = lrChartOption(data, log);
  preserveZoom(chart, opt);
  chart.setOption(opt, { notMerge: true });
}
