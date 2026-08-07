/* views/compare.js — 对比页 */

import { $, $$, api, esc, fmtNum, toast, hexA } from '../core/dom.js';
import { ST, C, MONO } from '../theme.js';
import { chartRegistry, mountChart, stageBase, epochAt } from '../core/charts.js';
import { cmpSel } from '../core/state.js';

let cmpCache = [];

export function getCmpSel() { return cmpSel; }

export function renderComparePicker() {
  const box = $('#cmp-list');
  box.innerHTML = cmpCache.map(r => {
    const st = ST[r.state] || { label: r.state };
    const checked = cmpSel.has(r.id);
    return `<div class="cmp-run" data-id="${r.id}" tabindex="0" role="checkbox" aria-checked="${checked}">
      <input type="checkbox" ${checked ? 'checked' : ''} tabindex="-1">
      <span class="cb"></span>
      <span class="cr-name">${esc(r.run_name)}</span>
      <span class="cr-meta">${esc(r.method || '—')} · ${esc(r.preset || '—')} · ${st.label} · ${fmtNum(r.final_avr_loss)}</span>
    </div>`;
  }).join('');
  $$('.cmp-run').forEach(el => el.addEventListener('click', () => {
    const id = el.dataset.id;
    if (cmpSel.has(id)) cmpSel.delete(id); else if (cmpSel.size < 8) cmpSel.add(id);
    renderComparePicker();
    window.dispatchEvent(new CustomEvent('ra:compare-picked', { detail: { ids: [...cmpSel] } }));
  }));
  $$('.cmp-run').forEach(el => el.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.click(); }
  }));
  $('#cmp-count').textContent = cmpSel.size;
  $('#cmp-go').disabled = cmpSel.size < 1;
}

export async function loadCompare() {
  if (!cmpCache.length) {
    try {
      const d = await api('/api/runs');
      cmpCache = d.runs;
    } catch (e) { toast(String(e.message || e)); return; }
  }
  renderComparePicker();
}

async function runCompare() {
  const ids = [...cmpSel];
  if (!ids.length) return;
  try {
    const d = await api('/api/compare', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids }),
    });
    renderCompareChart(d);
  } catch (e) { toast(String(e.message || e)); }
}

export function renderCompareChart(d) {
  const runs = d.runs || [];
  const palette = [C.paper, C.g7, C.g5, C.g3, C.g2, C.g4, C.g6, C.g1];
  const showEpAvg = $('#cmp-epavg').checked;
  const showSpans = $('#cmp-spans').checked;
  const opt = stageBase({});
  const seriesArr = [];
  const bandSeries = [];
  const labelSeries = [];

  /* 全局 y 域（显式锁定：custom 色带数据会污染 y 轴自动取值） */
  let globalMin = Infinity, globalMax = 1;
  runs.forEach(r => {
    (r.loss_average || []).forEach(p => {
      if (p[1] > globalMax) globalMax = p[1];
      if (p[1] < globalMin) globalMin = p[1];
    });
  });
  if (!isFinite(globalMin)) globalMin = 0.01;
  opt.yAxis.min = globalMin * 0.5;
  opt.yAxis.max = showSpans ? globalMax * 1.18 : globalMax * 1.08;

  /* 轮次分段：每运行一条半透明色带 custom series + E# 标注层 */
  if (showSpans) {
    runs.forEach((r, i) => {
      const color = palette[i % palette.length];
      const spans = r.epoch_spans || {};
      const eps = Object.keys(spans).sort((a, b) => +a - +b);
      if (!eps.length) return;
      const vals = (r.loss_average || []).map(p => p[1]);
      const yMin = vals.length ? Math.min(...vals) * 0.5 : 0.01;
      const yMax = vals.length ? Math.max(...vals) : 1;
      bandSeries.push({
        name: `${r.run_name} · epochs`, type: 'custom', z: 1, silent: true,
        tooltip: { show: false },
        data: eps.map(ep => [spans[ep][0], spans[ep][1], yMin, yMax]),
        renderItem: (p, api) => {
          const x0 = api.value(0), x1 = api.value(1), y0 = api.value(2), y1 = api.value(3);
          const a = api.coord([x0, y0]), b = api.coord([x1, y1]);
          return {
            type: 'rect',
            shape: {
              x: Math.min(a[0], b[0]), y: Math.min(a[1], b[1]),
              width: Math.max(Math.abs(b[0] - a[0]), 1), height: Math.max(Math.abs(b[1] - a[1]), 1),
            },
            style: { fill: hexA(color, 0.08) },
          };
        },
      });
      /* E# 标注统一置于图表顶部留白区（全局 max 之上，分两级错开防重叠） */
      const labelY = globalMax * (1.06 + 0.035 * (i % 2));
      labelSeries.push({
        name: `${r.run_name} · E#`, type: 'scatter', z: 8, symbol: 'none', silent: true,
        tooltip: { show: false },
        data: eps.map(ep => ({
          value: [(spans[ep][0] + spans[ep][1]) / 2, labelY],
          label: { show: true, formatter: `E${ep}`, color, fontSize: 10, fontFamily: MONO, position: 'top' },
        })),
      });
    });
  }
  seriesArr.push(...bandSeries);

  const lines = runs.map((r, i) => {
    const lineStyle = { width: 1.6, color: palette[i % palette.length] };
    const out = {
      name: r.run_name, type: 'line', data: r.loss_average || [], z: 4,
      symbol: 'none', lineStyle,
      emphasis: { lineStyle: { width: 2.4, color: C.volt } },
    };
    if (showEpAvg && (r.epoch_average || []).length) {
      out.markPoint = {
        symbol: 'rect', symbolSize: 5,
        itemStyle: { color: palette[i % palette.length] },
        label: {
          show: true,
          formatter: p => `E${p.dataIndex + 1}`,
          color: palette[i % palette.length], fontSize: 10, fontFamily: MONO,
          position: 'top',
        },
        data: (r.epoch_average || []).map(pt => ({ coord: pt })),
      };
    }
    return out;
  });
  seriesArr.push(...lines, ...labelSeries);
  opt.series = seriesArr;

  /* tooltip：每序列附所属轮次 */
  opt.tooltip.formatter = (params) => {
    const arr = Array.isArray(params) ? params : [params];
    const p0 = arr[0];
    if (!p0 || p0.value == null) return '';
    const step = p0.value[0];
    const rows = arr
      .filter(p => p.seriesName && !p.seriesName.includes(' · ') && p.value != null && Number.isFinite(p.value[1]))
      .map(p => {
        const run = runs.find(x => x.run_name === p.seriesName);
        let epStr = '';
        if (run && run.epoch_spans) {
          const ep = epochAt(step, run.epoch_spans);
          if (ep != null) epStr = ` · E${ep}`;
        }
        return `${p.marker}${p.seriesName}${epStr}<span style="float:right;margin-left:16px">${fmtNum(p.value[1])}</span>`;
      });
    return [`step ${step}`, ...rows].join('<br>');
  };

  const chart = mountChart('cmp-chart', opt);
  const tb = $('#cmp-table tbody');
  tb.innerHTML = runs.map(r => `<tr>
    <td>${esc(r.run_name)}</td>
    <td>${fmtNum(r.final_avr_loss)}</td>
    <td>${r.steps ?? '—'}</td>
    <td>${r.actual_epochs != null ? `${r.actual_epochs}/${r.total_epochs ?? '?'}` : '—'}</td>
    <td>${ST[r.state] ? ST[r.state].label : r.state}</td>
  </tr>`).join('');
}

function clearCompare() {
  cmpSel.clear();
  $('#cmp-epavg').checked = false;
  $('#cmp-spans').checked = true;
  const ch = chartRegistry.get('cmp-chart');
  if (ch) ch.clear();
  $('#cmp-table tbody').innerHTML = '';
  renderComparePicker();
  window.dispatchEvent(new CustomEvent('ra:compare-picked', { detail: { ids: [] } }));
}

export function initCompare() {
  $('#cmp-go').addEventListener('click', runCompare);
  $('#cmp-clear').addEventListener('click', clearCompare);
  const onChartOption = () => {
    if (!$('#cmp-chart').childElementCount) return;
    api('/api/compare', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids: [...cmpSel] }),
    }).then(renderCompareChart).catch(e => toast(String(e.message || e)));
  };
  $('#cmp-epavg').addEventListener('change', onChartOption);
  $('#cmp-spans').addEventListener('change', onChartOption);
}
