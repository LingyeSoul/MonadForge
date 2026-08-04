/* charts/loss.js — loss 图 option 构建（档案/监控共用） */

import { C, MONO } from '../theme.js';
import { fmtNum, hexA } from '../core/dom.js';
import { epochAt, rolling, stageBase } from '../core/charts.js';

export function lossChartOption(data, ui, opts = {}) {
  const series = data.series;
  const cur = series['loss/current'] || [];
  const avg = series['loss/average'] || [];
  const epAvg = series['loss/epoch_average'] || [];
  const spans = data.epoch_spans || {};
  const epochs = data.epochs || {};

  const opt = stageBase({ yLog: ui.log });
  const maxGs = Math.max(cur.length ? cur[cur.length - 1][0] : 0, avg.length ? avg[avg.length - 1][0] : 0, ...Object.values(spans).map(s => s[1] || 0));
  /* 每轮色带 — ECharts 5.5 的 markArea 在 item 只有 xAxis 时内部
     读取 coord 必崩，改用 custom series 画矩形 + scatter 文本层标注 */
  const spansList = Object.keys(spans).sort((a, b) => a - b);
  const allLoss = cur.concat(avg).map(p => p[1]);
  const yMin = allLoss.length ? Math.min(...allLoss) * 0.5 : 0.01;
  const yMax = allLoss.length ? Math.max(...allLoss) : 1;
  const levels = [1.15, 1.4, 1.7];
  /* bands 关闭时置空数据：custom series 空数据不参与轴取值，也不会绘制 */
  const bandData = ui.bands ? spansList.map(ep => [spans[ep][0], spans[ep][1], yMin, yMax]) : [];
  const labelData = ui.bands ? spansList.map((ep, i) => {
    const st = epochs[ep] || {};
    const mid = (spans[ep][0] + spans[ep][1]) / 2;
    return {
      value: [mid, (st.max ?? yMax) * levels[i % 3]],
      label: {
        show: true,
        formatter: `E${ep} ${fmtNum(st.min, 3)}~${fmtNum(st.max, 3)}`,
        color: C.g5, fontSize: 10, fontFamily: MONO,
        position: 'top',
      },
    };
  }) : [];
  const bandSeries = {
    name: 'epoch bands', type: 'custom', z: 1, silent: true, data: bandData,
    tooltip: { show: false },
    renderItem: (p, api) => {
      const x0 = api.value(0), x1 = api.value(1), y0 = api.value(2), y1 = api.value(3);
      const a = api.coord([x0, y0]), b = api.coord([x1, y1]);
      return {
        type: 'rect',
        shape: {
          x: Math.min(a[0], b[0]), y: Math.min(a[1], b[1]),
          width: Math.max(Math.abs(b[0] - a[0]), 1), height: Math.max(Math.abs(b[1] - a[1]), 1),
        },
        style: { fill: hexA(C.paper, 0.05) },
      };
    },
  };
  const labelSeries = {
    name: 'epoch range', type: 'scatter', z: 8, symbol: 'none', silent: true,
    data: labelData, tooltip: { show: false },
  };

  const mk = (name, pts, color, width, z, extra = {}) => ({
    name, type: 'line', data: pts, z,
    symbol: 'none',
    lineStyle: { width, color, type: 'solid' },
    emphasis: { lineStyle: { width: width + 1 } },
    ...extra,
  });

  const seriesArr = [bandSeries];

  /* D11：扫描光 — volt 竖线跟随最新 step（仅运行中） */
  if (opts.scan && cur.length) {
    const lastGs = cur[cur.length - 1][0];
    const scanSeries = mk('scan', [], C.volt, 0, 9, {
      silent: true, tooltip: { show: false },
      markLine: {
        silent: true, symbol: 'none',
        lineStyle: { color: C.volt, width: 1 },
        label: { show: true, formatter: 'SCAN', color: C.volt, fontSize: 10, fontFamily: MONO, position: 'insideEndTop' },
        data: [{ xAxis: lastGs }],
      },
    });
    seriesArr.push(scanSeries);
  }
  if (ui.raw && cur.length) seriesArr.push(mk('loss/current raw', cur, hexA(C.g5, 0.30), 0.6, 2));
  if (ui.show['loss/current'] && cur.length) seriesArr.push(mk('loss/current', rolling(cur, ui.smooth), C.g5, 1, 3));
  if (ui.show['loss/average'] && avg.length) {
    const avgSeries = mk('loss/average', rolling(avg, ui.smooth), C.paper, 1.8, 5);
    avgSeries.markLine = {
      silent: true, symbol: 'none',
      lineStyle: { color: C.g5, width: 1, type: 'dashed', opacity: 0.5 },
      data: (data.ckpts || []).map(c => ({ xAxis: c.global_step })),
    };
    if (data.state === 'stopped' || data.state === 'done') {
      const stopGs = data.run_end && data.run_end.final_step != null ? data.run_end.final_step : null;
      if (stopGs != null) {
        avgSeries.markLine.data = [...avgSeries.markLine.data, {
          xAxis: stopGs,
          lineStyle: { color: C.paper, width: 2, type: 'solid' },
          label: { show: true, formatter: 'STOP', color: C.paper, fontSize: 10, fontFamily: MONO, position: 'insideEndTop', backgroundColor: C.ink },
        }];
      }
    }
    seriesArr.push(avgSeries);
  }
  if (ui.show['loss/epoch_average'] && epAvg.length) {
    seriesArr.push(mk('loss/epoch_average', epAvg, C.volt, 1, 6, {
      symbol: 'rect', symbolSize: 5, showSymbol: true,
      lineStyle: { width: 1, color: hexA(C.volt, 0.45), type: 'dashed' },
    }));
  }
  seriesArr.push(labelSeries);
  /* 采样点：黄色方块覆盖层 */
  if (data.samples.length && avg.length) {
    const pts = data.samples.map(s => {
      const gs = s.global_step;
      let y = null;
      for (let i = avg.length - 1; i >= 0; i--) if (avg[i][0] <= gs) { y = avg[i][1]; break; }
      return { value: [gs, y == null ? 0 : y], itemStyle: { color: C.volt } };
    });
    seriesArr.push({
      name: 'sample', type: 'scatter', data: pts, z: 7,
      symbol: 'rect', symbolSize: 6,
      tooltip: { trigger: 'item', formatter: p => `SAMPLE @ step ${p.value[0]}` },
    });
  }

  /* 显式锁定 y 域：custom 色带 series 的 4 元数组数据会污染 y 轴自动取值
     （取 [1] 元素 = 轮次结束 step），必须始终由 loss 数据显式设定 */
  if (allLoss.length) {
    opt.yAxis.min = yMin;
    opt.yAxis.max = ui.bands ? yMax * 1.9 : yMax * 1.1;
  }
  /* tooltip：crosshair 顶部加所属轮次 */
  const tt = opt.tooltip;
  tt.formatter = (params) => {
    const arr = Array.isArray(params) ? params : [params];
    const p0 = arr[0];
    if (!p0 || p0.value == null) return '';
    const step = p0.value[0];
    const ep = epochAt(step, spans);
    const rows = arr
      .filter(p => p.seriesName && p.value != null && Number.isFinite(p.value[1]))
      .map(p => `${p.marker}${p.seriesName}<span style="float:right;margin-left:16px">${fmtNum(p.value[1])}</span>`);
    const header = `step ${step}${ep != null ? ` · E${ep}` : ''}`;
    return [header, ...rows].join('<br>');
  };
  opt.series = seriesArr;
  return { opt, maxGs };
}
