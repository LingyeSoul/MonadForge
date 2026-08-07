/* core/charts.js — ECharts 唯一入口（模块边界 R7）
   所有图表基建集中于此：舞台配置 / 实例注册 / 缩放 / 平滑 */

import { C, MONO } from '../theme.js';
import { hexA } from './dom.js';

const echarts = window.echarts;

export function stageBase({ yLog = false, xName = 'step' } = {}) {
  return {
    animation: false,
    backgroundColor: C.stage,
    textStyle: { fontFamily: MONO },
    grid: { left: 64, right: 28, top: 44, bottom: 52 },
    xAxis: {
      type: 'value', name: xName,
      nameTextStyle: { color: C.g5, fontSize: 10, fontFamily: MONO },
      axisLine: { lineStyle: { color: C.g8 } },
      axisTick: { lineStyle: { color: C.g8 } },
      axisLabel: { color: C.g5, fontSize: 10, fontFamily: MONO },
      splitLine: { show: false },
    },
    yAxis: {
      type: yLog ? 'log' : 'value',
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: C.g5, fontSize: 10, fontFamily: MONO },
      splitLine: { lineStyle: { color: C.g8, width: 1 } },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: C.paper,
      borderColor: C.ink, borderWidth: 1,
      borderRadius: 0,
      extraCssText: 'box-shadow: none;',
      padding: [8, 12],
      textStyle: { color: C.ink, fontSize: 12, fontFamily: MONO },
      axisPointer: {
        type: 'cross',
        lineStyle: { color: C.g5, width: 1 },
        crossStyle: { color: C.g5, width: 1 },
        label: { backgroundColor: C.ink, color: C.paper, fontSize: 10, fontFamily: MONO },
      },
    },
    dataZoom: [
      { type: 'inside', xAxisIndex: 0 },
      {
        type: 'slider', xAxisIndex: 0,
        height: 14, bottom: 12,
        borderColor: C.g8, backgroundColor: C.stage,
        fillerColor: hexA(C.volt, 0.10),
        handleStyle: { color: C.volt, borderColor: C.ink, borderWidth: 1 },
        moveHandleStyle: { color: C.g8 },
        emphasis: { handleStyle: { color: C.volt } },
        textStyle: { color: C.g5, fontSize: 10, fontFamily: MONO },
        dataBackground: { lineStyle: { color: C.g8 }, areaStyle: { color: C.g9 } },
        selectedDataBackground: { lineStyle: { color: C.volt, width: 1 }, areaStyle: { color: hexA(C.volt, 0.16) } },
      },
    ],
  };
}

export const chartRegistry = new Map();

export function mountChart(elId, option) {
  const el = document.getElementById(elId);
  let chart = chartRegistry.get(elId);
  /* 已注册实例若绑定在已移除的 DOM 上（跨运行导航），释放重建 */
  if (!chart || !chart.getDom() || !chart.getDom().isConnected) {
    if (chart) chart.dispose();
    chart = echarts.init(el, null, { renderer: 'canvas' });
    chartRegistry.set(elId, chart);
  }
  chart.setOption(option, { notMerge: true });
  return chart;
}

export function preserveZoom(chart, option) {
  try {
    const dz = chart.getOption().dataZoom;
    if (Array.isArray(dz) && dz[1] && option.dataZoom && option.dataZoom[1]) {
      if (dz[1].start != null) option.dataZoom[1].start = dz[1].start;
      if (dz[1].end != null) option.dataZoom[1].end = dz[1].end;
    }
  } catch (e) { /* keep defaults */ }
}

export function zoomTo(charts, gsStart, gsEnd, total) {
  const p0 = Math.max(0, (gsStart / total) * 100);
  const p1 = Math.min(100, (gsEnd / total) * 100);
  charts.forEach(ch => ch.dispatchAction({ type: 'dataZoom', start: p0, end: p1, dataZoomIndex: 1 }));
}

export function rolling(arr, win) {
  const out = [];
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i][1];
    if (i >= win) sum -= arr[i - win][1];
    const n = Math.min(i + 1, win);
    out.push([arr[i][0], sum / n]);
  }
  return out;
}

/* step → epoch 反查（用于 tooltip 显示轮次） */
export function epochAt(step, spans) {
  const eps = Object.keys(spans).sort((a, b) => +a - +b);
  let cur = null;
  for (const ep of eps) {
    const [s0, s1] = spans[ep];
    if (step >= s0 && step <= s1) return +ep;
    if (step < s0) break;
    cur = +ep;
  }
  return cur;
}
