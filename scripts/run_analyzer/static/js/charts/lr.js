/* charts/lr.js — 学习率图 option 构建 */

import { C } from '../theme.js';
import { stageBase } from '../core/charts.js';

export function lrChartOption(data, log = true) {
  const series = data.series;
  const lrTags = ((data.tags && data.tags.lr) || []).filter(t => series[t] && series[t].length);
  const palette = [C.paper, C.g5, C.g4, C.g6, C.g7, C.g2];
  const opt = stageBase({ yLog: log });
  opt.series = lrTags.map((t, i) => ({
    name: t, type: 'line', data: series[t], z: 4,
    symbol: 'none',
    lineStyle: { width: i === 0 ? 1.6 : 1, color: palette[i % palette.length] },
  }));
  if (lrTags.length === 1) opt.series[0].lineStyle.color = C.volt;
  return opt;
}
