/* charts/norm.js — norm/max_norm/vr 图 option 构建 */

import { C } from '../theme.js';
import { stageBase } from '../core/charts.js';

export function normChartOption(data) {
  const series = data.series;
  const tags = ((data.tags && data.tags.norm) || []).filter(t => series[t] && series[t].length);
  const palette = [C.paper, C.g5, C.g4, C.g6, C.g2];
  const opt = stageBase({ yLog: false });
  opt.series = tags.map((t, i) => ({
    name: t, type: 'line', data: series[t], z: 4,
    symbol: 'none',
    lineStyle: { width: 1, color: palette[i % palette.length] },
  }));
  return opt;
}
