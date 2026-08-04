/* components/kpi.js — KPI 数字墙渲染 + 圆弧进度（C7） */

export function kpiGridHtml(cells) {
  return cells.map(([l, v, cls]) => `
    <div class="kpi-cell"><div class="kpi-label">${l}</div><div class="kpi-value ${cls || ''}">${v}</div></div>`).join('');
}

/* C7：step 进度圆弧（黑图版 + volt 弧 + 中心数字） */
export function arcHtml({ step, total, epoch, totalEpochs }) {
  const pct = total ? Math.min(1, (step || 0) / total) : 0;
  const R = 62, circ = 2 * Math.PI * R;
  return `
  <div class="arc-block">
    <svg viewBox="0 0 180 180" class="arc-svg" aria-hidden="true">
      <circle class="arc-track" cx="90" cy="90" r="${R}"/>
      <circle class="arc-fill" cx="90" cy="90" r="${R}"
        stroke-dasharray="${(circ * pct).toFixed(1)} ${circ.toFixed(1)}"
        transform="rotate(-90 90 90)"/>
    </svg>
    <div class="arc-center">
      <div class="arc-step mono">${step ?? '—'}</div>
      <div class="arc-sub mono">E ${epoch ?? '—'} / ${totalEpochs ?? '—'}</div>
    </div>
  </div>`;
}
