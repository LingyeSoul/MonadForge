/* views/events.js — 事件视图（检查点 / 警告 / 校验 / stdout） */

import { $, $$, api, esc, fmtNum } from '../core/dom.js';

export function renderEvents(d) {
  const logs = d.logs || [];
  const ckpts = d.ckpts || [];
  const vals = d.vals || [];
  const warnRows = logs.filter(l => l.level === 'WARNING' || l.level === 'ERROR')
    .map(l => `<div class="ev-row ${l.level === 'ERROR' ? 'err' : 'warn'}"><span class="ev-level ${l.level}">${l.level}</span><span class="ev-step">${l.logger || ''}</span><span class="ev-msg">${esc(l.msg)}</span></div>`).join('')
    || '<div class="ev-row"><span class="ev-msg">无警告/错误</span></div>';
  const ckptRows = ckpts.map(c => `<div class="ev-row"><span class="ev-step">step ${c.global_step}</span><span class="ev-msg">${esc(c.path)}</span></div>`).join('')
    || '<div class="ev-row"><span class="ev-msg">无检查点记录</span></div>';
  const valRows = vals.map(v => `<div class="ev-row"><span class="ev-step">step ${v.global_step}</span><span class="ev-msg">CMMD ${fmtNum(v.cmmd)}</span></div>`).join('')
    || '<div class="ev-row"><span class="ev-msg">无校验记录</span></div>';

  $('#sec-events').innerHTML = `
    <div class="sec-head"><span class="sec-num">05</span><span class="sec-title">事件</span><span class="sec-sub">EVENTS</span></div>
    <div class="ev-grid">
      <div class="ev-block">
        <div class="meta-head">CHECKPOINTS · ${ckpts.length}</div>
        <div class="ev-list">${ckptRows}</div>
      </div>
      <div class="ev-block">
        <div class="meta-head">WARN / ERROR · ${logs.length}</div>
        <div class="ev-list">${warnRows}</div>
      </div>
      <div class="ev-block">
        <div class="meta-head">VALIDATION CMMD · ${vals.length}</div>
        <div class="ev-list">${valRows}</div>
      </div>
      <div class="ev-block full">
        <div class="meta-head">STDOUT · 标准输出</div>
        <div class="stdout-stage">
          <div class="stdout-head">
            <label class="search-box">
              <span class="search-ic mono">⌕</span>
              <input type="search" id="stdout-search" placeholder="过滤输出" autocomplete="off">
            </label>
            <div class="seg" id="seg-stdout">
              <button class="active" data-f="all">全部</button>
              <button data-f="WARNING">WARNING</button>
              <button data-f="ERROR">ERROR</button>
            </div>
          </div>
          <div class="stdout-body" id="stdout-body">加载中…</div>
        </div>
      </div>
    </div>`;
  let stdoutAll = [];
  const stdoutState = { q: '', f: 'all' };
  const renderStdout = () => {
    const lines = stdoutAll.filter(l => {
      if (stdoutState.f === 'WARNING' && !l.includes('WARNING')) return false;
      if (stdoutState.f === 'ERROR' && !(l.includes('ERROR') || l.includes('Traceback'))) return false;
      if (stdoutState.q && !l.toLowerCase().includes(stdoutState.q)) return false;
      return true;
    });
    const body = $('#stdout-body');
    body.innerHTML = lines.map(l => {
      let cls = '';
      if (l.includes('WARNING')) cls = 'l-warn';
      else if (l.includes('ERROR') || l.includes('Traceback')) cls = 'l-err';
      return `<div class="${cls}">${esc(l)}</div>`;
    }).join('') || '<div class="l-err">NO MATCH</div>';
  };
  api(`/api/runs/${encodeURIComponent(d.id)}/stdout?tail=2000`).then(res => {
    stdoutAll = res.lines || [];
    renderStdout();
  }).catch(() => {
    $('#stdout-body').innerHTML = '<div class="l-err">stdout 不可用</div>';
  });
  $('#stdout-search').addEventListener('input', e => { stdoutState.q = e.target.value.toLowerCase(); renderStdout(); });
  $$('#seg-stdout button').forEach(b => b.addEventListener('click', () => {
    $$('#seg-stdout button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    stdoutState.f = b.dataset.f;
    renderStdout();
  }));
}
