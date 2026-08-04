/* views/samples.js — 采样画廊 */

import { $, $$, esc } from '../core/dom.js';
import { openLightbox } from '../components/lightbox.js';
import { emptyHtml } from '../components/states.js';

export function renderSamples(d) {
  const items = d.samples || [];
  $('#sec-samples').innerHTML = `
    <div class="sec-head"><span class="sec-num">04</span><span class="sec-title">采样</span><span class="sec-sub">SAMPLES · ${items.length}</span></div>
    ${items.length ? `<div class="gallery" id="gallery">${items.map((s, i) => {
      const fn = s.path ? s.path.split('/').pop() : '';
      const ar = s.w && s.h ? ` style="--ar: ${s.w} / ${s.h}"` : '';
      const size = s.w && s.h ? `<span class="g-size">${s.w}×${s.h}</span>` : '';
      return `<div class="g-item" tabindex="0" data-i="${i}" role="button" aria-label="sample ${fn}"${ar}>
        <img src="/api/runs/${encodeURIComponent(d.id)}/samples/${encodeURIComponent(fn)}" alt="sample ${fn}" loading="lazy">
        <div class="g-cap"><span title="${esc(fn)}">${esc(fn)}</span>${size}${s.global_step ? `<span class="g-step">step ${s.global_step}</span>` : ''}</div>
      </div>`;
    }).join('')}</div>` : emptyHtml('NO SAMPLES — 无采样记录')}`;

  $$('.g-item').forEach(el => {
    const open = () => openLightbox(items[+el.dataset.i], d.id);
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
