/* components/lightbox.js — 采样预览遮罩（全局单例） */

import { $ } from '../core/dom.js';

let _cur = null;
let _curRunId = '';

function _metaBase() {
  const fn = _cur && _cur.path ? _cur.path.split('/').pop() : '';
  return `${fn}  ·  epoch ${(_cur && _cur.epoch) ?? '—'}  ·  step ${(_cur && _cur.global_step) ?? '—'}`;
}

export function sampleUrl(s, runId) {
  const fn = s && s.path ? s.path.split('/').pop() : '';
  const params = new URLSearchParams();
  if (s && s.attempt_id) params.set('attempt_id', s.attempt_id);
  const query = params.toString();
  return `/api/runs/${encodeURIComponent(runId)}/samples/${encodeURIComponent(fn)}${query ? `?${query}` : ''}`;
}

export function openLightbox(s, runId) {
  _cur = s;
  _curRunId = runId;
  const size = s && s.w && s.h ? `  ·  ${s.w}×${s.h}` : '';
  $('#lb-meta').textContent = _metaBase() + size;
  $('#lb-img').src = sampleUrl(s, runId);
  $('#lb-prompt').textContent = (s && s.prompt) || '';
  $('#lightbox').classList.remove('hidden');
  $('#lb-close').focus();
}

function _syncSizeFromImage() {
  const img = $('#lb-img');
  if (!_cur || !_curRunId || img.naturalWidth <= 0 || img.naturalHeight <= 0) return;
  if (_cur.w && _cur.h) return;
  if ($('#lb-meta').textContent.includes('×')) return;
  $('#lb-meta').textContent = `${_metaBase()}  ·  ${img.naturalWidth}×${img.naturalHeight}`;
}

export function closeLightbox() {
  $('#lightbox').classList.add('hidden');
  $('#lb-img').removeAttribute('src');
}

export function initLightbox() {
  $('#lb-close').addEventListener('click', closeLightbox);
  $('#lightbox').addEventListener('click', e => { if (e.target.id === 'lightbox') closeLightbox(); });
  $('#lb-img').addEventListener('load', _syncSizeFromImage);
  window.addEventListener('keydown', e => { if (e.key === 'Escape' && !$('#lightbox').classList.contains('hidden')) closeLightbox(); });
}
