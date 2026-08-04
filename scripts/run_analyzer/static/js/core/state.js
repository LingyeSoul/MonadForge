/* core/state.js — 共享选中集 + localStorage 记忆（A3 在批 1 扩展） */

export const cmpSel = new Set();

export const LS_KEY = 'ra:state';

export function saveState(partial) {
  try {
    const cur = loadState();
    localStorage.setItem(LS_KEY, JSON.stringify({ ...cur, ...partial }));
  } catch (e) { /* storage unavailable */ }
}

export function loadState() {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) || '{}');
  } catch (e) {
    return {};
  }
}
