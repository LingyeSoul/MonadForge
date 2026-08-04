/* theme.js — JS 侧设计令牌（与 css/tokens.css 同步） */

export const MONO = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
export const SANS = '"Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif';

export const C = {
  ink: '#191919', paper: '#FFFFFF', stage: '#000000',
  volt: '#FFFA00', aux: '#FFF000',
  g0: '#F7F7F7', g1: '#F2F2F2', g2: '#E5E5E5', g3: '#D9D9D9', g4: '#B3B3B3',
  g5: '#999999', g6: '#666666', g7: '#424242', g8: '#35373C', g9: '#1A1A1A',
};

export const ST = {
  running: { label: '进行中', cls: 'st-running' },
  queued: { label: '排队', cls: 'st-queued' },
  done: { label: '已完成', cls: 'st-done' },
  stopped: { label: '已停止', cls: 'st-stopped' },
  error: { label: '异常', cls: 'st-error' },
  orphan: { label: '孤立', cls: 'st-stopped' },
};
