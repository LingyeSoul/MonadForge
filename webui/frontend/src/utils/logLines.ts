// Shared log-line model for the streaming composables + LogStream.
//
// Lines carry their log level pre-computed at append time so the render
// path never re-classifies the whole buffer (the old per-render
// ``logLevel()`` walk was O(buffer) on every incoming message), and a
// monotonic ``seq`` so v-for keys stay stable across FIFO trims of the
// tail window.

export type LogLevel = '' | 'log-info' | 'log-warn' | 'log-error'

export interface LogLine {
  seq: number
  text: string
  level: LogLevel
}

export function classifyLine(text: string): LogLevel {
  const upper = text.toUpperCase()
  if (upper.includes('ERROR') || upper.includes('EXCEPTION') || upper.includes('TRACEBACK')) return 'log-error'
  if (upper.includes('WARN')) return 'log-warn'
  if (upper.includes('INFO')) return 'log-info'
  return ''
}

/** Client-side cap on rendered lines — matches the server's FIFO window
 * semantics; older entries are dropped so DOM size stays bounded. */
export const MAX_RENDERED_LINES = 2000

export function makeLogLine(seq: number, text: string): LogLine {
  return { seq, text, level: classifyLine(text) }
}
