import { ref, type Ref } from 'vue'
import { useTrainingStore } from '../stores/training'
import type { TrainingMetrics } from '../stores/training'
import { MAX_RENDERED_LINES, makeLogLine, type LogLine } from '../utils/logLines'
import type { LogStreamHandle } from './useTask'

interface WsMessage {
  type: 'connected' | 'log' | 'done' | 'cancelled' | 'error' | 'metrics' | 'wandb_url' | 'sample'
  line?: string
  task_id?: string
  exit_code?: number
  state?: string
  message?: string
  data?: Partial<TrainingMetrics>
  url?: string
  path?: string
  step?: number | null
  epoch?: number | null
  prompt?: string | null
  ts?: number | null
  attempt_id?: string | null
}

// Coalesce metrics snapshots (each carries the full loss/lr/step histories)
// into at most one store commit per window — the dashboard re-maps two chart
// datasets and re-renders two SVG charts per commit, so commit rate, not
// message rate, is what has to stay low.
const METRICS_COMMIT_INTERVAL = 500

export function useTrainingStream(taskId: string): LogStreamHandle & {
  exitCode: Ref<number | null>
  connect: () => Promise<void>
  disconnect: () => void
} {
  const store = useTrainingStore()
  const lines = ref<LogLine[]>([])
  const connected = ref(false)
  const done = ref(false)
  const exitCode = ref<number | null>(null)
  const activity = ref(0)
  const historyTotal = ref(0)
  let ws: WebSocket | null = null
  let reconnectTimer = 0
  let reconnectAttempt = 0
  let manuallyClosed = false
  let seq = 0

  // Skip WS replay lines already loaded via REST
  let replayRemaining = 0

  // rAF batching for log lines
  const pendingLines: string[] = []
  let rafId = 0

  function flushPending() {
    rafId = 0
    if (pendingLines.length === 0) return
    for (const line of pendingLines.splice(0)) {
      lines.value.push(makeLogLine(seq++, line))
    }
    const overflow = lines.value.length - MAX_RENDERED_LINES
    if (overflow > 0) lines.value.splice(0, overflow)
    activity.value++
  }

  function enqueueLine(line: string) {
    pendingLines.push(line)
    if (!rafId) {
      rafId = requestAnimationFrame(flushPending)
    }
  }

  // ── metrics commit throttle ────────────────────────────────────
  let pendingMetrics: Partial<TrainingMetrics> | null = null
  let metricsTimer = 0

  function scheduleMetricsCommit(data: Partial<TrainingMetrics>) {
    pendingMetrics = pendingMetrics ? { ...pendingMetrics, ...data } : data
    if (!metricsTimer) {
      metricsTimer = window.setTimeout(() => {
        metricsTimer = 0
        if (pendingMetrics) store.updateFromWs(pendingMetrics)
        pendingMetrics = null
      }, METRICS_COMMIT_INTERVAL)
    }
  }

  function flushMetricsCommit() {
    if (metricsTimer) {
      clearTimeout(metricsTimer)
      metricsTimer = 0
    }
    if (pendingMetrics) {
      store.updateFromWs(pendingMetrics)
      pendingMetrics = null
    }
  }

  function closeSocket() {
    if (!ws) return
    ws.onclose = null
    ws.onerror = null
    ws.close()
    ws = null
  }

  function scheduleReconnect() {
    if (manuallyClosed || done.value || reconnectTimer) return
    const delays = [1000, 2000, 5000]
    const delay = delays[Math.min(reconnectAttempt, delays.length - 1)]
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = 0
      reconnectAttempt += 1
      void open()
    }, delay)
  }

  async function open() {
    if (manuallyClosed) return
    closeSocket()

    // Load accumulated history + metrics before opening the live stream so a
    // reconnect fills the gap instead of duplicating or dropping messages.
    let historyCount = 0
    try {
      const res = await fetch(`/api/tasks/${taskId}/output`)
      if (res.ok) {
        const data = await res.json()
        if (Array.isArray(data.lines)) {
          historyCount = data.lines.length
          historyTotal.value = data.total ?? historyCount
          pendingLines.length = 0
          if (rafId) cancelAnimationFrame(rafId)
          rafId = 0
          lines.value = data.lines.slice(-MAX_RENDERED_LINES).map((line: string) => makeLogLine(seq++, line))
          activity.value++
        }
        if (data.state && data.state !== 'running' && data.state !== 'pending') {
          done.value = true
          exitCode.value = data.exit_code ?? null
          store.done = true
        }
      }
    } catch {
      // REST failed — WS replay handles it
    }

    // Load existing metrics snapshot
    await store.loadFromRest(taskId)

    replayRemaining = historyCount
    if (manuallyClosed || done.value) return

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    ws = new WebSocket(`${protocol}//${location.host}/ws/tasks/${taskId}`)

    ws.onopen = () => {
      reconnectAttempt = 0
      connected.value = true
      store.connected = true
    }

    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data)
        if (msg.type === 'log' && msg.line) {
          if (replayRemaining > 0) {
            replayRemaining--
            return
          }
          enqueueLine(msg.line)
        } else if (msg.type === 'metrics' && msg.data) {
          scheduleMetricsCommit(msg.data)
        } else if (msg.type === 'done') {
          flushMetricsCommit()
          done.value = true
          exitCode.value = msg.exit_code ?? null
          store.done = true
        } else if (msg.type === 'cancelled') {
          flushMetricsCommit()
          done.value = true
          exitCode.value = -1
          store.done = true
          enqueueLine('[cancelled]')
        } else if (msg.type === 'error' && msg.message) {
          enqueueLine(`[error] ${msg.message}`)
        } else if (msg.type === 'wandb_url' && msg.url) {
          store.updateFromWs({ wandb_run_url: msg.url })
        } else if (msg.type === 'sample' && msg.path) {
          // A new preview image landed in the training output dir. Append
          // to the gallery (the store dedupes by path).
          store.recordSample({
            attempt_id: msg.attempt_id ?? null,
            path: msg.path,
            step: msg.step ?? null,
            epoch: msg.epoch ?? null,
            prompt: msg.prompt ?? null,
            ts: msg.ts ?? null,
          })
        }
      } catch {
        enqueueLine(event.data)
      }
    }

    ws.onclose = () => {
      connected.value = false
      store.connected = false
      ws = null
      scheduleReconnect()
    }
    ws.onerror = () => {
      connected.value = false
      store.connected = false
      ws?.close()
    }
  }

  async function connect() {
    manuallyClosed = false
    reconnectAttempt = 0
    done.value = false
    exitCode.value = null
    store.reset()
    lines.value = []
    pendingLines.length = 0
    if (rafId) cancelAnimationFrame(rafId)
    rafId = 0
    if (metricsTimer) clearTimeout(metricsTimer)
    metricsTimer = 0
    pendingMetrics = null
    if (reconnectTimer) clearTimeout(reconnectTimer)
    reconnectTimer = 0
    await open()
  }

  function disconnect() {
    manuallyClosed = true
    if (reconnectTimer) clearTimeout(reconnectTimer)
    reconnectTimer = 0
    closeSocket()
    connected.value = false
    store.connected = false
    pendingLines.length = 0
    if (rafId) cancelAnimationFrame(rafId)
    rafId = 0
    if (metricsTimer) clearTimeout(metricsTimer)
    metricsTimer = 0
    pendingMetrics = null
  }

  return { lines, connected, done, exitCode, activity, historyTotal, connect, disconnect }
}
