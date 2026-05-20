import { ref, onUnmounted } from 'vue'
import { useTrainingStore } from '../stores/training'
import type { TrainingMetrics } from '../stores/training'

interface WsMessage {
  type: 'connected' | 'log' | 'done' | 'cancelled' | 'error' | 'metrics'
  line?: string
  task_id?: string
  exit_code?: number
  state?: string
  message?: string
  data?: Partial<TrainingMetrics>
}

export function useTrainingStream(taskId: string) {
  const store = useTrainingStore()
  const connected = ref(false)
  const done = ref(false)
  const exitCode = ref<number | null>(null)
  const logLines = ref<string[]>([])
  let ws: WebSocket | null = null

  // Skip WS replay lines already loaded via REST
  let replayRemaining = 0

  // rAF batching for log lines
  const pendingLines: string[] = []
  let rafId = 0

  function flushPending() {
    rafId = 0
    if (pendingLines.length === 0) return
    logLines.value.push(...pendingLines.splice(0))
  }

  function enqueueLine(line: string) {
    pendingLines.push(line)
    if (!rafId) {
      rafId = requestAnimationFrame(flushPending)
    }
  }

  async function connect() {
    store.reset()

    // 1. Load accumulated history + metrics via REST
    let historyCount = 0
    try {
      const res = await fetch(`/api/tasks/${taskId}/output`)
      if (res.ok) {
        const data = await res.json()
        if (data.lines?.length) {
          historyCount = data.lines.length
          logLines.value.push(...data.lines)
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

    // 2. Open WebSocket for live updates
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    ws = new WebSocket(`${protocol}//${location.host}/ws/tasks/${taskId}`)

    ws.onopen = () => {
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
          store.updateFromWs(msg.data)
        } else if (msg.type === 'done') {
          done.value = true
          exitCode.value = msg.exit_code ?? null
          store.done = true
        } else if (msg.type === 'cancelled') {
          done.value = true
          exitCode.value = -1
          store.done = true
          enqueueLine('[cancelled]')
        } else if (msg.type === 'error' && msg.message) {
          enqueueLine(`[error] ${msg.message}`)
        }
      } catch {
        enqueueLine(event.data)
      }
    }

    ws.onclose = () => {
      connected.value = false
      store.connected = false
    }
    ws.onerror = () => {
      connected.value = false
      store.connected = false
    }
  }

  function disconnect() {
    ws?.close()
    ws = null
    connected.value = false
    store.connected = false
    if (rafId) cancelAnimationFrame(rafId)
  }

  onUnmounted(disconnect)

  return { logLines, connected, done, exitCode, connect, disconnect }
}
