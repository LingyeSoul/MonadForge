import { ref, onUnmounted } from 'vue'

interface WsMessage {
  type: 'connected' | 'log' | 'done' | 'cancelled' | 'error'
  line?: string
  task_id?: string
  exit_code?: number
  state?: string
  message?: string
}

export function useTaskStream(taskId: string) {
  const messages = ref<string[]>([])
  const connected = ref(false)
  const done = ref(false)
  const exitCode = ref<number | null>(null)
  let ws: WebSocket | null = null

  // Number of WS "log" messages to skip (subscribe() replays these)
  let replayRemaining = 0

  // rAF batching — coalesce rapid WS messages into a single DOM update
  const pendingLines: string[] = []
  let rafId = 0

  function flushPending() {
    rafId = 0
    if (pendingLines.length === 0) return
    messages.value.push(...pendingLines.splice(0))
  }

  function enqueueLine(line: string) {
    pendingLines.push(line)
    if (!rafId) {
      rafId = requestAnimationFrame(flushPending)
    }
  }

  async function connect() {
    // 1. Fetch accumulated history via REST (single request, not N WS messages)
    let historyCount = 0
    try {
      const res = await fetch(`/api/tasks/${taskId}/output`)
      if (res.ok) {
        const data = await res.json()
        if (data.lines?.length) {
          historyCount = data.lines.length
          messages.value.push(...data.lines)
        }
        if (data.state && data.state !== 'running' && data.state !== 'pending') {
          done.value = true
          exitCode.value = data.exit_code ?? null
        }
      }
    } catch {
      // REST failed — let WS replay handle everything
    }

    // WS subscribe() will replay these same lines; skip them
    replayRemaining = historyCount

    // 2. Open WebSocket for live updates
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    ws = new WebSocket(`${protocol}//${location.host}/ws/tasks/${taskId}`)

    ws.onopen = () => {
      connected.value = true
    }
    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data)
        if (msg.type === 'log' && msg.line) {
          if (replayRemaining > 0) {
            // Part of the subscribe() replay — already loaded via REST
            replayRemaining--
            return
          }
          enqueueLine(msg.line)
        } else if (msg.type === 'done') {
          done.value = true
          exitCode.value = msg.exit_code ?? null
        } else if (msg.type === 'cancelled') {
          done.value = true
          exitCode.value = -1
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
    }
    ws.onerror = () => {
      connected.value = false
    }
  }

  function disconnect() {
    ws?.close()
    ws = null
    connected.value = false
    if (rafId) cancelAnimationFrame(rafId)
  }

  onUnmounted(disconnect)

  return { messages, connected, done, exitCode, connect, disconnect }
}
