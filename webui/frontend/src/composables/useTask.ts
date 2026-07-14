import { ref, watch, onUnmounted } from 'vue'

interface WsMessage {
  type: 'connected' | 'log' | 'done' | 'cancelled' | 'error'
  line?: string
  replace?: boolean
  task_id?: string
  exit_code?: number
  state?: string
  message?: string
}

export function useTaskStream(taskId: string | (() => string)) {
  const messages = ref<string[]>([])
  const connected = ref(false)
  const done = ref(false)
  const exitCode = ref<number | null>(null)
  let ws: WebSocket | null = null
  let reconnectTimer = 0
  let reconnectAttempt = 0
  let manuallyClosed = false
  let connectionVersion = 0
  let replayRemaining = 0

  const resolvedId = typeof taskId === 'function' ? taskId : () => taskId

  // rAF batching — coalesce rapid WS messages into a single DOM update
  const pendingLines: Array<{ line: string; replace: boolean }> = []
  let rafId = 0

  function flushPending() {
    rafId = 0
    if (pendingLines.length === 0) return
    const batch = pendingLines.splice(0)
    for (const { line, replace } of batch) {
      if (replace && messages.value.length > 0) {
        messages.value.splice(messages.value.length - 1, 1, line)
      } else {
        messages.value.push(line)
      }
    }
  }

  function enqueueLine(line: string, replace = false) {
    pendingLines.push({ line, replace })
    if (!rafId) {
      rafId = requestAnimationFrame(flushPending)
    }
  }

  function closeSocket() {
    if (!ws) return
    ws.onclose = null
    ws.onerror = null
    ws.close()
    ws = null
  }

  function scheduleReconnect(id: string, version: number) {
    if (manuallyClosed || done.value || reconnectTimer || version !== connectionVersion) return
    const delays = [1000, 2000, 5000]
    const delay = delays[Math.min(reconnectAttempt, delays.length - 1)]
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = 0
      reconnectAttempt += 1
      void open(id, version)
    }, delay)
  }

  async function open(id: string, version: number) {
    if (!id || manuallyClosed || version !== connectionVersion) return
    closeSocket()

    let historyCount = 0
    try {
      const res = await fetch(`/api/tasks/${id}/output`)
      if (res.ok) {
        const data = await res.json()
        if (Array.isArray(data.lines)) {
          historyCount = data.lines.length
          pendingLines.length = 0
          if (rafId) cancelAnimationFrame(rafId)
          rafId = 0
          messages.value = [...data.lines]
        }
        if (data.state && data.state !== 'running' && data.state !== 'pending') {
          done.value = true
          exitCode.value = data.exit_code ?? null
        }
      }
    } catch {
      // The WebSocket may still recover this tick.
    }

    if (manuallyClosed || done.value || version !== connectionVersion) return
    replayRemaining = historyCount
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    ws = new WebSocket(`${protocol}//${location.host}/ws/tasks/${id}`)

    ws.onopen = () => {
      reconnectAttempt = 0
      connected.value = true
    }
    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data)
        if (msg.type === 'log' && msg.line) {
          if (replayRemaining > 0) {
            replayRemaining--
            return
          }
          enqueueLine(msg.line, msg.replace === true)
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
      ws = null
      scheduleReconnect(id, version)
    }
    ws.onerror = () => {
      connected.value = false
      ws?.close()
    }
  }

  function disconnect() {
    connectionVersion += 1
    manuallyClosed = true
    if (reconnectTimer) clearTimeout(reconnectTimer)
    reconnectTimer = 0
    closeSocket()
    connected.value = false
    pendingLines.length = 0
    if (rafId) cancelAnimationFrame(rafId)
    rafId = 0
  }

  function reconnect(id: string) {
    disconnect()
    manuallyClosed = false
    reconnectAttempt = 0
    messages.value = []
    done.value = false
    exitCode.value = null
    const version = connectionVersion
    void open(id, version)
  }

  // Reconnect whenever the task ID changes
  watch(resolvedId, (id) => {
    reconnect(id)
  }, { immediate: true })

  onUnmounted(disconnect)

  return { messages, connected, done, exitCode, disconnect }
}
