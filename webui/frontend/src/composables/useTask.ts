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

  function _connect(id: string) {
    if (!id) return
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    ws = new WebSocket(`${protocol}//${location.host}/ws/tasks/${id}`)

    ws.onopen = () => {
      connected.value = true
    }
    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data)
        if (msg.type === 'log' && msg.line) {
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

  function reconnect(id: string) {
    disconnect()
    messages.value = []
    done.value = false
    exitCode.value = null
    _connect(id)
  }

  // Reconnect whenever the task ID changes
  watch(resolvedId, (id) => {
    reconnect(id)
  }, { immediate: true })

  onUnmounted(disconnect)

  return { messages, connected, done, exitCode, disconnect }
}
