import { ref, onUnmounted } from 'vue'

export function useTaskStream(taskId: string) {
  const messages = ref<string[]>([])
  const connected = ref(false)
  let ws: WebSocket | null = null

  function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    ws = new WebSocket(`${protocol}//${location.host}/ws/tasks/${taskId}`)

    ws.onopen = () => {
      connected.value = true
    }
    ws.onmessage = (event) => {
      messages.value.push(event.data)
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
  }

  onUnmounted(disconnect)

  return { messages, connected, connect, disconnect }
}
