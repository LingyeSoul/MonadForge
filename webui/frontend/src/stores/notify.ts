import { defineStore } from 'pinia'
import { ref } from 'vue'

export type NotifyType = 'success' | 'info' | 'warning' | 'error'

export interface NotifyItem {
  id: number
  message: string
  type: NotifyType
  timeout: number
}

let nextId = 0

export const useNotifyStore = defineStore('notify', () => {
  const current = ref<NotifyItem | null>(null)
  const queue: NotifyItem[] = []

  function show(message: string, type: NotifyType = 'info', timeout = 3000) {
    const item: NotifyItem = { id: nextId++, message, type, timeout }
    if (!current.value) {
      current.value = item
    } else {
      queue.push(item)
    }
  }

  function dismiss() {
    current.value = queue.shift() || null
  }

  return { current, show, dismiss }
})
