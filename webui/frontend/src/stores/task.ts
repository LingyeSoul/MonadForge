import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface TaskInfo {
  task_id: string
  command: string
  state: 'pending' | 'running' | 'success' | 'failed' | 'cancelled'
  pid: number | null
  started_at: string
  output_lines: number
}

export interface CommandInfo {
  name: string
  description: string
  category: string
}

export const useTaskStore = defineStore('task', () => {
  const tasks = ref<TaskInfo[]>([])
  const commands = ref<CommandInfo[]>([])
  const loading = ref(false)

  async function fetchTasks() {
    try {
      const res = await fetch('/api/tasks')
      tasks.value = await res.json()
    } catch {
      // silently ignore
    }
  }

  async function fetchCommands() {
    try {
      const res = await fetch('/api/tasks/commands')
      const data = await res.json()
      commands.value = data.commands || []
    } catch {
      // silently ignore
    }
  }

  async function startTask(command: string, args: string[] = []): Promise<string | null> {
    loading.value = true
    try {
      const res = await fetch('/api/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command, args }),
      })
      const data = await res.json()
      await fetchTasks()
      return data.task_id || null
    } catch {
      return null
    } finally {
      loading.value = false
    }
  }

  async function cancelTask(taskId: string) {
    try {
      await fetch(`/api/tasks/${taskId}`, { method: 'DELETE' })
      await fetchTasks()
    } catch {
      // silently ignore
    }
  }

  async function poll() {
    await fetchTasks()
  }

  return { tasks, commands, loading, fetchTasks, fetchCommands, startTask, cancelTask, poll }
})
