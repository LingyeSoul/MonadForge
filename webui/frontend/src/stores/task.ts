import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useNotifyStore } from './notify'
import { useI18n } from '../composables/useI18n'

export interface TaskInfo {
  task_id: string
  command: string
  state: 'pending' | 'running' | 'success' | 'failed' | 'cancelled'
  pid: number | null
  started_at: string | null
  output_lines: number
  category: 'training' | 'task'
  // Set by fetchQueueStatus (only meaningful for state === 'pending'): how
  // many queued jobs finish before this one starts (1, 2, … in FIFO order).
  // null/undefined for running or terminal tasks.
  queue_position?: number | null
}

export const useTaskStore = defineStore('task', () => {
  const tasks = ref<TaskInfo[]>([])
  const commands = ref<Record<string, string>>({})
  const loading = ref(false)
  // Queue state — driven by GET /api/tasks/queue/status (polled alongside
  // fetchTasks in the Tasks view).
  const daemonPaused = ref(false)
  const daemonUp = ref(true)
  // Tracks the daemon_up transition so we only toast once per down-edge
  // (polls every 5s — without this the toast would fire on every tick).
  const daemonWasDown = ref(false)

  const notify = useNotifyStore()
  const { t } = useI18n()

  async function fetchTasks() {
    try {
      const res = await fetch('/api/tasks')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      if (Array.isArray(data)) tasks.value = data
    } catch {
      // silently ignore
    }
  }

  async function fetchCommands() {
    try {
      const res = await fetch('/api/tasks/commands')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      commands.value = data.commands || {}
    } catch {
      // silently ignore
    }
  }

  async function fetchQueueStatus() {
    try {
      const res = await fetch('/api/tasks/queue/status')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const up = data.daemon_up !== false
      daemonUp.value = up
      // Toast once when the daemon transitions up → down (the key exists for
      // this; without the edge-guard the toast would repeat every 5s poll).
      if (!up && !daemonWasDown.value) {
        notify.show(t('notifyDaemonDown'), 'warning')
      }
      daemonWasDown.value = !up
      daemonPaused.value = !!data.paused
      // Merge positions onto the matching tasks. Only queued tasks carry a
      // position; running/terminal tasks get cleared so a job that just left
      // the queue doesn't keep a stale #N.
      const positions: Record<string, number> = data.positions || {}
      tasks.value = tasks.value.map((task) => ({
        ...task,
        queue_position: positions[task.task_id] ?? null,
      }))
    } catch {
      // daemon unreachable on this tick — leave the last known state.
    }
  }

  async function pauseQueue() {
    try {
      const res = await fetch('/api/tasks/queue/pause', { method: 'POST' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      daemonPaused.value = true
      notify.show(t('notifyQueuePaused'), 'info')
      await fetchQueueStatus()
    } catch {
      notify.show(t('notifyQueuePauseFailed'), 'error')
    }
  }

  async function resumeQueue() {
    try {
      const res = await fetch('/api/tasks/queue/resume', { method: 'POST' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      daemonPaused.value = false
      notify.show(t('notifyQueueResumed'), 'success')
      await fetchQueueStatus()
    } catch {
      notify.show(t('notifyQueueResumeFailed'), 'error')
    }
  }

  async function shutdownDaemon(killJobs: boolean = true): Promise<boolean> {
    try {
      await fetch('/api/tasks/daemon/shutdown', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kill_jobs: killJobs }),
      })
      // A clean response is best-case — but the daemon may host the WebUI as a
      // sidecar and tree-kill this server on shutdown, so a network error here
      // is also an expected outcome. Treat both as success.
      notify.show(t('notifyDaemonShutdown'), 'info')
      daemonUp.value = false
      return true
    } catch {
      // Connection reset = the daemon (and possibly this server) went down —
      // the shutdown did fire.
      notify.show(t('notifyDaemonShutdown'), 'info')
      daemonUp.value = false
      return true
    }
  }

  async function startTask(command: string, args: string[] = [], env?: Record<string, string>): Promise<string | null> {
    loading.value = true
    try {
      const res = await fetch('/api/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command, args, env: env || {} }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      await fetchTasks()
      await fetchQueueStatus()
      return data.task_id || null
    } catch {
      return null
    } finally {
      loading.value = false
    }
  }

  async function cancelTask(taskId: string) {
    try {
      const res = await fetch(`/api/tasks/${taskId}`, { method: 'DELETE' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      await fetchTasks()
      await fetchQueueStatus()
    } catch {
      // silently ignore
    }
  }

  async function poll() {
    await Promise.all([fetchTasks(), fetchQueueStatus()])
  }

  return {
    tasks,
    commands,
    loading,
    daemonPaused,
    daemonUp,
    fetchTasks,
    fetchCommands,
    fetchQueueStatus,
    pauseQueue,
    resumeQueue,
    shutdownDaemon,
    startTask,
    cancelTask,
    poll,
  }
})
