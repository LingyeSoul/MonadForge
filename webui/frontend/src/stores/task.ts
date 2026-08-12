import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useNotifyStore } from './notify'
import { useI18n } from '../composables/useI18n'

export interface TaskAttempt {
  job_id: string
  attempt_index: number
  state: string
  started_at: string | null
  ended_at: string | null
  recovery_step: number | null
  exit_code: number | null
  terminal_reason: string | null
}

export interface TaskInfo {
  task_id: string
  current_job_id: string | null
  attempt_count: number
  attempts: TaskAttempt[]
  command: string
  state: 'pending' | 'running' | 'stopping' | 'success' | 'failed' | 'cancelled'
  pid: number | null
  started_at: string | null
  output_lines: number
  category: 'training' | 'task'
  // Set by fetchQueueStatus (only meaningful for state === 'pending'): how
  // many queued jobs finish before this one starts (1, 2, … in FIFO order).
  // null/undefined for running or terminal tasks.
  queue_position?: number | null
  recovery_step?: number | null
  resume_state?: string | null
  terminal_reason?: string | null
  resumable?: boolean
  legacy?: boolean
  exit_code?: number | null
  last_progress?: {
    step?: number | null
    total_steps?: number | null
    epoch?: number | null
  }
}

export type TaskFilter = 'all' | 'active' | 'success' | 'failed' | 'cancelled'

export interface TaskPage {
  tasks: TaskInfo[]
  total: number
}

export const useTaskStore = defineStore('task', () => {
  const tasks = ref<TaskInfo[]>([])
  const commands = ref<Record<string, string>>({})
  const loading = ref(false)
  const taskListError = ref(false)
  // Queue state — driven by GET /api/tasks/queue/status (polled alongside
  // fetchTasks in the Tasks view).
  const daemonPaused = ref(false)
  const daemonUp = ref(true)
  const queuePositions = ref<Record<string, number>>({})
  // Tracks the daemon_up transition so we only toast once per down-edge
  // (polls every 5s — without this the toast would fire on every tick).
  const daemonWasDown = ref(false)

  const notify = useNotifyStore()
  const { t } = useI18n()

  async function fetchTaskPage(options: {
    state?: TaskFilter
    page?: number
    pageSize?: number
  } = {}): Promise<TaskPage> {
    const page = Math.max(1, options.page ?? 1)
    const pageSize = Math.max(1, options.pageSize ?? 25)
    const params = new URLSearchParams({
      limit: String(pageSize),
      offset: String((page - 1) * pageSize),
    })
    if (options.state && options.state !== 'all') params.set('state', options.state)
    const res = await fetch(`/api/tasks?${params.toString()}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    if (!Array.isArray(data)) throw new Error('invalid task page')
    const rawTotal = res.headers.get('X-Total-Count')
    const parsedTotal = rawTotal === null ? data.length : Number(rawTotal)
    return {
      tasks: data.map((task: TaskInfo) => ({
        ...task,
        queue_position: queuePositions.value[task.task_id] ?? null,
      })),
      total: Number.isFinite(parsedTotal) ? parsedTotal : data.length,
    }
  }

  async function fetchTasks() {
    try {
      const page = await fetchTaskPage({ page: 1, pageSize: 500 })
      tasks.value = page.tasks
      taskListError.value = false
    } catch {
      tasks.value = []
      taskListError.value = true
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
      queuePositions.value = positions
      tasks.value = tasks.value.map((task) => ({
        ...task,
        queue_position: positions[task.task_id] ?? null,
      }))
    } catch {
      daemonUp.value = false
      daemonPaused.value = false
      queuePositions.value = {}
      if (!daemonWasDown.value) {
        notify.show(t('notifyDaemonDown'), 'warning')
      }
      daemonWasDown.value = true
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

  async function shutdownDaemon(modeOrKill: boolean | 'detach' | 'cooperative-stop' | 'force' = true): Promise<boolean> {
    const mode = typeof modeOrKill === 'boolean'
      ? (modeOrKill ? 'force' : 'detach')
      : modeOrKill
    try {
      await fetch('/api/tasks/daemon/shutdown', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kill_jobs: mode === 'force', mode }),
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

  async function cancelTask(taskId: string): Promise<boolean> {
    try {
      const res = await fetch(`/api/tasks/${taskId}`, { method: 'DELETE' })
      if (!res.ok) {
        let detail = `HTTP ${res.status}`
        try {
          const body = await res.json()
          if (body?.detail) detail = body.detail
        } catch {
          // Keep the HTTP status when the server did not return JSON.
        }
        throw new Error(detail)
      }
      tasks.value = tasks.value.map((task) =>
        task.task_id === taskId ? { ...task, state: 'stopping' } : task,
      )
      notify.show(t('notifyTaskStopping'), 'info')
      await fetchTasks()
      await fetchQueueStatus()
      return true
    } catch {
      notify.show(t('notifyTaskCancelFailed'), 'error')
      return false
    }
  }

  async function resumeTask(taskId: string, sourceTask?: TaskInfo): Promise<boolean> {
    const source = sourceTask ?? tasks.value.find((task) => task.task_id === taskId)
    if (source?.legacy && !window.confirm(t('taskLegacyResumeConfirm'))) return false
    let failureDetail = ''
    try {
      const res = await fetch(`/api/tasks/${taskId}/resume`, { method: 'POST' })
      if (!res.ok) {
        failureDetail = `HTTP ${res.status}`
        try {
          const body = await res.json()
          if (body?.detail) failureDetail = String(body.detail)
        } catch {
          // Keep the HTTP status when the response is not JSON.
        }
        throw new Error(failureDetail)
      }
      await poll()
      notify.show(t('notifyTaskResumed'), 'success')
      return true
    } catch {
      notify.show(
        failureDetail ? `${t('notifyTaskResumeFailed')}: ${failureDetail}` : t('notifyTaskResumeFailed'),
        'error',
        6000,
      )
      return false
    }
  }

  async function deleteHistory(taskId: string): Promise<boolean> {
    if (!window.confirm(t('taskDeleteHistoryConfirm'))) return false
    try {
      const res = await fetch(`/api/tasks/${taskId}/history`, { method: 'DELETE' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      tasks.value = tasks.value.filter((task) => task.task_id !== taskId)
      await fetchTasks()
      notify.show(t('notifyTaskDeleted'), 'success')
      return true
    } catch {
      notify.show(t('notifyTaskDeleteFailed'), 'error')
      return false
    }
  }

  async function poll() {
    await Promise.all([fetchTasks(), fetchQueueStatus()])
  }

  return {
    tasks,
    commands,
    loading,
    taskListError,
    daemonPaused,
    daemonUp,
    fetchTasks,
    fetchTaskPage,
    fetchCommands,
    fetchQueueStatus,
    pauseQueue,
    resumeQueue,
    shutdownDaemon,
    startTask,
    cancelTask,
    resumeTask,
    deleteHistory,
    poll,
  }
})
