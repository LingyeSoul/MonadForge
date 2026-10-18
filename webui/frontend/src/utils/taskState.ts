/** A paused job still owns its task and can continue without a checkpoint. */
export function isTaskActive(state: string): boolean {
  return ['queued', 'pending', 'running', 'paused', 'stopping'].includes(state)
}

export function isTaskStoppable(state: string): boolean {
  return ['queued', 'pending', 'running', 'paused'].includes(state)
}

export function isTaskTerminal(state: string): boolean {
  return ['success', 'failed', 'cancelled'].includes(state)
}

export function canResumeTask(task: { state: string; resumable?: boolean }): boolean {
  return task.state === 'paused' || Boolean(task.resumable && ['failed', 'cancelled'].includes(task.state))
}
