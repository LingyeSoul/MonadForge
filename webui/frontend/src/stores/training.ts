import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface TrainingMetrics {
  step: number
  total_steps: number
  epoch: number
  total_epochs: number
  avr_loss: number
  loss_history: number[]
  step_history: number[]
  lr: number
  lr_history: number[]
  speed: string
  elapsed: string
  eta: string
  router_h: number | null
  keys_scaled: number | null
  avg_key_norm: number | null
  checkpoint_saved: boolean
  events: TrainingEvent[]
}

export interface TrainingEvent {
  type: 'epoch' | 'checkpoint'
  step: number
  epoch: number
  elapsed: string
  total_epochs?: number
  detail?: string
}

const emptyMetrics: TrainingMetrics = {
  step: 0,
  total_steps: 0,
  epoch: 0,
  total_epochs: 0,
  avr_loss: 0,
  loss_history: [],
  step_history: [],
  lr: 0,
  lr_history: [],
  speed: '',
  elapsed: '',
  eta: '',
  router_h: null,
  keys_scaled: null,
  avg_key_norm: null,
  checkpoint_saved: false,
  events: [],
}

export const useTrainingStore = defineStore('training', () => {
  const metrics = ref<TrainingMetrics>({ ...emptyMetrics })
  const connected = ref(false)
  const done = ref(false)

  function updateFromWs(data: Partial<TrainingMetrics>) {
    Object.assign(metrics.value, data)
  }

  function reset() {
    Object.assign(metrics.value, { ...emptyMetrics, events: [], loss_history: [], step_history: [], lr_history: [] })
    connected.value = false
    done.value = false
  }

  async function loadFromRest(taskId: string) {
    try {
      const res = await fetch(`/api/tasks/${taskId}/metrics`)
      if (res.ok) {
        const data = await res.json()
        Object.assign(metrics.value, data)
      }
    } catch {
      // ignore — WS will catch up
    }
  }

  return { metrics, connected, done, updateFromWs, reset, loadFromRest }
})
