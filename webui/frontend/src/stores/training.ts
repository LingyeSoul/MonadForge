import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface SampleInfo {
  path: string         // absolute on-disk path emitted by the training process
  filename: string     // basename used for the FileResponse URL
  step: number | null
  epoch: number | null
  prompt: string | null
  ts: number | null
  received_at: number  // client wall-clock ms when the event arrived
}

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
  wandb_run_url: string | null
  sample_history: SampleInfo[]
  latest_sample: SampleInfo | null
  sampling_enabled: boolean
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
  wandb_run_url: null,
  sample_history: [],
  latest_sample: null,
  sampling_enabled: false,
}

// Hard cap on the client-side gallery so a long run doesn't accumulate
// thousands of thumbnails. Older entries are dropped FIFO; the on-disk
// directory keeps the full set and can be browsed via the REST API.
const MAX_SAMPLE_HISTORY = 240

function basename(path: string): string {
  const i = path.lastIndexOf('/')
  const j = path.lastIndexOf('\\')
  return path.substring(Math.max(i, j) + 1)
}

export const useTrainingStore = defineStore('training', () => {
  const metrics = ref<TrainingMetrics>({ ...emptyMetrics })
  const connected = ref(false)
  const done = ref(false)

  // Set of on-disk paths we've already recorded — path is the natural
  // dedupe key (training process writes one PNG per sample event).
  const seenSamplePaths = new Set<string>()

  function recordSample(info: Omit<SampleInfo, 'filename' | 'received_at'>) {
    if (!info.path) return
    if (seenSamplePaths.has(info.path)) return
    seenSamplePaths.add(info.path)

    const sample: SampleInfo = {
      ...info,
      filename: basename(info.path),
      received_at: Date.now(),
    }
    metrics.value.sample_history.push(sample)
    metrics.value.latest_sample = sample

    // Drop oldest entries past the cap.
    const overflow = metrics.value.sample_history.length - MAX_SAMPLE_HISTORY
    if (overflow > 0) {
      const dropped = metrics.value.sample_history.splice(0, overflow)
      for (const d of dropped) seenSamplePaths.delete(d.path)
    }
  }

  function updateFromWs(data: Partial<TrainingMetrics>) {
    Object.assign(metrics.value, data)
  }

  function reset() {
    Object.assign(metrics.value, {
      ...emptyMetrics,
      events: [],
      loss_history: [],
      step_history: [],
      lr_history: [],
      sample_history: [],
    })
    seenSamplePaths.clear()
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

    // Backfill the sample history from disk so a late-joining viewer (or
    // a page reload) sees previews that were generated before the WS
    // connection was opened.
    try {
      const res = await fetch(
        `/api/preview/runs/${taskId}/samples?page=1&page_size=${MAX_SAMPLE_HISTORY}`,
      )
      if (!res.ok) return
      const data = await res.json()
      const items: any[] = data?.items || []
      // REST returns newest-first; reverse so sample_history is oldest-first
      // and ``latest_sample`` ends up pointing at the newest file.
      for (const it of items.slice().reverse()) {
        recordSample({
          path: it.path,
          step: null,
          epoch: null,
          prompt: null,
          ts: it.mtime_unix ? it.mtime_unix * 1000 : null,
        })
      }
    } catch {
      // ignore — gallery is best-effort
    }
  }

  return {
    metrics,
    connected,
    done,
    updateFromWs,
    recordSample,
    reset,
    loadFromRest,
  }
})
