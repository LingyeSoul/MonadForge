import { computed, ref, watch } from 'vue'
import { defineStore } from 'pinia'
import { useTaskStore, type TaskInfo } from './task'

export interface ResolutionStage {
  resolution: number
  ratio: number
  batch_size: number
  num_repeats: number
}

export interface StagedResolutionPlan {
  version: number
  method: string
  variant: string
  preset: string
  source_image_dir: string
  max_train_steps: number
  stages: ResolutionStage[]
}

export interface StageReadiness extends ResolutionStage {
  resized_dir: string
  cache_dir: string
  resized: number
  latents: number
  text_embeddings: number
  ready: boolean
}

export interface StagedResolutionStatus {
  profile: string
  source_image_dir: string
  source_exists: boolean
  source_images: number
  captions: number
  stages: StageReadiness[]
  all_ready: boolean
}

export function defaultStagedPlan(): StagedResolutionPlan {
  return {
    version: 1,
    method: 'lora',
    variant: 'lora',
    preset: 'default',
    source_image_dir: 'image_dataset',
    max_train_steps: 6000,
    stages: [
      { resolution: 512, ratio: 20, batch_size: 4, num_repeats: 1 },
      { resolution: 768, ratio: 30, batch_size: 2, num_repeats: 1 },
      { resolution: 1024, ratio: 50, batch_size: 1, num_repeats: 1 },
    ],
  }
}

async function responseDetail(res: Response): Promise<string> {
  try {
    const body = await res.json()
    return body.detail || `HTTP ${res.status}`
  } catch {
    return `HTTP ${res.status}`
  }
}

export const useStagedResolutionStore = defineStore('stagedResolution', () => {
  const taskStore = useTaskStore()
  const selectedProfileName = ref('default')
  const loadedProfileName = ref('')
  const profileName = selectedProfileName
  const profiles = ref<string[]>([])
  const plan = ref<StagedResolutionPlan>(defaultStagedPlan())
  const status = ref<StagedResolutionStatus | null>(null)
  const persisted = ref(false)
  const loading = ref(false)
  const saving = ref(false)
  const launching = ref<'preprocess' | 'train' | ''>('')
  const preprocessTaskId = ref<string | null>(null)
  const preprocessTaskState = ref<TaskInfo['state'] | null>(null)
  const error = ref('')
  const savedSnapshot = ref(JSON.stringify(plan.value))

  const profileAligned = computed(() =>
    selectedProfileName.value.trim() === loadedProfileName.value,
  )
  const dirty = computed(() =>
    !persisted.value ||
    !profileAligned.value ||
    JSON.stringify(plan.value) !== savedSnapshot.value,
  )

  watch(selectedProfileName, (name) => {
    if (name.trim() !== loadedProfileName.value) status.value = null
  }, { flush: 'sync' })

  async function fetchProfiles() {
    const res = await fetch('/api/staged-resolution/profiles')
    if (!res.ok) throw new Error(await responseDetail(res))
    const body = await res.json()
    profiles.value = Array.isArray(body.profiles) ? body.profiles : []
  }

  async function loadProfile(name = selectedProfileName.value) {
    loading.value = true
    error.value = ''
    const clean = name.trim() || 'default'
    selectedProfileName.value = clean
    status.value = null
    try {
      const res = await fetch(`/api/staged-resolution/profiles/${encodeURIComponent(clean)}`)
      if (!res.ok) throw new Error(await responseDetail(res))
      const body = await res.json()
      if (selectedProfileName.value.trim() !== clean) return
      selectedProfileName.value = body.name
      loadedProfileName.value = body.name
      plan.value = body.plan
      persisted.value = body.persisted === true
      savedSnapshot.value = JSON.stringify(body.plan)
      await fetchStatus(body.name)
    } catch (e) {
      error.value = String(e)
      throw e
    } finally {
      loading.value = false
    }
  }

  function newProfile(name: string) {
    selectedProfileName.value = name.trim()
    loadedProfileName.value = selectedProfileName.value
    plan.value = defaultStagedPlan()
    status.value = null
    persisted.value = false
    savedSnapshot.value = ''
    error.value = ''
  }

  async function saveProfile() {
    saving.value = true
    error.value = ''
    try {
      if (!profileAligned.value) throw new Error('Load the selected profile before saving')
      const name = selectedProfileName.value.trim()
      const res = await fetch(`/api/staged-resolution/profiles/${encodeURIComponent(name)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(plan.value),
      })
      if (!res.ok) throw new Error(await responseDetail(res))
      const body = await res.json()
      selectedProfileName.value = body.name
      loadedProfileName.value = body.name
      plan.value = body.plan
      persisted.value = body.persisted === true
      savedSnapshot.value = JSON.stringify(body.plan)
      await Promise.all([fetchProfiles(), fetchStatus()])
    } catch (e) {
      error.value = String(e)
      throw e
    } finally {
      saving.value = false
    }
  }

  async function fetchStatus(name = loadedProfileName.value) {
    const target = name.trim()
    if (!target || selectedProfileName.value.trim() !== target || loadedProfileName.value !== target) return
    try {
      const res = await fetch(`/api/staged-resolution/profiles/${encodeURIComponent(target)}/status`)
      if (!res.ok) throw new Error(await responseDetail(res))
      const body = await res.json()
      if (selectedProfileName.value.trim() === target && loadedProfileName.value === target) {
        status.value = body
      }
    } catch (e) {
      status.value = null
      error.value = String(e)
    }
  }

  async function start(action: 'preprocess' | 'train'): Promise<string | null> {
    launching.value = action
    error.value = ''
    try {
      if (!profileAligned.value) throw new Error('Load the selected profile before starting')
      if (dirty.value) await saveProfile()
      const name = selectedProfileName.value.trim()
      const res = await fetch(
        `/api/staged-resolution/profiles/${encodeURIComponent(name)}/${action}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ version: 1 }),
        },
      )
      if (!res.ok) throw new Error(await responseDetail(res))
      const body = await res.json()
      await taskStore.fetchTasks()
      await taskStore.fetchQueueStatus()
      const taskId = body.task_id || null
      if (action === 'preprocess' && taskId) {
        preprocessTaskId.value = taskId
        preprocessTaskState.value =
          taskStore.tasks.find(task => task.task_id === taskId)?.state || 'pending'
      }
      return taskId
    } catch (e) {
      error.value = String(e)
      return null
    } finally {
      launching.value = ''
    }
  }

  async function pollPreprocessTask() {
    const taskId = preprocessTaskId.value
    if (!taskId) return
    await taskStore.fetchTasks()
    const task = taskStore.tasks.find(item => item.task_id === taskId)
    if (!task) return
    preprocessTaskState.value = task.state
    if (task.state === 'pending' || task.state === 'running' || task.state === 'stopping') {
      await fetchStatus()
      return
    }
    await fetchStatus()
    preprocessTaskId.value = null
  }

  return {
    profileName,
    selectedProfileName,
    loadedProfileName,
    profiles,
    plan,
    status,
    persisted,
    loading,
    saving,
    launching,
    preprocessTaskId,
    preprocessTaskState,
    error,
    profileAligned,
    dirty,
    fetchProfiles,
    loadProfile,
    newProfile,
    saveProfile,
    fetchStatus,
    start,
    pollPreprocessTask,
  }
})
