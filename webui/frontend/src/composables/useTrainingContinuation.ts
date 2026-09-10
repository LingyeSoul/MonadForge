import { computed, onBeforeUnmount, ref, type Ref } from 'vue'
import { useAppStore } from '../stores/app'
import { useI18n } from './useI18n'
import { useConfigStore, type FieldMeta } from '../stores/config'

export interface ContinuationCandidate {
  task_id: string
  job_id: string
  name: string
  variant: string
  preset: string
  state: string
  submitted_at: number
  step: number
  epoch: number
  available: boolean
  reason: string | null
  budget_key: 'max_train_epochs' | 'max_train_steps'
  target: number
}

interface ContinuationDetail {
  candidate: ContinuationCandidate
  fields: FieldMeta[]
}

const copy = <T>(value: T): T => JSON.parse(JSON.stringify(value))

/** Keeps continuation drafts separate from the editable preset/variant. */
export function useTrainingContinuation(preset: Ref<string>, extraArgs: Ref<string>) {
  const config = useConfigStore()
  const { t } = useI18n()
  const app = useAppStore()
  const budgets = new Map<string, { value: unknown; dirty: boolean }>()
  const selected = ref<string | null>(null)
  const candidates = ref<ContinuationCandidate[]>([])
  const detail = ref<ContinuationDetail | null>(null)
  const loading = ref(false)
  const listLoading = ref(false)
  const ready = ref(false)
  let generation = 0
  let listGeneration = 0
  let draft: {
    fields: FieldMeta[]; edited: Record<string, unknown>; dirty: boolean
    preset: string; variant: string; extra: string; error: string
  } | null = null

  const active = computed(() => selected.value !== null)
  const budgetKey = computed(() => detail.value?.candidate.budget_key)
  const presetItems = computed(() => [...new Set([...config.presets, preset.value])])

  async function request(url: string, init?: RequestInit) {
    const res = await fetch(url, init)
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
    return data
  }

  async function fetchCandidates(variant: string) {
    const ticket = ++listGeneration
    listLoading.value = true
    try {
      const data = await request(`/api/tasks/continuation-candidates?variant=${encodeURIComponent(variant)}`)
      if (ticket !== listGeneration) return
      candidates.value = data.candidates
      if (selected.value) {
        const current = candidates.value.find(c => c.task_id === selected.value)
        if (!current?.available || (detail.value && current.job_id !== detail.value.candidate.job_id)) {
          ready.value = false
          config.error = current?.reason || t('cfgContinuationChanged')
        }
      }
    } catch (e: any) {
      if (ticket !== listGeneration) return
      candidates.value = []
      if (selected.value) ready.value = false
      config.error = e.message
    } finally {
      if (ticket === listGeneration) listLoading.value = false
    }
  }

  function draftKey() {
    return detail.value ? `monadforge.continuation.${detail.value.candidate.job_id}` : ''
  }

  function rememberBudget() {
    if (detail.value && budgetKey.value && ready.value) budgets.set(draftKey(), { value: config.getFieldValue(budgetKey.value), dirty: config.dirty })
  }

  function exit() {
    rememberBudget()
    ++generation
    selected.value = null
    detail.value = null
    ready.value = false
    loading.value = false
    if (draft) {
      config.fields = draft.fields
      config.editedValues = draft.edited
      config.dirty = draft.dirty
      config.preset = draft.preset
      config.variant = draft.variant
      config.error = draft.error
      preset.value = draft.preset
      extraArgs.value = draft.extra
      draft = null
    }
  }

  async function select(taskId: string | null) {
    if (!taskId) { exit(); return }
    rememberBudget()
    if (!draft) {
      draft = copy({ fields: config.fields, edited: config.editedValues,
        dirty: config.dirty, preset: preset.value, variant: config.variant,
        extra: extraArgs.value, error: config.error })
    }
    const ticket = ++generation
    selected.value = taskId
    detail.value = null
    ready.value = false
    loading.value = true
    config.error = ''
    // Do not leave the previous task's editable budget visible during loading.
    config.fields = config.fields.map(f => ({ ...f, read_only: true }))
    config.editedValues = {}
    config.dirty = false
    extraArgs.value = ''
    try {
      const data: ContinuationDetail = await request(`/api/tasks/${encodeURIComponent(taskId)}/continuation?lang=${encodeURIComponent(app.language)}`)
      if (ticket !== generation) return
      if (!data.candidate.available) throw new Error(data.candidate.reason || t('cfgContinuationUnavailable'))
      detail.value = data
      preset.value = data.candidate.preset
      config.preset = data.candidate.preset
      config.fields = data.fields.map(f => ({ ...f, read_only: f.key !== data.candidate.budget_key }))
      config.variant = data.candidate.variant
      try {
        const memory = budgets.get(draftKey())
        const saved = Number(memory?.value ?? localStorage.getItem(draftKey()))
        if (Number.isSafeInteger(saved) && saved >= data.candidate.target) {
          config.editedValues = { [data.candidate.budget_key]: saved }
          config.dirty = memory?.dirty ?? false
        }
      } catch { /* Storage may be disabled; current edits still work. */ }
      ready.value = true
    } catch (e: any) {
      if (ticket === generation) config.error = e.message
    } finally {
      if (ticket === generation) loading.value = false
    }
  }

  function saveDraft() {
    if (!ready.value || !budgetKey.value) throw new Error(t('cfgContinuationNotReady'))
    const value = Number(config.getFieldValue(budgetKey.value))
    if (!Number.isSafeInteger(value) || value <= 0) throw new Error(t('cfgContinuationPositive'))
    localStorage.setItem(draftKey(), String(value))
    config.dirty = false
  }

  async function submit() {
    if (!ready.value || !detail.value || !selected.value) throw new Error(t('cfgContinuationNotReady'))
    const candidate = detail.value.candidate
    const value = Number(config.getFieldValue(candidate.budget_key))
    if (!Number.isSafeInteger(value) || value < candidate.target) throw new Error(t('cfgContinuationShrink'))
    if (candidate.state === 'done' && value <= candidate.target) throw new Error(t('cfgContinuationCompleted', { target: candidate.target }))
    const base = `/api/tasks/${encodeURIComponent(selected.value)}/continuation`
    const prepared = await request(`${base}/prepare`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_job_id: candidate.job_id, budget_key: candidate.budget_key, target: value }),
    })
    const result = await request(base, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: prepared.token, idempotency_key: prepared.token }),
    })
    ready.value = false
    config.dirty = false
    config.fields = config.fields.map(f => ({ ...f, read_only: true }))
    return result
  }

  onBeforeUnmount(() => { ++listGeneration; exit() })
  return { selected, candidates, detail, loading, listLoading, ready, active,
    budgetKey, presetItems, fetchCandidates, select, exit, saveDraft, submit }
}
