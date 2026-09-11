import { computed, onBeforeUnmount, ref, watch, type Ref } from 'vue'
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
  reason_key?: string | null
  reason_params?: Record<string, string | number> | null
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

  /** Localize a daemon-side unavailability reason when it carries an i18n key. */
  function reasonLabel(candidate: ContinuationCandidate | null | undefined): string | null {
    if (!candidate?.reason) return null
    if (candidate.reason_key) {
      return t(`cfgContinuationReason_${candidate.reason_key}`, candidate.reason_params || {})
    }
    return candidate.reason
  }

  async function request(url: string, init?: RequestInit) {
    const res = await fetch(url, init)
    const data = await res.json().catch(() => ({} as Record<string, unknown>))
    if (!res.ok) {
      // Continuation endpoints return {message, key, params} on failure so the
      // toast follows the UI language instead of the daemon's Chinese text.
      const detail: unknown = (data as { detail?: unknown }).detail
      if (detail && typeof detail === 'object') {
        const { message, key, params } = detail as {
          message?: string; key?: string; params?: Record<string, string | number>
        }
        throw new Error(
          (key && t(`cfgContinuationReason_${key}`, params || {})) || message || `HTTP ${res.status}`
        )
      }
      throw new Error(
        (typeof detail === 'string' && detail) || `HTTP ${res.status}`
      )
    }
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
          config.error = reasonLabel(current) || t('cfgContinuationChanged')
        }
      }
    } catch (e: any) {
      if (ticket !== listGeneration) return
      candidates.value = []
      // Without a selection the config page is still a plain new-training
      // form; a daemon hiccup must not turn into an error banner there.
      if (selected.value) {
        ready.value = false
        config.error = e.message
      }
    } finally {
      if (ticket === listGeneration) listLoading.value = false
    }
  }

  function draftKey() {
    return detail.value ? `monadforge-continuation-${detail.value.candidate.job_id}` : ''
  }

  function rememberBudget() {
    if (detail.value && budgetKey.value && ready.value) budgets.set(draftKey(), { value: config.getFieldValue(budgetKey.value), dirty: config.dirty })
  }

  function exit() {
    rememberBudget()
    ++generation
    ++listGeneration
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
      if (!data.candidate.available) throw new Error(reasonLabel(data.candidate) || t('cfgContinuationUnavailable'))
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
    // The attempt is queued; hand the page back to the new-training draft so
    // the (now running) task cannot strand the selector in a locked state.
    ready.value = false
    exit()
    return result
  }

  // Field descriptions arrive pre-localized with the detail fetch; a language
  // switch while a task is selected needs a re-fetch to follow along.
  watch(() => app.language, () => {
    if (selected.value && !loading.value) void select(selected.value)
  })

  onBeforeUnmount(() => { ++listGeneration; exit() })
  return { selected, candidates, detail, loading, listLoading, ready, active,
    budgetKey, presetItems, reasonLabel, fetchCandidates, select, exit, saveDraft, submit }
}
