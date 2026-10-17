import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { useModelWorkspace, modelLabel } from './modelWorkspace.ts'
import {
  cacheStatus, checkedValues, draftKey, initialDraft, jobId, parseSchema, profileNames, readDraft, requestJson,
  type CacheStatus, type Phase, type QwenDraft, type QwenSchema, type Scalar, type Values,
} from '../utils/qwen21.ts'

export type QwenSection = 'models' | Phase
function createModelWorkspace(family: 'qwen21' | 'moss') {
 return defineStore(`${family}-workspace`, () => {
  const apiRoot = `/api/${family}`
  const storageKey = family === 'qwen21' ? 'monadforge-qwen-profile' : 'monadforge-moss-profile'
  const schema = ref<QwenSchema | null>(null)
  const draft = ref<QwenDraft | null>(null)
  const saved = ref('')
  const profile = ref(localStorage.getItem(storageKey) ?? 'default')
  const profiles = ref<string[]>([])
  const error = ref('')
  const loading = ref(false)
  const busy = ref(false)
  const cache = ref<CacheStatus | null>(null)
  const legacyAvailable = ref(family === 'qwen21' && localStorage.getItem(draftKey) !== null)
  const dirty = computed(() => draft.value !== null && JSON.stringify(draft.value) !== saved.value)
  let pending: Promise<void> | null = null
  let loadSeq = 0

  function update(section: QwenSection, name: string, value: Scalar): void {
    if (!draft.value) throw new Error(`${modelLabel(family)} workspace has not loaded`)
    draft.value = { ...draft.value, [section]: { ...draft.value[section], [name]: value } }
    cache.value = null
  }
  function values(phase: Phase): Values {
    if (!schema.value || !draft.value) throw new Error(`${modelLabel(family)} workspace has not loaded`)
    return checkedValues(schema.value[phase], { ...draft.value[phase], ...draft.value.models })
  }
  async function loadProfile(name: string): Promise<void> {
    if (!schema.value) throw new Error(`${modelLabel(family)} schema has not loaded`)
    const seq = ++loadSeq
    const result = await requestJson(`${apiRoot}/profiles/${encodeURIComponent(name)}`, {})
    if (seq !== loadSeq) return // a newer switch already superseded this request
    const parsed = readDraft(schema.value, JSON.stringify(result))
    localStorage.setItem(storageKey, name)
    draft.value = parsed
    saved.value = JSON.stringify(parsed)
    profile.value = name
    cache.value = null
  }
  async function initialize(): Promise<void> {
    loading.value = true
    try {
      schema.value = parseSchema(await requestJson(`${apiRoot}/schema`, {}))
      profiles.value = profileNames(await requestJson(`${apiRoot}/profiles`, {}))
      await loadProfile(profile.value)
    } finally { loading.value = false }
  }
  async function ensureLoaded(): Promise<void> {
    if (draft.value) return
    if (!pending) pending = initialize().finally(() => { pending = null })
    await pending
  }
  async function save(): Promise<void> {
    if (!draft.value || !schema.value) throw new Error(`${modelLabel(family)} workspace has not loaded`)
    const body = await requestJson(`${apiRoot}/profiles/${encodeURIComponent(profile.value)}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(draft.value),
    })
    draft.value = readDraft(schema.value, JSON.stringify(body))
    saved.value = JSON.stringify(draft.value)
    localStorage.setItem(storageKey, profile.value)
    profiles.value = profileNames(await requestJson(`${apiRoot}/profiles`, {}))
  }
  function createProfile(name: string): void {
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) throw new Error('Profile name: 1–64 letters, digits, underscores or hyphens')
    if (profiles.value.includes(name)) throw new Error(`Profile already exists: ${name}`)
    profile.value = name
    saved.value = ''
  }
  function discard(): void {
    if (!schema.value) throw new Error(`${modelLabel(family)} schema has not loaded`)
    draft.value = saved.value ? readDraft(schema.value, saved.value) : initialDraft(schema.value)
    cache.value = null
  }
  function importLegacy(): void {
    if (!schema.value) throw new Error(`${modelLabel(family)} schema has not loaded`)
    const text = localStorage.getItem(draftKey)
    if (text === null) throw new Error('No legacy Qwen form is stored')
    draft.value = readDraft(schema.value, text)
  }
  function resetDefaults(): void {
    if (!schema.value) throw new Error(`${modelLabel(family)} schema has not loaded`)
    draft.value = initialDraft(schema.value)
    cache.value = null
  }
  async function inspectCache(): Promise<void> {
    cache.value = cacheStatus(await requestJson(`${apiRoot}/cache-status`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values: values('cache') }),
    }))
  }
  async function preflight(): Promise<CacheStatus> {
    return cacheStatus(await requestJson(`${apiRoot}/preflight`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values: values('train') }),
    }))
  }
  async function submit(phase: Phase, overrides: Values): Promise<string> {
    const runtime_profile = await useModelWorkspace().runtimeForSubmit()
    return jobId(await requestJson(`${apiRoot}/jobs/${phase}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values: { ...values(phase), ...overrides }, runtime_profile }),
    }))
  }
  async function chain(): Promise<string> {
    const runtime_profile = await useModelWorkspace().runtimeForSubmit()
    return jobId(await requestJson(`${apiRoot}/chain`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cache: { values: values('cache') }, train: { values: values('train') }, runtime_profile }),
    }))
  }
  async function perform(action: () => Promise<void>): Promise<void> {
    if (busy.value) return
    error.value = ''
    busy.value = true
    try { await action() }
    catch (err) { error.value = err instanceof Error ? err.message : String(err) }
    finally { busy.value = false }
  }
  return { family, apiRoot, label: modelLabel(family), schema, draft, profile, profiles, dirty, loading, busy, error, cache, legacyAvailable, saved,
    update, values, loadProfile, ensureLoaded, save, createProfile, discard, importLegacy, resetDefaults,
    inspectCache, preflight, submit, chain, perform }
})

}
export const useQwenWorkspace = createModelWorkspace('qwen21')
export const useMossWorkspace = createModelWorkspace('moss')
export function useActiveModelWorkspace() {
  const workspace = useModelWorkspace()
  const qwen = useQwenWorkspace()
  const moss = useMossWorkspace()
  return computed(() => workspace.family === 'moss' ? moss : qwen)
}
