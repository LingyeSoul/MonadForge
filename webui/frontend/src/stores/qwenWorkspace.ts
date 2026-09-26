import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import {
  cacheStatus, checkedValues, draftKey, initialDraft, jobId, parseSchema, profileNames, readDraft, requestJson,
  type CacheStatus, type Phase, type QwenDraft, type QwenSchema, type Scalar, type Values,
} from '../utils/qwen21'

export type QwenSection = keyof QwenDraft
export const useQwenWorkspace = defineStore('qwen-workspace', () => {
  const schema = ref<QwenSchema | null>(null)
  const draft = ref<QwenDraft | null>(null)
  const saved = ref('')
  const profile = ref(localStorage.getItem('monadforge-qwen-profile') ?? 'default')
  const profiles = ref<string[]>([])
  const error = ref('')
  const loading = ref(false)
  const busy = ref(false)
  const cache = ref<CacheStatus | null>(null)
  const legacyAvailable = ref(localStorage.getItem(draftKey) !== null)
  const dirty = computed(() => draft.value !== null && JSON.stringify(draft.value) !== saved.value)
  let pending: Promise<void> | null = null

  function update(section: QwenSection, name: string, value: Scalar): void {
    if (!draft.value) throw new Error('Qwen workspace has not loaded')
    draft.value = { ...draft.value, [section]: { ...draft.value[section], [name]: value } }
    cache.value = null
  }
  function values(phase: Phase): Values {
    if (!schema.value || !draft.value) throw new Error('Qwen workspace has not loaded')
    return checkedValues(schema.value[phase], { ...draft.value[phase], ...draft.value.models })
  }
  async function loadProfile(name: string): Promise<void> {
    if (!schema.value) throw new Error('Qwen schema has not loaded')
    const result = await requestJson(`/api/qwen21/profiles/${encodeURIComponent(name)}`, {})
    const parsed = readDraft(schema.value, JSON.stringify(result))
    localStorage.setItem('monadforge-qwen-profile', name)
    draft.value = parsed
    saved.value = JSON.stringify(parsed)
    profile.value = name
    cache.value = null
  }
  async function initialize(): Promise<void> {
    loading.value = true
    try {
      schema.value = parseSchema(await requestJson('/api/qwen21/schema', {}))
      profiles.value = profileNames(await requestJson('/api/qwen21/profiles', {}))
      await loadProfile(profile.value)
    } finally { loading.value = false }
  }
  async function ensureLoaded(): Promise<void> {
    if (draft.value) return
    if (!pending) pending = initialize().finally(() => { pending = null })
    await pending
  }
  async function save(): Promise<void> {
    if (!draft.value || !schema.value) throw new Error('Qwen workspace has not loaded')
    const body = await requestJson(`/api/qwen21/profiles/${encodeURIComponent(profile.value)}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(draft.value),
    })
    draft.value = readDraft(schema.value, JSON.stringify(body))
    saved.value = JSON.stringify(draft.value)
    localStorage.setItem('monadforge-qwen-profile', profile.value)
    profiles.value = profileNames(await requestJson('/api/qwen21/profiles', {}))
  }
  function createProfile(name: string): void {
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) throw new Error('Profile name: 1–64 letters, digits, underscores or hyphens')
    if (profiles.value.includes(name)) throw new Error(`Profile already exists: ${name}`)
    profile.value = name
    saved.value = ''
  }
  function discard(): void {
    if (!schema.value) throw new Error('Qwen schema has not loaded')
    draft.value = saved.value ? readDraft(schema.value, saved.value) : initialDraft(schema.value)
    if (!saved.value) saved.value = JSON.stringify(draft.value)
    cache.value = null
  }
  function importLegacy(): void {
    if (!schema.value) throw new Error('Qwen schema has not loaded')
    const text = localStorage.getItem(draftKey)
    if (text === null) throw new Error('No legacy Qwen form is stored')
    draft.value = readDraft(schema.value, text)
  }
  function resetDefaults(): void {
    if (!schema.value) throw new Error('Qwen schema has not loaded')
    draft.value = initialDraft(schema.value)
    cache.value = null
  }
  async function inspectCache(): Promise<void> {
    cache.value = cacheStatus(await requestJson('/api/qwen21/cache-status', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values: values('cache') }),
    }))
  }
  async function preflight(): Promise<CacheStatus> {
    return cacheStatus(await requestJson('/api/qwen21/preflight', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values: values('train') }),
    }))
  }
  async function submit(phase: Phase, overrides: Values): Promise<string> {
    return jobId(await requestJson(`/api/qwen21/jobs/${phase}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values: { ...values(phase), ...overrides } }),
    }))
  }
  async function chain(): Promise<string> {
    return jobId(await requestJson('/api/qwen21/chain', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cache: { values: values('cache') }, train: { values: values('train') } }),
    }))
  }
  async function perform(action: () => Promise<void>): Promise<void> {
    error.value = ''
    busy.value = true
    try { await action() }
    catch (err) { error.value = err instanceof Error ? err.message : String(err) }
    finally { busy.value = false }
  }
  return { schema, draft, profile, profiles, dirty, loading, busy, error, cache, legacyAvailable,
    update, values, loadProfile, ensureLoaded, save, createProfile, discard, importLegacy, resetDefaults,
    inspectCache, preflight, submit, chain, perform }
})
