import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { useModelWorkspace, modelLabel } from './modelWorkspace.ts'
import { modelActionValues } from '../utils/modelConfig.ts'
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
  // The configuration chooser is session state, independent of the profile
  // loaded in the background for model paths and preprocessing.
  const configMethod = ref('')
  const configProfile = ref('')
  const profileAlgorithms = ref<Record<string, string>>({})
  const error = ref('')
  const loading = ref(false)
  const busy = ref(false)
  const cache = ref<CacheStatus | null>(null)
  const legacyAvailable = ref(family === 'qwen21' && localStorage.getItem(draftKey) !== null)
  const dirty = computed(() => draft.value !== null && JSON.stringify(draft.value) !== saved.value)
  let pending: Promise<void> | null = null
  let loadSeq = 0
  let catalogSeq = 0

  function validateMethod(method: string): void {
    if (!schema.value?.train.find(field => field.name === 'algo')?.choices.includes(method)) {
      throw new Error(`Unsupported ${modelLabel(family)} method: ${method}`)
    }
  }
  function checkSelectedMethod(value: QwenDraft, method: string): void {
    if (method && value.train.algo !== method) {
      throw new Error(`Profile method ${String(value.train.algo)} does not match selected method ${method}`)
    }
  }

  function update(section: QwenSection, name: string, value: Scalar): void {
    if (!draft.value) throw new Error(`${modelLabel(family)} workspace has not loaded`)
    if (configProfile.value && section === 'train' && name === 'algo' && value !== configMethod.value) {
      throw new Error('Choose a method before selecting or creating its configuration')
    }
    draft.value = { ...draft.value, [section]: { ...draft.value[section], [name]: value } }
    cache.value = null
  }
  function values(phase: Phase, overrides: Values = {}): Values {
    if (!schema.value || !draft.value) throw new Error(`${modelLabel(family)} workspace has not loaded`)
    return checkedValues(schema.value[phase], modelActionValues(family, phase, schema.value[phase], {
      ...draft.value[phase], ...draft.value.models, ...overrides,
    }))
  }
  async function loadProfileData(name: string, method: string): Promise<boolean> {
    if (!schema.value) throw new Error(`${modelLabel(family)} schema has not loaded`)
    const seq = ++loadSeq
    const result = await requestJson(`${apiRoot}/profiles/${encodeURIComponent(name)}`, {})
    if (seq !== loadSeq) return false // a newer switch already superseded this request
    const parsed = readDraft(schema.value, JSON.stringify(result))
    checkSelectedMethod(parsed, method)
    localStorage.setItem(storageKey, name)
    draft.value = parsed
    saved.value = JSON.stringify(parsed)
    profile.value = name
    cache.value = null
    return true
  }
  async function loadProfile(name: string): Promise<void> {
    if (await loadProfileData(name, configProfile.value ? configMethod.value : '')) {
      if (configProfile.value) configProfile.value = name
    }
  }
  async function loadProfileCatalog(): Promise<void> {
    await ensureLoaded()
    const seq = ++catalogSeq
    const names = profileNames(await requestJson(`${apiRoot}/profiles`, {}))
    const entries = await Promise.all(names.map(async name => {
      const result = await requestJson(`${apiRoot}/profiles/${encodeURIComponent(name)}`, {})
      const parsed = readDraft(schema.value!, JSON.stringify(result))
      const method = String(parsed.train.algo)
      validateMethod(method)
      return [name, method] as const
    }))
    if (seq !== catalogSeq) return
    profiles.value = names
    profileAlgorithms.value = Object.fromEntries(entries)
  }
  function selectConfigMethod(method: string): void {
    validateMethod(method)
    ++loadSeq
    configMethod.value = method
    configProfile.value = ''
  }
  async function selectConfigProfile(name: string): Promise<void> {
    const method = configMethod.value
    validateMethod(method)
    if (profileAlgorithms.value[name] !== method) {
      throw new Error(`Profile ${name} does not belong to selected method ${method}`)
    }
    if (await loadProfileData(name, method)) configProfile.value = name
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
    if (configProfile.value) checkSelectedMethod(draft.value, configMethod.value)
    const body = await requestJson(`${apiRoot}/profiles/${encodeURIComponent(profile.value)}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(draft.value),
    })
    const parsed = readDraft(schema.value, JSON.stringify(body))
    if (configProfile.value) checkSelectedMethod(parsed, configMethod.value)
    draft.value = parsed
    saved.value = JSON.stringify(draft.value)
    localStorage.setItem(storageKey, profile.value)
    ++catalogSeq
    profiles.value = [...new Set([...profiles.value, profile.value])].sort()
    if (typeof parsed.train.algo === 'string') {
      profileAlgorithms.value = { ...profileAlgorithms.value, [profile.value]: parsed.train.algo }
    }
  }
  function createProfile(name: string): void {
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) throw new Error('Profile name: 1–64 letters, digits, underscores or hyphens')
    if (profiles.value.includes(name)) throw new Error(`Profile already exists: ${name}`)
    profile.value = name
    saved.value = ''
  }
  function beginConfigProfile(name: string, copyCurrent = true): void {
    validateMethod(configMethod.value)
    const nextDraft = copyCurrent && configProfile.value && draft.value
      ? readDraft(schema.value!, JSON.stringify(draft.value))
      : initialDraft(schema.value!)
    nextDraft.train.algo = configMethod.value
    createProfile(name)
    ++loadSeq
    draft.value = nextDraft
    configProfile.value = name
    cache.value = null
  }
  async function createConfigProfile(name: string, copyCurrent = true): Promise<void> {
    const previous = {
      profile: profile.value, draft: draft.value, saved: saved.value, cache: cache.value,
      configMethod: configMethod.value, configProfile: configProfile.value,
      profiles: profiles.value, profileAlgorithms: profileAlgorithms.value,
    }
    try {
      beginConfigProfile(name, copyCurrent)
      await save()
    } catch (cause) {
      profile.value = previous.profile
      draft.value = previous.draft
      saved.value = previous.saved
      cache.value = previous.cache
      configMethod.value = previous.configMethod
      configProfile.value = previous.configProfile
      profiles.value = previous.profiles
      profileAlgorithms.value = previous.profileAlgorithms
      throw cause
    }
  }
  function discard(): void {
    if (!schema.value) throw new Error(`${modelLabel(family)} schema has not loaded`)
    const restored = saved.value ? readDraft(schema.value, saved.value) : initialDraft(schema.value)
    if (configProfile.value) {
      if (saved.value) checkSelectedMethod(restored, configMethod.value)
      else restored.train.algo = configMethod.value
    }
    draft.value = restored
    cache.value = null
  }
  function importLegacy(): void {
    if (!schema.value) throw new Error(`${modelLabel(family)} schema has not loaded`)
    const text = localStorage.getItem(draftKey)
    if (text === null) throw new Error('No legacy Qwen form is stored')
    const imported = readDraft(schema.value, text)
    if (configProfile.value) checkSelectedMethod(imported, configMethod.value)
    draft.value = imported
  }
  function resetDefaults(): void {
    if (!schema.value) throw new Error(`${modelLabel(family)} schema has not loaded`)
    draft.value = initialDraft(schema.value)
    if (configProfile.value) draft.value.train.algo = configMethod.value
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
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values: values(phase, overrides), runtime_profile }),
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
  return { family, apiRoot, label: modelLabel(family), schema, draft, profile, profiles, configMethod, configProfile, profileAlgorithms, dirty, loading, busy, error, cache, legacyAvailable, saved,
    update, values, loadProfile, ensureLoaded, save, createProfile, discard, importLegacy, resetDefaults,
    loadProfileCatalog, selectConfigMethod, selectConfigProfile, beginConfigProfile, createConfigProfile,
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
