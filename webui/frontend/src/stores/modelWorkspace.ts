import { defineStore } from 'pinia'
import { ref } from 'vue'

export type ModelFamily = 'anima' | 'qwen21' | 'moss'
export type RuntimeProfileName = 'default' | 'curation' | 'v100'
export interface RuntimeProfile { name: RuntimeProfileName; available: boolean; python: string | null; error: string | null }
export const modelLabel = (family: ModelFamily): string => family === 'moss' ? 'MOSS' : family === 'qwen21' ? 'Qwen-Image 2.1' : 'Anima'

export const useModelWorkspace = defineStore('model-workspace', () => {
  const stored = localStorage.getItem('monadforge-model-family')
  const family = ref<ModelFamily>(stored === 'qwen21' || stored === 'moss' ? stored : 'anima')
  const storedRuntime = localStorage.getItem('monadforge-runtime-profile')
  const runtimeProfile = ref<RuntimeProfileName>(storedRuntime === 'curation' || storedRuntime === 'v100' ? storedRuntime : 'default')
  const runtimes = ref<RuntimeProfile[]>([])
  const runtimeLoading = ref(false)
  const runtimeError = ref('')
  let runtimeRequest: Promise<void> | null = null
  async function loadRuntimes(): Promise<void> {
    if (runtimeRequest) return runtimeRequest
    runtimeLoading.value = true
    runtimeError.value = ''
    runtimeRequest = (async () => {
      try {
        const response = await fetch('/api/system/runtimes')
        const result = await response.json()
        if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`)
        if (!Array.isArray(result.runtimes) || result.runtimes.some((entry: RuntimeProfile) => !['default', 'curation', 'v100'].includes(entry.name) || typeof entry.available !== 'boolean')) throw new Error('Invalid runtime catalog')
        runtimes.value = result.runtimes
        const selected = runtimes.value.find(entry => entry.name === runtimeProfile.value)
        if (!selected?.available) runtimeError.value = selected?.error || `Runtime ${runtimeProfile.value} is unavailable`
      } catch (cause) {
        runtimeError.value = cause instanceof Error ? cause.message : String(cause)
        throw cause
      } finally { runtimeLoading.value = false; runtimeRequest = null }
    })()
    return runtimeRequest
  }
  function selectRuntime(name: RuntimeProfileName): void {
    const selected = runtimes.value.find(entry => entry.name === name)
    if (!selected?.available) throw new Error(selected?.error || `Runtime ${name} is unavailable`)
    runtimeProfile.value = name
    localStorage.setItem('monadforge-runtime-profile', name)
    runtimeError.value = ''
  }
  async function runtimeForSubmit(): Promise<RuntimeProfileName> {
    if (!runtimes.value.length) await loadRuntimes()
    const selected = runtimes.value.find(entry => entry.name === runtimeProfile.value)
    if (!selected?.available) throw new Error(selected?.error || `Runtime ${runtimeProfile.value} is unavailable`)
    return selected.name
  }
  function select(value: ModelFamily): void {
    localStorage.setItem('monadforge-model-family', value)
    family.value = value
  }
  return { family, select, runtimeProfile, runtimes, runtimeLoading, runtimeError, loadRuntimes, selectRuntime, runtimeForSubmit }
})

export function taskModelFamily(command: string): ModelFamily {
  return command.startsWith('moss-') ? 'moss' : command.startsWith('qwen21-') ? 'qwen21' : 'anima'
}
