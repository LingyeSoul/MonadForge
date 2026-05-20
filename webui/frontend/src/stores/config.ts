import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useAppStore } from './app'
import { getFrontendTranslations } from '../i18n'

export interface FieldMeta {
  key: string
  value: unknown
  default_value: unknown
  field_type: string
  description: string
  description_en?: string
  origin: string
  group: string
  is_virtual: boolean
  read_only: boolean
  options?: string[]
}

const _GROUP_LABELS: Record<string, Record<string, string>> = {
  en: {
    Architecture: 'Architecture',
    Training: 'Training',
    Performance: 'Performance',
    Paths: 'Paths',
    'Preview Sampling': 'Preview Sampling',
  },
  cn: {
    Architecture: '模型架构',
    Training: '训练参数',
    Performance: '性能优化',
    Paths: '路径配置',
    'Preview Sampling': '预览采样',
  },
}

function _t(key: string, lang: string): string {
  const msgs = getFrontendTranslations(lang)
  return msgs[key] ?? key
}

export const useConfigStore = defineStore('config', () => {
  const appStore = useAppStore()

  const fields = ref<FieldMeta[]>([])
  const variant = ref('')
  const preset = ref('default')
  const methods = ref<string[]>([])
  const variants = ref<string[]>([])
  const variantLabels = ref<Record<string, string>>({})
  const presets = ref<string[]>([])
  const loading = ref(false)
  const dirty = ref(false)
  const error = ref('')
  const editedValues = ref<Record<string, unknown>>({})

  const basicFields = computed(() => fields.value.filter(f => f.group === 'basic'))
  const advancedFields = computed(() => fields.value.filter(f => f.group !== 'basic'))

  const groupedAdvanced = computed(() => {
    const lang = appStore.language
    const labels = _GROUP_LABELS[lang] ?? _GROUP_LABELS.en
    const groups: Record<string, FieldMeta[]> = {}
    for (const f of advancedFields.value) {
      const raw = f.group || 'Other'
      const g = labels[raw] ?? raw
      if (!groups[g]) groups[g] = []
      groups[g].push(f)
    }
    return groups
  })

  async function fetchMethods() {
    try {
      const res = await fetch('/api/config/methods')
      const data = await res.json()
      methods.value = data.methods || []
    } catch (e) {
      const lang = appStore.language
      error.value = _t('cfgErrLoadMethods', lang).replace('{error}', String(e))
    }
  }

  async function fetchVariants(method: string) {
    try {
      const res = await fetch(`/api/config/variants?method=${encodeURIComponent(method)}`)
      const data = await res.json()
      variants.value = data.variants || []
      variantLabels.value = data.labels || {}
    } catch (e) {
      const lang = appStore.language
      error.value = _t('cfgErrLoadVariants', lang).replace('{error}', String(e))
    }
  }

  async function fetchPresets() {
    try {
      const res = await fetch('/api/config/presets')
      const data = await res.json()
      presets.value = data.presets || []
    } catch (e) {
      const lang = appStore.language
      error.value = _t('cfgErrLoadPresets', lang).replace('{error}', String(e))
    }
  }

  async function fetchMerged(v: string, pre: string) {
    loading.value = true
    error.value = ''
    try {
      const lang = appStore.language
      const res = await fetch(`/api/config/merged?variant=${encodeURIComponent(v)}&preset=${encodeURIComponent(pre)}&lang=${encodeURIComponent(lang)}`)
      const data = await res.json()
      fields.value = data.fields || []
      variant.value = v
      preset.value = pre
      editedValues.value = {}
      dirty.value = false
    } catch (e) {
      const lang = appStore.language
      error.value = _t('cfgErrLoadConfig', lang).replace('{error}', String(e))
    } finally {
      loading.value = false
    }
  }

  function setFieldValue(key: string, value: unknown) {
    editedValues.value[key] = value
    dirty.value = true
  }

  function getFieldValue(key: string): unknown {
    if (key in editedValues.value) return editedValues.value[key]
    const field = fields.value.find(f => f.key === key)
    return field?.value
  }

  async function save() {
    if (!dirty.value) return
    loading.value = true
    error.value = ''
    try {
      const lang = appStore.language
      const res = await fetch(`/api/config/method?variant=${encodeURIComponent(variant.value)}&preset=${encodeURIComponent(preset.value)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: editedValues.value }),
      })
      if (!res.ok) {
        const data = await res.json()
        const detail = data.detail || _t('cfgErrSaveDetail', lang)
        throw new Error(detail)
      }
      await fetchMerged(variant.value, preset.value)
    } catch (e) {
      const lang = appStore.language
      error.value = _t('cfgErrSave', lang).replace('{error}', String(e))
    } finally {
      loading.value = false
    }
  }

  async function createPreset(name: string, data: Record<string, unknown>): Promise<boolean> {
    try {
      const res = await fetch('/api/config/presets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, data }),
      })
      if (!res.ok) {
        const body = await res.json()
        throw new Error(body.detail || 'Failed to create preset')
      }
      const body = await res.json()
      presets.value = body.presets || []
      return true
    } catch (e) {
      error.value = String(e)
      return false
    }
  }

  async function deletePreset(name: string): Promise<boolean> {
    try {
      const res = await fetch(`/api/config/presets/${encodeURIComponent(name)}`, {
        method: 'DELETE',
      })
      if (!res.ok) {
        const body = await res.json()
        throw new Error(body.detail || 'Failed to delete preset')
      }
      const body = await res.json()
      presets.value = body.presets || []
      return true
    } catch (e) {
      error.value = String(e)
      return false
    }
  }

  return {
    fields, variant, preset, methods, variants, variantLabels, presets,
    loading, dirty, error, editedValues,
    basicFields, advancedFields, groupedAdvanced,
    fetchMethods, fetchVariants, fetchPresets, fetchMerged,
    setFieldValue, getFieldValue, save,
    createPreset, deletePreset,
  }
})
