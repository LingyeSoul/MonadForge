import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useAppStore } from './app'
import { getFrontendTranslations } from '../i18n'

export interface FieldMeta {
  label?: string
  nullable?: boolean
  path_kind?: 'file' | 'directory' | 'either'
  path_extensions?: string
  multiline?: boolean
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
  visible_when?: Record<string, unknown[]>[] | null
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
  ja: {
    Architecture: 'アーキテクチャ',
    Training: '学習パラメータ',
    Performance: 'パフォーマンス',
    Paths: 'パス設定',
    'Preview Sampling': 'プレビュー采样',
  },
  ko: {
    Architecture: '아키텍처',
    Training: '학습 매개변수',
    Performance: '성능 최적화',
    Paths: '경로 설정',
    'Preview Sampling': '미리보기 샘플링',
  },
}

// Groups localized via the i18n message catalog (cfgGroup* keys) rather than the
// _GROUP_LABELS table. 'Other' is the fallback for ungrouped fields; 'Resume &
// Warm-start' is the new resume/warm-start group. Routing them through
// getFrontendTranslations lets the active language (incl. en/cn) drive the label.
const _GROUP_I18N_KEYS: Record<string, string> = {
  Other: 'cfgGroupOther',
  'Resume & Warm-start': 'cfgGroupResume',
}

function _t(key: string, lang: string): string {
  const msgs = getFrontendTranslations(lang)
  return msgs[key] ?? key
}

export const useConfigStore = defineStore('config', () => {
  const appStore = useAppStore()

  const fields = ref<FieldMeta[]>([])
  const method = ref('')
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

  function isFieldVisible(field: FieldMeta): boolean {
    if (!field.visible_when) return true
    const normalize = (value: unknown): unknown => {
      if (value == null) return null
      if (typeof value !== 'string') return value
      if (/^(true|yes|1)$/i.test(value.trim())) return true
      if (/^(false|no|0)$/i.test(value.trim())) return false
      return value.trim()
    }
    // Explicit network_args win over top-level keys in the training intake.
    const networkArgs = getFieldValue('network_args')
    const overrides = new Map<string, string>()
    if (Array.isArray(networkArgs)) for (const item of networkArgs) {
      if (typeof item !== 'string' || !item.includes('=')) continue
      const index = item.indexOf('=')
      overrides.set(item.slice(0, index).trim(), item.slice(index + 1).trim())
    }
    return field.visible_when.some(clause => Object.entries(clause).every(([key, values]) => {
      const actual = normalize(overrides.has(key) ? overrides.get(key) : getFieldValue(key))
      return values.some(value => Object.is(normalize(value), actual))
    }))
  }

  const basicFields = computed(() => fields.value.filter(f => f.group === 'basic' && isFieldVisible(f)))
  const advancedFields = computed(() => fields.value.filter(f => f.group !== 'basic' && isFieldVisible(f)))

  const groupedAdvanced = computed(() => {
    const lang = appStore.language
    const labels = _GROUP_LABELS[lang] ?? _GROUP_LABELS.en
    const groups: Record<string, FieldMeta[]> = {}
    for (const f of advancedFields.value) {
      const raw = f.group || 'Other'
      // Prefer the i18n catalog (cfgGroup*) for groups that have a message key
      // (Other, Resume & Warm-start); fall back to the language label table,
      // then the raw backend group name.
      const i18nKey = _GROUP_I18N_KEYS[raw]
      const g = i18nKey ? _t(i18nKey, lang) : (labels[raw] ?? raw)
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

  async function fetchVariants(m: string) {
    method.value = m
    try {
      const res = await fetch(`/api/config/variants?method=${encodeURIComponent(m)}`)
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

  async function readMerged(v: string, pre: string): Promise<FieldMeta[]> {
    const lang = appStore.language
    const res = await fetch(`/api/config/merged?variant=${encodeURIComponent(v)}&preset=${encodeURIComponent(pre)}&lang=${encodeURIComponent(lang)}`)
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
    if (!Array.isArray(data.fields)) throw new Error('Invalid configuration response')
    return data.fields
  }

  async function fetchMerged(v: string, pre: string) {
    loading.value = true
    error.value = ''
    try {
      const nextFields = await readMerged(v, pre)
      fields.value = nextFields
      variant.value = v
      preset.value = pre
      editedValues.value = {}
      dirty.value = false
    } catch (e) {
      const lang = appStore.language
      error.value = _t('cfgErrLoadConfig', lang).replace('{error}', String(e))
      throw e
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

  async function save(extraArgs?: string) {
    if (!dirty.value && !extraArgs) return
    loading.value = true
    error.value = ''
    const savedVariant = variant.value
    const savedPreset = preset.value
    const submitted = JSON.parse(JSON.stringify(editedValues.value)) as Record<string, unknown>
    try {
      const lang = appStore.language
      const body: Record<string, unknown> = { data: submitted }
      if (extraArgs) body.data = { ...submitted, extra_args: extraArgs }
      const res = await fetch(`/api/config/method?variant=${encodeURIComponent(savedVariant)}&preset=${encodeURIComponent(savedPreset)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const data = await res.json()
        const detail = data.detail || _t('cfgErrSaveDetail', lang)
        throw new Error(detail)
      }
      const nextFields = await readMerged(savedVariant, savedPreset)
      if (variant.value !== savedVariant || preset.value !== savedPreset) return
      // A caller can edit the shared store while this request is pending.
      // Only acknowledge the values actually sent to the server.
      const pending = Object.fromEntries(Object.entries(editedValues.value)
        .filter(([key, value]) => JSON.stringify(value) !== JSON.stringify(submitted[key])))
      fields.value = nextFields
      editedValues.value = pending
      dirty.value = Object.keys(pending).length > 0
    } catch (e) {
      const lang = appStore.language
      error.value = _t('cfgErrSave', lang).replace('{error}', String(e))
      throw e
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
    fields, method, variant, preset, methods, variants, variantLabels, presets,
    loading, dirty, error, editedValues,
    basicFields, advancedFields, groupedAdvanced,
    fetchMethods, fetchVariants, fetchPresets, fetchMerged,
    setFieldValue, getFieldValue, isFieldVisible, save,
    createPreset, deletePreset,
  }
})
