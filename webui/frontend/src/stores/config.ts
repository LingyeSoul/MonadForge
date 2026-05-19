import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface FieldMeta {
  key: string
  value: unknown
  default_value: unknown
  field_type: string
  description: string
  origin: string
  group: string
  is_virtual: boolean
}

export const useConfigStore = defineStore('config', () => {
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
    const groups: Record<string, FieldMeta[]> = {}
    for (const f of advancedFields.value) {
      const g = f.group || 'Other'
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
      error.value = `Failed to load methods: ${e}`
    }
  }

  async function fetchVariants(method: string) {
    try {
      const res = await fetch(`/api/config/variants?method=${encodeURIComponent(method)}`)
      const data = await res.json()
      variants.value = data.variants || []
      variantLabels.value = data.labels || {}
    } catch (e) {
      error.value = `Failed to load variants: ${e}`
    }
  }

  async function fetchPresets() {
    try {
      const res = await fetch('/api/config/presets')
      const data = await res.json()
      presets.value = data.presets || []
    } catch (e) {
      error.value = `Failed to load presets: ${e}`
    }
  }

  async function fetchMerged(v: string, pre: string) {
    loading.value = true
    error.value = ''
    try {
      const res = await fetch(`/api/config/merged?variant=${encodeURIComponent(v)}&preset=${encodeURIComponent(pre)}`)
      const data = await res.json()
      fields.value = data.fields || []
      variant.value = v
      preset.value = pre
      editedValues.value = {}
      dirty.value = false
    } catch (e) {
      error.value = `Failed to load config: ${e}`
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
      const res = await fetch(`/api/config/method`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ variant: variant.value, values: editedValues.value }),
      })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Save failed')
      }
      await fetchMerged(variant.value, preset.value)
    } catch (e) {
      error.value = `Save failed: ${e}`
    } finally {
      loading.value = false
    }
  }

  return {
    fields, variant, preset, methods, variants, variantLabels, presets,
    loading, dirty, error, editedValues,
    basicFields, advancedFields, groupedAdvanced,
    fetchMethods, fetchVariants, fetchPresets, fetchMerged,
    setFieldValue, getFieldValue, save,
  }
})
