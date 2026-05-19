import { ref, computed } from 'vue'
import { useAppStore } from '../stores/app'
import { getFrontendTranslations, type TranslationMessages } from '../i18n'

const backendTranslations = ref<TranslationMessages>({})
const languages = ref<string[]>(['en', 'cn'])
const loaded = ref(false)

export async function initI18n() {
  const appStore = useAppStore()
  try {
    const langRes = await fetch('/api/i18n')
    const langData = await langRes.json()
    languages.value = langData.languages || ['en', 'cn']
  } catch {
    // fallback to en/cn
  }
  await loadBackendTranslations(appStore.language)
  loaded.value = true
}

async function loadBackendTranslations(lang: string) {
  try {
    const res = await fetch(`/api/i18n/${lang}`)
    backendTranslations.value = await res.json()
  } catch {
    backendTranslations.value = {}
  }
}

export function useI18n() {
  const appStore = useAppStore()

  const frontend = computed(() => getFrontendTranslations(appStore.language))

  /**
   * Translate a key. Frontend translations take priority over backend.
   * Supports {param} interpolation.
   */
  function t(key: string, params?: Record<string, string | number>): string {
    let text = frontend.value[key] ?? backendTranslations.value[key] ?? key
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        text = text.replace(`{${k}}`, String(v))
      }
    }
    return text
  }

  async function setLanguage(lang: string) {
    appStore.setLanguage(lang)
    await loadBackendTranslations(lang)
  }

  return { t, setLanguage, languages, loaded }
}
