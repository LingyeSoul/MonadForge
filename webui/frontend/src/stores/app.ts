import { defineStore } from 'pinia'
import { ref } from 'vue'

export type ThemeName = 'dark' | 'light'

export interface WallpaperSettings {
  enabled: boolean
  src: string
}

export const useAppStore = defineStore('app', () => {
  const language = ref('en')
  const theme = ref<ThemeName>('dark')
  const glass = ref(false)
  const wallpaper = ref<WallpaperSettings>({ enabled: false, src: '' })

  function init() {
    const saved = localStorage.getItem('monadforge-lang')
    if (saved) language.value = saved
    const savedTheme = localStorage.getItem('monadforge-theme')
    if (savedTheme === 'dark' || savedTheme === 'light') theme.value = savedTheme
    glass.value = localStorage.getItem('monadforge-glass') === 'true'
    try {
      const savedWallpaper = JSON.parse(localStorage.getItem('monadforge-wallpaper') ?? 'null')
      if (
        savedWallpaper &&
        typeof savedWallpaper.enabled === 'boolean' &&
        typeof savedWallpaper.src === 'string'
      ) {
        wallpaper.value = { enabled: savedWallpaper.enabled, src: savedWallpaper.src }
      }
    } catch { /* corrupt entry — fall back to defaults */ }
  }

  function setLanguage(lang: string) {
    language.value = lang
    localStorage.setItem('monadforge-lang', lang)
  }

  function setTheme(name: ThemeName) {
    theme.value = name
    localStorage.setItem('monadforge-theme', name)
  }

  function setGlass(value: boolean) {
    glass.value = value
    localStorage.setItem('monadforge-glass', String(value))
  }

  function setWallpaper(value: WallpaperSettings) {
    wallpaper.value = value
    localStorage.setItem('monadforge-wallpaper', JSON.stringify(value))
  }

  return {
    language,
    theme,
    glass,
    wallpaper,
    init,
    setLanguage,
    setTheme,
    setGlass,
    setWallpaper,
  }
})
