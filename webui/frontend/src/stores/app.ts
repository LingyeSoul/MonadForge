import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import { makeWallpaperDisplay } from '../utils/wallpaper'

export type ThemeName = 'dark' | 'light'

export interface WallpaperSettings {
  enabled: boolean
  src: string
  blur: number
}

const DEFAULT_WALLPAPER: WallpaperSettings = { enabled: false, src: '', blur: 0 }

/** Parse and clamp a stored number; anything unusable falls back to `fallback`. */
function readNumber(raw: number | string | null | undefined, min: number, max: number, fallback: number) {
  if (raw === null || raw === undefined || raw === '') return fallback
  const n = Number(raw)
  if (!Number.isFinite(n)) return fallback
  return Math.min(max, Math.max(min, n))
}

export const useAppStore = defineStore('app', () => {
  const language = ref('en')
  const theme = ref<ThemeName>('dark')
  const glass = ref(false)
  const wallpaper = ref<WallpaperSettings>({ ...DEFAULT_WALLPAPER })
  // Glass surface opacity scale in percent — 100 keeps the default look.
  const surfaceOpacity = ref(100)
  // Wallpaper veil density in percent (dark theme; light theme adds +10).
  const veilOpacity = ref(55)
  // The blurred/prepared image actually painted as wallpaper. Kept in memory
  // only — `wallpaper.src` holds the persisted original, and `wallpaperDisplay`
  // is re-derived from it whenever src or blur changes. Empty when disabled.
  const wallpaperDisplay = ref('')

  let processStamp = 0
  async function reprocessWallpaper() {
    const { enabled, src, blur } = wallpaper.value
    const stamp = ++processStamp
    if (!enabled || !src) {
      wallpaperDisplay.value = ''
      return
    }
    try {
      const out = await makeWallpaperDisplay(src, blur)
      if (stamp === processStamp) wallpaperDisplay.value = out
    } catch {
      // Cross-origin sources the canvas cannot read back: fall back to the
      // sharp original rather than losing the wallpaper entirely.
      if (stamp === processStamp) wallpaperDisplay.value = src
    }
  }
  watch(
    () => [wallpaper.value.enabled, wallpaper.value.src, wallpaper.value.blur],
    reprocessWallpaper,
    { immediate: true },
  )

  function init() {
    const saved = localStorage.getItem('monadforge-lang')
    if (saved) language.value = saved
    const savedTheme = localStorage.getItem('monadforge-theme')
    if (savedTheme === 'dark' || savedTheme === 'light') theme.value = savedTheme
    glass.value = localStorage.getItem('monadforge-glass') === 'true'
    surfaceOpacity.value = readNumber(localStorage.getItem('monadforge-glass-surface'), 20, 100, 100)
    veilOpacity.value = readNumber(localStorage.getItem('monadforge-veil'), 0, 90, 55)
    try {
      const savedWallpaper = JSON.parse(localStorage.getItem('monadforge-wallpaper') ?? 'null')
      if (
        savedWallpaper &&
        typeof savedWallpaper.enabled === 'boolean' &&
        typeof savedWallpaper.src === 'string'
      ) {
        wallpaper.value = {
          enabled: savedWallpaper.enabled,
          src: savedWallpaper.src,
          blur: readNumber(savedWallpaper.blur, 0, 30, 0),
        }
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

  function setSurfaceOpacity(value: number) {
    surfaceOpacity.value = readNumber(value, 20, 100, 100)
    localStorage.setItem('monadforge-glass-surface', String(surfaceOpacity.value))
  }

  function setVeilOpacity(value: number) {
    veilOpacity.value = readNumber(value, 0, 90, 55)
    localStorage.setItem('monadforge-veil', String(veilOpacity.value))
  }

  function setWallpaper(value: WallpaperSettings) {
    wallpaper.value = {
      enabled: value.enabled,
      src: value.src,
      blur: readNumber(value.blur, 0, 30, 0),
    }
    localStorage.setItem('monadforge-wallpaper', JSON.stringify(wallpaper.value))
  }

  return {
    language,
    theme,
    glass,
    wallpaper,
    wallpaperDisplay,
    surfaceOpacity,
    veilOpacity,
    init,
    setLanguage,
    setTheme,
    setGlass,
    setSurfaceOpacity,
    setVeilOpacity,
    setWallpaper,
  }
})
