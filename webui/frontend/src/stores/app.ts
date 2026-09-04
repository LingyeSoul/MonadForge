import { defineStore } from 'pinia'
import { ref } from 'vue'

export type ThemeName = 'dark' | 'light'

export const useAppStore = defineStore('app', () => {
  const language = ref('en')
  const theme = ref<ThemeName>('dark')

  function init() {
    const saved = localStorage.getItem('monadforge-lang')
    if (saved) language.value = saved
    const savedTheme = localStorage.getItem('monadforge-theme')
    if (savedTheme === 'dark' || savedTheme === 'light') theme.value = savedTheme
  }

  function setLanguage(lang: string) {
    language.value = lang
    localStorage.setItem('monadforge-lang', lang)
  }

  function setTheme(name: ThemeName) {
    theme.value = name
    localStorage.setItem('monadforge-theme', name)
  }

  return { language, theme, init, setLanguage, setTheme }
})
