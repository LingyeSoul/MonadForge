import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useAppStore = defineStore('app', () => {
  const language = ref('en')
  const theme = ref<'dark' | 'light'>('dark')

  function init() {
    const saved = localStorage.getItem('monadforge-lang')
    if (saved) language.value = saved
  }

  function setLanguage(lang: string) {
    language.value = lang
    localStorage.setItem('monadforge-lang', lang)
  }

  return { language, theme, init, setLanguage }
})
