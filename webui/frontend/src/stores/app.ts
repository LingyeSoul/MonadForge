import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useAppStore = defineStore('app', () => {
  const language = ref('en')
  const theme = ref<'dark' | 'light'>('dark')
  const taskDrawer = ref(false)

  function init() {
    const saved = localStorage.getItem('monadforge-lang')
    if (saved) language.value = saved
  }

  function setLanguage(lang: string) {
    language.value = lang
    localStorage.setItem('monadforge-lang', lang)
  }

  function toggleTaskDrawer() {
    taskDrawer.value = !taskDrawer.value
  }

  return { language, theme, taskDrawer, init, setLanguage, toggleTaskDrawer }
})
