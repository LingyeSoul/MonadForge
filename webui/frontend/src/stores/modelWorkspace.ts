import { defineStore } from 'pinia'
import { ref } from 'vue'

export type ModelFamily = 'anima' | 'qwen21'
export const modelLabel = (family: ModelFamily): string => family === 'qwen21' ? 'Qwen-Image 2.1' : 'Anima'

export const useModelWorkspace = defineStore('model-workspace', () => {
  const stored = localStorage.getItem('monadforge-model-family')
  if (stored !== null && stored !== 'anima' && stored !== 'qwen21') throw new Error('Invalid saved model family')
  const family = ref<ModelFamily>(stored ?? 'anima')
  function select(value: ModelFamily): void {
    localStorage.setItem('monadforge-model-family', value)
    family.value = value
  }
  return { family, select }
})
